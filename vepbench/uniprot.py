"""UniProt REST access: canonical sequence, domains and membrane topology."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import requests

from vepbench import HTTP_HEADERS
from vepbench.config import PATHS, Paths, read_json, write_json

log = logging.getLogger(__name__)

UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
INTERPRO_API = "https://www.ebi.ac.uk/interpro/api"
DEFAULT_TIMEOUT = 60

_FEATURE_TYPES = {
    "Transmembrane": "transmembrane",
    "Intramembrane": "intramembrane",
    "Topological domain": "topology",
    "Domain": "domain",
    "Region": "region",
    "Signal": "signal_peptide",
    "Repeat": "repeat",
    "Coiled coil": "coiled_coil",
}


class UniProtError(RuntimeError):
    pass


@dataclass
class SequenceFeature:
    kind: str
    description: str
    start: int
    end: int

    def as_dict(self) -> dict:
        return {"kind": self.kind, "description": self.description,
                "start": self.start, "end": self.end}


@dataclass
class UniProtRecord:
    uniprot_id: str
    sequence: str
    entry_name: str = ""
    protein_name: str = ""
    organism: str = ""
    sequence_version: int | None = None
    sequence_checksum: str = ""
    features: list[SequenceFeature] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def features_of(self, kind: str) -> list[SequenceFeature]:
        return [f for f in self.features if f.kind == kind]

    @property
    def transmembrane_segments(self) -> list[tuple[int, int]]:
        return [(f.start, f.end) for f in self.features_of("transmembrane")]

    @property
    def is_membrane_protein(self) -> bool:
        return bool(self.transmembrane_segments) or bool(self.features_of("intramembrane"))

    def as_dict(self) -> dict:
        return {
            "uniprot_id": self.uniprot_id,
            "entry_name": self.entry_name,
            "protein_name": self.protein_name,
            "organism": self.organism,
            "sequence": self.sequence,
            "sequence_length": self.length,
            "sequence_version": self.sequence_version,
            "sequence_checksum": self.sequence_checksum,
            "features": [f.as_dict() for f in self.features],
            "is_membrane_protein": self.is_membrane_protein,
            "transmembrane_segments": self.transmembrane_segments,
        }


def _parse_entry(payload: dict) -> UniProtRecord:
    block = payload.get("sequence") or {}
    sequence = (block.get("value") or "").upper()
    if not sequence:
        raise UniProtError("UniProt entry has no sequence")

    description = payload.get("proteinDescription") or {}
    protein_name = ((description.get("recommendedName") or {}).get("fullName") or {}).get(
        "value", ""
    )
    if not protein_name:
        submitted = description.get("submissionNames") or []
        if submitted:
            protein_name = (submitted[0].get("fullName") or {}).get("value", "")

    features: list[SequenceFeature] = []
    for raw in payload.get("features", []) or []:
        kind = _FEATURE_TYPES.get(raw.get("type", ""))
        if kind is None:
            continue
        location = raw.get("location") or {}
        start = (location.get("start") or {}).get("value")
        end = (location.get("end") or {}).get("value")
        if start is None or end is None:
            continue  # fuzzy boundaries are not usable for numbering
        features.append(SequenceFeature(
            kind=kind, description=raw.get("description", "") or "",
            start=int(start), end=int(end),
        ))

    return UniProtRecord(
        uniprot_id=payload.get("primaryAccession", ""),
        sequence=sequence,
        entry_name=payload.get("uniProtkbId", "") or "",
        protein_name=protein_name,
        organism=((payload.get("organism") or {}).get("scientificName") or ""),
        sequence_version=block.get("version"),
        sequence_checksum=block.get("crc64", "") or block.get("md5", "") or "",
        features=features,
    )


def fetch_uniprot(
    uniprot_id: str, *, paths: Paths = PATHS, force: bool = False,
    offline: bool = False, timeout: int = DEFAULT_TIMEOUT,
) -> UniProtRecord:
    """Fetch and cache the canonical UniProt entry."""
    cache_path = paths.annotations / f"{uniprot_id}.json"
    # offline outranks force: forcing a prediction recompute must not re-fetch
    # inputs that were deliberately pinned to the cache.
    if cache_path.exists() and (not force or offline):
        return _record_from_cache(read_json(cache_path))
    if offline:
        raise UniProtError(
            f"{uniprot_id} is not cached and --offline was requested; "
            f"expected cache at {cache_path}"
        )

    log.info("fetching UniProt entry %s", uniprot_id)
    response = requests.get(
        f"{UNIPROT_REST}/{uniprot_id}.json", timeout=timeout,
        headers={**HTTP_HEADERS, "Accept": "application/json"},
    )
    if response.status_code == 404:
        raise UniProtError(f"UniProt accession {uniprot_id!r} not found")
    response.raise_for_status()
    record = _parse_entry(response.json())
    if not record.uniprot_id:
        record.uniprot_id = uniprot_id
    write_json(cache_path, record.as_dict())
    return record


def _record_from_cache(payload: dict) -> UniProtRecord:
    return UniProtRecord(
        uniprot_id=payload["uniprot_id"],
        sequence=payload["sequence"],
        entry_name=payload.get("entry_name", ""),
        protein_name=payload.get("protein_name", ""),
        organism=payload.get("organism", ""),
        sequence_version=payload.get("sequence_version"),
        sequence_checksum=payload.get("sequence_checksum", ""),
        features=[SequenceFeature(**f) for f in payload.get("features", [])],
    )


def read_fasta(path: Path) -> str:
    lines = Path(path).read_text().splitlines()
    seq = "".join(ln.strip() for ln in lines if ln and not ln.startswith(">"))
    if not seq:
        raise UniProtError(f"no sequence found in {path}")
    return seq.upper()


def fetch_interpro_domains(
    uniprot_id: str, *, paths: Paths = PATHS, force: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[SequenceFeature]:
    """Pfam domain boundaries, used when UniProt has no Domain feature.

    Returns an empty list when InterPro is unreachable; the caller then falls
    back to UniProt annotations or to the full-length sequence.
    """
    cache_path = paths.annotations / f"{uniprot_id}.interpro.json"
    if cache_path.exists() and not force:
        return [SequenceFeature(**f) for f in read_json(cache_path)["features"]]

    url = f"{INTERPRO_API}/entry/pfam/protein/uniprot/{uniprot_id}/"
    try:
        response = requests.get(
            url, timeout=timeout, headers={**HTTP_HEADERS, "Accept": "application/json"}
        )
        if response.status_code in (204, 404):
            features: list[SequenceFeature] = []
        else:
            response.raise_for_status()
            features = _parse_interpro(response.json())
    except requests.RequestException as exc:
        log.warning("InterPro lookup for %s failed: %s", uniprot_id, exc)
        return []

    write_json(cache_path, {"features": [f.as_dict() for f in features]})
    return features


def _parse_interpro(payload: dict) -> list[SequenceFeature]:
    features: list[SequenceFeature] = []
    for result in payload.get("results", []) or []:
        metadata = result.get("metadata") or {}
        label = f"{metadata.get('accession', '')} {metadata.get('name', '')}".strip()
        for protein in result.get("proteins", []) or []:
            for location in protein.get("entry_protein_locations", []) or []:
                for fragment in location.get("fragments", []) or []:
                    features.append(SequenceFeature(
                        kind="pfam_domain", description=label,
                        start=int(fragment["start"]), end=int(fragment["end"]),
                    ))
    return features


def choose_domain(
    record: UniProtRecord, *, override: tuple[int, int] | None = None,
    prefer: str = "auto", paths: Paths = PATHS,
) -> tuple[tuple[int, int], str]:
    """Pick the window used by the domain-level predictors.

    Returns the window and a note saying where it came from, recorded in
    metadata.json. For a membrane protein, ``prefer='transmembrane'`` spans the
    first to the last transmembrane segment, which is usually the window a
    biogenesis assay probes.
    """
    if override is not None:
        start, end = override
        if start < 1 or end > record.length or start >= end:
            raise UniProtError(
                f"domain override {start}-{end} is outside the sequence "
                f"(length {record.length})"
            )
        return (start, end), "user override (--domain)"

    if prefer == "transmembrane":
        segments = record.transmembrane_segments
        if segments:
            return (min(s for s, _ in segments), max(e for _, e in segments)), (
                f"span of {len(segments)} UniProt transmembrane segments"
            )

    uniprot_domains = record.features_of("domain")
    if uniprot_domains:
        longest = max(uniprot_domains, key=lambda f: f.end - f.start)
        return (longest.start, longest.end), f"UniProt Domain feature: {longest.description}"

    pfam = fetch_interpro_domains(record.uniprot_id, paths=paths)
    if pfam:
        longest = max(pfam, key=lambda f: f.end - f.start)
        return (longest.start, longest.end), f"Pfam/InterPro: {longest.description}"

    segments = record.transmembrane_segments
    if segments:
        return (min(s for s, _ in segments), max(e for _, e in segments)), (
            f"fallback: span of {len(segments)} UniProt transmembrane segments"
        )
    return (1, record.length), "fallback: full-length (no domain annotation found)"
