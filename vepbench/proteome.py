"""Defining a protein set and recording it as a manifest.

Everything downstream is driven by a manifest: a table with one row per
protein, written once and then treated as fixed. That matters because the
analysis takes days of compute spread over thousands of array tasks, and a
set that quietly changes between stages makes the result impossible to
interpret. The manifest records the exact UniProt query and the date it ran,
so a rerun either reproduces the same set or says plainly that it did not.

The default set is the reviewed human proteins carrying UniProt's
Transmembrane keyword, KW-0812. The alternative, the subcellular-location
term for 'Membrane', is a smaller and looser set; KW-0812 is backed by
per-residue TRANSMEM features, which are the same annotations
``features.py`` already uses for the topology columns.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from vepbench import HTTP_HEADERS
from vepbench.config import PATHS, Paths, write_json

log = logging.getLogger(__name__)

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_TIMEOUT = 300

#: Named protein sets. The key is what a user passes to ``--set``.
QUERIES: dict[str, str] = {
    "human_membrane": "(keyword:KW-0812) AND (organism_id:9606) AND (reviewed:true)",
    "yeast_membrane": "(keyword:KW-0812) AND (organism_id:559292) AND (reviewed:true)",
    "human_multipass": (
        "(keyword:KW-0812) AND (keyword:KW-0472) AND (organism_id:9606) AND (reviewed:true)"
    ),
    "human_gpcr": "(family:\"g-protein coupled receptor 1 family\") AND (organism_id:9606) AND (reviewed:true)",
    "human_reviewed": "(organism_id:9606) AND (reviewed:true)",
}

_FIELDS = (
    "accession,id,gene_primary,protein_name,length,"
    "ft_transmem,ft_intramem,cc_subcellular_location"
)

_TRANSMEM = re.compile(r"TRANSMEM (\d+)\.\.(\d+)")
_INTRAMEM = re.compile(r"INTRAMEM (\d+)\.\.(\d+)")
# ECO:0000269 is experimental evidence; ECO:0000255 is a sequence-model
# prediction. Most membrane topology is the latter, and it is worth being
# able to stratify on which.
_EXPERIMENTAL = re.compile(r"ECO:0000269")

MANIFEST_COLUMNS = [
    "uniprot_id", "entry_name", "gene", "protein_name", "length",
    "n_tm_segments", "tm_residues", "n_intramem_segments",
    "tm_evidence", "membrane_topology",
]


class ProteomeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestInfo:
    """What a manifest is and where it came from."""

    name: str
    query: str
    n_proteins: int
    n_residues: int
    n_substitutions: int
    retrieved: str
    path: Path

    def summary(self) -> str:
        return (
            f"{self.name}: {self.n_proteins:,} proteins, {self.n_residues:,} residues, "
            f"{self.n_substitutions:,} possible substitutions ({self.retrieved})"
        )


#: Accessions per UniProt request when building a manifest from a list.
#: 150 is refused outright; 100 works. The batch is verified either way,
#: because a request can come back 200 with only part of what was asked
#: for, and a manifest quietly missing proteins is worse than a slow one.
ACCESSION_BATCH = 100


def read_accession_list(path: Path) -> list[str]:
    """Read a file of accessions, one per line, ignoring blanks and comments."""
    accessions: list[str] = []
    seen: set[str] = set()
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Tolerate a TSV or a FASTA-style header, taking the first field.
        accession = line.split()[0].split("|")[1] if line.startswith(">sp|") else line.split()[0]
        if accession not in seen:
            seen.add(accession)
            accessions.append(accession)
    if not accessions:
        raise ProteomeError(f"no accessions found in {path}")
    return accessions


def fetch_manifest_for(
    accessions: list[str], name: str, *, paths: Paths = PATHS, force: bool = False,
    timeout: int = DEFAULT_TIMEOUT, min_length: int = 1, max_length: int | None = None,
) -> tuple[pd.DataFrame, ManifestInfo]:
    """Build a manifest for an explicit set of accessions.

    The same fields as a query-derived manifest, so everything downstream
    behaves identically. Accessions UniProt does not return are reported
    rather than silently dropped: an obsolete or mistyped one would
    otherwise shrink the set without saying so.
    """
    directory = paths.outputs / "proteome" / name
    manifest_path = directory / "manifest.tsv"
    if manifest_path.exists() and not force:
        frame = pd.read_csv(manifest_path, sep="\t")
        log.info("reusing the manifest at %s (%d proteins); --force to rebuild",
                 manifest_path, len(frame))
        return frame, _info_from_frame(name, f"{len(accessions)} accessions",
                                       frame, manifest_path, "cached")

    rows: list[dict] = []
    found: set[str] = set()
    for start in range(0, len(accessions), ACCESSION_BATCH):
        batch = accessions[start : start + ACCESSION_BATCH]
        log.info("fetching %d-%d of %d accessions",
                 start + 1, start + len(batch), len(accessions))
        for row in _fetch_accession_batch(batch, timeout=timeout):
            rows.append(row)
            found.add(row["uniprot_id"])

        # A request can answer 200 with only part of the batch. Ask for
        # anything that did not come back one at a time, where a failure is
        # unambiguous.
        stragglers = [a for a in batch if a not in found]
        if stragglers:
            log.info("  %d of %d did not come back; retrying individually",
                     len(stragglers), len(batch))
            for accession in stragglers:
                for row in _fetch_accession_batch([accession], timeout=timeout):
                    rows.append(row)
                    found.add(row["uniprot_id"])

    missing = [a for a in accessions if a not in found]
    if missing:
        log.warning(
            "%d of %d accessions were not returned by UniProt and are not in the "
            "manifest (obsolete, demerged or mistyped): %s",
            len(missing), len(accessions), ", ".join(missing[:8]),
        )

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    before = len(frame)
    frame = frame.loc[frame["length"] >= min_length]
    if max_length is not None:
        frame = frame.loc[frame["length"] <= max_length]
    if len(frame) != before:
        log.info("length filter kept %d of %d proteins", len(frame), before)

    frame = frame.sort_values("uniprot_id").reset_index(drop=True)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tsv.tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    tmp.replace(manifest_path)

    info = _info_from_frame(
        name, f"{len(accessions)} accessions from a list", frame, manifest_path,
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(directory / "manifest.json", {
        "name": info.name, "query": info.query, "retrieved": info.retrieved,
        "n_proteins": info.n_proteins, "n_residues": info.n_residues,
        "n_substitutions": info.n_substitutions,
        "n_requested": len(accessions), "not_found": missing,
        "min_length": min_length, "max_length": max_length, "fields": _FIELDS,
    })
    log.info("wrote %s", info.summary())
    return frame, info


def _fetch_accession_batch(batch: list[str], *, timeout: int) -> list[dict]:
    """One UniProt request for a batch of accessions, parsed into rows."""
    query = " OR ".join(f"accession:{a}" for a in batch)
    response = requests.get(
        UNIPROT_STREAM,
        params={"query": query, "fields": _FIELDS, "format": "tsv"},
        headers=HTTP_HEADERS, timeout=timeout,
    )
    if response.status_code != 200:
        log.warning("UniProt answered %d for a batch of %d accessions",
                    response.status_code, len(batch))
        return []
    lines = response.text.splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    out = []
    for line in lines[1:]:
        row = _parse_row(dict(zip(header, line.split("\t"), strict=False)))
        if row["uniprot_id"]:
            out.append(row)
    return out


def resolve_query(name_or_query: str) -> tuple[str, str]:
    """Turn a set name or a raw UniProt query into ``(name, query)``."""
    if name_or_query in QUERIES:
        return name_or_query, QUERIES[name_or_query]
    if "(" in name_or_query or ":" in name_or_query:
        return "custom", name_or_query
    raise ProteomeError(
        f"unknown protein set {name_or_query!r}; known sets are "
        f"{', '.join(sorted(QUERIES))}, or pass a UniProt query string"
    )


def count_matches(query: str, *, timeout: int = 60) -> int:
    """How many entries the query returns, without downloading them."""
    response = requests.get(
        UNIPROT_SEARCH, params={"query": query, "size": 1},
        headers=HTTP_HEADERS, timeout=timeout,
    )
    response.raise_for_status()
    total = response.headers.get("x-total-results")
    if total is None:
        raise ProteomeError("UniProt did not report a result count for this query")
    return int(total)


def _topology_label(location: str, n_tm: int) -> str:
    """A coarse class for stratifying plots, from the subcellular-location text."""
    text = (location or "").lower()
    if "multi-pass" in text or n_tm > 1:
        return "multipass"
    if "single-pass" in text or n_tm == 1:
        return "singlepass"
    if "peripheral" in text:
        return "peripheral"
    if "lipid-anchor" in text or "gpi-anchor" in text:
        return "lipid_anchored"
    return "unclassified"


def _parse_row(row: dict) -> dict:
    transmem = row.get("Transmembrane") or ""
    intramem = row.get("Intramembrane") or ""
    segments = [(int(a), int(b)) for a, b in _TRANSMEM.findall(transmem)]
    n_tm = len(segments)
    return {
        "uniprot_id": row.get("Entry", ""),
        "entry_name": row.get("Entry Name", "") or "",
        "gene": row.get("Gene Names (primary)", "") or "",
        "protein_name": (row.get("Protein names", "") or "").split(" (")[0],
        "length": int(row.get("Length") or 0),
        "n_tm_segments": n_tm,
        "tm_residues": sum(b - a + 1 for a, b in segments),
        "n_intramem_segments": len(_INTRAMEM.findall(intramem)),
        "tm_evidence": "experimental" if _EXPERIMENTAL.search(transmem) else "predicted",
        "membrane_topology": _topology_label(
            row.get("Subcellular location [CC]", "") or "", n_tm
        ),
    }


def fetch_manifest(
    name_or_query: str = "human_membrane", *, paths: Paths = PATHS,
    force: bool = False, timeout: int = DEFAULT_TIMEOUT,
    min_length: int = 1, max_length: int | None = None,
) -> tuple[pd.DataFrame, ManifestInfo]:
    """Fetch the protein set and write ``proteome/<name>/manifest.tsv``.

    An existing manifest is reused unless ``force`` is given, because every
    later stage keys off it and re-fetching mid-analysis would silently move
    the goalposts.
    """
    name, query = resolve_query(name_or_query)
    directory = paths.outputs / "proteome" / name
    manifest_path = directory / "manifest.tsv"
    info_path = directory / "manifest.json"

    if manifest_path.exists() and not force:
        frame = pd.read_csv(manifest_path, sep="\t")
        log.info("reusing the manifest at %s (%d proteins); --force to refetch",
                 manifest_path, len(frame))
        return frame, _info_from_frame(name, query, frame, manifest_path, "cached")

    log.info("fetching %s from UniProt: %s", name, query)
    response = requests.get(
        UNIPROT_STREAM,
        params={"query": query, "fields": _FIELDS, "format": "tsv"},
        headers=HTTP_HEADERS, timeout=timeout,
    )
    response.raise_for_status()

    lines = response.text.splitlines()
    if len(lines) < 2:
        raise ProteomeError(f"UniProt returned no entries for {query!r}")
    header = lines[0].split("\t")
    rows = [_parse_row(dict(zip(header, line.split("\t"), strict=False))) for line in lines[1:]]
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    frame = frame.loc[frame["uniprot_id"].astype(bool)]

    before = len(frame)
    frame = frame.loc[frame["length"] >= min_length]
    if max_length is not None:
        frame = frame.loc[frame["length"] <= max_length]
    if len(frame) != before:
        log.info("length filter kept %d of %d proteins", len(frame), before)

    frame = frame.sort_values("uniprot_id").reset_index(drop=True)
    directory.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".tsv.tmp")
    frame.to_csv(tmp, sep="\t", index=False)
    tmp.replace(manifest_path)

    info = _info_from_frame(name, query, frame, manifest_path,
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(info_path, {
        "name": info.name, "query": info.query, "retrieved": info.retrieved,
        "n_proteins": info.n_proteins, "n_residues": info.n_residues,
        "n_substitutions": info.n_substitutions,
        "min_length": min_length, "max_length": max_length,
        "fields": _FIELDS,
    })
    log.info("wrote %s", info.summary())
    return frame, info


def _info_from_frame(
    name: str, query: str, frame: pd.DataFrame, path: Path, retrieved: str
) -> ManifestInfo:
    residues = int(frame["length"].sum()) if len(frame) else 0
    return ManifestInfo(
        name=name, query=query, n_proteins=len(frame), n_residues=residues,
        n_substitutions=residues * 19, retrieved=retrieved, path=path,
    )


def load_manifest(name: str = "human_membrane", *, paths: Paths = PATHS) -> pd.DataFrame:
    """Read a manifest written earlier, with a useful error when it is absent."""
    path = paths.outputs / "proteome" / name / "manifest.tsv"
    if not path.exists():
        raise ProteomeError(
            f"no manifest for {name!r} at {path}; run `vepbench proteome define --set {name}`"
        )
    return pd.read_csv(path, sep="\t")


def manifest_directory(name: str = "human_membrane", *, paths: Paths = PATHS) -> Path:
    return paths.outputs / "proteome" / name
