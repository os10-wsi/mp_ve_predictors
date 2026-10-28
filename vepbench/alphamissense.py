"""AlphaMissense precomputed substitution scores.

AlphaMissense ships every possible substitution for the human proteome as
one 1.2 GB table keyed on ``uniprot_id`` and ``protein_variant`` -- exactly
the keys vepbench joins on -- so there is no model to install and no
inference to run. What there is instead is one streaming pass that slices
the table into a file per accession, after which scoring a protein is a
dictionary lookup.

The slicing is worth doing once rather than filtering the whole table per
protein: 5,000 passes over 1.2 GB of gzip is hours of pointless IO.

Licence: the AlphaMissense predictions are CC BY-NC-SA 4.0. That is
non-commercial, unlike the permissive licences on most of the other
predictors here, and it is recorded in the registry so it reaches
PREDICTORS.md and anything derived from it.
"""

from __future__ import annotations

import csv
import gzip
import logging
import shutil
import urllib.request
from pathlib import Path

from vepbench.config import PATHS, Paths

log = logging.getLogger(__name__)

ALPHAMISSENSE_URL = (
    "https://storage.googleapis.com/dm_alphamissense/"
    "AlphaMissense_aa_substitutions.tsv.gz"
)
DEFAULT_TIMEOUT = 3600
LICENSE = "CC BY-NC-SA 4.0 (non-commercial)"


class AlphaMissenseError(RuntimeError):
    pass


def slices_directory(paths: Paths = PATHS) -> Path:
    """Where the per-accession slices live, beside the other model data."""
    return paths.weights / "alphamissense"


def slice_path(uniprot_id: str, *, paths: Paths = PATHS) -> Path:
    return slices_directory(paths) / f"{uniprot_id}.csv"


def download(destination: Path | None = None, *, paths: Paths = PATHS,
             url: str = ALPHAMISSENSE_URL, force: bool = False) -> Path:
    """Fetch the substitution table, skipping it if already present."""
    destination = Path(destination) if destination else (
        slices_directory(paths) / "AlphaMissense_aa_substitutions.tsv.gz"
    )
    if destination.exists() and not force:
        log.info("using the cached AlphaMissense table at %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading AlphaMissense from %s (1.2 GB; this takes a while)", url)
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT) as response, \
                tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=4 << 20)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise AlphaMissenseError(
            f"could not download AlphaMissense from {url}: {exc}. If this machine "
            f"cannot reach storage.googleapis.com, fetch the file elsewhere and pass "
            f"--alphamissense-file."
        ) from exc
    tmp.replace(destination)
    return destination


def build_slices(
    table: Path, accessions: list[str] | None = None, *, paths: Paths = PATHS,
    force: bool = False,
) -> dict[str, int]:
    """Split the table into one CSV per accession. Returns rows written per accession.

    ``accessions`` restricts the slicing to a protein set, which for the
    membrane proteome is a quarter of the file. Passing None slices
    everything, which is ~20,000 files and worth thinking about on a
    filesystem with an inode quota.

    The upstream table is sorted by accession, but nothing here depends on
    that: files are opened in append mode and the header is written once per
    accession, so an unsorted input still produces correct slices.
    """
    table = Path(table)
    if not table.exists():
        raise AlphaMissenseError(f"no AlphaMissense table at {table}")
    directory = slices_directory(paths)
    directory.mkdir(parents=True, exist_ok=True)

    wanted = set(accessions) if accessions is not None else None
    if wanted is not None and not force:
        remaining = {a for a in wanted if not slice_path(a, paths=paths).exists()}
        if not remaining:
            log.info("every requested AlphaMissense slice is already present")
            return {a: -1 for a in sorted(wanted)}
        log.info("%d of %d slices still to build", len(remaining), len(wanted))
        wanted = remaining

    counts: dict[str, int] = {}
    started: set[str] = set()
    current: str | None = None
    handle = None
    writer = None

    opener = gzip.open if table.name.endswith(".gz") else open
    with opener(table, "rt", encoding="utf-8", errors="replace") as source:
        reader = csv.reader(
            (line for line in source if not line.startswith("#")), delimiter="\t"
        )
        header = next(reader, None)
        if header is None or "uniprot_id" not in header:
            raise AlphaMissenseError(
                f"{table} does not look like AlphaMissense_aa_substitutions.tsv.gz "
                f"(header: {header})"
            )
        index = {name: i for i, name in enumerate(header)}
        try:
            for row in reader:
                if len(row) <= index["am_pathogenicity"]:
                    continue
                accession = row[index["uniprot_id"]]
                if wanted is not None and accession not in wanted:
                    continue
                if accession != current:
                    if handle is not None:
                        handle.close()
                    path = slice_path(accession, paths=paths)
                    fresh = accession not in started
                    handle = path.open("w" if fresh else "a", newline="")
                    writer = csv.writer(handle)
                    if fresh:
                        writer.writerow(["variant", "am_pathogenicity", "am_class"])
                        started.add(accession)
                        counts.setdefault(accession, 0)
                    current = accession
                writer.writerow([
                    row[index["protein_variant"]],
                    row[index["am_pathogenicity"]],
                    row[index["am_class"]],
                ])
                counts[accession] += 1
        finally:
            if handle is not None:
                handle.close()

    log.info("wrote %d AlphaMissense slices into %s", len(counts), directory)
    return counts


def load_slice(uniprot_id: str, *, paths: Paths = PATHS) -> dict[str, tuple[float, str]]:
    """Read one protein's scores as ``{variant: (pathogenicity, class)}``."""
    path = slice_path(uniprot_id, paths=paths)
    if not path.exists():
        return {}
    out: dict[str, tuple[float, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                out[row["variant"].strip().upper()] = (
                    float(row["am_pathogenicity"]), row.get("am_class", ""),
                )
            except (KeyError, ValueError):
                continue
    return out
