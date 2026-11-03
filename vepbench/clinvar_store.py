"""ClinVar labels, sliced one file per protein.

Parsing variant_summary takes minutes and covers the whole genome, so
doing it inside every array task would be absurd. It is parsed once and
written out per accession, exactly as the AlphaMissense table is, after
which a protein's labels are a single small read.

The mapping itself -- which is the part that can be wrong without looking
wrong -- stays in ``clinvar.py``. This module only decides where the
results live.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from vepbench.config import PATHS, Paths

log = logging.getLogger(__name__)

COLUMNS = ["variant", "position", "wt_aa", "mut_aa", "clinical_significance",
           "label", "review_stars", "phenotypes"]


def directory(paths: Paths = PATHS) -> Path:
    return paths.cache / "clinvar"


def slice_path(uniprot_id: str, *, paths: Paths = PATHS) -> Path:
    return directory(paths) / f"{uniprot_id}.tsv"


def write_slices(frame: pd.DataFrame, *, paths: Paths = PATHS) -> int:
    """Split a mapped ClinVar table into one file per protein."""
    target = directory(paths)
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    for accession, group in frame.groupby("uniprot_id"):
        path = slice_path(str(accession), paths=paths)
        keep = [c for c in COLUMNS if c in group.columns]
        tmp = path.with_name(path.name + ".tmp")
        group.loc[:, keep].to_csv(tmp, sep="\t", index=False)
        tmp.replace(path)
        written += 1
    log.info("wrote ClinVar slices for %d proteins into %s", written, target)
    return written


def load_for_protein(uniprot_id: str, *, paths: Paths = PATHS) -> pd.DataFrame | None:
    """One protein's ClinVar variants, or None when it has no slice.

    None and an empty table mean different things: no slice means ClinVar
    was never sliced for this protein, while an empty one means it was and
    the protein has no missense variants. Both end up unlabelled, but only
    the first is a reason to go and run the fetch.
    """
    path = slice_path(uniprot_id, paths=paths)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, sep="\t")
    except (OSError, ValueError) as exc:
        log.warning("unreadable ClinVar slice at %s (%s)", path, exc)
        return None
    if "label" in frame.columns:
        frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    return frame


def available(paths: Paths = PATHS) -> int:
    """How many proteins have a ClinVar slice."""
    target = directory(paths)
    return len(list(target.glob("*.tsv"))) if target.exists() else 0
