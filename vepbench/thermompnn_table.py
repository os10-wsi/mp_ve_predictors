"""Published ThermoMPNN predictions, and deciding when to compute instead.

A precomputed table covers some proteins and not others, so every protein
asks the same question: is it in there? If yes the published numbers are
used, which is both free and closer to what the authors released. If no,
ThermoMPNN runs.

That question has to be answered honestly. A table that covers a protein
only partially is the dangerous case: taking it at face value gives a
column that looks complete and silently stops at whatever the table
happened to hold. So coverage is measured against the variants actually
asked for, and anything below ``MIN_COVERAGE`` is treated as absent and
computed instead.

The archive's internal layout is not fixed by us, so the reader accepts
the several shapes these tables come in rather than demanding one. What
it will not do is guess: a file whose columns it cannot identify raises
rather than returning an empty table that would read as "not covered".
"""

from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vepbench.config import PATHS, Paths

log = logging.getLogger(__name__)

ZENODO_RECORD = "https://zenodo.org/records/18381534"

#: Below this share of the requested variants, the table is not worth
#: using and ThermoMPNN is run instead. A partial column is worse than no
#: column: it looks complete.
MIN_COVERAGE = 0.5

#: Column names these tables are published under, lowercased.
_ID_COLUMNS = ("uniprot_id", "uniprot", "accession", "protein", "uniprotkb", "id")
_VARIANT_COLUMNS = ("variant", "mutation", "mutant", "substitution", "aa_change")
_SCORE_COLUMNS = ("ddg", "ddg_pred", "thermompnn", "ddG", "prediction", "score",
                  "predicted_ddg")
_POSITION_COLUMNS = ("position", "pos", "resnum", "residue_number")
_WT_COLUMNS = ("wt_aa", "wt", "wildtype", "wild_type", "ref")
_MUT_COLUMNS = ("mut_aa", "mut", "mutant_aa", "alt", "variant_aa")

_VARIANT_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


class ThermoMPNNTableError(RuntimeError):
    pass


@dataclass
class Coverage:
    """What the table had for one protein."""

    uniprot_id: str
    requested: int
    matched: int

    @property
    def fraction(self) -> float:
        return self.matched / self.requested if self.requested else 0.0

    @property
    def usable(self) -> bool:
        return self.fraction >= MIN_COVERAGE

    def describe(self) -> str:
        return (f"{self.uniprot_id}: the published table covers {self.matched:,} of "
                f"{self.requested:,} variants ({100 * self.fraction:.1f}%)")


def archive_path(paths: Paths = PATHS) -> Path:
    """Where the downloaded archive lives."""
    return paths.weights / "thermompnn" / "thermompnn_predictions.tsv.gz"


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_table(path: Path) -> pd.DataFrame:
    """Read the published table into uniprot_id, position, wt_aa, mut_aa, ddg.

    Accepts either a single ``variant`` column such as ``A123V`` or separate
    position and residue columns, and either tab or comma separation.
    """
    path = Path(path)
    if not path.exists():
        raise ThermoMPNNTableError(
            f"no ThermoMPNN table at {path}. Download it from {ZENODO_RECORD} "
            f"or pass --thermompnn-table."
        )
    with _open(path) as handle:
        head = handle.readline()
    separator = "\t" if head.count("\t") >= head.count(",") else ","

    frame = pd.read_csv(path, sep=separator, dtype=str, na_filter=False,
                        compression="gzip" if str(path).endswith(".gz") else None)
    columns = list(frame.columns)

    id_column = _pick(columns, _ID_COLUMNS)
    score_column = _pick(columns, _SCORE_COLUMNS)
    if id_column is None or score_column is None:
        raise ThermoMPNNTableError(
            f"{path} does not look like a ThermoMPNN table: expected an accession "
            f"column (one of {', '.join(_ID_COLUMNS)}) and a ddG column (one of "
            f"{', '.join(_SCORE_COLUMNS)}), found {columns}"
        )

    out = pd.DataFrame({"uniprot_id": frame[id_column].str.strip()})
    variant_column = _pick(columns, _VARIANT_COLUMNS)
    if variant_column is not None:
        parsed = frame[variant_column].str.strip().str.upper().str.extract(
            _VARIANT_PATTERN)
        out["wt_aa"], out["position"], out["mut_aa"] = parsed[0], parsed[1], parsed[2]
    else:
        position_column = _pick(columns, _POSITION_COLUMNS)
        wt_column = _pick(columns, _WT_COLUMNS)
        mut_column = _pick(columns, _MUT_COLUMNS)
        if not (position_column and wt_column and mut_column):
            raise ThermoMPNNTableError(
                f"{path} has no variant column and no position/wt/mut triple; "
                f"found {columns}"
            )
        out["position"] = frame[position_column]
        out["wt_aa"] = frame[wt_column].str.strip().str.upper()
        out["mut_aa"] = frame[mut_column].str.strip().str.upper()

    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    out["ddg"] = pd.to_numeric(frame[score_column], errors="coerce")
    out = out.dropna(subset=["uniprot_id", "position", "wt_aa", "mut_aa", "ddg"])
    out["position"] = out["position"].astype(int)
    log.info("read %d ThermoMPNN predictions covering %d proteins from %s",
             len(out), out["uniprot_id"].nunique(), path)
    return out


def index_by_protein(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """``{uniprot_id: {variant: ddg}}`` for fast per-protein lookup."""
    index: dict[str, dict[str, float]] = {}
    for accession, group in table.groupby("uniprot_id"):
        index[str(accession)] = {
            f"{w}{int(p)}{m}": float(d) for w, p, m, d in zip(
                group["wt_aa"], group["position"], group["mut_aa"], group["ddg"],
                strict=True)
        }
    return index


def lookup(
    variants: pd.DataFrame, published: dict[str, float], uniprot_id: str
) -> tuple[pd.Series, Coverage]:
    """Score the requested variants from a protein's published entries."""
    keys = [f"{w}{int(p)}{m}" for w, p, m in
            zip(variants["wt_aa"], variants["position"], variants["mut_aa"],
                strict=True)]
    values = pd.Series([published.get(k, float("nan")) for k in keys],
                       index=variants.index, dtype=float)
    coverage = Coverage(uniprot_id=uniprot_id, requested=len(variants),
                        matched=int(values.notna().sum()))
    return values, coverage
