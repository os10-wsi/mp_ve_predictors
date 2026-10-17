"""Join predictions onto an existing DMS table.

Rows that cannot be matched are never dropped silently. Every input row comes
back with a ``match_status`` saying what happened and, for failures, a
``match_detail`` explaining why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vepbench.schema import HARMONIZED_SUFFIX, WIDE_KEY_COLUMNS
from vepbench.variants import VariantParseError, parse_variant

log = logging.getLogger(__name__)

MATCHED = "matched"
NO_PREDICTION = "no_prediction"
WT_MISMATCH = "wt_mismatch"
OUT_OF_RANGE = "out_of_range"
MULTI_MUTANT = "multi_mutant"
UNPARSEABLE = "unparseable"
SYNONYMOUS = "synonymous"
NONSENSE = "nonsense"

FAILURE_STATUSES = (
    NO_PREDICTION, WT_MISMATCH, OUT_OF_RANGE, MULTI_MUTANT, UNPARSEABLE, NONSENSE,
)


class MergeError(ValueError):
    pass


@dataclass
class MergeReport:
    total_rows: int
    counts: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def matched(self) -> int:
        return self.counts.get(MATCHED, 0)

    @property
    def unmatched(self) -> int:
        return self.total_rows - self.matched

    def as_dict(self) -> dict:
        return {"total_rows": self.total_rows, "matched": self.matched,
                "unmatched": self.unmatched, "counts": self.counts,
                "examples": self.examples}

    def summary(self) -> str:
        lines = [f"{self.matched}/{self.total_rows} DMS rows matched a prediction"]
        for status, count in sorted(self.counts.items()):
            if status == MATCHED or count == 0:
                continue
            example = ", ".join(self.examples.get(status, [])[:3])
            lines.append(f"  {status}: {count}" + (f" (e.g. {example})" if example else ""))
        return "\n".join(lines)


def _normalise_dms(
    dms: pd.DataFrame, *, variant_col: str | None, position_col: str | None,
    wt_col: str | None, mut_col: str | None,
) -> pd.DataFrame:
    out = dms.copy()
    positions: list[int | None] = []
    wts: list[str | None] = []
    muts: list[str | None] = []
    statuses: list[str] = []
    details: list[str] = []

    if variant_col is not None:
        if variant_col not in out.columns:
            raise MergeError(f"column {variant_col!r} not found in the DMS table")
        for raw in out[variant_col]:
            try:
                parsed = parse_variant(raw)
            except VariantParseError as exc:
                positions.append(None)
                wts.append(None)
                muts.append(None)
                message = str(exc)
                statuses.append(MULTI_MUTANT if "multi-mutant" in message else UNPARSEABLE)
                details.append(message)
                continue
            positions.append(parsed.position)
            wts.append(parsed.wt_aa)
            muts.append(parsed.mut_aa)
            statuses.append("")
            details.append("")
    else:
        for column, label in ((position_col, "position"), (wt_col, "wild type"),
                              (mut_col, "mutant")):
            if column is None:
                raise MergeError(
                    "give either --variant-col, or all of --position-col, --wt-col and "
                    f"--mut-col (missing the {label} column)"
                )
            if column not in out.columns:
                raise MergeError(f"column {column!r} not found in the DMS table")
        for raw_position, raw_wt, raw_mut in zip(
            out[position_col], out[wt_col], out[mut_col], strict=True
        ):
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                positions.append(None)
                wts.append(None)
                muts.append(None)
                statuses.append(UNPARSEABLE)
                details.append(f"position {raw_position!r} is not an integer")
                continue
            wt = str(raw_wt).strip().upper()
            mut = str(raw_mut).strip().upper()
            if len(wt) == 3:
                try:
                    wt = parse_variant(f"{wt}{position}Ala").wt_aa
                except VariantParseError:
                    pass
            if len(mut) == 3:
                try:
                    mut = parse_variant(f"Ala{position}{mut}").mut_aa
                except VariantParseError:
                    pass
            positions.append(position)
            wts.append(wt)
            muts.append(mut)
            statuses.append("")
            details.append("")

    out["_position"] = positions
    out["_wt_aa"] = wts
    out["_mut_aa"] = muts
    out["_status"] = statuses
    out["_detail"] = details
    return out


def merge_with_dms(
    dms: pd.DataFrame, predictions_wide: pd.DataFrame, sequence: str, *,
    variant_col: str | None = None, position_col: str | None = None,
    wt_col: str | None = None, mut_col: str | None = None,
    uniprot_id: str | None = None, keep_unmatched: bool = True,
) -> tuple[pd.DataFrame, MergeReport]:
    """Join the wide prediction table onto a DMS table.

    ``sequence`` is the canonical UniProt sequence, used to validate every
    row's wild-type residue. With ``keep_unmatched`` (the default), unmatched
    rows are kept with empty prediction columns so nothing disappears.
    """
    prepared = _normalise_dms(
        dms, variant_col=variant_col, position_col=position_col,
        wt_col=wt_col, mut_col=mut_col,
    )

    length = len(sequence)
    for index, row in prepared.iterrows():
        if row["_status"]:
            continue
        position = row["_position"]
        position = None if pd.isna(position) else int(position)
        wt, mut = row["_wt_aa"], row["_mut_aa"]
        if position is None or position < 1 or position > length:
            prepared.at[index, "_status"] = OUT_OF_RANGE
            prepared.at[index, "_detail"] = (
                f"position {position} is outside the sequence (length {length})"
            )
            continue
        expected = sequence[position - 1]
        if wt != expected:
            prepared.at[index, "_status"] = WT_MISMATCH
            prepared.at[index, "_detail"] = (
                f"table says wild type {wt} at {position}, UniProt sequence has {expected}"
            )
            continue
        if mut == "*":
            prepared.at[index, "_status"] = NONSENSE
            prepared.at[index, "_detail"] = "nonsense variant; no substitution score exists"
            continue
        if mut == wt:
            prepared.at[index, "_status"] = SYNONYMOUS
            prepared.at[index, "_detail"] = "synonymous; scored only when --include-wt was used"

    prepared["_position"] = pd.array(prepared["_position"], dtype="Int64")
    prediction_columns = [c for c in predictions_wide.columns if c not in WIDE_KEY_COLUMNS]

    lookup = predictions_wide.copy()
    lookup["position"] = pd.array(lookup["position"], dtype="Int64")
    lookup = lookup.set_index(["position", "wt_aa", "mut_aa"])

    merged = prepared.merge(
        lookup.reset_index(), how="left",
        left_on=["_position", "_wt_aa", "_mut_aa"],
        right_on=["position", "wt_aa", "mut_aa"],
        suffixes=("", "_vepbench"),
    )

    has_prediction = (
        merged[prediction_columns].notna().any(axis=1) if prediction_columns
        else pd.Series(False, index=merged.index)
    )
    for index in merged.index:
        if merged.at[index, "_status"]:
            continue
        merged.at[index, "_status"] = MATCHED if has_prediction.iloc[index] else NO_PREDICTION
        if not has_prediction.iloc[index]:
            merged.at[index, "_detail"] = "no predictor produced a score for this variant"

    merged = merged.rename(columns={"_status": "match_status", "_detail": "match_detail"})
    merged["variant_normalised"] = [
        f"{w}{int(p)}{m}" if (w and not pd.isna(p)) else ""
        for w, p, m in zip(merged["_wt_aa"], merged["_position"], merged["_mut_aa"], strict=True)
    ]
    if uniprot_id is not None:
        merged["uniprot_id"] = uniprot_id

    drop = ["_position", "_wt_aa", "_mut_aa"]
    drop += [c for c in ("position", "wt_aa", "mut_aa", "variant")
             if c in merged.columns and c not in dms.columns]
    merged = merged.drop(columns=[c for c in drop if c in merged.columns])

    counts = merged["match_status"].value_counts().to_dict()
    examples: dict[str, list[str]] = {}
    for status in FAILURE_STATUSES + (SYNONYMOUS,):
        subset = merged.loc[merged["match_status"] == status]
        if len(subset):
            examples[status] = subset["match_detail"].head(3).tolist()
    report = MergeReport(total_rows=len(dms), counts=counts, examples=examples)

    if not keep_unmatched:
        merged = merged.loc[merged["match_status"] == MATCHED].reset_index(drop=True)
    return merged, report


def load_table(path: Path) -> pd.DataFrame:
    """Read a CSV or TSV DMS table, guessing the separator from the suffix."""
    path = Path(path)
    if path.suffix.lower() in (".tsv", ".tab", ".txt"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def harmonized_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.endswith(HARMONIZED_SUFFIX)]
