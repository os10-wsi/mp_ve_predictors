"""Output schema, score harmonization and validation.

The rule that matters downstream: ``score_harmonized`` is always oriented so
that higher means more damaging or more destabilizing, in the raw units. The
raw ``score`` column is never altered, so nothing is lost.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

MORE_DELETERIOUS = "more_deleterious"
MORE_TOLERATED = "more_tolerated"
VALID_DIRECTIONS = (MORE_DELETERIOUS, MORE_TOLERATED)

LONG_COLUMNS = [
    "uniprot_id", "position", "wt_aa", "mut_aa", "variant", "predictor",
    "score", "score_type", "higher_is", "score_harmonized", "model_version",
]
WIDE_KEY_COLUMNS = ["uniprot_id", "position", "wt_aa", "mut_aa", "variant"]
HARMONIZED_SUFFIX = "__harmonized"


class SchemaError(ValueError):
    """Raised when a predictor returns something that does not fit the schema."""


@dataclass(frozen=True)
class ScoreConvention:
    """How to read a predictor's raw number.

    Attributes:
        score_type: unit or kind of the raw score, such as ``ddG_kcal_mol``.
        higher_is: whether a larger raw score means more deleterious or more
            tolerated.
        verified: whether the direction was confirmed against the method's own
            paper or repository, rather than inferred from secondary sources.
        source: where the convention was read from, for the audit trail.
    """

    score_type: str
    higher_is: str
    verified: bool
    source: str

    def __post_init__(self) -> None:
        if self.higher_is not in VALID_DIRECTIONS:
            raise SchemaError(
                f"higher_is must be one of {VALID_DIRECTIONS}, got {self.higher_is!r}"
            )


def harmonize(score: float, higher_is: str) -> float:
    """Flip a raw score so that higher always means more damaging."""
    if higher_is == MORE_DELETERIOUS:
        return score
    if higher_is == MORE_TOLERATED:
        return -score
    raise SchemaError(f"unknown higher_is value {higher_is!r}")


def harmonize_series(scores: pd.Series, higher_is: str) -> pd.Series:
    if higher_is == MORE_DELETERIOUS:
        return scores.astype(float)
    if higher_is == MORE_TOLERATED:
        return -scores.astype(float)
    raise SchemaError(f"unknown higher_is value {higher_is!r}")


def make_variant(wt_aa: str, position: int, mut_aa: str) -> str:
    return f"{wt_aa}{position}{mut_aa}"


def build_long_frame(
    *, uniprot_id: str, frame: pd.DataFrame, predictor: str,
    convention: ScoreConvention, model_version: str,
) -> pd.DataFrame:
    """Turn a predictor's minimal output into a full long-format frame.

    A predictor returns only position, wt_aa, mut_aa and score. Everything else
    is filled in here, so no wrapper can get harmonization or column order wrong.
    """
    required = {"position", "wt_aa", "mut_aa", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise SchemaError(f"predictor {predictor!r} is missing columns: {sorted(missing)}")

    out = frame.loc[:, ["position", "wt_aa", "mut_aa", "score"]].copy()
    out["position"] = out["position"].astype(int)
    out["wt_aa"] = out["wt_aa"].astype(str).str.upper()
    out["mut_aa"] = out["mut_aa"].astype(str).str.upper()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out.insert(0, "uniprot_id", uniprot_id)
    out["variant"] = [
        make_variant(w, p, m)
        for w, p, m in zip(out["wt_aa"], out["position"], out["mut_aa"], strict=True)
    ]
    out["predictor"] = predictor
    out["score_type"] = convention.score_type
    out["higher_is"] = convention.higher_is
    out["score_harmonized"] = harmonize_series(out["score"], convention.higher_is)
    out["model_version"] = model_version
    return out.loc[:, LONG_COLUMNS]


def validate_long_frame(frame: pd.DataFrame, sequence: str | None = None) -> None:
    """Check a long frame for schema and numbering errors.

    When ``sequence`` is given, every row's ``wt_aa`` must match the residue at
    that 1-based position. Numbering drift between UniProt and a structure is
    the commonest silent error here, so this is an assertion, not a warning.
    """
    missing = [c for c in LONG_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaError(f"long frame is missing columns: {missing}")

    bad = set(frame["higher_is"].unique()) - set(VALID_DIRECTIONS)
    if bad:
        raise SchemaError(f"invalid higher_is values: {sorted(bad)}")

    if sequence is None:
        return
    length = len(sequence)
    out_of_range = frame.loc[(frame["position"] < 1) | (frame["position"] > length)]
    if len(out_of_range):
        raise SchemaError(
            f"{len(out_of_range)} rows fall outside the sequence (length {length}); "
            f"examples: {out_of_range['variant'].head(5).tolist()}"
        )
    expected = frame["position"].map(lambda p: sequence[p - 1])
    mismatched = frame.loc[frame["wt_aa"] != expected]
    if len(mismatched):
        detail = ", ".join(
            f"{row.variant} (sequence has {sequence[row.position - 1]})"
            for row in mismatched.head(5).itertuples()
        )
        raise SchemaError(
            f"{len(mismatched)} rows disagree with the UniProt sequence: {detail}"
        )


def to_wide(long_frame: pd.DataFrame, feature_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pivot to one row per variant, one column per predictor.

    Raw scores first, then a matching block of ``<predictor>__harmonized``
    columns, then any per-variant features.
    """
    if long_frame.empty:
        return pd.DataFrame(columns=WIDE_KEY_COLUMNS)

    raw = long_frame.pivot_table(
        index=WIDE_KEY_COLUMNS, columns="predictor", values="score", aggfunc="first"
    )
    harm = long_frame.pivot_table(
        index=WIDE_KEY_COLUMNS, columns="predictor", values="score_harmonized",
        aggfunc="first",
    )
    harm.columns = [f"{c}{HARMONIZED_SUFFIX}" for c in harm.columns]
    wide = raw.join(harm).reset_index()
    wide.columns.name = None
    wide = wide.sort_values(["position", "mut_aa"]).reset_index(drop=True)

    if feature_frame is not None and not feature_frame.empty:
        keys = [c for c in ("position", "wt_aa", "mut_aa") if c in feature_frame.columns]
        if keys:
            wide = wide.merge(feature_frame, on=keys, how="left")
    return wide


def empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LONG_COLUMNS)


def concat_long(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return empty_long_frame()
    return pd.concat(frames, ignore_index=True).loc[:, LONG_COLUMNS]
