"""Enumerating saturation variants and parsing variant notation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from vepbench.constants import AA20, AA_SET, THREE_TO_ONE

_SHORT_RE = re.compile(r"^(?:p\.)?([A-Z])(\d+)([A-Z*])$", re.IGNORECASE)
_LONG_RE = re.compile(r"^(?:p\.)?([A-Za-z]{3})(\d+)([A-Za-z]{3})$")
_STOP_TOKENS = {"*", "TER", "X"}


class VariantParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedVariant:
    wt_aa: str
    position: int
    mut_aa: str

    @property
    def variant(self) -> str:
        return f"{self.wt_aa}{self.position}{self.mut_aa}"


def parse_variant(text: str) -> ParsedVariant:
    """Parse ``A123G``, ``p.A123G``, ``p.Ala123Gly`` and similar.

    Multi-mutants are rejected here; the merge code reports them as unmatched
    rather than guessing which substitution was meant.
    """
    if text is None:
        raise VariantParseError("empty variant")
    raw = str(text).strip()
    if not raw:
        raise VariantParseError("empty variant")
    if any(sep in raw for sep in (",", ";", ":", "/", "+", "_")):
        raise VariantParseError(f"looks like a multi-mutant: {raw!r}")

    match = _SHORT_RE.match(raw)
    if match:
        wt, pos, mut = match.group(1).upper(), int(match.group(2)), match.group(3).upper()
    else:
        match = _LONG_RE.match(raw)
        if not match:
            raise VariantParseError(f"unrecognised variant notation: {raw!r}")
        wt_raw, pos, mut_raw = match.group(1).upper(), int(match.group(2)), match.group(3).upper()
        if wt_raw not in THREE_TO_ONE:
            raise VariantParseError(f"unknown wild-type residue {wt_raw!r} in {raw!r}")
        wt = THREE_TO_ONE[wt_raw]
        if mut_raw in _STOP_TOKENS:
            mut = "*"
        elif mut_raw in THREE_TO_ONE:
            mut = THREE_TO_ONE[mut_raw]
        else:
            raise VariantParseError(f"unknown mutant residue {mut_raw!r} in {raw!r}")

    if mut in _STOP_TOKENS:
        mut = "*"
    if wt not in AA_SET:
        raise VariantParseError(f"wild-type residue {wt!r} is not a standard amino acid")
    if mut != "*" and mut not in AA_SET:
        raise VariantParseError(f"mutant residue {mut!r} is not a standard amino acid")
    if pos < 1:
        raise VariantParseError(f"position must be 1-based and positive, got {pos}")
    return ParsedVariant(wt_aa=wt, position=pos, mut_aa=mut)


def saturation_variants(
    sequence: str, *, include_wt: bool = False, positions: list[int] | None = None
) -> pd.DataFrame:
    """Every single substitution at every position: L x 19, or L x 20 with WT."""
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("sequence is empty")
    unknown = set(sequence) - AA_SET
    if unknown:
        raise ValueError(
            f"sequence contains non-standard residues {sorted(unknown)}; "
            "these positions cannot be scored"
        )

    wanted = positions if positions is not None else range(1, len(sequence) + 1)
    rows = []
    for pos in wanted:
        if pos < 1 or pos > len(sequence):
            raise ValueError(f"position {pos} is outside the sequence (length {len(sequence)})")
        wt = sequence[pos - 1]
        for mut in AA20:
            if mut == wt and not include_wt:
                continue
            rows.append((pos, wt, mut, f"{wt}{pos}{mut}", mut == wt))
    return pd.DataFrame(rows, columns=["position", "wt_aa", "mut_aa", "variant", "is_wt"])


def check_against_sequence(frame: pd.DataFrame, sequence: str) -> pd.Series:
    """Boolean mask of rows whose ``wt_aa`` matches the sequence."""
    in_range = (frame["position"] >= 1) & (frame["position"] <= len(sequence))
    matches = pd.Series(False, index=frame.index)
    sub = frame.loc[in_range]
    matches.loc[sub.index] = [
        sequence[p - 1] == w for p, w in zip(sub["position"], sub["wt_aa"], strict=True)
    ]
    return matches
