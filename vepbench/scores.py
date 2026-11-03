"""The ten scores, and the one thing they must agree on.

Every score here is eventually fed to the same ROC curve, so all of them
have to point the same way: higher means more likely pathogenic. The raw
number is kept as the method reported it and a separate harmonized column
carries the oriented value, so nothing is silently flipped and the original
is always recoverable.

Three kinds of score live here. Five are arithmetic on the two residues and
the structure, and run in-process. Two are lookups into a published table.
Three need a model, and run in an isolated environment over a subprocess.
The registry below is the only place that knows which is which.

The physicochemical deltas need one decision spelled out. A substitution
that makes a residue much larger and one that makes it much smaller are
both disruptive, so the damage signal is the *magnitude* of the change,
not its direction. These therefore score ``abs(delta)``; the signed value
is kept alongside, under its own name, for anyone who wants it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from vepbench.constants import (
    KD_HYDROPATHY,
    RESIDUE_CHARGE,
    RESIDUE_VOLUME,
    blosum62,
)

log = logging.getLogger(__name__)

#: What a score needs before it can run.
SEQUENCE = "sequence"
STRUCTURE = "structure"
HUMAN = "human"

#: Where the number comes from, which decides how it is computed.
BUILTIN = "builtin"
TABLE = "table"
MODEL = "model"


class ScoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class Score:
    """One column of the output table."""

    name: str
    kind: str
    needs: frozenset[str]
    #: True when a larger raw number already means more damaging. False for
    #: a score where larger means better tolerated, which is then negated
    #: to build the harmonized column.
    higher_is_damaging: bool
    units: str
    description: str
    env: str = ""
    runner: str = ""
    options: dict = field(default_factory=dict)
    #: Only for kind == BUILTIN.
    compute: Callable[[pd.DataFrame, dict], pd.Series] | None = None
    #: Scoring every substitution is wasteful for a method that costs
    #: minutes per variant and is only ever read at ClinVar positions.
    default_scope: str = "saturation"

    def harmonize(self, raw: pd.Series) -> pd.Series:
        values = pd.to_numeric(raw, errors="coerce")
        return values if self.higher_is_damaging else -values


def _delta(table: dict[str, float], frame: pd.DataFrame) -> pd.Series:
    """Signed change in a per-residue property."""
    return pd.Series(
        [table.get(m, float("nan")) - table.get(w, float("nan"))
         for w, m in zip(frame["wt_aa"], frame["mut_aa"], strict=True)],
        index=frame.index, dtype=float,
    )


def _abs_delta(table: dict[str, float]):
    return lambda frame, context: _delta(table, frame).abs()


def _blosum62(frame: pd.DataFrame, context: dict) -> pd.Series:
    return pd.Series(
        [blosum62(w, m) for w, m in zip(frame["wt_aa"], frame["mut_aa"], strict=True)],
        index=frame.index, dtype=float,
    )


def _solvent_exposure(frame: pd.DataFrame, context: dict) -> pd.Series:
    """Relative solvent accessibility at the position.

    A property of the position rather than the substitution, so every
    substitution at one residue shares a value. That is the point: it asks
    whether buried positions carry more pathogenic variants, which is a
    real and well-attested effect worth measuring beside the predictors.
    """
    rsasa = context.get("rsasa")
    if not rsasa:
        raise ScoreError("solvent_exposure needs a structure")
    return frame["position"].map(rsasa).astype(float)


SCORES: dict[str, Score] = {}


def register(score: Score) -> Score:
    if score.name in SCORES:
        raise ScoreError(f"duplicate score {score.name!r}")
    SCORES[score.name] = score
    return score


# --- arithmetic, in-process -------------------------------------------
register(Score(
    name="blosum62", kind=BUILTIN, needs=frozenset({SEQUENCE}),
    higher_is_damaging=False, units="substitution_matrix_score",
    description="BLOSUM62 exchangeability; positive means a favoured substitution",
    compute=_blosum62,
))
register(Score(
    name="delta_hydrophobicity", kind=BUILTIN, needs=frozenset({SEQUENCE}),
    higher_is_damaging=True, units="kd_units",
    description="absolute change in Kyte-Doolittle hydropathy",
    compute=_abs_delta(KD_HYDROPATHY),
))
register(Score(
    name="delta_volume", kind=BUILTIN, needs=frozenset({SEQUENCE}),
    higher_is_damaging=True, units="cubic_angstrom",
    description="absolute change in residue volume",
    compute=_abs_delta(RESIDUE_VOLUME),
))
register(Score(
    name="delta_charge", kind=BUILTIN, needs=frozenset({SEQUENCE}),
    higher_is_damaging=True, units="elementary_charge",
    description="absolute change in residue charge at pH 7",
    compute=_abs_delta(RESIDUE_CHARGE),
))
register(Score(
    name="solvent_exposure", kind=BUILTIN, needs=frozenset({SEQUENCE, STRUCTURE}),
    higher_is_damaging=False, units="relative_sasa",
    description="relative solvent accessibility; buried positions are less tolerant",
    compute=_solvent_exposure,
))

# --- published tables --------------------------------------------------
register(Score(
    name="alphamissense", kind=TABLE, needs=frozenset({SEQUENCE, HUMAN}),
    higher_is_damaging=True, units="pathogenicity_probability",
    description="AlphaMissense published table; human only, and calibrated "
                "against ClinVar, so not independent of a ClinVar benchmark",
))
register(Score(
    name="thermompnn", kind=TABLE, needs=frozenset({SEQUENCE, STRUCTURE}),
    higher_is_damaging=True, units="ddG_kcal_mol",
    description="ThermoMPNN ddG; read from the published table when it covers "
                "the protein, otherwise computed",
    env="thermompnn", runner="thermompnn.py",
))

# --- models in isolated environments ------------------------------------
register(Score(
    name="esm1v", kind=MODEL, needs=frozenset({SEQUENCE}),
    higher_is_damaging=False, units="log_likelihood_ratio",
    description="ESM-1v masked marginal; positive favours the mutant",
    env="esm", runner="esm_masked_marginals.py",
    options={"model": "esm1v_t33_650M_UR90S_1"},
))
register(Score(
    name="saprot", kind=MODEL, needs=frozenset({SEQUENCE, STRUCTURE}),
    higher_is_damaging=False, units="log_likelihood_ratio",
    description="SaProt structure-aware masked marginal",
    env="huggingface", runner="saprot.py",
    options={"model": "westlake-repl/SaProt_650M_AF2"},
))
register(Score(
    name="foldx", kind=MODEL, needs=frozenset({SEQUENCE, STRUCTURE}),
    higher_is_damaging=True, units="ddG_kcal_mol",
    description="FoldX BuildModel ddG; positive is destabilising",
    env="foldx", runner="foldx.py", options={"repair": True, "num_runs": 1},
    # Minutes per variant. At saturation one protein is days of CPU, and
    # every score outside a ClinVar position would go unread.
    default_scope="clinvar",
))


def all_names() -> list[str]:
    return list(SCORES)


def resolve(names: list[str] | None) -> list[Score]:
    """Turn names, or None for everything, into Score objects in a fixed order."""
    if not names or names == ["all"]:
        return list(SCORES.values())
    unknown = [n for n in names if n not in SCORES]
    if unknown:
        raise ScoreError(
            f"unknown score(s) {', '.join(unknown)}; available: {', '.join(SCORES)}"
        )
    return [SCORES[n] for n in names]


def applicable(score: Score, *, is_human: bool, has_structure: bool) -> tuple[bool, str]:
    """Whether this score can run for a protein, and why not when it cannot."""
    if HUMAN in score.needs and not is_human:
        return False, "human only"
    if STRUCTURE in score.needs and not has_structure:
        return False, "no structure available"
    return True, ""
