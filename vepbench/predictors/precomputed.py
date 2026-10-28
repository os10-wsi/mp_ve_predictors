"""Predictors that ship their scores rather than their model.

Some methods publish every substitution for the human proteome. For those,
running the model locally reproduces a table that already exists, slowly and
with some risk of reproducing it wrong. Reading the published table is both
faster and closer to what the authors reported, so these predictors are
lookups. They need no isolated environment, no GPU and no network at run
time; what they need is a one-off download, which ``vepbench proteome
fetch-data`` performs.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from vepbench.alphamissense import LICENSE, load_slice, slice_path
from vepbench.predictors.base import Availability, Predictor, PredictorSpec, register
from vepbench.protein import ProteinContext
from vepbench.schema import MORE_DELETERIOUS, ScoreConvention

log = logging.getLogger(__name__)


class AlphaMissense(Predictor):
    """Look up AlphaMissense pathogenicity for each substitution.

    A variant missing from the table scores NaN rather than zero: absent is
    not the same as benign, and a zero here would read as a confident call.
    """

    def is_available(self, protein: ProteinContext | None = None) -> Availability:
        if protein is None:
            return Availability(True)
        path = slice_path(protein.uniprot_id)
        if not path.exists():
            return Availability(False, (
                f"no AlphaMissense slice for {protein.uniprot_id} at {path}. Run "
                f"`vepbench proteome fetch-data --only alphamissense` to download and "
                f"slice the published table."
            ))
        return Availability(True)

    def _predict_raw(self, protein: ProteinContext, variants: pd.DataFrame) -> pd.DataFrame:
        lookup = load_slice(protein.uniprot_id)
        out = variants.loc[:, ["position", "wt_aa", "mut_aa"]].copy()
        scores: list[float] = []
        classes: list[str] = []
        for position, wt, mut in zip(
            out["position"], out["wt_aa"], out["mut_aa"], strict=True
        ):
            if wt == mut:
                scores.append(float("nan"))
                classes.append("")
                continue
            hit = lookup.get(f"{wt}{position}{mut}".upper())
            scores.append(float("nan") if hit is None else hit[0])
            classes.append("" if hit is None else hit[1])
        out["score"] = scores
        out["am_class"] = classes
        missing = sum(1 for s in scores if math.isnan(s))
        if missing:
            log.info("AlphaMissense has no score for %d of %d variants in %s",
                     missing, len(scores), protein.uniprot_id)
        return out

    def model_version(self, protein: ProteinContext) -> str:
        return "AlphaMissense 2023 (published substitution table)"


register(AlphaMissense(PredictorSpec(
    name="alphamissense", display_name="AlphaMissense", category="pathogenicity",
    convention=ScoreConvention(
        score_type="pathogenicity_probability", higher_is=MORE_DELETERIOUS, verified=True,
        source="Cheng et al. 2023: am_pathogenicity runs 0 to 1, higher is more "
               "likely pathogenic",
    ),
    citation="Cheng J et al. Accurate proteome-wide missense variant effect "
             "prediction with AlphaMissense. Science 2023;381:eadg7492",
    repo_url="https://github.com/google-deepmind/alphamissense", license=LICENSE,
    requires=frozenset({"sequence"}), model_version="AlphaMissense 2023",
    trained_on_clinvar=True,
    notes="Read from the published proteome-wide table rather than run locally. "
          "Weakly supervised on population frequency and calibrated against "
          "ClinVar, so it is not independent of a ClinVar benchmark; report it "
          "alongside the unsupervised predictors, not as one of them.",
    install_hint="python3 -m vepbench proteome fetch-data --only alphamissense",
)))
