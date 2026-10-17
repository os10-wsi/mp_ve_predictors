"""In-repo baselines.

These are not among the published methods being benchmarked. They are naive
references that any serious predictor should beat, and they are the only ones
that run with no weights, no network and no isolated environment, which makes
them useful for exercising the pipeline end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vepbench.constants import AA20, blosum62
from vepbench.msa import read_alignment, to_matched_columns
from vepbench.predictors.base import Predictor, PredictorSpec, register
from vepbench.protein import ProteinContext
from vepbench.schema import MORE_TOLERATED, ScoreConvention

_AA_INDEX = {aa: i for i, aa in enumerate(AA20)}


class Blosum62Baseline(Predictor):
    """Substitution-matrix score: the cheapest possible reference."""

    def _predict_raw(self, protein: ProteinContext, variants: pd.DataFrame) -> pd.DataFrame:
        out = variants.loc[:, ["position", "wt_aa", "mut_aa"]].copy()
        out["score"] = [
            blosum62(w, m) for w, m in zip(out["wt_aa"], out["mut_aa"], strict=True)
        ]
        return out


class PssmIndependentBaseline(Predictor):
    """Independent-sites model: log f(mutant) - log f(wild type) from the MSA.

    This is the independent model used as a baseline in the EVmutation paper,
    computed from sequence-reweighted column frequencies with a Laplace
    pseudocount. Higher means better tolerated.
    """

    def __init__(self, spec: PredictorSpec, *, identity_threshold: float = 0.8,
                 pseudocount: float = 1.0) -> None:
        super().__init__(spec)
        self.identity_threshold = identity_threshold
        self.pseudocount = pseudocount

    def _column_frequencies(self, protein: ProteinContext) -> np.ndarray:
        matrix = to_matched_columns(read_alignment(protein.msa.path))
        matrix = matrix[:, np.where(matrix[0] != "-")[0]]

        codes = np.frombuffer(matrix.astype("S1").tobytes(), dtype=np.uint8).reshape(matrix.shape)
        length = codes.shape[1]
        weights = np.ones(codes.shape[0])
        block = 512
        for start in range(0, codes.shape[0], block):
            chunk = codes[start : start + block]
            same = (chunk[:, None, :] == codes[None, :, :]).sum(axis=2) / max(length, 1)
            weights[start : start + block] = 1.0 / (same >= self.identity_threshold).sum(axis=1)

        counts = np.full((matrix.shape[1], 20), self.pseudocount, dtype=float)
        for aa, index in _AA_INDEX.items():
            counts[:, index] += ((matrix == aa) * weights[:, None]).sum(axis=0)
        return counts / counts.sum(axis=1, keepdims=True)

    def _predict_raw(self, protein: ProteinContext, variants: pd.DataFrame) -> pd.DataFrame:
        frequencies = self._column_frequencies(protein)
        n_columns = frequencies.shape[0]
        out = variants.loc[:, ["position", "wt_aa", "mut_aa"]].copy()
        scores = []
        for position, wt, mut in zip(
            out["position"], out["wt_aa"], out["mut_aa"], strict=True
        ):
            if position < 1 or position > n_columns:
                scores.append(float("nan"))
                continue
            column = frequencies[position - 1]
            scores.append(float(np.log(column[_AA_INDEX[mut]]) - np.log(column[_AA_INDEX[wt]])))
        out["score"] = scores
        return out

    def model_version(self, protein: ProteinContext) -> str:
        msa = protein.msa
        return (f"independent-sites (Neff {msa.n_effective:.0f}, {msa.backend})"
                if msa else "n/a")


register(Blosum62Baseline(PredictorSpec(
    name="blosum62_baseline", display_name="BLOSUM62 baseline", category="baseline",
    convention=ScoreConvention(
        score_type="substitution_matrix_score", higher_is=MORE_TOLERATED, verified=True,
        source="Henikoff & Henikoff 1992: positive scores are favoured substitutions",
    ),
    citation="Henikoff S, Henikoff JG. PNAS 1992;89:10915-10919",
    repo_url="", license="public domain", requires=frozenset({"sequence"}),
    model_version="BLOSUM62",
    notes="Naive reference implemented in this repository, not a published predictor.",
)))

register(PssmIndependentBaseline(PredictorSpec(
    name="pssm_independent", display_name="Independent-sites MSA baseline",
    category="baseline",
    convention=ScoreConvention(
        score_type="log_likelihood_ratio", higher_is=MORE_TOLERATED, verified=True,
        source="log f(mut) - log f(wt); positive means the mutant is the commoner residue",
    ),
    citation="Hopf TA et al. Nat Biotechnol 2017;35:128-135 (independent model baseline)",
    repo_url="", license="MIT (this implementation)",
    requires=frozenset({"sequence", "msa"}), model_version="independent-sites",
    notes="Naive reference implemented in this repository, not a published predictor. "
          "Useful for checking whether an alignment is deep enough to be informative.",
)))
