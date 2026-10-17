"""Inverse-folding predictors: ESM-IF1 and ProteinMPNN.

Both score a substitution as the change in the model's conditional
log-likelihood for the residue given the backbone, so positive means the
backbone prefers the mutant and higher is more tolerated.

Both were trained on soluble protein structures, from CATH and the PDB, which
is a real caveat for membrane proteins: a lipid-facing surface looks to them
like exposed hydrophobic surface in water.
"""

from __future__ import annotations

from vepbench.predictors.base import PredictorSpec, SubprocessPredictor, register
from vepbench.schema import MORE_TOLERATED, ScoreConvention

_INVERSE_FOLDING = ScoreConvention(
    score_type="log_likelihood_ratio", higher_is=MORE_TOLERATED, verified=True,
    source="conditional log-likelihood difference log p(mut|backbone) - log p(wt|backbone)",
)

register(SubprocessPredictor(
    PredictorSpec(
        name="esm_if1", display_name="ESM-IF1", category="inverse_folding",
        convention=_INVERSE_FOLDING,
        citation="Hsu C et al. Learning inverse folding from millions of predicted "
                 "structures. ICML 2022",
        repo_url="https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding",
        license="MIT code; CC-BY-NC 4.0 weights",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=2.0, env="esm",
        model_version="esm_if1_gvp4_t16_142M_UR50",
        notes="The only model here still distributed solely through fair-esm, so its "
              "weights come from dl.fbaipublicfiles.com rather than Hugging Face. "
              "Weights are CC-BY-NC, so commercial use is restricted.",
        install_hint="python -m vepbench install-env esm",
    ),
    runner_name="esm_if1.py",
    options={"scoring": "conditional"},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="proteinmpnn", display_name="ProteinMPNN", category="inverse_folding",
        convention=_INVERSE_FOLDING,
        citation="Dauparas J et al. Robust deep learning-based protein sequence design "
                 "with ProteinMPNN. Science 2022;378:49-56",
        repo_url="https://github.com/dauparas/ProteinMPNN", license="MIT",
        requires=frozenset({"sequence", "structure"}), approximate_vram_gb=1.0,
        env="proteinmpnn", model_version="v_48_020",
        notes="Scored with conditional probabilities averaged over random decoding "
              "orders, the inverse-folding analogue of a masked marginal. Small enough "
              "to run comfortably on CPU. A soluble-weights variant is available "
              "through the use_soluble_weights option.",
        install_hint="python -m vepbench install-env proteinmpnn",
    ),
    runner_name="proteinmpnn.py",
    options={"model_name": "v_48_020", "use_soluble_weights": False},
))
