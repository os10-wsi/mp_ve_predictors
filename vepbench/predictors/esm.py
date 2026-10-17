"""ESM-family predictors: ESM-1v, ESM-2, ESM C and ESM3.

All are scored with masked marginals, the convention from Meier et al. 2021:
``log p(mutant | context) - log p(wild type | context)``. Positive means the
substitution is better tolerated, so ``higher_is`` is ``more_tolerated``
throughout and the orchestrator flips the sign for ``score_harmonized``.

ESM-1v and ESM-2 are loaded through Hugging Face ``transformers`` rather than
``fair-esm``. The weights are identical, but transformers gives clean device
and dtype control for GPU runs, and avoids depending on a second weight host.
"""

from __future__ import annotations

from vepbench.predictors.base import PredictorSpec, SubprocessPredictor, register
from vepbench.schema import MORE_TOLERATED, ScoreConvention

_MASKED_MARGINAL = ScoreConvention(
    score_type="log_likelihood_ratio", higher_is=MORE_TOLERATED, verified=True,
    source="Meier et al. 2021 (NeurIPS), masked marginals: log p(mut) - log p(wt)",
)

ESM1V_MODELS = [f"facebook/esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)]

register(SubprocessPredictor(
    PredictorSpec(
        name="esm1v_full", display_name="ESM-1v (full length, 5-model ensemble)",
        category="language_model", convention=_MASKED_MARGINAL,
        citation="Meier J et al. Language models enable zero-shot prediction of the "
                 "effects of mutations on protein function. NeurIPS 2021",
        repo_url="https://github.com/facebookresearch/esm", license="MIT",
        requires=frozenset({"sequence"}), gpu_recommended=True,
        approximate_vram_gb=3.0, env="esm",
        model_version="esm1v_t33_650M_UR90S_1..5",
        notes="ESM-1v has a 1022-residue context. Longer proteins are tiled, each "
              "position scored in the window where it is most central, and the window "
              "used is written to the per-member CSV. Replicates are loaded one at a "
              "time, so peak memory is one 650M model, not five.",
        install_hint="python3 -m vepbench install-env esm",
    ),
    runner_name="esm_masked_marginals.py",
    options={"models": ESM1V_MODELS},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="esm1v_domain", display_name="ESM-1v (domain only, 5-model ensemble)",
        category="language_model", convention=_MASKED_MARGINAL,
        citation="Meier J et al. NeurIPS 2021",
        repo_url="https://github.com/facebookresearch/esm", license="MIT",
        requires=frozenset({"sequence", "domain"}), gpu_recommended=True,
        approximate_vram_gb=3.0, env="esm",
        model_version="esm1v_t33_650M_UR90S_1..5 (domain window)",
        notes="Scores only the chosen domain window, with the model seeing the domain "
              "sequence alone. For a membrane protein the window defaults to the span "
              "of the UniProt transmembrane segments; override with --domain.",
        install_hint="python3 -m vepbench install-env esm",
    ),
    runner_name="esm_masked_marginals.py",
    options={"models": ESM1V_MODELS, "use_domain": True},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="esm2_650m", display_name="ESM-2 650M", category="language_model",
        convention=_MASKED_MARGINAL,
        citation="Lin Z et al. Evolutionary-scale prediction of atomic-level protein "
                 "structure with a language model. Science 2023;379:1123-1130",
        repo_url="https://github.com/facebookresearch/esm", license="MIT",
        requires=frozenset({"sequence"}), gpu_recommended=True,
        approximate_vram_gb=3.0, env="esm", model_version="facebook/esm2_t33_650M_UR50D",
        notes="Likelihood-ratio baseline; the same 1022-residue tiling applies.",
        install_hint="python3 -m vepbench install-env esm",
    ),
    runner_name="esm_masked_marginals.py",
    options={"models": ["facebook/esm2_t33_650M_UR50D"]},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="esm2_3b", display_name="ESM-2 3B", category="language_model",
        convention=_MASKED_MARGINAL,
        citation="Lin Z et al. Science 2023;379:1123-1130",
        repo_url="https://github.com/facebookresearch/esm", license="MIT",
        requires=frozenset({"sequence"}), gpu_recommended=True,
        approximate_vram_gb=13.0, env="esm", model_version="facebook/esm2_t36_3B_UR50D",
        notes="Needs roughly 13 GB of VRAM in float32 and about 7 GB in float16. Pass "
              "--dtype float16 on a smaller card, or --device cpu to trade speed for "
              "memory.",
        install_hint="python3 -m vepbench install-env esm",
    ),
    runner_name="esm_masked_marginals.py",
    options={"models": ["facebook/esm2_t36_3B_UR50D"]},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="esmc_600m", display_name="ESM C 600M", category="language_model",
        convention=_MASKED_MARGINAL,
        citation="ESM Team. ESM Cambrian: revealing the mysteries of proteins with "
                 "unsupervised learning. EvolutionaryScale 2024",
        repo_url="https://github.com/evolutionaryscale/esm",
        license="Cambrian Open licence; non-commercial terms apply",
        requires=frozenset({"sequence"}), gpu_recommended=True,
        approximate_vram_gb=3.0, env="esm", model_version="esmc_600m",
        notes="Likelihood-ratio score from the ESM Cambrian encoder. Gated on Hugging "
              "Face, so HF_TOKEN must be set.",
        install_hint="python3 -m vepbench install-env esm, then export HF_TOKEN",
    ),
    runner_name="esmc_masked_marginals.py",
    options={"model": "esmc_600m"},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="esm3_open", display_name="ESM3 (open small, 1.4B)",
        category="language_model", convention=_MASKED_MARGINAL,
        citation="Hayes T et al. Simulating 500 million years of evolution with a "
                 "language model. Science 2025;387:850-858",
        repo_url="https://github.com/evolutionaryscale/esm",
        license="EvolutionaryScale Community licence; non-commercial, accept on "
                "Hugging Face",
        requires=frozenset({"sequence"}), gpu_recommended=True,
        approximate_vram_gb=7.0, env="esm", model_version="esm3-sm-open-v1",
        notes="Sequence-track masked marginals. The structure track can be added as "
              "conditioning, which changes the score, so it is off here and exposed "
              "separately as esm3_dg_masked_marginal.",
        install_hint="Accept the licence at huggingface.co/EvolutionaryScale/"
                     "esm3-sm-open-v1, export HF_TOKEN, then install-env esm",
    ),
    runner_name="esm3_masked_marginals.py",
    options={"model": "esm3-sm-open-v1"},
))
