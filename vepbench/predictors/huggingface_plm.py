"""Transformer protein language models served through Hugging Face.

SaProt and ProSST are structure-aware and consume a structural alphabet
alongside the sequence. ProtT5 and Ankh are sequence-only and are included as
likelihood baselines.
"""

from __future__ import annotations

from vepbench.predictors.base import PredictorSpec, SubprocessPredictor, register
from vepbench.schema import MORE_TOLERATED, ScoreConvention

_MASKED_MARGINAL = ScoreConvention(
    score_type="log_likelihood_ratio", higher_is=MORE_TOLERATED, verified=True,
    source="masked marginals: log p(mut) - log p(wt)",
)

register(SubprocessPredictor(
    PredictorSpec(
        name="saprot", display_name="SaProt 650M (AF2 structural alphabet)",
        category="language_model", convention=_MASKED_MARGINAL,
        citation="Su J et al. SaProt: protein language modeling with structure-aware "
                 "vocabulary. ICLR 2024",
        repo_url="https://github.com/westlake-repl/SaProt", license="MIT",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=3.0, env="huggingface", model_version="SaProt_650M_AF2",
        notes="Each token pairs a residue with its Foldseek 3Di structural letter, so "
              "a structure is mandatory and foldseek must be on PATH. The 3Di alphabet "
              "was derived from soluble-protein structures.",
        install_hint="python3 -m vepbench install-env huggingface && "
                     "bash scripts/install_foldseek.sh",
    ),
    runner_name="saprot.py",
    options={"model": "westlake-repl/SaProt_650M_AF2"},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="saprot_1p3b", display_name="SaProt 1.3B", category="language_model",
        convention=_MASKED_MARGINAL,
        citation="Su J et al. ICLR 2024",
        repo_url="https://github.com/westlake-repl/SaProt", license="MIT",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=6.0, env="huggingface", model_version="SaProt_1.3B_AF2",
        notes="The larger SaProt. Same scoring and the same foldseek requirement; "
              "enable it when you have the VRAM.",
        install_hint="python3 -m vepbench install-env huggingface && "
                     "bash scripts/install_foldseek.sh",
    ),
    runner_name="saprot.py",
    options={"model": "westlake-repl/SaProt_1.3B_AF2"},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="prosst", display_name="ProSST 2048", category="language_model",
        convention=_MASKED_MARGINAL,
        citation="Li M et al. ProSST: protein language modeling with quantized "
                 "structure and disentangled attention. NeurIPS 2024",
        repo_url="https://github.com/ai4protein/ProSST", license="MIT",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=2.0, env="huggingface", model_version="AI4Protein/ProSST-2048",
        notes="Quantises local structure into a discrete vocabulary, then scores with "
              "disentangled attention over sequence and structure tokens.",
        install_hint="python3 -m vepbench install-env huggingface",
    ),
    runner_name="prosst.py",
    options={"model": "AI4Protein/ProSST-2048"},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="prott5", display_name="ProtT5-XL UniRef50", category="language_model",
        convention=_MASKED_MARGINAL,
        citation="Elnaggar A et al. ProtTrans: towards cracking the language of life's "
                 "code. IEEE TPAMI 2022;44:7112-7127",
        repo_url="https://github.com/agemagician/ProtTrans",
        license="Academic use per the model card",
        requires=frozenset({"sequence"}), gpu_recommended=True,
        approximate_vram_gb=6.0, env="huggingface",
        model_version="Rostlab/prot_t5_xl_uniref50",
        notes="An encoder-decoder trained with span denoising, so a reconstruction "
              "likelihood at a sentinel token stands in for a masked-LM head. That is "
              "a baseline of this repository's choosing, not a scheme the ProtT5 "
              "authors proposed for variant effects.",
        install_hint="python3 -m vepbench install-env huggingface",
    ),
    runner_name="t5_reconstruction.py",
    options={"model": "Rostlab/prot_t5_xl_uniref50"},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="ankh", display_name="Ankh large", category="language_model",
        convention=_MASKED_MARGINAL,
        citation="Elnaggar A et al. Ankh: optimized protein language model unlocks "
                 "general-purpose modelling. arXiv 2023:2301.06568",
        repo_url="https://github.com/agemagician/Ankh", license="CC-BY-NC-SA 4.0",
        requires=frozenset({"sequence"}), gpu_recommended=True,
        approximate_vram_gb=5.0, env="huggingface", model_version="ElnaggarLab/ankh-large",
        notes="Same caveat as ProtT5: a denoising reconstruction likelihood used as a "
              "baseline rather than an authored variant-effect score.",
        install_hint="python3 -m vepbench install-env huggingface",
    ),
    runner_name="t5_reconstruction.py",
    options={"model": "ElnaggarLab/ankh-large"},
))
