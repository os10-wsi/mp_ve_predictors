"""Newer predictors, including the six whose existence needed checking.

What the literature search found, recorded here so the provenance lives in the
code rather than only in a commit message:

* **S3F** exists. Zhang Z, Notin P, Huang Y, Lozano AC, Chenthamarakshan V,
  Marks D, Das P, Tang J. *Multi-Scale Representation Learning for Protein
  Fitness Prediction*, NeurIPS 2024 (arXiv:2412.01108). Code:
  github.com/DeepGraphLearning/S3F. Sequence, structure and surface, as
  described.
* **VenusREM** exists. Tan Y et al., *From high-throughput evaluation to
  wet-lab studies: advancing mutation effect prediction with a
  retrieval-enhanced model*, Bioinformatics 2025;41(Suppl 1):i401-i409. Code:
  github.com/ai4protein/VenusREM, mirrored at tyang816/VenusREM.
* **ProMEP** exists. Cheng P et al., Cell Research 2024;34:630-647. Code:
  github.com/wenjiegroup/ProMEP.
* **METL** exists. Gelman S et al., *Biophysics-based protein language models
  for protein engineering*, Nature Methods 2025. Code: gitter-lab/metl and
  gitter-lab/metl-pretrained.
* **SaProt-dG and ESM3-dG** are not separate publications. The closest real
  implementation is github.com/yehlincho/absolute-stability-predictor, which
  fine-tunes SaProt 650M and ESM3 with LoRA plus a stability head to predict
  absolute dG and ddG, with weights on Hugging Face under
  Yehlin/absolute-stability. That is supervised regression, not a
  masked-marginal likelihood difference. Both are therefore registered twice,
  once against that implementation and once as an explicitly labelled
  masked-marginal fallback, so the distinction is never blurred in the output.
"""

from __future__ import annotations

from vepbench.predictors.base import PredictorSpec, SubprocessPredictor, register
from vepbench.schema import MORE_DELETERIOUS, MORE_TOLERATED, ScoreConvention

_FITNESS_TOLERATED = ScoreConvention(
    score_type="log_likelihood_ratio", higher_is=MORE_TOLERATED, verified=True,
    source="fitness log-likelihood ratio of mutant to wild type; positive favours "
           "the mutant",
)

register(SubprocessPredictor(
    PredictorSpec(
        name="s3f", display_name="S3F (sequence-structure-surface)",
        category="structure_aware", convention=_FITNESS_TOLERATED,
        citation="Zhang Z, Notin P, Huang Y, Lozano AC, Chenthamarakshan V, Marks D, "
                 "Das P, Tang J. Multi-Scale Representation Learning for Protein "
                 "Fitness Prediction. NeurIPS 2024. arXiv:2412.01108",
        repo_url="https://github.com/DeepGraphLearning/S3F",
        license="Check the repository; not stated in the paper",
        requires=frozenset({"sequence", "structure"}), gpu_required=True,
        approximate_vram_gb=8.0, env="s3f", model_version="S3F",
        notes="Combines a protein language model with residue-graph and "
              "molecular-surface encoders. The S3F-MSA variant also consumes an "
              "alignment; select it with the use_msa option.",
        install_hint="python -m vepbench install-env s3f",
    ),
    runner_name="s3f.py", options={"use_msa": False},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="venusrem", display_name="VenusREM", category="structure_aware",
        convention=_FITNESS_TOLERATED,
        citation="Tan Y et al. From high-throughput evaluation to wet-lab studies: "
                 "advancing mutation effect prediction with a retrieval-enhanced "
                 "model. Bioinformatics 2025;41(Suppl 1):i401-i409",
        repo_url="https://github.com/ai4protein/VenusREM",
        license="Check the repository",
        requires=frozenset({"sequence", "structure", "msa"}), gpu_recommended=True,
        approximate_vram_gb=4.0, env="venusrem", model_version="VenusREM",
        trained_on_proteingym=True,
        notes="Retrieval-enhanced: a structure-aware language model combined with "
              "alignment-derived residue preferences. Reported state of the art on 217 "
              "ProteinGym assays, so treat ProteinGym overlap as leakage.",
        install_hint="python -m vepbench install-env venusrem",
    ),
    runner_name="venusrem.py",
))

register(SubprocessPredictor(
    PredictorSpec(
        name="promep", display_name="ProMEP", category="structure_aware",
        convention=_FITNESS_TOLERATED,
        citation="Cheng P et al. Zero-shot prediction of mutation effects with "
                 "multimodal deep representation learning guides protein engineering. "
                 "Cell Research 2024;34:630-647",
        repo_url="https://github.com/wenjiegroup/ProMEP",
        license="Check the repository",
        requires=frozenset({"sequence", "structure"}), gpu_required=True,
        approximate_vram_gb=8.0, env="promep", model_version="ProMEP",
        notes="Alignment-free multimodal model trained on sequence and structure from "
              "roughly 160 million AlphaFold DB entries. Needs SE(3)-Transformer and "
              "DGL, the most fragile install here.",
        install_hint="python -m vepbench install-env promep",
    ),
    runner_name="promep.py",
))

register(SubprocessPredictor(
    PredictorSpec(
        name="metl_global", display_name="METL-Global", category="structure_aware",
        convention=ScoreConvention(
            score_type="rosetta_total_score_delta", higher_is=MORE_DELETERIOUS,
            verified=False,
            source="METL-Global predicts Rosetta biophysical attributes, and the total "
                   "score follows Rosetta's convention where higher energy is worse. "
                   "The fine-tuned METL-Local models instead predict assay-specific "
                   "fitness, whose direction is whatever the training assay used. "
                   "Confirm before benchmarking.",
        ),
        citation="Gelman S et al. Biophysics-based protein language models for protein "
                 "engineering. Nature Methods 2025",
        repo_url="https://github.com/gitter-lab/metl-pretrained", license="MIT",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=2.0, env="metl", model_version="METL-G-20M-3D",
        notes="Pretrained on synthetic Rosetta data. Zero-shot in that it needs no "
              "assay data, but it predicts biophysical attributes rather than fitness. "
              "Local models exist only for proteins the authors simulated, none of "
              "them a membrane protein.",
        install_hint="python -m vepbench install-env metl",
    ),
    runner_name="metl.py", options={"model": "metl-g-20m-3d"},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="saprot_dg", display_name="SaProt-dG (absolute stability predictor)",
        category="stability",
        convention=ScoreConvention(
            score_type="ddG_kcal_mol", higher_is=MORE_DELETERIOUS, verified=False,
            source="The absolute-stability-predictor repository predicts dG and ddG "
                   "with a LoRA stability head. Its sign convention is not stated in "
                   "the sources used to build this entry; confirm with "
                   "`vepbench check-conventions` before benchmarking.",
        ),
        citation="Cho Y et al. absolute-stability-predictor: fine-tuned SaProt and "
                 "ESM3 for absolute protein stability. github.com/yehlincho/"
                 "absolute-stability-predictor",
        repo_url="https://github.com/yehlincho/absolute-stability-predictor",
        license="Check the repository; SaProt itself is MIT",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=4.0, env="stability_dg",
        model_version="SaProt 650M + LoRA r=4 stability head, ensemble of 3",
        trained_on_megascale=True, supervised=True,
        notes="Supervised on experimental dG and ddG, so not zero-shot, and "
              "overlapping benchmark data is leakage. This is the published "
              "implementation; see saprot_dg_masked_marginal for the likelihood "
              "alternative.",
        install_hint="python -m vepbench install-env stability_dg",
    ),
    runner_name="stability_dg.py", options={"backbone": "saprot", "ensemble": 3},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="esm3_dg", display_name="ESM3-dG (absolute stability predictor)",
        category="stability",
        convention=ScoreConvention(
            score_type="ddG_kcal_mol", higher_is=MORE_DELETERIOUS, verified=False,
            source="Same repository and same caveat as saprot_dg.",
        ),
        citation="Cho Y et al. absolute-stability-predictor. github.com/yehlincho/"
                 "absolute-stability-predictor",
        repo_url="https://github.com/yehlincho/absolute-stability-predictor",
        license="Check the repository; ESM3 weights are under the EvolutionaryScale "
                "Community licence",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=8.0, env="stability_dg",
        model_version="ESM3 + LoRA stability head",
        trained_on_megascale=True, supervised=True,
        notes="Requires accepting the ESM3 licence on Hugging Face. See "
              "esm3_dg_masked_marginal for the likelihood alternative.",
        install_hint="python -m vepbench install-env stability_dg, then export HF_TOKEN",
    ),
    runner_name="stability_dg.py", options={"backbone": "esm3", "ensemble": 3},
))

# The masked-marginal readings of the same two backbones. These are NOT the
# published dG models; they are the standard likelihood-difference score, kept
# separate so the two can never be confused in a merged table.
register(SubprocessPredictor(
    PredictorSpec(
        name="saprot_dg_masked_marginal",
        display_name="SaProt masked-marginal (labelled dG fallback)",
        category="stability",
        convention=ScoreConvention(
            score_type="log_likelihood_ratio", higher_is=MORE_TOLERATED, verified=True,
            source="standard masked marginal log p(mut) - log p(wt) from SaProt",
        ),
        citation="Su J et al. SaProt. ICLR 2024 (scoring from Meier et al. 2021)",
        repo_url="https://github.com/westlake-repl/SaProt", license="MIT",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=3.0, env="huggingface",
        model_version="SaProt_650M_AF2 masked marginals",
        notes="NOT a free-energy estimate and NOT the published SaProt-dG. This is the "
              "standard likelihood-ratio score, provided as the documented fallback. "
              "Units are log-likelihood, not kcal/mol.",
        install_hint="python -m vepbench install-env huggingface",
    ),
    runner_name="saprot.py", options={"model": "westlake-repl/SaProt_650M_AF2"},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="esm3_dg_masked_marginal",
        display_name="ESM3 masked-marginal (labelled dG fallback)",
        category="stability",
        convention=ScoreConvention(
            score_type="log_likelihood_ratio", higher_is=MORE_TOLERATED, verified=True,
            source="standard masked marginal log p(mut) - log p(wt) from ESM3",
        ),
        citation="Hayes T et al. Science 2025 (scoring from Meier et al. 2021)",
        repo_url="https://github.com/evolutionaryscale/esm",
        license="EvolutionaryScale Community licence",
        requires=frozenset({"sequence", "structure"}), gpu_recommended=True,
        approximate_vram_gb=7.0, env="esm",
        model_version="esm3-sm-open-v1 masked marginals, structure conditioned",
        notes="NOT a free-energy estimate. Same caveat as "
              "saprot_dg_masked_marginal; here the structure track is used as "
              "conditioning, which is what distinguishes it from esm3_open.",
        install_hint="python -m vepbench install-env esm, then export HF_TOKEN",
    ),
    runner_name="esm3_masked_marginals.py",
    options={"model": "esm3-sm-open-v1", "structure_conditioned": True},
))
