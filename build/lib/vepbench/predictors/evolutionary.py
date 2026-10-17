"""Evolutionary-model predictors, all of which need an alignment.

Sign conventions differ across this group and are the easiest thing to get
wrong, so each entry records where its convention was read from.
"""

from __future__ import annotations

from vepbench.predictors.base import PredictorSpec, SubprocessPredictor, register
from vepbench.schema import MORE_DELETERIOUS, MORE_TOLERATED, ScoreConvention

_LLR_TOLERATED = ScoreConvention(
    score_type="log_likelihood_ratio", higher_is=MORE_TOLERATED, verified=True,
    source="log-likelihood ratio of mutant to wild type; positive favours the mutant",
)
_EVE_INDEX = ScoreConvention(
    score_type="evolutionary_index", higher_is=MORE_DELETERIOUS, verified=True,
    source="Frazer et al. 2021 Nature: the evolutionary index is the negative of the "
           "mutant-to-wild-type log-likelihood ratio, so higher is more pathogenic",
)

register(SubprocessPredictor(
    PredictorSpec(
        name="eve_full", display_name="EVE (full length)", category="evolutionary",
        convention=_EVE_INDEX,
        citation="Frazer J et al. Disease variant prediction with deep generative "
                 "models of evolutionary data. Nature 2021;599:91-95",
        repo_url="https://github.com/OATML-Markslab/EVE", license="MIT",
        requires=frozenset({"sequence", "msa"}), gpu_required=True,
        approximate_vram_gb=8.0, env="eve", model_version="EVE VAE, trained per protein",
        notes="A VAE is trained from scratch on this protein's alignment, so the cost "
              "is GPU hours per protein rather than per variant. Unreliable on a "
              "shallow alignment; the pipeline warns below Neff 100.",
        install_hint="python -m vepbench install-env eve",
    ),
    runner_name="eve.py", options={"use_domain": False},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="eve_domain", display_name="EVE (domain only)", category="evolutionary",
        convention=_EVE_INDEX,
        citation="Frazer J et al. Nature 2021;599:91-95",
        repo_url="https://github.com/OATML-Markslab/EVE", license="MIT",
        requires=frozenset({"sequence", "msa", "domain"}), gpu_required=True,
        approximate_vram_gb=8.0, env="eve", model_version="EVE VAE, trained per domain",
        notes="The domain-level variant EVE was published with. The alignment is "
              "trimmed to the domain window before training, which usually deepens it.",
        install_hint="python -m vepbench install-env eve",
    ),
    runner_name="eve.py", options={"use_domain": True},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="popeve", display_name="popEVE", category="evolutionary",
        convention=ScoreConvention(
            score_type="popeve_score", higher_is=MORE_TOLERATED, verified=True,
            source="Orenbuch et al. 2023: popEVE scores are calibrated so that more "
                   "negative is more damaging",
        ),
        citation="Orenbuch R et al. popEVE. Marks lab, 2023 preprint",
        repo_url="https://github.com/debbiemarkslab/popEVE", license="MIT",
        requires=frozenset({"sequence", "msa"}), gpu_required=True,
        approximate_vram_gb=4.0, env="eve",
        model_version="popEVE (ESM-1v + EVE Gaussian process)",
        notes="Proteome-wide precomputed scores exist for human proteins and are used "
              "when available. A local run needs both EVE and ESM-1v scores first.",
        install_hint="python -m vepbench install-env eve",
    ),
    runner_name="popeve.py", options={"use_precomputed": True},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="tranception", display_name="Tranception L (retrieval)",
        category="evolutionary", convention=_LLR_TOLERATED,
        citation="Notin P et al. Tranception: protein fitness prediction with "
                 "autoregressive transformers and inference-time retrieval. ICML 2022",
        repo_url="https://github.com/OATML-Markslab/Tranception", license="MIT",
        requires=frozenset({"sequence", "msa"}), gpu_recommended=True,
        approximate_vram_gb=6.0, env="tranception", model_version="Tranception_Large",
        notes="Run with inference-time retrieval from the alignment, the configuration "
              "the paper benchmarks. Without retrieval it degrades to the "
              "autoregressive model alone.",
        install_hint="python -m vepbench install-env tranception",
    ),
    runner_name="tranception.py", options={"retrieval": True},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="trancepteve", display_name="TranceptEVE L", category="evolutionary",
        convention=_LLR_TOLERATED,
        citation="Notin P et al. TranceptEVE: combining family-specific and "
                 "family-agnostic models of protein sequences. NeurIPS 2022 (LMRL)",
        repo_url="https://github.com/OATML-Markslab/Tranception", license="MIT",
        requires=frozenset({"sequence", "msa"}), gpu_required=True,
        approximate_vram_gb=8.0, env="tranception", model_version="TranceptEVE_L",
        notes="Needs EVE trained for the same protein first: it mixes the EVE family "
              "model with Tranception's autoregressive scores.",
        install_hint="python -m vepbench install-env tranception, after install-env eve",
    ),
    runner_name="trancepteve.py",
))

register(SubprocessPredictor(
    PredictorSpec(
        name="gemme", display_name="GEMME", category="evolutionary",
        convention=ScoreConvention(
            score_type="gemme_score", higher_is=MORE_TOLERATED, verified=True,
            source="Laine et al. 2019: GEMME predicted effects are negative, with more "
                   "negative meaning more deleterious",
        ),
        citation="Laine E, Karami Y, Carbone A. GEMME: a simple and fast global "
                 "epistatic model predicting mutational effects. Mol Biol Evol "
                 "2019;36:2604-2619",
        repo_url="https://github.com/elolaine/GEMME",
        license="Free for academic use",
        requires=frozenset({"sequence", "msa"}), env="gemme",
        model_version="GEMME 1.1 with JET2",
        notes="Needs Java. Fast, no per-protein training, and one of the stronger "
              "alignment-only methods on ProteinGym.",
        install_hint="python -m vepbench install-env gemme",
    ),
    runner_name="gemme.py",
))

register(SubprocessPredictor(
    PredictorSpec(
        name="evmutation", display_name="EVmutation (epistatic Potts model)",
        category="evolutionary",
        convention=ScoreConvention(
            score_type="potts_energy_difference", higher_is=MORE_TOLERATED, verified=True,
            source="Hopf et al. 2017: the epistatic score is E(mutant) - E(wild type) "
                   "in the statistical energy, where more negative is more deleterious",
        ),
        citation="Hopf TA et al. Mutation effects predicted from sequence co-variation. "
                 "Nat Biotechnol 2017;35:128-135",
        repo_url="https://github.com/debbiemarkslab/EVcouplings", license="MIT",
        requires=frozenset({"sequence", "msa"}), env="evcouplings",
        model_version="plmc asymmetric, lambda_J scaled by (N-1)",
        notes="Classic baseline. Fits a pairwise undirected graphical model with plmc "
              "and needs a deep alignment to fit couplings reliably.",
        install_hint="python -m vepbench install-env evcouplings",
    ),
    runner_name="evmutation.py",
))

register(SubprocessPredictor(
    PredictorSpec(
        name="deepsequence", display_name="DeepSequence", category="evolutionary",
        convention=ScoreConvention(
            score_type="elbo_difference", higher_is=MORE_TOLERATED, verified=True,
            source="Riesselman et al. 2018: the effect is the ELBO of the mutant minus "
                   "the ELBO of the wild type, so higher is more tolerated",
        ),
        citation="Riesselman AJ, Ingraham JB, Marks DS. Deep generative models of "
                 "genetic variation capture the effects of mutations. Nat Methods "
                 "2018;15:816-822",
        repo_url="https://github.com/debbiemarkslab/DeepSequence", license="MIT",
        requires=frozenset({"sequence", "msa"}), gpu_recommended=True,
        approximate_vram_gb=4.0, env="deepsequence",
        model_version="DeepSequence VAE, 2000 ELBO samples",
        notes="Published against Python 2.7 and Theano. The container in "
              "envs/deepsequence/Dockerfile is the practical way to run it today.",
        install_hint="docker build -t vepbench-deepsequence envs/deepsequence",
    ),
    runner_name="deepsequence.py",
))

register(SubprocessPredictor(
    PredictorSpec(
        name="vespa", display_name="VESPA", category="evolutionary",
        convention=ScoreConvention(
            score_type="effect_probability", higher_is=MORE_DELETERIOUS, verified=True,
            source="Marquet et al. 2022: VESPA outputs a probability in [0,1] that the "
                   "substitution has an effect, so higher is more deleterious",
        ),
        citation="Marquet C et al. Embeddings from protein language models predict "
                 "conservation and variant effects. Hum Genet 2022;141:1629-1647",
        repo_url="https://github.com/Rostlab/VESPA", license="AFL-3.0",
        requires=frozenset({"sequence"}), gpu_recommended=True,
        approximate_vram_gb=6.0, env="vespa",
        model_version="VESPA (ProtT5 embeddings + conservation)", supervised=True,
        notes="Supervised on variant-effect data. VESPA-light is the faster variant, "
              "selected with the light option.",
        install_hint="python -m vepbench install-env vespa",
    ),
    runner_name="vespa.py", options={"light": False},
))
