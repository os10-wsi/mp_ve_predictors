"""Stability predictors (delta delta G).

Every method here was trained or validated largely on soluble, globular
proteins, mostly small monomeric domains. Membrane proteins are out of
distribution for all of them. The pipeline does not alter the methods; it
flags the caveat in metadata.json and in the README.

Sign conventions are not uniform across this group, which is why each entry
records its source and whether the direction has been confirmed.
"""

from __future__ import annotations

import os

from vepbench.predictors.base import (
    Availability,
    PredictorSpec,
    SubprocessPredictor,
    register,
)
from vepbench.protein import ProteinContext
from vepbench.schema import MORE_DELETERIOUS, ScoreConvention

# Rosetta-style: ddG = G(mutant) - G(wild type), positive is destabilising.
_DDG_POSITIVE_DESTABILISING = ScoreConvention(
    score_type="ddG_kcal_mol", higher_is=MORE_DELETERIOUS, verified=True,
    source="ddG = G(mutant) - G(wild type); positive is destabilising, the "
           "Rosetta and FoldX convention",
)


class FoldXPredictor(SubprocessPredictor):
    """FoldX needs a licensed binary that cannot be redistributed."""

    def _extra_availability(self, protein: ProteinContext, *, paths=None) -> Availability:
        base = super()._extra_availability(protein, paths=paths)
        if not base.available:
            return base
        binary = os.environ.get("FOLDX_BINARY")
        if not binary or not os.path.exists(binary):
            return Availability(
                False,
                "FoldX requires a licensed binary. Obtain it from foldxsuite.crg.eu and "
                "set FOLDX_BINARY to its path, plus FOLDX_ROTABASE if your version "
                "needs rotabase.txt. The binary is deliberately not committed here.",
            )
        return Availability(True)


register(SubprocessPredictor(
    PredictorSpec(
        name="thermompnn", display_name="ThermoMPNN", category="stability",
        convention=ScoreConvention(
            score_type="ddG_kcal_mol", higher_is=MORE_DELETERIOUS, verified=False,
            source="Dieckhaus et al. 2024 report ddG with positive meaning "
                   "destabilising, but the raw Megascale data uses the opposite sign "
                   "and some forks flip it. Confirm with `vepbench check-conventions` "
                   "before benchmarking.",
        ),
        citation="Dieckhaus H, Brocidiacono M, Randolph NZ, Kuhlman B. Transfer "
                 "learning to leverage larger datasets for improved prediction of "
                 "protein stability changes. PNAS 2024;121:e2314853121",
        repo_url="https://github.com/Kuhlman-Lab/ThermoMPNN", license="MIT",
        requires=frozenset({"sequence", "structure"}), approximate_vram_gb=1.0,
        env="proteinmpnn", model_version="ThermoMPNN v1 (transfer-learned on Megascale)",
        trained_on_megascale=True, supervised=True,
        notes="Trained on Tsuboyama et al. 2023, so any benchmark overlapping that "
              "dataset is leaked. Built on ProteinMPNN features, hence the shared "
              "environment. Scores a whole saturation scan in one pass.",
        install_hint="python3 -m vepbench install-env proteinmpnn",
    ),
    runner_name="thermompnn.py",
))

register(FoldXPredictor(
    PredictorSpec(
        name="foldx", display_name="FoldX BuildModel", category="stability",
        convention=_DDG_POSITIVE_DESTABILISING,
        citation="Schymkowitz J et al. The FoldX web server: an online force field. "
                 "Nucleic Acids Res 2005;33:W382-W388",
        repo_url="https://foldxsuite.crg.eu/",
        license="Proprietary; free academic licence on request",
        requires=frozenset({"sequence", "structure"}), env="foldx",
        model_version="set by FOLDX_BINARY at run time",
        notes="RepairPDB runs once per structure and is cached, then BuildModel runs "
              "per variant. This is the slow path: seconds to minutes per variant, so "
              "a saturation scan of a large protein is expensive. Results are cached "
              "per variant, so an interrupted scan resumes. The empirical force field "
              "was parameterised on soluble proteins.",
        install_hint="export FOLDX_BINARY=/path/to/foldx",
    ),
    runner_name="foldx.py", options={"repair": True, "num_runs": 1},
))

register(SubprocessPredictor(
    PredictorSpec(
        name="rasp", display_name="RaSP", category="stability",
        convention=_DDG_POSITIVE_DESTABILISING,
        citation="Blaabjerg LM et al. Rapid protein stability prediction using deep "
                 "learning representations. eLife 2023;12:e82593",
        repo_url="https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg",
        license="Apache-2.0", requires=frozenset({"sequence", "structure"}),
        approximate_vram_gb=2.0, env="rasp",
        model_version="RaSP cavity model + downstream ddG head", supervised=True,
        notes="Trained to reproduce Rosetta cartesian_ddg, so it inherits Rosetta's "
              "sign convention and its soluble-protein bias. Fast enough for "
              "saturation scans, which is its advantage over FoldX and Rosetta.",
        install_hint="python3 -m vepbench install-env rasp",
    ),
    runner_name="rasp.py",
))
