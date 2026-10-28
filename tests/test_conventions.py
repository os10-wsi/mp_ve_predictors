"""Registry integrity and sign conventions.

Getting a sign wrong silently inverts a benchmark, turning the best predictor
into the worst. These tests pin the direction of every registered predictor
and assert that harmonization flips exactly what it should.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vepbench.predictors import REGISTRY
from vepbench.predictors.base import RUNNER_DIR, SubprocessPredictor
from vepbench.schema import MORE_DELETERIOUS, MORE_TOLERATED, VALID_DIRECTIONS, harmonize

# The direction each method's raw score runs in, with the reason recorded in
# the predictor entry. Changing one of these is a deliberate act, not a typo.
EXPECTED_DIRECTION = {
    # Likelihood ratios: positive favours the mutant, so higher is tolerated.
    "esm1v_full": MORE_TOLERATED, "esm1v_domain": MORE_TOLERATED,
    "esm2_650m": MORE_TOLERATED, "esm2_3b": MORE_TOLERATED,
    "esmc_600m": MORE_TOLERATED, "esm3_open": MORE_TOLERATED,
    "esm_if1": MORE_TOLERATED, "proteinmpnn": MORE_TOLERATED,
    "saprot": MORE_TOLERATED, "saprot_1p3b": MORE_TOLERATED,
    "prosst": MORE_TOLERATED, "prott5": MORE_TOLERATED, "ankh": MORE_TOLERATED,
    "tranception": MORE_TOLERATED, "trancepteve": MORE_TOLERATED,
    "gemme": MORE_TOLERATED, "evmutation": MORE_TOLERATED,
    "deepsequence": MORE_TOLERATED, "popeve": MORE_TOLERATED,
    "s3f": MORE_TOLERATED, "venusrem": MORE_TOLERATED, "promep": MORE_TOLERATED,
    "saprot_dg_masked_marginal": MORE_TOLERATED,
    "esm3_dg_masked_marginal": MORE_TOLERATED,
    "blosum62_baseline": MORE_TOLERATED, "pssm_independent": MORE_TOLERATED,
    # EVE's evolutionary index is the negative log-likelihood ratio.
    "eve_full": MORE_DELETERIOUS, "eve_domain": MORE_DELETERIOUS,
    # VESPA outputs a probability that the substitution has an effect.
    "vespa": MORE_DELETERIOUS,
    # AlphaMissense's am_pathogenicity runs 0 to 1, higher being more likely
    # pathogenic.
    "alphamissense": MORE_DELETERIOUS,
    # Rosetta and FoldX: positive ddG is destabilising.
    "foldx": MORE_DELETERIOUS, "rasp": MORE_DELETERIOUS,
    "thermompnn": MORE_DELETERIOUS, "saprot_dg": MORE_DELETERIOUS,
    "esm3_dg": MORE_DELETERIOUS, "metl_global": MORE_DELETERIOUS,
}

# Conventions not confirmed against the method's own documentation. Shrinking
# this set is good; growing it silently is not.
EXPECTED_UNVERIFIED = {"thermompnn", "metl_global", "saprot_dg", "esm3_dg"}


def test_every_predictor_has_an_expected_direction():
    assert set(EXPECTED_DIRECTION) == set(REGISTRY), (
        "a predictor was added or removed without updating EXPECTED_DIRECTION"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_DIRECTION))
def test_direction_matches_expectation(name):
    assert REGISTRY[name].spec.convention.higher_is == EXPECTED_DIRECTION[name]


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_harmonization_flips_exactly_when_it_should(name):
    convention = REGISTRY[name].spec.convention
    assert convention.higher_is in VALID_DIRECTIONS
    if convention.higher_is == MORE_DELETERIOUS:
        assert harmonize(3.0, convention.higher_is) == 3.0
        assert harmonize(-3.0, convention.higher_is) == -3.0
    else:
        assert harmonize(3.0, convention.higher_is) == -3.0
        assert harmonize(-3.0, convention.higher_is) == 3.0


def test_unverified_set_has_not_grown_silently():
    actual = {n for n, p in REGISTRY.items() if not p.spec.convention.verified}
    assert actual == EXPECTED_UNVERIFIED


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_convention_cites_a_source(name):
    source = REGISTRY[name].spec.convention.source
    assert source and len(source) > 20, f"{name} has no usable convention source"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_predictor_has_metadata(name):
    spec = REGISTRY[name].spec
    assert spec.citation, f"{name} has no citation"
    assert spec.license, f"{name} has no licence"
    assert spec.category
    assert "sequence" in spec.requires


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_subprocess_predictors_have_a_runner_and_an_environment(name):
    predictor = REGISTRY[name]
    if not isinstance(predictor, SubprocessPredictor):
        return
    assert predictor.spec.env, f"{name} has no environment"
    assert (RUNNER_DIR / predictor.runner_name).exists(), (
        f"{name} refers to a missing runner {predictor.runner_name}"
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_gpu_models_declare_their_memory(name):
    """Anything that wants a GPU should say roughly how much memory it needs."""
    spec = REGISTRY[name].spec
    if spec.gpu_required or spec.gpu_recommended:
        assert spec.approximate_vram_gb, f"{name} wants a GPU but declares no VRAM"
        assert 0.5 <= spec.approximate_vram_gb <= 80


def test_leakage_flags_are_set_for_megascale_models():
    for name in ("thermompnn", "saprot_dg", "esm3_dg"):
        assert REGISTRY[name].spec.trained_on_megascale
    for name in ("esm1v_full", "gemme"):
        assert not REGISTRY[name].spec.trained_on_megascale


def test_proteingym_flag_is_set_where_claimed():
    assert REGISTRY["venusrem"].spec.trained_on_proteingym


def test_stability_methods_report_energy_units():
    for name in ("foldx", "rasp", "thermompnn", "saprot_dg", "esm3_dg"):
        assert REGISTRY[name].spec.convention.score_type == "ddG_kcal_mol"


def test_masked_marginal_fallbacks_are_not_labelled_as_energies():
    """The dG fallbacks must never claim kcal/mol units."""
    for name in ("saprot_dg_masked_marginal", "esm3_dg_masked_marginal"):
        spec = REGISTRY[name].spec
        assert spec.convention.score_type == "log_likelihood_ratio"
        assert "NOT a free-energy estimate" in spec.notes


def test_predictors_md_is_up_to_date():
    """PREDICTORS.md is generated; a stale copy means someone edited by hand."""
    root = Path(__file__).resolve().parents[1]
    target = root / "PREDICTORS.md"
    assert target.exists(), "PREDICTORS.md is missing"
    generator = root / "scripts" / "generate_predictors_md.py"
    spec = importlib.util.spec_from_file_location("generate_predictors_md", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build() == target.read_text(), (
        "PREDICTORS.md is out of date; regenerate it with "
        "`python scripts/generate_predictors_md.py`"
    )


def test_upstream_entry_points_are_not_run_in_isolated_mode():
    """`-I` implies `-P`, which keeps the script's own directory off sys.path.

    An upstream research repository's entry point then cannot import its
    siblings: ProteinMPNN's protein_mpnn_utils, EVE's model code. `-s` gives
    the isolation that matters (no user site-packages) without that.
    """
    runners = sorted((Path(__file__).parent.parent / "scripts" / "runners").glob("*.py"))
    assert runners, "no runners found"
    offenders = [
        f"{path.name}:{number}"
        for path in runners
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if 'sys.executable, "-I"' in line
    ]
    assert not offenders, (
        "use -s for upstream entry points, not -I:\n" + "\n".join(offenders)
    )
