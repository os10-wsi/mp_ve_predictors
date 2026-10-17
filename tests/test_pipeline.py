"""End-to-end runs, caching, failure isolation and the output fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vepbench.config import RunConfig
from vepbench.runner import STATUS_CACHED, STATUS_OK, run_protein

FIXTURES = Path(__file__).parent / "fixtures"
ACCESSION = "P26678"
EXPECTED_LONG = FIXTURES / "P26678_blosum62_baseline_long.csv"


def _config(paths, structure_path, predictors, **kwargs):
    return RunConfig(predictors=list(predictors), structure_override=structure_path,
                     msa_backend="none", offline=True, paths=paths, **kwargs)


def test_saturation_output_shape(paths, structure_path, sequence):
    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    assert len(result.long_frame) == len(sequence) * 19
    assert len(result.wide_frame) == len(sequence) * 19
    assert result.succeeded == ["blosum62_baseline"]
    assert result.failed == []


def test_include_wt_adds_synonymous_rows(paths, structure_path, sequence):
    result = run_protein(ACCESSION, _config(paths, structure_path,
                                            ["blosum62_baseline"], include_wt=True))
    assert len(result.long_frame) == len(sequence) * 20


def test_every_row_matches_the_uniprot_sequence(paths, structure_path, sequence):
    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    for row in result.long_frame.itertuples():
        assert sequence[row.position - 1] == row.wt_aa


def test_outputs_are_written(paths, structure_path):
    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    for name in ("predictions_long.csv", "predictions_wide.csv",
                 "metadata.json", "status.json"):
        assert (result.output_dir / name).exists(), f"{name} was not written"


def test_status_json_records_every_predictor(paths, structure_path):
    result = run_protein(ACCESSION, _config(paths, structure_path,
                                            ["blosum62_baseline", "esm1v_full"]))
    payload = json.loads((result.output_dir / "status.json").read_text())
    assert {e["predictor"] for e in payload["predictors"]} == {
        "blosum62_baseline", "esm1v_full"
    }
    assert payload["counts"]["ok"] == 1
    assert payload["counts"]["unavailable"] == 1


def test_unavailable_predictor_does_not_stop_the_others(paths, structure_path):
    result = run_protein(ACCESSION, _config(
        paths, structure_path, ["esm1v_full", "blosum62_baseline", "foldx"]
    ))
    assert "blosum62_baseline" in result.succeeded
    assert set(result.unavailable) == {"esm1v_full", "foldx"}
    assert result.failed == []
    outcomes = {o.name: o for o in result.outcomes}
    assert "not installed" in outcomes["esm1v_full"].message


def test_unavailable_predictors_contribute_no_rows(paths, structure_path, sequence):
    result = run_protein(ACCESSION, _config(paths, structure_path,
                                            ["esm1v_full", "blosum62_baseline"]))
    assert set(result.long_frame["predictor"]) == {"blosum62_baseline"}
    assert len(result.long_frame) == len(sequence) * 19


def test_results_are_cached_and_reused(paths, structure_path):
    first = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    assert first.outcomes[0].status == STATUS_OK
    second = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    assert second.outcomes[0].status == STATUS_CACHED
    pd.testing.assert_frame_equal(first.long_frame, second.long_frame)


def test_force_recomputes(paths, structure_path):
    run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    forced = run_protein(ACCESSION, _config(paths, structure_path,
                                            ["blosum62_baseline"], force=True))
    assert forced.outcomes[0].status == STATUS_OK


def test_cache_survives_an_unrelated_domain_choice(paths, structure_path):
    """Choosing a domain for a domain-level model must not evict other caches."""
    run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    mixed = run_protein(ACCESSION, _config(paths, structure_path,
                                           ["blosum62_baseline", "eve_domain"]))
    assert {o.name: o.status for o in mixed.outcomes}["blosum62_baseline"] == STATUS_CACHED


def test_cache_survives_a_hardware_change(paths, structure_path):
    """Moving a model to a GPU must not invalidate a cached score."""
    run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    again = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"],
                                           device="cpu", dtype="float32", batch_size=1))
    assert again.outcomes[0].status == STATUS_CACHED


def test_metadata_records_provenance(paths, structure_path, sequence):
    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    metadata = json.loads((result.output_dir / "metadata.json").read_text())
    assert metadata["uniprot"]["sequence"] == sequence
    assert metadata["uniprot"]["sequence_length"] == len(sequence)
    assert metadata["structure"]["uniprot_coverage"] == 1.0
    assert metadata["n_variants"] == len(sequence) * 19
    assert "git_hash" in metadata and "vepbench_version" in metadata
    assert metadata["predictors"]["blosum62_baseline"]["higher_is"] == "more_tolerated"


def test_membrane_caveat_is_recorded(paths, structure_path):
    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    caveats = " ".join(result.metadata["caveats"])
    assert "membrane protein" in caveats
    assert "soluble" in caveats


def test_alphafold_caveat_is_recorded(paths):
    """A predicted model has no bilayer; the run should say so."""
    config = RunConfig(predictors=["blosum62_baseline"], msa_backend="none",
                       offline=True, paths=paths)
    config.predictors = ["blosum62_baseline", "proteinmpnn"]  # pulls in a structure
    result = run_protein(ACCESSION, config)
    caveats = " ".join(result.metadata["caveats"])
    assert "AlphaFold prediction" in caveats


def test_leakage_caveat_is_recorded(paths, structure_path):
    result = run_protein(ACCESSION, _config(paths, structure_path,
                                            ["thermompnn", "blosum62_baseline"]))
    caveats = " ".join(result.metadata["caveats"])
    assert "Tsuboyama" in caveats
    assert "thermompnn" in caveats


def test_unknown_predictor_is_rejected(paths, structure_path):
    with pytest.raises(ValueError, match="unknown predictors"):
        run_protein(ACCESSION, _config(paths, structure_path, ["not_a_predictor"]))


def test_group_names_expand(paths, structure_path):
    result = run_protein(ACCESSION, _config(paths, structure_path, ["stability"]))
    assert {"foldx", "rasp", "thermompnn", "ddmut"} <= {o.name for o in result.outcomes}


def test_no_msa_group_excludes_alignment_models(paths, structure_path):
    result = run_protein(ACCESSION, _config(paths, structure_path, ["no_msa"]))
    names = {o.name for o in result.outcomes}
    assert "gemme" not in names
    assert "esm2_650m" in names


def test_output_matches_the_stored_fixture(paths, structure_path):
    """Regression fixture: the schema and the scores must not drift."""
    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    expected = pd.read_csv(EXPECTED_LONG)
    actual = result.long_frame.reset_index(drop=True)
    assert list(actual.columns) == list(expected.columns)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-9)


# --- the MSA path --------------------------------------------------------

def test_msa_based_predictor_runs_end_to_end(paths, structure_path, alignment, sequence):
    config = _config(paths, structure_path, ["pssm_independent"])
    config.msa_override = alignment
    result = run_protein(ACCESSION, config)
    assert result.succeeded == ["pssm_independent"]
    assert len(result.long_frame) == len(sequence) * 19
    assert result.long_frame["score"].notna().all()


def test_conservation_drives_the_msa_baseline(paths, structure_path, uniprot_record):
    """A conserved column must penalise substitutions more than a variable one.

    The alignment is built here rather than taken from the shared fixture, so
    the transmembrane helix is deliberately conserved and the cytoplasmic
    region deliberately variable, which is the pattern the score should find.
    """
    import random

    from vepbench.benchmark import spearman

    sequence = uniprot_record.sequence
    start, end = uniprot_record.transmembrane_segments[0]
    random.seed(11)
    lines = [f">query\n{sequence}"]
    for index in range(200):
        mutated = list(sequence)
        for position in range(len(mutated)):
            conserved = start - 1 <= position < end
            if random.random() < (0.04 if conserved else 0.30):
                mutated[position] = random.choice("ACDEFGHIKLMNPQRSTVWY")
        lines.append(f">h{index}\n{''.join(mutated)}")
    path = paths.msas / "crafted.a3m"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

    config = _config(paths, structure_path, ["pssm_independent"])
    config.msa_override = path
    config.features = ["entropy", "membrane"]
    wide = run_protein(ACCESSION, config).wide_frame

    by_segment = wide.groupby("in_transmembrane")["pssm_independent__harmonized"].mean()
    assert by_segment[True] > by_segment[False]
    entropy = wide.groupby("in_transmembrane")["msa_entropy_bits"].mean()
    assert entropy[True] < entropy[False]
    assert spearman(wide["pssm_independent__harmonized"].to_numpy(),
                    wide["msa_entropy_bits"].to_numpy()) < -0.5


def test_shallow_alignment_is_flagged_as_a_caveat(paths, structure_path, sequence):
    path = paths.msas / "shallow.a3m"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">query\n{sequence}\n>a\n{sequence}\n")
    config = _config(paths, structure_path, ["pssm_independent"])
    config.msa_override = path
    result = run_protein(ACCESSION, config)
    assert "shallow" in " ".join(result.metadata["caveats"]).lower()
