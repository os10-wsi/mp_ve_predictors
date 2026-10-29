"""A run adds its predictors to the output files instead of replacing them.

The batch these guard: fourteen predictors run first, then a single extra one
is run weeks later, sometimes while the first batch is still finishing. Both
sets of columns have to survive.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pytest

from vepbench.config import RunConfig
from vepbench.outputs import (
    load_existing_long,
    merge_long,
    merge_status_entries,
    output_lock,
    write_csv,
)
from vepbench.runner import run_protein
from vepbench.schema import MORE_DELETERIOUS, ScoreConvention, build_long_frame, to_wide
from vepbench.variants import saturation_variants

FIXTURES = Path(__file__).parent / "fixtures"
ACCESSION = "P26678"


def _config(paths, structure_path, predictors, **kwargs):
    return RunConfig(predictors=list(predictors), structure_override=structure_path,
                     msa_backend="none", offline=True, paths=paths, **kwargs)


def _long_frame(sequence: str, predictor: str, score: float = 1.0) -> pd.DataFrame:
    """A stand-in for a predictor this container cannot run."""
    frame = saturation_variants(sequence).loc[:, ["position", "wt_aa", "mut_aa"]].copy()
    frame["score"] = score
    return build_long_frame(
        uniprot_id=ACCESSION, frame=frame, predictor=predictor,
        convention=ScoreConvention(score_type="ddG_kcal_mol", higher_is=MORE_DELETERIOUS,
                                   verified=True, source="test fixture"),
        model_version="test",
    )


def _seed(paths, sequence, predictor):
    """Write an earlier run's output files, as the first batch would leave them."""
    output_dir = paths.output_dir(ACCESSION)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_frame = _long_frame(sequence, predictor)
    write_csv(output_dir / "predictions_long.csv", long_frame)
    write_csv(output_dir / "predictions_wide.csv", to_wide(long_frame))
    (output_dir / "status.json").write_text(json.dumps({
        "uniprot_id": ACCESSION, "generated": 0,
        "predictors": [{"predictor": predictor, "status": "ok", "seconds": 1.0}],
        "counts": {"ok": 1, "cached": 0, "unavailable": 0, "failed": 0},
    }))
    return output_dir


def test_a_narrow_run_keeps_the_earlier_columns(paths, structure_path, sequence):
    """The whole point: running one predictor must not drop the other fourteen."""
    _seed(paths, sequence, "pretend_rasp")

    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))

    assert "pretend_rasp" in result.wide_frame.columns
    assert "blosum62_baseline" in result.wide_frame.columns
    on_disk = pd.read_csv(result.output_dir / "predictions_wide.csv")
    assert {"pretend_rasp", "blosum62_baseline"} <= set(on_disk.columns)
    assert len(on_disk) == len(sequence) * 19


def test_the_harmonized_block_survives_too(paths, structure_path, sequence):
    _seed(paths, sequence, "pretend_rasp")
    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))
    assert "pretend_rasp__harmonized" in result.wide_frame.columns
    assert result.wide_frame["pretend_rasp"].notna().all()


def test_rerunning_a_predictor_replaces_its_rows(paths, structure_path, sequence):
    """Otherwise a repeated run would stack duplicate rows in the long table."""
    _seed(paths, sequence, "blosum62_baseline")
    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))

    counts = result.long_frame["predictor"].value_counts()
    assert counts["blosum62_baseline"] == len(sequence) * 19
    # The seeded placeholder scored everything 1.0; the real run must have won.
    assert not (result.long_frame["score"] == 1.0).all()


def test_status_json_accumulates_across_runs(paths, structure_path, sequence):
    output_dir = _seed(paths, sequence, "pretend_rasp")
    run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))

    payload = json.loads((output_dir / "status.json").read_text())
    assert {e["predictor"] for e in payload["predictors"]} == {
        "pretend_rasp", "blosum62_baseline"
    }
    assert payload["counts"]["ok"] == 2


def test_a_failed_predictor_leaves_the_earlier_result_standing(paths, structure_path, sequence):
    """A re-run that fails must not delete the column it was meant to refresh."""
    _seed(paths, sequence, "pretend_rasp")
    result = run_protein(ACCESSION, _config(
        paths, structure_path, ["blosum62_baseline", "esm1v_full"]
    ))
    assert "esm1v_full" in result.unavailable
    assert "pretend_rasp" in result.wide_frame.columns


def test_a_table_for_another_sequence_is_set_aside_not_merged(paths, sequence):
    """Mixing isoforms would silently disagree about the wild-type residue."""
    output_dir = paths.output_dir(ACCESSION)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predictions_long.csv"
    wrong = _long_frame(sequence, "stale")
    wrong["wt_aa"] = "W"          # no longer matches the real sequence
    write_csv(path, wrong)

    kept = load_existing_long(path, sequence)

    assert kept.empty
    assert path.with_suffix(".csv.superseded").exists()


def test_an_unreadable_table_does_not_kill_the_run(paths, structure_path, sequence):
    output_dir = paths.output_dir(ACCESSION)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions_long.csv").write_text("not,a,valid\nlong,table\n")

    result = run_protein(ACCESSION, _config(paths, structure_path, ["blosum62_baseline"]))

    assert "blosum62_baseline" in result.wide_frame.columns


def test_merge_long_keeps_predictors_this_run_did_not_touch(sequence):
    existing = pd.concat([_long_frame(sequence, "a"), _long_frame(sequence, "b")])
    merged = merge_long(existing, _long_frame(sequence, "b", score=9.0))

    assert set(merged["predictor"]) == {"a", "b"}
    assert (merged.loc[merged["predictor"] == "b", "score"] == 9.0).all()


def test_merge_long_with_nothing_computed_keeps_everything(sequence):
    existing = _long_frame(sequence, "a")
    merged = merge_long(existing, existing.iloc[0:0])
    assert set(merged["predictor"]) == {"a"}


def test_merge_status_entries_prefers_the_newer_record():
    merged = merge_status_entries(
        [{"predictor": "a", "status": "ok"}, {"predictor": "b", "status": "failed"}],
        [{"predictor": "b", "status": "ok"}],
    )
    assert merged == [{"predictor": "a", "status": "ok"}, {"predictor": "b", "status": "ok"}]


# --- concurrency -----------------------------------------------------------

def _append_under_lock(args):
    """One worker's read-modify-write, the same shape run_protein performs."""
    output_dir, name = Path(args[0]), args[1]
    path = output_dir / "predictions_long.csv"
    with output_lock(output_dir):
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["predictor"])
        frame = pd.concat([existing, pd.DataFrame({"predictor": [name]})], ignore_index=True)
        write_csv(path, frame)


def test_concurrent_writers_do_not_lose_each_others_rows(tmp_path):
    """Without the lock both would read the same table and one write would vanish."""
    output_dir = tmp_path / "P26678"
    output_dir.mkdir()
    names = [f"predictor_{i}" for i in range(12)]

    with mp.get_context("spawn").Pool(6) as pool:
        pool.map(_append_under_lock, [(str(output_dir), n) for n in names])

    written = pd.read_csv(output_dir / "predictions_long.csv")
    assert sorted(written["predictor"]) == sorted(names)


def test_write_csv_leaves_no_partial_file(tmp_path, sequence):
    path = tmp_path / "predictions_long.csv"
    write_csv(path, _long_frame(sequence, "a"))
    assert not list(tmp_path.glob("*.tmp"))
    assert len(pd.read_csv(path)) == len(sequence) * 19


class TestConcurrentWrites:
    """A job array and `proteome fetch-data` write the same caches at once.

    The race these guard is hard to reproduce on a local filesystem, where
    one large write() usually lands whole. On Lustre it is striped across
    servers and does not, which is where this code runs. So the property
    tested is the one that is actually decidable: each write gets a private
    scratch file, which is what makes the rename meaningful.
    """

    def test_each_write_uses_a_scratch_path_of_its_own(self, tmp_path):
        from vepbench.config import unique_temp

        target = tmp_path / "P00001.json"
        names = {unique_temp(target) for _ in range(200)}
        assert len(names) == 200
        assert all(p.parent == target.parent for p in names)
        assert all(p.name.endswith(".tmp") for p in names)

    def test_the_scratch_name_includes_the_process(self, tmp_path):
        import os

        from vepbench.config import unique_temp

        assert str(os.getpid()) in unique_temp(tmp_path / "x.json").name

    def test_no_scratch_files_are_left_behind(self, tmp_path):
        from vepbench.config import write_json

        write_json(tmp_path / "thing.json", {"a": 1})
        assert list(tmp_path.glob("*.tmp")) == []

    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path):
        import json

        from vepbench.config import write_json

        target = tmp_path / "thing.json"
        write_json(target, {"good": True})
        # default=str swallows most unserialisable values, so this needs a
        # payload json genuinely cannot encode.
        circular: dict = {}
        circular["self"] = circular
        with pytest.raises(ValueError):
            write_json(target, circular)
        assert json.loads(target.read_text()) == {"good": True}
        assert list(tmp_path.glob("*.tmp")) == []

    def test_many_writers_of_one_file_leave_it_readable(self, tmp_path):
        """A smoke test: whatever wins, the file parses and is not a blend."""
        import json
        import multiprocessing

        from vepbench.config import write_json

        target = tmp_path / "P00001.json"

        def writer(letter, length, times):
            for _ in range(times):
                write_json(target, {"sequence": letter * length})

        processes = [
            multiprocessing.Process(target=writer, args=("A", 50, 40)),
            multiprocessing.Process(target=writer, args=("B", 500_000, 40)),
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=60)

        payload = json.loads(target.read_text())
        assert set(payload["sequence"]) in ({"A"}, {"B"})
        assert list(tmp_path.glob("*.tmp")) == []
