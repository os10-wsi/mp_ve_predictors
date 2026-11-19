"""Building a family sample, and comparing predictors across families."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vepbench import family_report
from vepbench.families import (
    FAMILIES,
    FamilyError,
    accession_of,
    count_sequences,
    membership,
    sample,
)


class TestFamilyDefinitions:
    def test_twenty_families_are_defined(self):
        assert len(FAMILIES) == 20

    def test_every_one_is_a_pfam_accession(self):
        assert all(k.startswith("PF") and k[2:].isdigit() for k in FAMILIES)

    def test_each_carries_a_human_readable_name(self):
        assert all(v for v in FAMILIES.values())


class TestFastaHandling:
    @pytest.mark.parametrize(("header", "expected"), [
        (">sp|P26678|PPLA_HUMAN Phospholamban", "P26678"),
        (">tr|A0A0U1RQS6|A0A0U1RQS6_HUMAN Uncharacterized", "A0A0U1RQS6"),
        (">P12345 bare accession", "P12345"),
    ])
    def test_accessions_come_out_of_either_header_style(self, header, expected):
        assert accession_of(header) == expected

    def test_sequences_are_counted_by_header(self, tmp_path):
        path = tmp_path / "x.fasta"
        path.write_text(">sp|A|A\nMKV\n>sp|B|B\nMKV\nAAA\n")
        assert count_sequences(path) == 2


class TestSampling:
    def test_the_same_seed_gives_the_same_sample(self):
        pool = [f"P{i:05d}" for i in range(200)]
        assert sample(pool, 10, seed=3) == sample(pool, 10, seed=3)

    def test_a_different_seed_gives_a_different_sample(self):
        pool = [f"P{i:05d}" for i in range(200)]
        assert sample(pool, 10, seed=1) != sample(pool, 10, seed=2)

    def test_input_order_does_not_change_the_result(self):
        # mmseqs makes no promise about the order it emits representatives,
        # so the sample must not depend on it.
        pool = [f"P{i:05d}" for i in range(200)]
        assert sample(pool, 10, seed=1) == sample(list(reversed(pool)), 10, seed=1)

    def test_a_small_family_returns_everything_it_has(self):
        assert sample(["P1", "P2"], 10) == ["P1", "P2"]

    def test_duplicates_are_collapsed(self):
        assert sample(["P1", "P1", "P2"], 10) == ["P1", "P2"]


class TestMembership:
    def test_every_built_family_contributes(self, paths):
        root = paths.outputs / "families"
        for pfam, members in (("PF00001", ["A", "B"]), ("PF07690", ["C"])):
            (root / pfam).mkdir(parents=True)
            (root / pfam / "accessions.txt").write_text("\n".join(members) + "\n")
        assert membership(paths) == {"A": "PF00001", "B": "PF00001", "C": "PF07690"}

    def test_a_shared_accession_is_assigned_once(self, paths):
        # A protein in two families must not contribute twice to a
        # comparison between them.
        root = paths.outputs / "families"
        for pfam in ("PF00001", "PF07690"):
            (root / pfam).mkdir(parents=True)
            (root / pfam / "accessions.txt").write_text("SHARED\n")
        assert list(membership(paths).values()).count("PF00001") == 1
        assert "PF07690" not in membership(paths).values()

    def test_nothing_built_is_an_empty_map(self, paths):
        assert membership(paths) == {}

    def test_clustering_without_mmseqs_says_how_to_get_it(self, paths, tmp_path,
                                                          monkeypatch):
        from vepbench import families

        monkeypatch.setattr(families, "have_mmseqs", lambda *a, **k: None)
        with pytest.raises(FamilyError, match="install-env clustering"):
            families.cluster(tmp_path / "x.fasta", paths=paths)

    def test_mmseqs_is_found_inside_its_own_environment(self, paths):
        # A conda environment under envs/ is not on PATH, so looking only
        # there would make `install-env clustering` appear to do nothing.
        from vepbench.families import have_mmseqs

        binary = paths.envs / "clustering" / ".venv" / "bin" / "mmseqs"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\n")
        assert have_mmseqs(paths) == str(binary)


def planted(shift_by_family, predictors=("alpha", "beta"), n=300, seed=0):
    """Tables where the transmembrane effect is known per family."""
    rng = np.random.default_rng(seed)
    frames = []
    for family, shift in shift_by_family.items():
        for index in range(12):
            tm = rng.random(n) < 0.5
            frame = pd.DataFrame({
                "uniprot_id": f"{family}_{index}", "family": family,
                "position": rng.integers(1, 200, n), "in_transmembrane": tm,
            })
            shared = rng.normal(0, 1, n)
            for name in predictors:
                frame[f"{name}{family_report.SUFFIX}"] = (
                    shared * 0.7 + rng.normal(0, 0.7, n) + tm * shift)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TestComparison:
    def test_distributions_are_reported_per_family_and_predictor(self):
        frame = planted({"PF00001": 1.0, "PF07690": 0.2})
        out = family_report.distributions(frame, ["alpha", "beta"])
        assert len(out) == 4
        assert set(out["family"]) == {"PF00001", "PF07690"}

    def test_the_topology_difference_recovers_what_was_planted(self):
        frame = planted({"PF00001": 1.2, "PF07690": 0.3})
        out = family_report.topology_split(frame, ["alpha"]).set_index("family")
        assert out.at["PF00001", "difference"] == pytest.approx(1.2, abs=0.2)
        assert out.at["PF07690", "difference"] == pytest.approx(0.3, abs=0.2)

    def test_agreement_is_reported_for_every_pair_once(self):
        frame = planted({"PF00001": 1.0}, predictors=("a", "b", "c"))
        out = family_report.agreement(frame, ["a", "b", "c"])
        assert len(out) == 3
        assert (out["spearman"] > 0).all()

    def test_a_predictor_tracking_another_shows_high_agreement(self):
        frame = planted({"PF00001": 0.5})
        frame["copy__damaging"] = frame["alpha__damaging"]
        out = family_report.agreement(frame, ["alpha", "copy"])
        assert out.iloc[0]["spearman"] == pytest.approx(1.0)

    def test_topology_is_skipped_when_nothing_is_annotated(self):
        frame = planted({"PF00001": 1.0}).drop(columns=["in_transmembrane"])
        assert family_report.topology_split(frame, ["alpha"]).empty

    def test_an_auc_is_produced_only_where_labels_exist(self):
        frame = planted({"PF00001": 1.0})
        assert family_report.labelled_auc(frame, ["alpha"]).empty

        rng = np.random.default_rng(1)
        frame["label"] = np.where(rng.random(len(frame)) < 0.1,
                                  rng.integers(0, 2, len(frame)), np.nan)
        out = family_report.labelled_auc(frame, ["alpha"])
        assert len(out) == 1
        assert out.iloc[0]["family"] == "PF00001"

    def test_the_auc_says_how_many_proteins_it_rests_on(self):
        # A family resting on two proteins is not evidence, and the table
        # has to say so rather than look like the rest.
        frame = planted({"PF00001": 1.0})
        frame["label"] = np.nan
        first = frame["uniprot_id"].isin(["PF00001_0", "PF00001_1"])
        rng = np.random.default_rng(2)
        frame.loc[first, "label"] = rng.integers(0, 2, int(first.sum()))
        out = family_report.labelled_auc(frame, ["alpha"])
        assert out.iloc[0]["n_proteins"] == 2


class TestLoading:
    def test_only_the_needed_columns_are_kept(self, paths):
        # A family run is tens of millions of rows; holding the full tables
        # would not fit.
        directory = paths.output_dir("A1")
        directory.mkdir(parents=True)
        pd.DataFrame({
            "uniprot_id": ["A1"], "position": [1], "wt_aa": ["M"], "mut_aa": ["A"],
            "variant": ["M1A"], "alpha": [1.0], "alpha__damaging": [-1.0],
            "in_transmembrane": [True],
        }).to_csv(directory / "scores.tsv", sep="\t", index=False)
        frame = family_report.load_scored(["A1"], {"A1": "PF1"},
                                          outputs=paths.outputs)
        assert "wt_aa" not in frame.columns
        assert "alpha" not in frame.columns
        assert "alpha__damaging" in frame.columns
        assert frame.iloc[0]["family"] == "PF1"

    def test_an_unscored_accession_is_skipped(self, paths):
        assert family_report.load_scored(["NOPE"], {}, outputs=paths.outputs).empty
