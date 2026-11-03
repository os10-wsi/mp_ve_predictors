"""The three verbs, driven end to end."""

from __future__ import annotations

import pandas as pd
import pytest

from vepbench.cli import main

BUILTINS = ["blosum62", "delta_volume", "delta_charge", "delta_hydrophobicity"]


def clinvar_slice(paths, accession, n_path=8, n_benign=8):
    """A slice with enough of both classes for an AUC."""
    from vepbench.clinvar_store import slice_path
    from vepbench.config import read_json

    sequence = read_json(paths.annotations / f"{accession}.json")["sequence"]
    rows = []
    for index in range(n_path + n_benign):
        position = index + 1
        wt = sequence[position - 1]
        mut = "W" if wt != "W" else "A"
        pathogenic = index < n_path
        rows.append({
            "variant": f"{wt}{position}{mut}", "position": position,
            "wt_aa": wt, "mut_aa": mut,
            "clinical_significance": "Pathogenic" if pathogenic else "Benign",
            "label": 1 if pathogenic else 0,
        })
    path = slice_path(accession, paths=paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


class TestTargets:
    def test_accessions_come_from_a_file(self, paths, tmp_path, capsys):
        listing = tmp_path / "ids.txt"
        listing.write_text("# a note\n\nP26678\n")
        assert main(["score", "--uniprot-list", str(listing), "--offline",
                     "--scores", "blosum62"]) == 0

    def test_naming_nothing_is_refused_with_the_options(self, paths):
        with pytest.raises(SystemExit, match="uniprot"):
            main(["score", "--scores", "blosum62"])

    def test_a_missing_list_file_says_which(self, paths):
        with pytest.raises(SystemExit, match="no such file"):
            main(["score", "--uniprot-list", "/nope.txt"])

    def test_an_unknown_score_lists_what_exists(self, paths):
        with pytest.raises(SystemExit, match="blosum62"):
            main(["score", "--uniprot", "P26678", "--scores", "nonsense"])


class TestScore:
    def test_it_writes_a_table_per_protein(self, paths, capsys):
        assert main(["score", "--uniprot", "P26678", "--offline",
                     "--scores", *BUILTINS]) == 0
        path = paths.output_dir("P26678") / "scores.tsv"
        assert path.exists()
        assert "wrote" in capsys.readouterr().out

    def test_one_protein_failing_does_not_stop_the_list(self, paths, capsys):
        code = main(["score", "--uniprot", "P26678", "NOTREAL", "--offline",
                     "--scores", "blosum62"])
        assert code == 0
        assert (paths.output_dir("P26678") / "scores.tsv").exists()
        assert "1 of 2" in capsys.readouterr().err

    def test_every_protein_failing_is_an_error(self, paths):
        assert main(["score", "--uniprot", "NOTREAL", "--offline",
                     "--scores", "blosum62"]) == 1


class TestReport:
    def _scored(self, paths):
        clinvar_slice(paths, "P26678")
        main(["score", "--uniprot", "P26678", "--offline", "--scores", *BUILTINS])

    def test_it_writes_the_three_tables(self, paths, tmp_path, capsys):
        self._scored(paths)
        out = tmp_path / "report"
        assert main(["report", "--uniprot", "P26678", "--out", str(out),
                     "--scores", *BUILTINS, "--no-plots"]) == 0
        for stem in ("auc_by_protein", "auc_long", "auc_summary"):
            assert (out / f"{stem}.tsv").exists(), stem

    def test_the_wide_table_has_one_row_per_protein(self, paths, tmp_path):
        self._scored(paths)
        out = tmp_path / "report"
        main(["report", "--uniprot", "P26678", "--out", str(out),
              "--scores", *BUILTINS, "--no-plots"])
        wide = pd.read_csv(out / "auc_by_protein.tsv", sep="\t")
        assert len(wide) == 1
        for name in BUILTINS:
            assert f"{name}_auc" in wide.columns
            assert f"{name}_se" in wide.columns

    def test_it_draws_the_roc_panels(self, paths, tmp_path):
        pytest.importorskip("matplotlib")
        self._scored(paths)
        out = tmp_path / "report"
        main(["report", "--uniprot", "P26678", "--out", str(out),
              "--scores", *BUILTINS])
        assert (out / "roc_curves.pdf").exists()
        assert (out / "auc_ranking.png").exists()

    def test_nothing_labelled_says_what_to_do(self, paths, tmp_path, capsys):
        main(["score", "--uniprot", "P26678", "--offline", "--scores", "blosum62"])
        assert main(["report", "--uniprot", "P26678", "--out", str(tmp_path / "r"),
                     "--scores", "blosum62"]) == 1
        assert "fetch clinvar" in capsys.readouterr().err

    def test_an_unscored_accession_is_counted_not_fatal(self, paths, tmp_path,
                                                        capsys):
        self._scored(paths)
        main(["report", "--uniprot", "P26678", "NEVERRAN", "--out",
              str(tmp_path / "r"), "--scores", *BUILTINS, "--no-plots"])
        assert "1 accessions had no scores.tsv" in capsys.readouterr().out


class TestStatus:
    def test_it_names_the_interpreters_and_every_score(self, paths, capsys):
        from vepbench.scores import all_names

        assert main(["status"]) == 0
        out = capsys.readouterr().out
        assert "orchestrator" in out
        for name in all_names():
            assert name in out
