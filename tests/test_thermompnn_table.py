"""Reading the published ThermoMPNN table, and deciding when not to."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from vepbench import thermompnn_table as table


def write(path, body, gzipped=False):
    if gzipped:
        with gzip.open(path, "wt") as handle:
            handle.write(body)
    else:
        path.write_text(body)
    return path


class TestReading:
    def test_a_variant_column_is_parsed(self, tmp_path):
        path = write(tmp_path / "t.tsv.gz",
                     "uniprot_id\tvariant\tddG\nP1\tA10V\t1.5\nP1\tL20P\t3.2\n",
                     gzipped=True)
        frame = table.read_table(path)
        assert len(frame) == 2
        assert frame.iloc[0]["position"] == 10
        assert frame.iloc[0]["wt_aa"] == "A"
        assert frame.iloc[0]["ddg"] == 1.5

    def test_separate_position_and_residue_columns_work(self, tmp_path):
        path = write(tmp_path / "t.csv",
                     "accession,pos,wt,alt,ddg_pred\nP1,10,A,V,1.5\n")
        frame = table.read_table(path)
        assert frame.iloc[0]["mut_aa"] == "V"

    def test_alternative_column_names_are_accepted(self, tmp_path):
        path = write(tmp_path / "t.tsv",
                     "protein\tmutation\tprediction\nP1\tA10V\t1.5\n")
        assert len(table.read_table(path)) == 1

    def test_an_unrecognisable_file_raises_rather_than_reading_as_empty(self, tmp_path):
        # Returning nothing would be indistinguishable from "this protein is
        # not covered", and would silently run the model for everything.
        path = write(tmp_path / "t.tsv", "alpha\tbeta\n1\t2\n")
        with pytest.raises(table.ThermoMPNNTableError, match="does not look like"):
            table.read_table(path)

    def test_a_missing_file_names_where_to_get_it(self, tmp_path):
        with pytest.raises(table.ThermoMPNNTableError, match="zenodo"):
            table.read_table(tmp_path / "absent.tsv.gz")

    def test_unparseable_rows_are_dropped_not_guessed(self, tmp_path):
        path = write(tmp_path / "t.tsv",
                     "uniprot_id\tvariant\tddG\nP1\tA10V\t1.5\nP1\tnonsense\t2.0\n"
                     "P1\tL20P\tnotanumber\n")
        assert len(table.read_table(path)) == 1


class TestCoverage:
    def _index(self, tmp_path):
        path = write(tmp_path / "t.tsv",
                     "uniprot_id\tvariant\tddG\nP1\tA10V\t1.5\nP1\tL20P\t3.2\n")
        return table.index_by_protein(table.read_table(path))

    def _variants(self, n=4):
        return pd.DataFrame({"position": [10, 20, 30, 40][:n],
                             "wt_aa": list("ALGG")[:n],
                             "mut_aa": list("VPWW")[:n]})

    def test_a_fully_covered_protein_is_usable(self, tmp_path):
        values, coverage = table.lookup(self._variants(2),
                                        self._index(tmp_path)["P1"], "P1")
        assert coverage.fraction == 1.0
        assert coverage.usable
        assert values.tolist() == [1.5, 3.2]

    def test_a_protein_absent_from_the_table_is_not_usable(self, tmp_path):
        _, coverage = table.lookup(self._variants(), {}, "P9")
        assert coverage.fraction == 0.0
        assert not coverage.usable

    def test_a_thinly_covered_protein_is_refused(self, tmp_path):
        # A partial column is worse than none: it looks complete.
        variants = pd.DataFrame({"position": list(range(1, 11)),
                                 "wt_aa": ["A"] * 10, "mut_aa": ["V"] * 10})
        _, coverage = table.lookup(variants, self._index(tmp_path)["P1"], "P1")
        assert not coverage.usable

    def test_coverage_is_measured_against_what_was_asked_for(self, tmp_path):
        _, coverage = table.lookup(self._variants(4),
                                   self._index(tmp_path)["P1"], "P1")
        assert coverage.requested == 4
        assert coverage.matched == 2
        assert "2 of 4" in coverage.describe()

    def test_an_empty_request_does_not_divide_by_zero(self):
        _, coverage = table.lookup(
            pd.DataFrame({"position": [], "wt_aa": [], "mut_aa": []}), {}, "P1")
        assert coverage.fraction == 0.0
