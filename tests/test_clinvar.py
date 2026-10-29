"""ClinVar parsing, mapping onto UniProt coordinates, and deduplication.

The mapping is the one join that can be wrong without looking wrong, so
these tests concentrate on the ways a row should be refused.
"""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from vepbench import clinvar

HEADER = [
    "Type", "Name", "GeneSymbol", "ClinicalSignificance", "ReviewStatus",
    "NumberSubmitters", "PhenotypeList", "VariationID", "Origin", "OriginSimple",
    "Assembly",
]
# 1=M 2=G 3=R 4=G 5=L 6=M 7=F 8=V
SEQUENCE = "MGRGLMFV"


def row(name, *, gene="ABCA1", significance="Pathogenic",
        review="criteria provided, single submitter", submitters="1",
        variation_id="1", origin="germline", assembly="GRCh38",
        kind="single nucleotide variant"):
    return [kind, name, gene, significance, review, submitters, "Cond",
            variation_id, origin, origin, assembly]


def write(tmp_path, rows):
    path = tmp_path / "variant_summary.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("\t".join(HEADER) + "\n")
        for item in rows:
            handle.write("\t".join(item) + "\n")
    return path


def parse(tmp_path, rows, **kwargs):
    manifest = pd.DataFrame({"gene": ["ABCA1"], "uniprot_id": ["P00001"]})
    gene_to_accession, ambiguous = clinvar.gene_index(manifest)
    return clinvar.parse(
        write(tmp_path, rows), {"P00001": SEQUENCE}, gene_to_accession,
        ambiguous_genes=ambiguous, **kwargs,
    )


class TestMapping:
    def test_a_matching_missense_row_maps(self, tmp_path):
        frame, report = parse(tmp_path, [row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)")])
        assert frame["variant"].tolist() == ["R3Q"]
        assert frame["uniprot_id"].tolist() == ["P00001"]
        assert report.counts[clinvar.MAPPED] == 1

    def test_a_disagreeing_wild_type_residue_is_refused_not_coerced(self, tmp_path):
        # UniProt has G at position 2; ClinVar claims W. Accepting this would
        # score the right protein at the wrong coordinates.
        frame, report = parse(tmp_path, [row("NM_1.1(ABCA1):c.4A>G (p.Trp2Cys)")])
        assert frame.empty
        assert report.counts[clinvar.WT_MISMATCH] == 1
        assert "has G at 2" in report.examples[clinvar.WT_MISMATCH][0]

    def test_a_position_past_the_end_is_refused(self, tmp_path):
        frame, report = parse(tmp_path, [row("NM_1.1(ABCA1):c.1A>G (p.Met99Val)")])
        assert frame.empty
        assert report.counts[clinvar.OUT_OF_RANGE] == 1

    def test_a_gene_outside_the_manifest_is_recorded(self, tmp_path):
        frame, report = parse(tmp_path, [row("NM_9(NOPE):c.1A>G (p.Ala1Val)", gene="NOPE")])
        assert frame.empty
        assert report.counts[clinvar.UNKNOWN_GENE] == 1

    def test_a_gene_shared_by_two_accessions_is_excluded(self):
        manifest = pd.DataFrame({"gene": ["DUP", "DUP", "SOLO"],
                                 "uniprot_id": ["P1", "P2", "P3"]})
        mapping, ambiguous = clinvar.gene_index(manifest)
        assert mapping == {"SOLO": "P3"}
        assert ambiguous == frozenset({"DUP"})

    def test_a_blank_gene_symbol_does_not_become_a_key(self):
        manifest = pd.DataFrame({"gene": ["", None, "REAL"],
                                 "uniprot_id": ["P1", "P2", "P3"]})
        mapping, _ = clinvar.gene_index(manifest)
        assert mapping == {"REAL": "P3"}

    def test_non_substitution_changes_are_skipped_silently(self, tmp_path):
        # Frameshifts, indels and nonsense have no substitution score to
        # compare against, so they are not failures to report.
        frame, report = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.20del (p.Phe7fs)"),
            row("NM_1.1(ABCA1):c.21A>T (p.Arg3Ter)"),
            row("NM_1.1(ABCA1):c.22A>T (p.Arg3=)"),
        ])
        assert frame.empty
        assert report.total == 0

    def test_a_synonymous_substitution_is_skipped(self, tmp_path):
        frame, _ = parse(tmp_path, [row("NM_1.1(ABCA1):c.1A>G (p.Arg3Arg)")])
        assert frame.empty


class TestFiltering:
    def test_somatic_rows_are_excluded_by_default(self, tmp_path):
        frame, _ = parse(tmp_path, [row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)", origin="somatic")])
        assert frame.empty

    def test_somatic_rows_can_be_kept(self, tmp_path):
        frame, _ = parse(tmp_path, [row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)", origin="somatic")],
                         germline_only=False)
        assert len(frame) == 1

    def test_other_assemblies_are_excluded(self, tmp_path):
        frame, _ = parse(tmp_path, [row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)", assembly="GRCh37")])
        assert frame.empty

    def test_indel_types_are_excluded(self, tmp_path):
        frame, _ = parse(tmp_path, [row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)", kind="Deletion")])
        assert frame.empty


class TestLabelling:
    @pytest.mark.parametrize(("significance", "expected"), [
        ("Pathogenic", 1), ("Likely pathogenic", 1), ("Pathogenic/Likely pathogenic", 1),
        ("Benign", 0), ("Likely benign", 0), ("Benign/Likely benign", 0),
        ("Uncertain significance", None),
        ("Conflicting classifications of pathogenicity", None),
        ("drug response", None),
    ])
    def test_only_the_confident_calls_become_labels(self, significance, expected):
        assert clinvar._label(significance) == expected

    @pytest.mark.parametrize(("status", "stars"), [
        ("practice guideline", 4), ("reviewed by expert panel", 3),
        ("criteria provided, multiple submitters, no conflicts", 2),
        ("criteria provided, single submitter", 1),
        ("no assertion criteria provided", 0), ("something new upstream", 0),
    ])
    def test_review_status_becomes_a_star_rating(self, status, stars):
        assert clinvar._stars(status) == stars

    def test_uncertain_rows_are_kept_but_left_unlabelled(self, tmp_path):
        frame, _ = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)", significance="Uncertain significance"),
        ])
        assert len(frame) == 1
        assert pd.isna(frame["label"].iloc[0])


class TestDeduplication:
    def test_the_best_reviewed_record_of_a_substitution_wins(self, tmp_path):
        frame, _ = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)",
                review="criteria provided, single submitter", variation_id="1"),
            row("NM_1.1(ABCA1):c.2G>A (p.Arg3Gln)",
                review="reviewed by expert panel", variation_id="2"),
        ])
        assert len(frame) == 1
        assert frame["review_stars"].iloc[0] == 3
        assert frame["variation_id"].iloc[0] == "2"

    def test_records_disagreeing_on_pathogenicity_lose_their_label(self, tmp_path):
        frame, _ = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)", significance="Pathogenic",
                variation_id="1"),
            row("NM_1.1(ABCA1):c.2G>A (p.Arg3Gln)", significance="Benign",
                variation_id="2"),
        ])
        assert len(frame) == 1
        assert pd.isna(frame["label"].iloc[0])
        assert frame["clinical_significance"].iloc[0] == "Conflicting (vepbench)"

    def test_agreeing_duplicates_keep_their_label(self, tmp_path):
        frame, _ = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)", significance="Pathogenic",
                variation_id="1"),
            row("NM_1.1(ABCA1):c.2G>A (p.Arg3Gln)", significance="Likely pathogenic",
                variation_id="2"),
        ])
        assert frame["label"].tolist() == [1]

    def test_distinct_substitutions_are_not_collapsed(self, tmp_path):
        frame, _ = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)"),
            row("NM_1.1(ABCA1):c.9T>C (p.Leu5Pro)", significance="Benign"),
        ])
        assert sorted(frame["variant"]) == ["L5P", "R3Q"]


class TestReporting:
    def test_the_summary_names_the_failures_with_examples(self, tmp_path):
        _, report = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)"),
            row("NM_1.1(ABCA1):c.4A>G (p.Trp2Cys)"),
        ])
        text = report.summary()
        assert "1 of 2" in text
        assert clinvar.WT_MISMATCH in text

    def test_the_per_gene_frame_accounts_for_every_row(self, tmp_path):
        _, report = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)"),
            row("NM_1.1(ABCA1):c.4A>G (p.Trp2Cys)"),
        ])
        frame = report.as_frame()
        assert frame.loc[frame["gene"] == "ABCA1", "total"].iloc[0] == 2

    def test_label_counts_describe_the_table(self, tmp_path):
        frame, _ = parse(tmp_path, [
            row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)", significance="Pathogenic"),
            row("NM_1.1(ABCA1):c.9T>C (p.Leu5Pro)", significance="Benign"),
            row("NM_1.1(ABCA1):c.12A>C (p.Gly4Ser)", significance="Uncertain significance"),
        ])
        assert clinvar.label_counts(frame) == {
            "pathogenic": 1, "benign": 1, "unlabelled": 1, "proteins": 1,
        }

    def test_an_empty_table_counts_as_empty_rather_than_failing(self):
        assert clinvar.label_counts(pd.DataFrame())["proteins"] == 0


class TestMappingRate:
    """The rate has to be measured against the rows that were candidates."""

    def test_genes_outside_the_set_do_not_count_against_the_rate(self, tmp_path):
        # ClinVar covers the whole genome. On the membrane proteome three
        # quarters of it is out of scope, and counting that as failure makes
        # a healthy run read as a broken one.
        rows = [row("NM_1.1(ABCA1):c.1A>G (p.Arg3Gln)")]
        rows += [row(f"NM_9(OTHER):c.{i}A>G (p.Ala1Val)", gene="OTHER",
                     variation_id=str(100 + i)) for i in range(20)]
        _, report = parse(tmp_path, rows)

        assert report.in_scope == 1
        assert report.total == 21
        text = report.summary()
        assert "1 of 1" in text
        assert "100.0%" in text
        assert "20 rows were in genes outside the set" in text

    def test_out_of_scope_rows_are_not_listed_as_a_failure_mode(self, tmp_path):
        rows = [row("NM_9(OTHER):c.1A>G (p.Ala1Val)", gene="OTHER")]
        _, report = parse(tmp_path, rows)
        assert "gene_not_in_manifest:" not in report.summary()

    def test_a_set_with_nothing_in_scope_does_not_divide_by_zero(self, tmp_path):
        _, report = parse(tmp_path, [row("NM_9(OTHER):c.1A>G (p.Ala1Val)", gene="OTHER")])
        assert "0.0%" in report.summary()


class TestSuspectGenes:
    """A gene symbol can resolve to the wrong protein entirely."""

    def _report(self, per_gene):
        from collections import Counter

        from vepbench.clinvar import MappingReport

        report = MappingReport()
        report.per_gene = {g: Counter(c) for g, c in per_gene.items()}
        for counter in report.per_gene.values():
            report.counts.update(counter)
        return report

    def test_a_gene_failing_most_attempts_is_flagged(self):
        # PRNP's shape: the manifest holds the 73-residue alternative reading
        # frame, because the main prion protein is GPI-anchored and carries
        # no transmembrane keyword. Rows that pass did so by coincidence.
        report = self._report({
            "PRNP": {clinvar.MAPPED: 4, clinvar.OUT_OF_RANGE: 180},
        })
        assert report.suspect_genes() == [("PRNP", 4, 184)]

    def test_a_gene_mapping_nearly_everything_is_not_flagged(self):
        report = self._report({
            "ABCA1": {clinvar.MAPPED: 900, clinvar.WT_MISMATCH: 10},
        })
        assert report.suspect_genes() == []

    def test_a_gene_that_mapped_nothing_is_not_flagged(self):
        # Nothing got through, so there is nothing to mistrust; it shows up
        # in the failure counts instead.
        report = self._report({"GONE": {clinvar.OUT_OF_RANGE: 50}})
        assert report.suspect_genes() == []

    def test_the_summary_names_the_suspect_genes(self):
        report = self._report({
            "PRNP": {clinvar.MAPPED: 4, clinvar.OUT_OF_RANGE: 180},
        })
        text = report.summary()
        assert "wrong isoform" in text
        assert "PRNP: 4 of 184" in text

    def test_the_per_gene_frame_carries_the_fraction_and_the_flag(self):
        report = self._report({
            "PRNP": {clinvar.MAPPED: 4, clinvar.OUT_OF_RANGE: 180},
            "ABCA1": {clinvar.MAPPED: 900, clinvar.WT_MISMATCH: 10},
        })
        frame = report.as_frame().set_index("gene")
        assert bool(frame.at["PRNP", "suspect"])
        assert not bool(frame.at["ABCA1", "suspect"])
        assert frame.at["ABCA1", "mapped_fraction"] > 0.98

    def test_a_gene_with_no_attempts_gets_no_fraction(self):
        report = self._report({"ODD": {clinvar.NON_STANDARD: 3}})
        frame = report.as_frame().set_index("gene")
        assert pd.isna(frame.at["ODD", "mapped_fraction"])
        assert not bool(frame.at["ODD", "suspect"])
