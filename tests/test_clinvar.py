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
