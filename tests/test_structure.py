"""Structure parsing, UniProt renumbering and surface geometry."""

from __future__ import annotations

from vepbench.constants import MAX_ASA_TIEN
from vepbench.structure import (
    assign_uniprot_numbering,
    contact_number,
    parse_pdb,
    relative_sasa,
    shrake_rupley,
)

OFFSET_FIXTURE = "P26678_offset_numbering_TESTONLY.pdb"


def test_parse_pdb_reads_every_residue(structure_path, sequence):
    residues, chain = parse_pdb(structure_path)
    assert chain == "A"
    assert len(residues) == len(sequence)
    assert "".join(r.aa for r in residues) == sequence


def test_direct_numbering_is_used_when_it_matches(structure, sequence):
    assert "matches UniProt directly" in structure.mapping_note
    assert structure.coverage == 1.0
    mapped = structure.by_uniprot_position()
    assert len(mapped) == len(sequence)
    for position, residue in mapped.items():
        assert sequence[position - 1] == residue.aa


def test_offset_numbering_is_recovered(uniprot_record, structure_path, paths):
    from vepbench.structure import load_structure

    record = load_structure(
        "P26678", uniprot_record.sequence,
        override=structure_path.parent / OFFSET_FIXTURE, paths=paths,
    )
    assert "offset" in record.mapping_note
    for position, residue in record.by_uniprot_position().items():
        assert uniprot_record.sequence[position - 1] == residue.aa


def test_alignment_route_refuses_to_map_mismatches():
    from vepbench.structure import Residue

    residues = [Residue(chain="A", seq_id=i + 1, insertion_code="", aa=aa)
                for i, aa in enumerate("MKWLA")]
    note, coverage = assign_uniprot_numbering(residues, "MKVLA")
    # W at position 3 does not exist in the target, so it must not be mapped
    assert residues[2].uniprot_position is None
    assert coverage < 1.0
    assert "alignment" in note or "substring" in note


def test_real_alphafold_model_parses_and_numbers(uniprot_record, alphafold_path, paths):
    from vepbench.structure import load_structure

    record = load_structure("P26678", uniprot_record.sequence,
                            override=alphafold_path, paths=paths)
    assert len(record.residues) == len(uniprot_record.sequence)
    assert record.coverage == 1.0
    assert record.sequence == uniprot_record.sequence


def test_plddt_only_for_predicted_structures(structure, uniprot_record, alphafold_path,
                                             paths):
    from vepbench.structure import StructureRecord, parse_pdb

    # A user-supplied file is not treated as a prediction, so no pLDDT.
    assert structure.is_predicted is False
    assert structure.plddt() == {}

    # The same file read as an AlphaFold model does give pLDDT, and the
    # transmembrane helix must be more confident than the flexible terminus.
    residues, chain = parse_pdb(alphafold_path)
    note, coverage = assign_uniprot_numbering(residues, uniprot_record.sequence)
    model = StructureRecord(
        path=alphafold_path, source="alphafold", identifier="AF-P26678-F1",
        residues=residues, chain=chain, is_predicted=True, mapping_note=note,
        coverage=coverage,
    )
    plddt = model.plddt()
    assert len(plddt) == len(uniprot_record.sequence)
    helix = sum(plddt[p] for p in range(32, 53)) / 21
    terminus = sum(plddt[p] for p in range(1, 16)) / 15
    assert helix > terminus
    assert 0 <= min(plddt.values()) and max(plddt.values()) <= 100


def test_sasa_is_physically_plausible(structure):
    absolute = shrake_rupley(structure.residues, n_points=120)
    assert len(absolute) == len(structure.residues)
    for index, residue in enumerate(structure.residues):
        assert absolute[index] >= 0.0
        assert absolute[index] < 3 * MAX_ASA_TIEN[residue.aa]


def test_relative_sasa_keyed_by_uniprot_position(structure, sequence):
    values = relative_sasa(structure, n_points=120)
    assert set(values) == set(range(1, len(sequence) + 1))
    assert all(0.0 <= v < 3.0 for v in values.values())


def test_terminal_residues_are_more_exposed_than_the_middle(structure):
    """On a single helix the ends are necessarily more exposed."""
    values = relative_sasa(structure, n_points=120)
    length = len(values)
    ends = (values[1] + values[length]) / 2
    middle = sum(values[p] for p in range(length // 2 - 2, length // 2 + 3)) / 5
    assert ends > middle


def test_contact_number_is_sane(structure):
    counts = contact_number(structure, cutoff=10.0)
    assert all(count >= 0 for count in counts.values())
    assert counts[len(counts) // 2] > counts[1]
