"""Shared fixtures. Every test runs offline against cached inputs."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TEST_ACCESSION = "P26678"          # phospholamban: 52 residues, one TM helix


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """A throwaway data and outputs tree seeded with the cached inputs."""
    from vepbench.config import Paths

    monkeypatch.setenv("VEPBENCH_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("VEPBENCH_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("VEPBENCH_WEIGHTS", str(tmp_path / "weights"))
    monkeypatch.setenv("VEPBENCH_THIRD_PARTY", str(tmp_path / "third_party"))
    instance = Paths(root=tmp_path)
    instance.ensure()
    shutil.copy(FIXTURES / f"{TEST_ACCESSION}.uniprot.json",
                instance.annotations / f"{TEST_ACCESSION}.json")
    shutil.copy(FIXTURES / "AF-P26678-F1-model_v6.pdb",
                instance.structures / "AF-P26678-F1-model_v6.pdb")
    return instance


@pytest.fixture
def uniprot_record(paths):
    from vepbench.uniprot import fetch_uniprot

    return fetch_uniprot(TEST_ACCESSION, paths=paths, offline=True)


@pytest.fixture
def sequence(uniprot_record):
    return uniprot_record.sequence


@pytest.fixture
def structure_path():
    """The synthetic helix, for geometry tests that must be deterministic."""
    return FIXTURES / f"{TEST_ACCESSION}_synthetic_helix_TESTONLY.pdb"


@pytest.fixture
def alphafold_path():
    """The real AlphaFold model, for pLDDT and numbering tests."""
    return FIXTURES / "AF-P26678-F1-model_v6.pdb"


@pytest.fixture
def structure(uniprot_record, structure_path, paths):
    from vepbench.structure import load_structure

    return load_structure(TEST_ACCESSION, uniprot_record.sequence,
                          override=structure_path, paths=paths)


@pytest.fixture
def alignment(tmp_path, sequence):
    """A synthetic alignment diverse enough to exercise the MSA code."""
    import random

    random.seed(7)
    lines = [f">query\n{sequence}"]
    # Mutate roughly a third of each sequence, so members fall into genuinely
    # distinct clusters rather than one 80%-identity blob.
    for index in range(60):
        mutated = list(sequence)
        for position in range(len(mutated)):
            if random.random() < 0.35:
                mutated[position] = random.choice("ACDEFGHIKLMNPQRSTVWY")
        lines.append(f">seq{index}\n{''.join(mutated)}")
    path = tmp_path / "test.a3m"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def protein(uniprot_record, structure, tmp_path):
    from vepbench.protein import ProteinContext

    return ProteinContext(
        uniprot=uniprot_record, work_dir=tmp_path / "work", structure=structure,
        domain=(32, 52), domain_note="UniProt transmembrane segment",
    )
