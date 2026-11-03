"""The per-protein bundle handed to every score."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vepbench.structure import StructureRecord
from vepbench.uniprot import UniProtRecord

HUMAN = "Homo sapiens"


@dataclass
class ProteinContext:
    """Everything a score may need about one protein.

    Scores declare what they require and the runner skips one whose
    requirements are unmet, rather than letting it fail halfway through.
    """

    uniprot: UniProtRecord
    work_dir: Path
    structure: StructureRecord | None = None
    weights_dir: Path | None = None
    third_party_dir: Path | None = None
    extras: dict = field(default_factory=dict)

    @property
    def uniprot_id(self) -> str:
        return self.uniprot.uniprot_id

    @property
    def sequence(self) -> str:
        return self.uniprot.sequence

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def is_human(self) -> bool:
        """Whether human-only resources apply.

        AlphaMissense published the human proteome and ClinVar is a human
        database, so this decides whether either can contribute. The check
        is on the organism UniProt reports, not on the accession.
        """
        return self.uniprot.organism.startswith(HUMAN)

    @property
    def has_structure(self) -> bool:
        return self.structure is not None

    def as_dict(self) -> dict:
        return {
            "uniprot_id": self.uniprot_id,
            "organism": self.uniprot.organism,
            "is_human": self.is_human,
            "length": self.length,
            "structure": self.structure.as_dict() if self.structure else None,
        }
