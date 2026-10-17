"""The per-protein bundle handed to every predictor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vepbench.msa import MSARecord
from vepbench.structure import StructureRecord
from vepbench.uniprot import UniProtRecord


@dataclass
class ProteinContext:
    """Everything a predictor may need about one protein.

    Predictors declare which of these they require, and the runner skips one
    whose requirements are unmet rather than letting it fail halfway through.
    """

    uniprot: UniProtRecord
    work_dir: Path
    structure: StructureRecord | None = None
    msa: MSARecord | None = None
    domain: tuple[int, int] | None = None
    domain_note: str = ""
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
        return self.uniprot.length

    @property
    def is_membrane_protein(self) -> bool:
        return self.uniprot.is_membrane_protein

    def domain_sequence(self) -> str:
        if self.domain is None:
            return self.sequence
        start, end = self.domain
        return self.sequence[start - 1 : end]

    def has(self, requirement: str) -> bool:
        return {
            "sequence": True,
            "structure": self.structure is not None,
            "msa": self.msa is not None,
            "domain": self.domain is not None,
        }.get(requirement, False)

    def as_dict(self) -> dict:
        return {
            "uniprot": self.uniprot.as_dict(),
            "structure": self.structure.as_dict() if self.structure else None,
            "msa": self.msa.as_dict() if self.msa else None,
            "domain": list(self.domain) if self.domain else None,
            "domain_note": self.domain_note,
        }
