"""Per-position and per-variant features added as extra columns.

These are descriptors, not predictions. They sit in the wide table so a DMS
analysis can condition on burial, confidence, conservation or membrane
topology without a second join.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from vepbench.constants import (
    KD_HYDROPATHY,
    RESIDUE_CHARGE,
    RESIDUE_VOLUME,
    blosum62,
    grantham_distance,
)
from vepbench.msa import per_position_entropy, per_position_gap_fraction, read_alignment
from vepbench.protein import ProteinContext
from vepbench.structure import contact_number, relative_sasa, residue_depth

log = logging.getLogger(__name__)

DEFAULT_FEATURES = ("rsasa", "plddt", "physchem", "blosum", "grantham", "membrane")
ALL_FEATURES = DEFAULT_FEATURES + (
    "entropy", "gaps", "depth", "contacts", "secondary_structure", "bilayer_depth",
)

# Which inputs each feature needs. The runner reads this so that asking for a
# structural feature fetches a structure, even when no requested predictor
# would have pulled one in on its own.
FEATURE_REQUIREMENTS = {
    "rsasa": "structure",
    "plddt": "structure",
    "depth": "structure",
    "contacts": "structure",
    "secondary_structure": "structure",
    "bilayer_depth": "structure",
    "entropy": "msa",
    "gaps": "msa",
}


def inputs_for_features(requested: list[str] | None) -> set[str]:
    """The set of inputs needed to compute the requested features."""
    names = requested if requested is not None else list(DEFAULT_FEATURES)
    return {FEATURE_REQUIREMENTS[f] for f in names if f in FEATURE_REQUIREMENTS}


def _per_variant_physchem(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    pairs = list(zip(frame["wt_aa"], frame["mut_aa"], strict=True))
    out["delta_hydropathy"] = [
        KD_HYDROPATHY.get(m, float("nan")) - KD_HYDROPATHY.get(w, float("nan"))
        for w, m in pairs
    ]
    out["delta_volume"] = [
        RESIDUE_VOLUME.get(m, float("nan")) - RESIDUE_VOLUME.get(w, float("nan"))
        for w, m in pairs
    ]
    out["delta_charge"] = [
        RESIDUE_CHARGE.get(m, float("nan")) - RESIDUE_CHARGE.get(w, float("nan"))
        for w, m in pairs
    ]
    out["introduces_proline"] = [m == "P" for _w, m in pairs]
    out["removes_glycine"] = [w == "G" for w, _m in pairs]
    out["introduces_charge"] = [
        RESIDUE_CHARGE.get(w, 0.0) == 0.0 and abs(RESIDUE_CHARGE.get(m, 0.0)) >= 1.0
        for w, m in pairs
    ]
    return out


def _membrane_annotations(protein: ProteinContext, positions: pd.Series) -> pd.DataFrame:
    segments = protein.uniprot.transmembrane_segments
    topology = {
        (f.start, f.end): f.description for f in protein.uniprot.features_of("topology")
    }

    def segment_index(position: int) -> int:
        for index, (start, end) in enumerate(segments, start=1):
            if start <= position <= end:
                return index
        return 0

    def topology_label(position: int) -> str:
        for (start, end), label in topology.items():
            if start <= position <= end:
                return label
        return "transmembrane" if segment_index(position) else ""

    out = pd.DataFrame(index=positions.index)
    out["tm_segment"] = [segment_index(p) for p in positions]
    out["in_transmembrane"] = out["tm_segment"] > 0
    out["topology"] = [topology_label(p) for p in positions]
    if segments:
        def distance_to_tm(position: int) -> int:
            if segment_index(position):
                return 0
            return min(min(abs(position - s), abs(position - e)) for s, e in segments)
        out["residues_to_nearest_tm"] = [distance_to_tm(p) for p in positions]
    return out


def _secondary_structure(protein: ProteinContext) -> dict[int, str]:
    """DSSP secondary structure, if mkdssp or dssp is installed.

    Returns an empty mapping when DSSP is absent; the column is then omitted
    rather than filled with a guess.
    """
    if protein.structure is None:
        return {}
    executable = shutil.which("mkdssp") or shutil.which("dssp")
    if executable is None:
        log.warning("secondary_structure requested but neither mkdssp nor dssp is on PATH")
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "out.dssp"
        try:
            subprocess.run([executable, str(protein.structure.path), str(output)],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            log.warning("DSSP failed: %s", exc.stderr.decode()[-400:])
            return {}
        by_seq_id = {r.seq_id: r.uniprot_position for r in protein.structure.residues}
        mapping: dict[int, str] = {}
        started = False
        for line in output.read_text().splitlines():
            if line.startswith("  #  RESIDUE"):
                started = True
                continue
            if not started or len(line) < 17 or line[13] == "!":
                continue
            try:
                seq_id = int(line[5:10])
            except ValueError:
                continue
            position = by_seq_id.get(seq_id)
            if position is not None:
                mapping[position] = line[16].strip() or "-"
        return mapping


def build_features(
    protein: ProteinContext, variants: pd.DataFrame, requested: list[str] | None = None
) -> pd.DataFrame:
    """Assemble the feature block for the wide output.

    A feature whose inputs are missing is skipped with a warning, never filled
    with a placeholder value.
    """
    requested = list(requested if requested is not None else DEFAULT_FEATURES)
    unknown = [f for f in requested if f not in ALL_FEATURES]
    if unknown:
        raise ValueError(f"unknown features {unknown}; available: {sorted(ALL_FEATURES)}")

    frame = variants.loc[:, ["position", "wt_aa", "mut_aa"]].copy()
    blocks: list[pd.DataFrame] = [frame]

    if "physchem" in requested:
        blocks.append(_per_variant_physchem(frame))
    if "blosum" in requested:
        blocks.append(pd.DataFrame(
            {"blosum62": [blosum62(w, m) for w, m in
                          zip(frame["wt_aa"], frame["mut_aa"], strict=True)]},
            index=frame.index,
        ))
    if "grantham" in requested:
        blocks.append(pd.DataFrame(
            {"grantham": [grantham_distance(w, m) for w, m in
                          zip(frame["wt_aa"], frame["mut_aa"], strict=True)]},
            index=frame.index,
        ))

    structure = protein.structure
    if structure is not None:
        if "rsasa" in requested:
            values = relative_sasa(structure)
            blocks.append(pd.DataFrame({"rsasa": frame["position"].map(values)},
                                       index=frame.index))
        if "plddt" in requested:
            values = structure.plddt()
            if values:
                blocks.append(pd.DataFrame({"plddt": frame["position"].map(values)},
                                           index=frame.index))
            else:
                log.info("pLDDT skipped: the structure is not an AlphaFold prediction")
        if "contacts" in requested:
            values = contact_number(structure)
            blocks.append(pd.DataFrame({"contact_number": frame["position"].map(values)},
                                       index=frame.index))
        if "depth" in requested:
            values = residue_depth(structure)
            blocks.append(pd.DataFrame({"residue_depth": frame["position"].map(values)},
                                       index=frame.index))
        if "secondary_structure" in requested:
            values = _secondary_structure(protein)
            if values:
                blocks.append(pd.DataFrame(
                    {"secondary_structure": frame["position"].map(values)}, index=frame.index
                ))
        if "bilayer_depth" in requested:
            values = protein.extras.get("bilayer_depth")
            if values:
                blocks.append(pd.DataFrame({"bilayer_depth": frame["position"].map(values)},
                                           index=frame.index))
            else:
                log.warning(
                    "bilayer_depth requested but no membrane-embedded structure was "
                    "supplied; pass an OPM or PDBTM model with --pdb"
                )
    else:
        skipped = [f for f in requested if f in
                   ("rsasa", "plddt", "contacts", "depth", "secondary_structure",
                    "bilayer_depth")]
        if skipped:
            log.warning("no structure available, skipping features: %s", skipped)

    if protein.msa is not None:
        records = None
        if "entropy" in requested or "gaps" in requested:
            records = read_alignment(protein.msa.path)
        if "entropy" in requested:
            values = per_position_entropy(records)
            blocks.append(pd.DataFrame({"msa_entropy_bits": frame["position"].map(values)},
                                       index=frame.index))
        if "gaps" in requested:
            values = per_position_gap_fraction(records)
            blocks.append(pd.DataFrame({"msa_gap_fraction": frame["position"].map(values)},
                                       index=frame.index))
    elif any(f in requested for f in ("entropy", "gaps")):
        log.warning("no MSA available, skipping conservation features")

    if "membrane" in requested:
        blocks.append(_membrane_annotations(protein, frame["position"]))

    return pd.concat(blocks, axis=1)


def membrane_depth_from_structure(
    protein: ProteinContext, normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> dict[int, float]:
    """Signed distance along the membrane normal, for OPM or PDBTM models.

    OPM places the bilayer centre at z = 0 with the normal along z, so the
    default is correct for an OPM download.
    """
    import numpy as np

    if protein.structure is None:
        return {}
    axis = np.asarray(normal, dtype=float)
    axis = axis / np.linalg.norm(axis)
    out: dict[int, float] = {}
    for residue in protein.structure.residues:
        ca = residue.ca_coord
        if residue.uniprot_position is None or ca is None:
            continue
        out[residue.uniprot_position] = float(ca @ axis)
    return out
