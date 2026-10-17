"""Structures: AlphaFold download, PDB parsing, UniProt numbering, rSASA.

No structural-biology dependency is needed. The PDB reader, the Shrake-Rupley
solvent accessibility and the alignment used for renumbering are implemented
here on numpy, which keeps the orchestrator installable anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

from vepbench import HTTP_HEADERS
from vepbench.config import PATHS, Paths
from vepbench.constants import (
    DEFAULT_VDW_RADIUS,
    MAX_ASA_TIEN,
    THREE_TO_ONE,
    VDW_RADII,
    WATER_PROBE_RADIUS,
)

log = logging.getLogger(__name__)

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
RCSB_DOWNLOAD = "https://files.rcsb.org/download"
DEFAULT_TIMEOUT = 120


class StructureError(RuntimeError):
    pass


@dataclass
class Residue:
    chain: str
    seq_id: int
    insertion_code: str
    aa: str
    atoms: list[tuple[str, str, np.ndarray]] = field(default_factory=list)
    b_factors: list[float] = field(default_factory=list)
    uniprot_position: int | None = None

    @property
    def mean_b_factor(self) -> float:
        return float(np.mean(self.b_factors)) if self.b_factors else float("nan")

    @property
    def ca_coord(self) -> np.ndarray | None:
        for name, _element, coord in self.atoms:
            if name == "CA":
                return coord
        return None


@dataclass
class StructureRecord:
    path: Path
    source: str
    identifier: str
    residues: list[Residue]
    chain: str
    version: str = ""
    is_predicted: bool = False
    mapping_note: str = ""
    coverage: float = 0.0

    @property
    def sequence(self) -> str:
        return "".join(r.aa for r in self.residues)

    def by_uniprot_position(self) -> dict[int, Residue]:
        return {r.uniprot_position: r for r in self.residues if r.uniprot_position is not None}

    def plddt(self) -> dict[int, float]:
        """Per-residue pLDDT, keyed by UniProt position.

        Only meaningful for predicted models. For an experimental structure the
        B-factor is a different quantity, so nothing is returned.
        """
        if not self.is_predicted:
            return {}
        return {p: r.mean_b_factor for p, r in self.by_uniprot_position().items()}

    def as_dict(self) -> dict:
        return {
            "path": str(self.path), "source": self.source,
            "identifier": self.identifier, "chain": self.chain,
            "version": self.version, "is_predicted": self.is_predicted,
            "n_residues": len(self.residues), "mapping_note": self.mapping_note,
            "uniprot_coverage": round(self.coverage, 4),
        }


def parse_pdb(path: Path, chain: str | None = None) -> tuple[list[Residue], str]:
    """Minimal PDB reader for ATOM records.

    Only the first model is read, and only altloc A or blank, so alternate
    conformations do not duplicate residues.
    """
    residues: dict[tuple[str, int, str], Residue] = {}
    order: list[tuple[str, int, str]] = []
    chains_seen: list[str] = []

    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", "A"):
            continue
        aa = THREE_TO_ONE.get(line[17:20].strip().upper())
        if aa is None:
            continue
        chain_id = line[21].strip() or "A"
        if chain_id not in chains_seen:
            chains_seen.append(chain_id)
        if chain is not None and chain_id != chain:
            continue
        try:
            seq_id = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        icode = line[26].strip()
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name[:1]).upper()
        try:
            b_factor = float(line[60:66])
        except ValueError:
            b_factor = float("nan")

        key = (chain_id, seq_id, icode)
        if key not in residues:
            residues[key] = Residue(chain=chain_id, seq_id=seq_id, insertion_code=icode, aa=aa)
            order.append(key)
        residues[key].atoms.append((atom_name, element, np.array([x, y, z])))
        if atom_name == "CA":
            residues[key].b_factors.append(b_factor)

    if not order:
        raise StructureError(f"no usable ATOM records in {path}")

    chosen = chain or order[0][0]
    if chain is None and len(chains_seen) > 1:
        log.warning("structure %s has chains %s; using %s. Pass --chain to choose another.",
                    path.name, ",".join(chains_seen), chosen)
        order = [k for k in order if k[0] == chosen]
    return [residues[k] for k in order], chosen


def _needleman_wunsch(a: str, b: str, gap: float = -8.0) -> list[tuple[int | None, int | None]]:
    """Global alignment returning index pairs. Match +2, mismatch -1.

    Used only to renumber a structure onto UniProt, where the two sequences are
    nearly identical, so the scoring is deliberately simple.
    """
    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1))
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    pointer = np.zeros((n + 1, m + 1), dtype=np.int8)
    pointer[:, 0] = 1
    pointer[0, :] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (2.0 if ai == b[j - 1] else -1.0)
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            best = max(diag, up, left)
            score[i, j] = best
            pointer[i, j] = 0 if best == diag else (1 if best == up else 2)

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = pointer[i, j]
        if i > 0 and j > 0 and move == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and move == 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def assign_uniprot_numbering(residues: list[Residue], uniprot_sequence: str) -> tuple[str, float]:
    """Map structure residues onto UniProt positions.

    Tries the trivial cases first (author numbering already correct, then a
    constant offset) and only then aligns. Returns a note describing the route
    taken and the fraction of UniProt positions covered.
    """
    structure_sequence = "".join(r.aa for r in residues)
    length = len(uniprot_sequence)

    direct = all(
        1 <= r.seq_id <= length and uniprot_sequence[r.seq_id - 1] == r.aa for r in residues
    )
    if direct and not any(r.insertion_code for r in residues):
        for r in residues:
            r.uniprot_position = r.seq_id
        return ("author numbering matches UniProt directly",
                len({r.uniprot_position for r in residues}) / length)

    index = uniprot_sequence.find(structure_sequence)
    if index != -1 and structure_sequence:
        for offset, r in enumerate(residues):
            r.uniprot_position = index + offset + 1
        return (f"contiguous substring match at UniProt offset {index + 1}",
                len(structure_sequence) / length)

    pairs = _needleman_wunsch(structure_sequence, uniprot_sequence)
    matched = 0
    for struct_idx, uni_idx in pairs:
        if struct_idx is None or uni_idx is None:
            continue
        residue = residues[struct_idx]
        if uniprot_sequence[uni_idx] == residue.aa:
            residue.uniprot_position = uni_idx + 1
            matched += 1
        else:
            residue.uniprot_position = None  # mismatch: refuse to map it
    return (f"global alignment, {matched}/{len(residues)} structure residues mapped",
            matched / length)


def alphafold_metadata(uniprot_id: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    response = requests.get(
        f"{ALPHAFOLD_API}/{uniprot_id}", timeout=timeout,
        headers={**HTTP_HEADERS, "Accept": "application/json"},
    )
    if response.status_code == 404:
        raise StructureError(f"no AlphaFold DB model for {uniprot_id}")
    response.raise_for_status()
    payload = response.json()
    if not payload:
        raise StructureError(f"no AlphaFold DB model for {uniprot_id}")
    return payload[0]


def fetch_alphafold(
    uniprot_id: str, *, paths: Paths = PATHS, force: bool = False,
    offline: bool = False, timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Path, str]:
    """Download the AlphaFold DB model, returning the file path and version.

    AlphaFold's file host answers 403 to a default client user agent, which
    looks like a network block but is not one, so a real User-Agent is sent.
    """
    paths.structures.mkdir(parents=True, exist_ok=True)
    existing = sorted(paths.structures.glob(f"AF-{uniprot_id}-F1-model_v*.pdb"))
    if existing and (not force or offline):
        path = existing[-1]
        return path, path.stem.split("model_")[-1]
    if offline:
        raise StructureError(
            f"no cached AlphaFold model for {uniprot_id} and --offline was requested "
            f"(looked in {paths.structures})"
        )

    meta = alphafold_metadata(uniprot_id, timeout=timeout)
    url = meta.get("pdbUrl")
    if not url:
        raise StructureError(f"AlphaFold entry for {uniprot_id} has no PDB URL")
    version = f"v{meta.get('latestVersion', '?')}"
    destination = paths.structures / Path(url).name

    log.info("downloading AlphaFold model %s", url)
    response = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
    if response.status_code != 200:
        raise StructureError(
            f"AlphaFold download failed for {uniprot_id}: HTTP {response.status_code} "
            f"from {url}. Download it elsewhere and pass it with --pdb if this "
            f"environment restricts outbound hosts."
        )
    destination.write_bytes(response.content)
    return destination, version


def fetch_pdb(
    pdb_id: str, *, paths: Paths = PATHS, force: bool = False, timeout: int = DEFAULT_TIMEOUT
) -> Path:
    """Download an experimental structure from the RCSB by its four-letter ID."""
    paths.structures.mkdir(parents=True, exist_ok=True)
    destination = paths.structures / f"{pdb_id.lower()}.pdb"
    if destination.exists() and not force:
        return destination
    url = f"{RCSB_DOWNLOAD}/{pdb_id.upper()}.pdb"
    response = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
    if response.status_code != 200:
        raise StructureError(
            f"could not download {pdb_id} from the RCSB (HTTP {response.status_code}). "
            f"Download it yourself and pass it with --pdb."
        )
    destination.write_bytes(response.content)
    return destination


def load_structure(
    uniprot_id: str, uniprot_sequence: str, *, override: Path | None = None,
    pdb_id: str | None = None, chain: str | None = None, paths: Paths = PATHS,
    force: bool = False, offline: bool = False,
) -> StructureRecord:
    """Get a structure for this protein and map it onto UniProt numbering."""
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise StructureError(f"structure file not found: {path}")
        source, identifier, version, predicted = "user", path.stem, "", False
    elif pdb_id is not None:
        path = fetch_pdb(pdb_id, paths=paths, force=force)
        source, identifier, version, predicted = "rcsb", pdb_id.upper(), "", False
    else:
        path, version = fetch_alphafold(
            uniprot_id, paths=paths, force=force, offline=offline
        )
        source, identifier, predicted = "alphafold", f"AF-{uniprot_id}-F1", True

    residues, chosen_chain = parse_pdb(path, chain=chain)
    note, coverage = assign_uniprot_numbering(residues, uniprot_sequence)
    log.info("structure %s: %s (coverage %.1f%%)", path.name, note, 100 * coverage)
    return StructureRecord(
        path=path, source=source, identifier=identifier, residues=residues,
        chain=chosen_chain, version=version, is_predicted=predicted,
        mapping_note=note, coverage=coverage,
    )


def _sphere_points(n: int) -> np.ndarray:
    """Golden-spiral points on the unit sphere."""
    indices = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    return np.column_stack(
        (np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))
    )


def shrake_rupley(residues: list[Residue], n_points: int = 240) -> dict[int, float]:
    """Per-residue solvent accessible surface area in square angstroms.

    Shrake & Rupley (1973) numerical integration with a 1.4 angstrom probe.
    Keyed by index into ``residues``.
    """
    coords, radii, owner = [], [], []
    for idx, residue in enumerate(residues):
        for _name, element, coord in residue.atoms:
            coords.append(coord)
            radii.append(VDW_RADII.get(element, DEFAULT_VDW_RADIUS) + WATER_PROBE_RADIUS)
            owner.append(idx)
    if not coords:
        return {}

    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    owner = np.asarray(owner, dtype=int)
    sphere = _sphere_points(n_points)
    max_reach = radii.max()

    areas = np.zeros(len(residues))
    for i in range(len(coords)):
        distance = np.linalg.norm(coords - coords[i], axis=1)
        neighbours = np.where((distance < radii[i] + max_reach) & (distance > 0))[0]
        test_points = coords[i] + radii[i] * sphere
        if len(neighbours):
            diff = test_points[:, None, :] - coords[neighbours][None, :, :]
            inside = (np.einsum("ijk,ijk->ij", diff, diff) < radii[neighbours] ** 2).any(axis=1)
            accessible = int((~inside).sum())
        else:
            accessible = n_points
        areas[owner[i]] += 4.0 * np.pi * radii[i] ** 2 * accessible / n_points
    return {idx: float(areas[idx]) for idx in range(len(residues))}


def relative_sasa(structure: StructureRecord, n_points: int = 240) -> dict[int, float]:
    """Relative SASA per UniProt position, using Tien et al. (2013) maxima.

    Values above 1 are possible and are left unclipped, so nothing is silently
    altered.
    """
    absolute = shrake_rupley(structure.residues, n_points=n_points)
    out: dict[int, float] = {}
    for idx, residue in enumerate(structure.residues):
        if residue.uniprot_position is None:
            continue
        reference = MAX_ASA_TIEN.get(residue.aa)
        if reference:
            out[residue.uniprot_position] = absolute.get(idx, float("nan")) / reference
    return out


def contact_number(structure: StructureRecord, cutoff: float = 10.0) -> dict[int, int]:
    """Number of other residues with a C-alpha within ``cutoff`` angstroms."""
    positions, coords = [], []
    for residue in structure.residues:
        ca = residue.ca_coord
        if residue.uniprot_position is None or ca is None:
            continue
        positions.append(residue.uniprot_position)
        coords.append(ca)
    if not coords:
        return {}
    coords = np.asarray(coords)
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    counts = ((distance < cutoff).sum(axis=1) - 1).astype(int)
    return dict(zip(positions, counts.tolist(), strict=True))


def residue_depth(structure: StructureRecord, n_points: int = 120) -> dict[int, float]:
    """Distance from each C-alpha to the nearest accessible surface point.

    A cheap stand-in for MSMS residue depth: the surface is approximated by the
    accessible probe points of all atoms.
    """
    coords, radii = [], []
    for residue in structure.residues:
        for _name, element, coord in residue.atoms:
            coords.append(coord)
            radii.append(VDW_RADII.get(element, DEFAULT_VDW_RADIUS) + WATER_PROBE_RADIUS)
    if not coords:
        return {}
    coords_arr = np.asarray(coords, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)
    sphere = _sphere_points(n_points)

    surface = []
    for i in range(len(coords_arr)):
        test_points = coords_arr[i] + radii_arr[i] * sphere
        diff = test_points[:, None, :] - coords_arr[None, :, :]
        distance_sq = np.einsum("ijk,ijk->ij", diff, diff)
        distance_sq[:, i] = np.inf
        exposed = (distance_sq >= radii_arr ** 2).all(axis=1)
        if exposed.any():
            surface.append(test_points[exposed])
    if not surface:
        return {}
    surface_points = np.vstack(surface)

    out: dict[int, float] = {}
    for residue in structure.residues:
        ca = residue.ca_coord
        if residue.uniprot_position is None or ca is None:
            continue
        out[residue.uniprot_position] = float(np.linalg.norm(surface_points - ca, axis=1).min())
    return out
