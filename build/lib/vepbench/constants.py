"""Amino-acid reference data used for features and validation.

Everything here is a published constant; the source is named next to each
table so the numbers can be checked rather than trusted.
"""

from __future__ import annotations

import math

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = frozenset(AA20)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common modified or alternate residues, mapped to the parent amino acid.
    "MSE": "M", "SEC": "C", "PYL": "K", "HSD": "H", "HSE": "H", "HSP": "H",
}
ONE_TO_THREE = {v: k for k, v in list(THREE_TO_ONE.items())[:20]}

# --- Grantham (1974) Science 185:862 -------------------------------------
# Composition, polarity and molecular volume with the published weights. The
# formula reproduces the published 20x20 matrix, so it is computed rather than
# embedded.
_GRANTHAM_CPV = {
    "S": (1.42, 9.2, 32), "R": (0.65, 10.5, 124), "L": (0.00, 4.9, 111),
    "P": (0.39, 8.0, 32.5), "T": (0.71, 8.6, 61), "A": (0.00, 8.1, 31),
    "V": (0.00, 5.9, 84), "G": (0.74, 9.0, 3), "I": (0.00, 5.2, 111),
    "F": (0.00, 5.2, 132), "Y": (0.20, 6.2, 136), "C": (2.75, 5.5, 55),
    "H": (0.58, 10.4, 96), "Q": (0.89, 10.5, 85), "N": (1.33, 11.6, 56),
    "K": (0.33, 11.3, 119), "D": (1.38, 13.0, 54), "E": (0.92, 12.3, 83),
    "M": (0.00, 5.7, 105), "W": (0.13, 5.4, 170),
}
_ALPHA, _BETA, _GAMMA, _SCALE = 1.833, 0.1018, 0.000399, 50.723


def grantham_distance(wt: str, mut: str) -> float:
    """Grantham chemical distance between two amino acids."""
    if wt not in _GRANTHAM_CPV or mut not in _GRANTHAM_CPV:
        return float("nan")
    c1, p1, v1 = _GRANTHAM_CPV[wt]
    c2, p2, v2 = _GRANTHAM_CPV[mut]
    return _SCALE * math.sqrt(
        _ALPHA * (c1 - c2) ** 2 + _BETA * (p1 - p2) ** 2 + _GAMMA * (v1 - v2) ** 2
    )


# --- BLOSUM62 (Henikoff & Henikoff 1992, PNAS 89:10915) ------------------
_BLOSUM62_RAW = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def _parse_blosum(raw: str) -> dict:
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    header = lines[0].split()
    out: dict[tuple[str, str], int] = {}
    for line in lines[1:]:
        parts = line.split()
        for col_aa, value in zip(header, parts[1:], strict=True):
            out[(parts[0], col_aa)] = int(value)
    return out


BLOSUM62 = _parse_blosum(_BLOSUM62_RAW)


def blosum62(wt: str, mut: str) -> float:
    return float(BLOSUM62.get((wt, mut), float("nan")))


# --- Kyte & Doolittle (1982) JMB 157:105 ---------------------------------
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

# --- Zamyatnin (1972) residue volumes, cubic angstroms -------------------
RESIDUE_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}

# Formal side-chain charge at pH 7; histidine takes the usual partial value.
RESIDUE_CHARGE = {aa: 0.0 for aa in AA20}
RESIDUE_CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})

# --- Tien et al. (2013) PLoS ONE 8:e80635, theoretical maximum ASA -------
MAX_ASA_TIEN = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

# Bondi (1964) van der Waals radii, angstroms, for the SASA calculation.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}
DEFAULT_VDW_RADIUS = 1.70
WATER_PROBE_RADIUS = 1.4
