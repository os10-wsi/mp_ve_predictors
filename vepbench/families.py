"""Protein families: which sequences, and which of them to actually score.

A large Pfam family holds hundreds of thousands of sequences, nearly all
of them unreviewed and most of them near-duplicates of each other. Scoring
them all would be mostly redundant computation, so each family is
clustered and a sample of the cluster representatives is scored instead.

The clustering is the point, not an optimisation. A family's membership is
heavily biased by which organisms happen to have been sequenced, so a
random sample of raw members would mostly measure that bias. One sequence
per cluster flattens it.

Note what this cannot give you: family members are overwhelmingly TrEMBL,
so there are no ClinVar labels and therefore no ROC curves for them. The
family comparison is about how the score distributions differ, which is a
different question from how well a predictor separates known variants.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

from vepbench import HTTP_HEADERS
from vepbench.config import PATHS, Paths

log = logging.getLogger(__name__)

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
DEFAULT_TIMEOUT = 1800

#: The twenty largest transmembrane Pfam families, each verified to be
#: 99-100% transmembrane by UniProt's own keyword. Member counts are from
#: November 2026 and move slowly.
FAMILIES: dict[str, str] = {
    "PF07690": "MFS_1 (major facilitator superfamily)",
    "PF00528": "BPD_transp_1 (ABC transporter permease)",
    "PF00001": "7tm_1 (class A GPCR)",
    "PF00664": "ABC_membrane",
    "PF00083": "Sugar_tr (sugar transporter)",
    "PF00892": "EamA",
    "PF00520": "Ion_trans (voltage-gated ion channel)",
    "PF13520": "AA_permease_2",
    "PF01061": "ABC2_membrane",
    "PF00999": "Na_H_Exchanger",
    "PF01554": "MatE",
    "PF00873": "ACR_tran (RND efflux)",
    "PF00474": "SSF (sodium:solute symporter)",
    "PF00324": "AA_permease",
    "PF00230": "MIP (aquaporin)",
    "PF00860": "Xan_ur_permease",
    "PF00209": "SNF (neurotransmitter symporter)",
    "PF00654": "Voltage_CLC",
    "PF03176": "MMPL",
    "PF03547": "Mem_trans",
}

#: Sequence identity for clustering. 50% collapses orthologues and close
#: paralogues while keeping the genuinely distinct architectures apart.
DEFAULT_IDENTITY = 0.5
DEFAULT_SAMPLE = 1000


class FamilyError(RuntimeError):
    pass


@dataclass
class FamilySet:
    """One family's members, representatives, and the sample scored."""

    pfam_id: str
    name: str
    n_members: int
    n_clusters: int
    accessions: list[str]
    identity: float
    seed: int

    def summary(self) -> str:
        return (f"{self.pfam_id} {self.name}: {self.n_members:,} members, "
                f"{self.n_clusters:,} clusters at {100 * self.identity:.0f}% identity, "
                f"{len(self.accessions):,} sampled")

    def as_dict(self) -> dict:
        return {
            "pfam_id": self.pfam_id, "name": self.name,
            "n_members": self.n_members, "n_clusters": self.n_clusters,
            "n_sampled": len(self.accessions), "identity": self.identity,
            "seed": self.seed,
        }


def directory(pfam_id: str, *, paths: Paths = PATHS) -> Path:
    return paths.outputs / "families" / pfam_id


def fetch_members(
    pfam_id: str, *, paths: Paths = PATHS, force: bool = False,
    timeout: int = DEFAULT_TIMEOUT, reviewed_only: bool = False,
) -> Path:
    """Stream a family's sequences to a FASTA file.

    The largest families are several hundred megabytes, so the response is
    streamed to disk rather than held in memory, and a complete download is
    reused unless forced.
    """
    target = directory(pfam_id, paths=paths)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "members.fasta"
    if path.exists() and path.stat().st_size > 0 and not force:
        log.info("reusing %s", path)
        return path

    query = f"xref:pfam-{pfam_id}"
    if reviewed_only:
        query += " AND reviewed:true"
    log.info("downloading %s members from UniProt", pfam_id)
    tmp = path.with_name(path.name + ".tmp")
    with requests.get(
        UNIPROT_STREAM, params={"query": query, "format": "fasta"},
        headers=HTTP_HEADERS, timeout=timeout, stream=True,
    ) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    tmp.replace(path)
    return path


def count_sequences(path: Path) -> int:
    with Path(path).open() as handle:
        return sum(1 for line in handle if line.startswith(">"))


def accession_of(header: str) -> str:
    """``sp|P12345|NAME_ORG ...`` or ``tr|A0A123|...`` to the accession."""
    text = header.lstrip(">").split()[0]
    parts = text.split("|")
    return parts[1] if len(parts) >= 2 else text


def have_mmseqs(paths: Paths = PATHS) -> str | None:
    """mmseqs on PATH, or inside the environment that installs it.

    A conda environment built under envs/ is not on PATH, so looking only
    there would make `install-env clustering` appear to do nothing.
    """
    found = shutil.which("mmseqs")
    if found:
        return found
    bundled = paths.envs / "clustering" / ".venv" / "bin" / "mmseqs"
    return str(bundled) if bundled.exists() else None


def cluster(
    fasta: Path, *, identity: float = DEFAULT_IDENTITY, coverage: float = 0.8,
    threads: int = 8, workdir: Path | None = None, paths: Paths = PATHS,
) -> list[str]:
    """Cluster a FASTA with mmseqs2, returning one accession per cluster."""
    executable = have_mmseqs(paths)
    if executable is None:
        raise FamilyError(
            f"mmseqs was not found on PATH or at "
            f"{paths.envs / 'clustering' / '.venv' / 'bin' / 'mmseqs'}. Install it "
            f"with `vepbench install-env clustering --backend conda`, or "
            f"`conda install -c bioconda mmseqs2`, then rerun."
        )
    fasta = Path(fasta)
    workdir = Path(workdir) if workdir else fasta.parent / "mmseqs"
    workdir.mkdir(parents=True, exist_ok=True)
    prefix = workdir / "cluster"

    command = [
        executable, "easy-cluster", str(fasta), str(prefix), str(workdir / "tmp"),
        "--min-seq-id", str(identity), "-c", str(coverage),
        "--cov-mode", "0", "--threads", str(threads),
    ]
    log.info("clustering %s at %.0f%% identity", fasta.name, 100 * identity)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise FamilyError(
            f"mmseqs easy-cluster failed (exit {completed.returncode}): "
            + " | ".join(completed.stderr.strip().splitlines()[-4:])
        )

    representatives = prefix.with_name(prefix.name + "_rep_seq.fasta")
    if not representatives.exists():
        raise FamilyError(f"mmseqs produced no representatives at {representatives}")
    with representatives.open() as handle:
        return [accession_of(line) for line in handle if line.startswith(">")]


def sample(accessions: list[str], n: int, seed: int = 0) -> list[str]:
    """Take n representatives, deterministically.

    Sorted before sampling so the same seed gives the same set whatever
    order mmseqs happened to emit, which it does not promise.
    """
    ordered = sorted(set(accessions))
    if len(ordered) <= n:
        return ordered
    return sorted(random.Random(seed).sample(ordered, n))


def build(
    pfam_id: str, *, n: int = DEFAULT_SAMPLE, identity: float = DEFAULT_IDENTITY,
    seed: int = 0, paths: Paths = PATHS, force: bool = False, threads: int = 8,
) -> FamilySet:
    """Download, cluster and sample one family. Writes the accession list."""
    if pfam_id not in FAMILIES and not pfam_id.startswith("PF"):
        raise FamilyError(f"{pfam_id!r} is not a Pfam accession")
    target = directory(pfam_id, paths=paths)
    listing = target / "accessions.txt"
    record = target / "family.json"

    if listing.exists() and record.exists() and not force:
        stored = json.loads(record.read_text())
        log.info("reusing %s (%d accessions); --force to rebuild",
                 listing, stored.get("n_sampled", 0))
        return FamilySet(
            pfam_id=pfam_id, name=stored.get("name", ""),
            n_members=stored.get("n_members", 0),
            n_clusters=stored.get("n_clusters", 0),
            accessions=listing.read_text().split(),
            identity=stored.get("identity", identity), seed=stored.get("seed", seed),
        )

    fasta = fetch_members(pfam_id, paths=paths, force=force)
    n_members = count_sequences(fasta)
    representatives = cluster(fasta, identity=identity, threads=threads,
                              workdir=target / "mmseqs", paths=paths)
    chosen = sample(representatives, n, seed=seed)

    listing.write_text("\n".join(chosen) + "\n")
    result = FamilySet(
        pfam_id=pfam_id, name=FAMILIES.get(pfam_id, ""), n_members=n_members,
        n_clusters=len(set(representatives)), accessions=chosen,
        identity=identity, seed=seed,
    )
    record.write_text(json.dumps(result.as_dict(), indent=2))
    log.info("%s", result.summary())
    return result


def membership(paths: Paths = PATHS) -> dict[str, str]:
    """``{accession: pfam_id}`` across every family built so far.

    An accession in two families is assigned to the first by Pfam id, so a
    protein never contributes twice to a comparison between families.
    """
    out: dict[str, str] = {}
    root = paths.outputs / "families"
    if not root.exists():
        return out
    for listing in sorted(root.glob("PF*/accessions.txt")):
        pfam_id = listing.parent.name
        for accession in listing.read_text().split():
            out.setdefault(accession, pfam_id)
    return out
