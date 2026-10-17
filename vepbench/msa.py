"""Multiple sequence alignments: generation, caching and quality statistics.

EVE, GEMME, EVmutation, DeepSequence, TranceptEVE, popEVE and the
retrieval-augmented models all need an alignment. Generating one is the
slowest and most environment-dependent step here, so alignments are cached by
(accession, backend, settings) and can always be supplied with ``--msa``.

Backends, in the order you would normally reach for them:

``ebi``
    EBI Job Dispatcher BLAST or PSI-BLAST against UniProtKB, converted to a
    query-anchored A3M. Needs no local database and no extra host beyond
    ``www.ebi.ac.uk``. Jobs are queued server-side and take minutes, and the
    alignments are shallower than a profile HMM search, so Neff is reported
    and a thin alignment is flagged.
``colabfold``
    MMseqs2 against UniRef30 plus environmental databases on the public
    ColabFold server. Deepest alignments for the least effort, but needs
    ``api.colabfold.com`` reachable.
``jackhmmer`` / ``hhblits``
    Local search against a database you supply. The EVE and EVcouplings route,
    and the most sensitive, but UniRef90 is roughly 60 GB and UniRef30 roughly
    90 GB.
``none``
    No alignment; MSA-dependent predictors are skipped with a reason.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from vepbench import HTTP_HEADERS
from vepbench.config import PATHS, Paths, cache_key

log = logging.getLogger(__name__)

COLABFOLD_API = "https://api.colabfold.com"
EBI_REST = "https://www.ebi.ac.uk/Tools/services/rest"
DEFAULT_TIMEOUT = 120

# Below this effective sequence count the evolutionary models are unreliable,
# and we say so loudly rather than quietly producing numbers.
LOW_NEFF_THRESHOLD = 100.0
LOW_DEPTH_THRESHOLD = 50


class MSAError(RuntimeError):
    pass


@dataclass
class MSARecord:
    path: Path
    backend: str
    n_sequences: int
    n_effective: float
    query_length: int
    mean_coverage: float
    mean_identity: float
    database: str = ""
    seconds: float = 0.0

    @property
    def is_shallow(self) -> bool:
        return self.n_effective < LOW_NEFF_THRESHOLD or self.n_sequences < LOW_DEPTH_THRESHOLD

    @property
    def neff_per_length(self) -> float:
        return self.n_effective / self.query_length if self.query_length else 0.0

    def as_dict(self) -> dict:
        return {
            "path": str(self.path), "backend": self.backend, "database": self.database,
            "n_sequences": self.n_sequences,
            "n_effective": round(self.n_effective, 2),
            "neff_per_length": round(self.neff_per_length, 3),
            "query_length": self.query_length,
            "mean_coverage": round(self.mean_coverage, 4),
            "mean_identity": round(self.mean_identity, 4),
            "is_shallow": self.is_shallow, "seconds": round(self.seconds, 1),
        }


# --------------------------------------------------------------------------
# Parsing and statistics
# --------------------------------------------------------------------------

def read_a3m(path: Path) -> list[tuple[str, str]]:
    """Read FASTA or A3M. Lower-case insertion columns are kept as written."""
    records: list[tuple[str, str]] = []
    header, chunks = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:].strip(), []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise MSAError(f"no sequences in {path}")
    return records


def read_stockholm(path: Path) -> list[tuple[str, str]]:
    sequences: dict[str, list[str]] = {}
    order: list[str] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("#") or line == "//":
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name, chunk = parts
        if name not in sequences:
            sequences[name] = []
            order.append(name)
        sequences[name].append(chunk.strip())
    if not order:
        raise MSAError(f"no sequences in {path}")
    return [(name, "".join(sequences[name])) for name in order]


def read_alignment(path: Path) -> list[tuple[str, str]]:
    path = Path(path)
    if path.suffix.lower() in (".sto", ".stockholm"):
        return read_stockholm(path)
    return read_a3m(path)


def to_matched_columns(records: list[tuple[str, str]]) -> np.ndarray:
    """Project an A3M onto the query's columns.

    Insertions relative to the query (lower case in A3M) are dropped, so every
    row has exactly the query length.
    """
    query = records[0][1]
    keep = [i for i, c in enumerate(query) if c.isupper() or c == "-"]
    rows = []
    for _name, raw_seq in records:
        seq = raw_seq.ljust(len(query), "-") if len(raw_seq) < len(query) else raw_seq
        rows.append([seq[i].upper() if i < len(seq) else "-" for i in keep])
    return np.array(rows, dtype="<U1")


def alignment_statistics(
    records: list[tuple[str, str]], *, identity_threshold: float = 0.8,
    max_sequences: int = 20000,
) -> tuple[int, float, float, float]:
    """Return (n_sequences, Neff, mean coverage, mean identity to the query).

    Neff is the standard reweighted count: each sequence contributes
    1 / (number of sequences within ``identity_threshold`` identity of it). The
    pairwise step is capped by subsampling and scaled back, so very deep
    alignments stay tractable.
    """
    matrix = to_matched_columns(records)
    n_sequences = matrix.shape[0]
    query = matrix[0]
    ungapped = query != "-"

    coverage = (matrix[:, ungapped] != "-").mean(axis=1)
    identity = (matrix[:, ungapped] == query[ungapped]).mean(axis=1)

    sample, scale = matrix, 1.0
    if n_sequences > max_sequences:
        rng = np.random.default_rng(0)
        sample = matrix[rng.choice(n_sequences, max_sequences, replace=False)]
        scale = n_sequences / max_sequences

    codes = np.frombuffer(sample.astype("S1").tobytes(), dtype=np.uint8).reshape(sample.shape)
    length = codes.shape[1]
    neighbours = np.ones(codes.shape[0], dtype=float)
    block = 512
    for start in range(0, codes.shape[0], block):
        chunk = codes[start : start + block]
        same = (chunk[:, None, :] == codes[None, :, :]).sum(axis=2) / max(length, 1)
        neighbours[start : start + block] = (same >= identity_threshold).sum(axis=1)
    n_effective = float((1.0 / neighbours).sum() * scale)

    return n_sequences, n_effective, float(coverage.mean()), float(identity.mean())


def describe_alignment(
    path: Path, backend: str, database: str = "", seconds: float = 0.0
) -> MSARecord:
    records = read_alignment(path)
    n_sequences, n_effective, coverage, identity = alignment_statistics(records)
    query_length = len([c for c in records[0][1] if c != "-" and not c.islower()])
    record = MSARecord(
        path=Path(path), backend=backend, n_sequences=n_sequences,
        n_effective=n_effective, query_length=query_length, mean_coverage=coverage,
        mean_identity=identity, database=database, seconds=seconds,
    )
    if record.is_shallow:
        log.warning(
            "shallow alignment for %s: %d sequences, Neff %.1f (%.2f per residue). "
            "Evolutionary models (EVE, GEMME, EVmutation, DeepSequence) are "
            "unreliable at this depth.",
            path.name, record.n_sequences, record.n_effective, record.neff_per_length,
        )
    return record


def per_position_entropy(
    records: list[tuple[str, str]], *, pseudocount: float = 1e-9
) -> dict[int, float]:
    """Shannon entropy per query position, in bits, gaps excluded."""
    matrix = to_matched_columns(records)
    query = matrix[0]
    out: dict[int, float] = {}
    position = 0
    for column_index, query_char in enumerate(query):
        if query_char == "-":
            continue
        position += 1
        column = matrix[:, column_index]
        column = column[(column != "-") & (column != "X")]
        if column.size == 0:
            out[position] = float("nan")
            continue
        _, counts = np.unique(column, return_counts=True)
        frequency = counts / counts.sum()
        out[position] = float(-(frequency * np.log2(frequency + pseudocount)).sum())
    return out


def per_position_gap_fraction(records: list[tuple[str, str]]) -> dict[int, float]:
    matrix = to_matched_columns(records)
    query = matrix[0]
    out: dict[int, float] = {}
    position = 0
    for column_index, query_char in enumerate(query):
        if query_char == "-":
            continue
        position += 1
        out[position] = float((matrix[:, column_index] == "-").mean())
    return out


# --------------------------------------------------------------------------
# EBI Job Dispatcher backend
# --------------------------------------------------------------------------

def hsp_to_a3m_row(
    query_length: int, qseq: str, hseq: str, query_from: int, query_to: int
) -> str:
    """Convert one BLAST pairwise alignment into a query-anchored A3M row.

    Columns where the query has a residue carry the hit residue in upper case.
    Columns inserted in the hit relative to the query become lower case, which
    is exactly the A3M convention. Query positions outside the aligned span are
    gaps.
    """
    pieces = ["-" * (query_from - 1)]
    for q_char, h_char in zip(qseq, hseq, strict=True):
        if q_char == "-":
            if h_char != "-":
                pieces.append(h_char.lower())     # insertion relative to the query
        else:
            pieces.append(h_char.upper() if h_char != "-" else "-")
    pieces.append("-" * (query_length - query_to))
    return "".join(pieces)


def blast_json_to_a3m(payload: dict, sequence: str, destination: Path) -> int:
    """Write an A3M from an EBI Job Dispatcher BLAST result.

    Only the best-scoring HSP per hit is used, because a query-anchored A3M
    cannot represent two disjoint alignments of the same subject on one row.
    """
    query_length = len(sequence)
    lines = [">query", sequence]
    seen: set[str] = set()
    written = 0
    for hit in payload.get("hits", []) or []:
        hsps = hit.get("hit_hsps") or []
        if not hsps:
            continue
        best = max(hsps, key=lambda h: float(h.get("hsp_bit_score", 0)))
        qseq, hseq = best.get("hsp_qseq"), best.get("hsp_hseq")
        if not qseq or not hseq:
            continue
        name = str(hit.get("hit_acc") or hit.get("hit_id") or f"hit{written}")
        if name in seen:
            continue
        seen.add(name)
        row = hsp_to_a3m_row(
            query_length, qseq, hseq,
            int(best.get("hsp_query_from", 1)), int(best.get("hsp_query_to", query_length)),
        )
        description = str(hit.get("hit_uni_de") or hit.get("hit_desc") or "").replace("\n", " ")
        lines.append(f">{name} {description[:120]}".rstrip())
        lines.append(row)
        written += 1
    destination.write_text("\n".join(lines) + "\n")
    return written


def _run_ebi_job(
    program: str, parameters: dict, email: str, timeout: int, poll_seconds: int = 20,
    max_wait_seconds: int = 5400,
) -> dict:
    """Submit a Job Dispatcher job, wait for it, and return the JSON result."""
    run_url = f"{EBI_REST}/{program}/run"
    payload = {**parameters, "email": email}
    response = requests.post(run_url, data=payload, timeout=timeout, headers=HTTP_HEADERS)
    if response.status_code != 200:
        raise MSAError(
            f"EBI {program} submission failed (HTTP {response.status_code}): "
            f"{response.text[:400]}"
        )
    job_id = response.text.strip()
    log.info("EBI %s job %s submitted; these are queued server-side and take minutes",
             program, job_id)

    deadline = time.time() + max_wait_seconds
    last = ""
    while time.time() < deadline:
        status = requests.get(
            f"{EBI_REST}/{program}/status/{job_id}", timeout=timeout, headers=HTTP_HEADERS
        ).text.strip()
        if status != last:
            log.info("  EBI %s %s: %s", program, job_id, status)
            last = status
        if status == "FINISHED":
            break
        if status in ("ERROR", "FAILURE", "NOT_FOUND"):
            raise MSAError(f"EBI {program} job {job_id} ended with status {status}")
        time.sleep(poll_seconds)
    else:
        raise MSAError(
            f"EBI {program} job {job_id} did not finish within "
            f"{max_wait_seconds // 60} minutes. It may still complete; re-run to pick it up, "
            f"or use a different --msa-backend."
        )

    result = requests.get(
        f"{EBI_REST}/{program}/result/{job_id}/json", timeout=600, headers=HTTP_HEADERS
    )
    if result.status_code != 200:
        raise MSAError(
            f"could not retrieve JSON for EBI job {job_id} (HTTP {result.status_code})"
        )
    return result.json()


def _run_ebi(
    sequence: str, destination: Path, *, program: str, database: str, email: str,
    max_hits: int, iterations: int, expect: float, timeout: int,
) -> str:
    fasta = f">query\n{sequence}"
    if program == "psiblast":
        parameters = {
            "database": database, "stype": "protein", "sequence": fasta,
            "expthr": str(expect), "psithr": str(expect), "iterations": str(iterations),
            "alignments": str(max_hits), "scores": str(max_hits),
        }
    elif program == "ncbiblast":
        parameters = {
            "program": "blastp", "database": database, "stype": "protein",
            "sequence": fasta, "exp": str(expect),
            "alignments": str(max_hits), "scores": str(max_hits),
        }
    else:
        raise MSAError(f"unknown EBI program {program!r}; use 'ncbiblast' or 'psiblast'")

    payload = _run_ebi_job(program, parameters, email, timeout)
    written = blast_json_to_a3m(payload, sequence, destination)
    if written == 0:
        raise MSAError(f"EBI {program} returned no alignable hits for this sequence")
    log.info("wrote %d aligned homologues to %s", written, destination.name)
    return f"EBI {program} vs {database} (E <= {expect}, {written} hits)"


# --------------------------------------------------------------------------
# Other backends
# --------------------------------------------------------------------------

def build_msa(
    uniprot_id: str, sequence: str, *, backend: str = "ebi", paths: Paths = PATHS,
    override: Path | None = None, force: bool = False, offline: bool = False,
    database: str | None = None, email: str = "", max_hits: int = 1000,
    iterations: int = 3, expect: float = 1e-3, jackhmmer_db: Path | None = None,
    hhblits_db: Path | None = None, jackhmmer_bitscore_per_residue: float = 0.5,
) -> MSARecord:
    """Obtain an alignment, from cache when possible."""
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise MSAError(f"MSA file not found: {path}")
        return describe_alignment(path, backend="user-supplied")

    if backend == "none":
        raise MSAError(
            "no MSA backend selected. Pass --msa with an alignment, or choose "
            "--msa-backend ebi (no local database needed)."
        )

    paths.msas.mkdir(parents=True, exist_ok=True)
    key = cache_key(uniprot_id, sequence, backend, database, max_hits, iterations,
                    expect, str(jackhmmer_db), str(hhblits_db),
                    jackhmmer_bitscore_per_residue)
    destination = paths.msas / f"{uniprot_id}.{backend}.{key}.a3m"
    if destination.exists() and (not force or offline):
        log.info("using cached MSA %s", destination.name)
        return describe_alignment(destination, backend=backend)
    if offline:
        raise MSAError(
            f"no cached MSA for {uniprot_id} and --offline was requested "
            f"(looked for {destination})"
        )

    started = time.time()
    if backend == "ebi":
        if not email:
            raise MSAError(
                "the EBI Job Dispatcher requires a contact email address. Pass "
                "--email, or set VEPBENCH_EMAIL. EBI uses it to contact you about "
                "misbehaving jobs; it is their documented requirement, not ours."
            )
        resolved = _run_ebi(
            sequence, destination, program=database_to_program(database),
            database=database_to_db(database), email=email, max_hits=max_hits,
            iterations=iterations, expect=expect, timeout=DEFAULT_TIMEOUT,
        )
    elif backend == "colabfold":
        resolved = _run_colabfold(sequence, destination)
    elif backend == "jackhmmer":
        resolved = _run_jackhmmer(sequence, destination, jackhmmer_db, iterations,
                                  jackhmmer_bitscore_per_residue)
    elif backend == "hhblits":
        resolved = _run_hhblits(sequence, destination, hhblits_db)
    else:
        raise MSAError(f"unknown MSA backend {backend!r}")
    return describe_alignment(destination, backend=backend, database=resolved,
                              seconds=time.time() - started)


def database_to_program(database: str | None) -> str:
    """``psiblast`` is more sensitive; ``ncbiblast`` finishes faster."""
    if not database or ":" not in database:
        return "psiblast"
    return database.split(":", 1)[0]


def database_to_db(database: str | None) -> str:
    if not database:
        return "uniprotkb"
    return database.split(":", 1)[1] if ":" in database else database


def _run_colabfold(sequence: str, destination: Path, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Submit to the public ColabFold MMseqs2 server and poll until ready."""
    try:
        response = requests.post(
            f"{COLABFOLD_API}/ticket/msa", data={"q": f">query\n{sequence}", "mode": "env"},
            timeout=timeout, headers=HTTP_HEADERS,
        )
        response.raise_for_status()
        ticket = response.json()
    except requests.RequestException as exc:
        raise MSAError(
            f"could not reach the ColabFold MMseqs2 server ({exc}). Use "
            f"--msa-backend ebi, which needs no extra host, or supply --msa."
        ) from exc

    identifier = ticket.get("id")
    if not identifier:
        raise MSAError(f"ColabFold did not return a ticket: {ticket}")

    deadline = time.time() + 3600
    while time.time() < deadline:
        status = requests.get(
            f"{COLABFOLD_API}/ticket/{identifier}", timeout=timeout, headers=HTTP_HEADERS
        ).json().get("status")
        if status == "COMPLETE":
            break
        if status in ("ERROR", "UNKNOWN", "RATELIMIT"):
            raise MSAError(f"ColabFold MSA job failed with status {status}")
        time.sleep(10)
    else:
        raise MSAError("ColabFold MSA job timed out after one hour")

    result = requests.get(
        f"{COLABFOLD_API}/result/download/{identifier}", timeout=600, headers=HTTP_HEADERS
    )
    result.raise_for_status()
    archive = destination.with_suffix(".tar.gz")
    archive.write_bytes(result.content)
    _extract_a3m(archive, destination)
    archive.unlink(missing_ok=True)
    return "ColabFold MMseqs2 (UniRef30 + ColabFoldDB environmental)"


def _extract_a3m(archive: Path, destination: Path) -> None:
    import tarfile

    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".a3m")]
        if not members:
            raise MSAError(f"no .a3m inside {archive}")
        extracted = tar.extractfile(max(members, key=lambda m: m.size))
        if extracted is None:
            raise MSAError(f"could not read the alignment from {archive}")
        destination.write_bytes(extracted.read())


def _run_jackhmmer(
    sequence: str, destination: Path, database: Path | None, iterations: int,
    bitscore_per_residue: float,
) -> str:
    if database is None:
        raise MSAError(
            "the jackhmmer backend needs a sequence database: pass --msa-database "
            "with a path to, for example, a UniRef90 FASTA (about 60 GB)."
        )
    if shutil.which("jackhmmer") is None:
        raise MSAError("jackhmmer is not on PATH; see envs/README.md")

    query = destination.with_suffix(".query.fasta")
    query.write_text(f">query\n{sequence}\n")
    stockholm = destination.with_suffix(".sto")
    threshold = bitscore_per_residue * len(sequence)
    command = [
        "jackhmmer", "-N", str(iterations), "--incT", f"{threshold:.1f}",
        "-T", f"{threshold:.1f}", "--cpu", "4", "-A", str(stockholm),
        str(query), str(database),
    ]
    log.info("running %s", " ".join(command))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    records = read_stockholm(stockholm)
    with destination.open("w") as handle:
        handle.write(f">query\n{sequence}\n")
        for name, seq in records:
            handle.write(f">{name}\n{seq}\n")
    query.unlink(missing_ok=True)
    return f"jackhmmer {iterations} iterations vs {database.name} (bitscore >= {threshold:.0f})"


def _run_hhblits(sequence: str, destination: Path, database: Path | None) -> str:
    if database is None:
        raise MSAError(
            "the hhblits backend needs --msa-database with a UniRef30 prefix "
            "(about 90 GB)."
        )
    if shutil.which("hhblits") is None:
        raise MSAError("hhblits is not on PATH; see envs/README.md")
    query = destination.with_suffix(".query.fasta")
    query.write_text(f">query\n{sequence}\n")
    command = [
        "hhblits", "-i", str(query), "-d", str(database), "-oa3m", str(destination),
        "-n", "3", "-cpu", "4", "-maxfilt", "100000", "-diff", "inf",
    ]
    log.info("running %s", " ".join(command))
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    query.unlink(missing_ok=True)
    return f"hhblits 3 iterations vs {database.name}"
