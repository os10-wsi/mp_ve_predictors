#!/usr/bin/env python3
"""ProteinMPNN inverse-folding scoring.

Drives the upstream ``protein_mpnn_run.py`` with ``--conditional_probs_only``,
which returns p(residue_i | the rest of the sequence, backbone) averaged over
random decoding orders. That is the inverse-folding analogue of a masked
marginal, and the score is ``log p(mut) - log p(wt)``.

Running the upstream script rather than reimplementing the model keeps the
published behaviour, including its decoding-order averaging.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import fail, parse_job, read_variants, write_meta, write_scores  # noqa: E402
from _third_party import repo_path  # noqa: E402

# ProteinMPNN's output alphabet, in the order the model emits.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWXY"


def main() -> int:
    import numpy as np

    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])

    if not job.get("structure_path"):
        fail("ProteinMPNN needs a structure")

    repo = repo_path(job, "ProteinMPNN", "PROTEINMPNN_DIR",
                     "https://github.com/dauparas/ProteinMPNN",
                     "scripts/install_proteinmpnn.sh")
    script = repo / "protein_mpnn_run.py"
    if not script.exists():
        fail(f"expected {script} in the ProteinMPNN checkout")

    output_dir = work_dir / "mpnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-I", str(script),
        "--pdb_path", job["structure_path"],
        "--pdb_path_chains", job.get("structure_chain") or "A",
        "--out_folder", str(output_dir),
        "--conditional_probs_only", "1",
        "--num_seq_per_target", str(int(options.get("num_samples", 10))),
        "--batch_size", "1",
        "--seed", str(int(options.get("seed", 37))),
        "--model_name", str(options.get("model_name", "v_48_020")),
    ]
    if options.get("use_soluble_weights"):
        command.append("--use_soluble_model")
    if options.get("device") == "cpu":
        command.append("--device")
        command.append("cpu")

    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=str(repo), check=False
    )
    if completed.returncode != 0:
        fail(f"protein_mpnn_run.py failed:\n{completed.stderr[-2000:]}")

    produced = list((output_dir / "conditional_probs_only").glob("*.npz"))
    if not produced:
        fail(f"ProteinMPNN produced no conditional_probs_only output in {output_dir}")

    data = np.load(produced[0])
    log_p = data["log_p"]
    if log_p.ndim == 3:
        log_p = log_p.mean(axis=0)          # average over decoding orders
    structure_sequence = (
        "".join(MPNN_ALPHABET[i] for i in data["S"].reshape(-1)[: log_p.shape[0]])
        if "S" in data else ""
    )
    offset = sequence.find(structure_sequence) if structure_sequence else 0
    if offset == -1:
        offset = 0

    index_of = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}
    rows = []
    for position, wt, mut in variants:
        index = position - 1 - offset
        if index < 0 or index >= log_p.shape[0]:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
            continue
        column = log_p[index]
        score = 0.0 if mut == wt else float(column[index_of[mut]] - column[index_of[wt]])
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": score})

    scored = sum(1 for r in rows if r["score"] != "")
    write_scores(job, rows)
    write_meta(job, {
        "model_version": f"ProteinMPNN {options.get('model_name', 'v_48_020')}"
                         + (" soluble" if options.get("use_soluble_weights") else ""),
        "scoring": "conditional probabilities, log p(mut) - log p(wt)",
        "structure_offset": offset, "scored_variants": scored,
        "unscored_variants": len(rows) - scored, "command": " ".join(command),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
