#!/usr/bin/env python3
"""Tranception scoring, with inference-time retrieval from the alignment.

Retrieval is the configuration the paper benchmarks; without it the model is\nthe autoregressive transformer alone, which is weaker. Set retrieval to false\nto do that deliberately.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repo_driver import run_command, write_variant_list
from _runner_common import (
    describe_hardware,
    fail,
    parse_job,
    pick_device,
    read_variants,
    write_meta,
    write_scores,
)
from _third_party import repo_path


def main() -> int:
    job = parse_job()
    options = job.get("options", {})
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])
    device = pick_device(options)
    retrieval = bool(options.get("retrieval", True))

    if retrieval and not job.get("msa_path"):
        fail("Tranception needs an MSA")

    repo = repo_path(job, "Tranception", "TRANCEPTION_DIR",
                     "https://github.com/OATML-Markslab/Tranception",
                     "scripts/install_tranception.sh")
    checkpoint = Path(job["weights_dir"]) / "tranception" / options.get(
        "checkpoint", "Tranception_Large"
    )
    if not checkpoint.exists():
        fail(
            f"Tranception checkpoint not found at {checkpoint}. Download it with "
            f"bash scripts/download_weights_tranception.sh"
        )

    variant_file = write_variant_list(work_dir / "variants.csv", variants, style="csv")
    msa_dir = work_dir / "msa"
    if job.get("msa_path"):
        msa_dir.mkdir(exist_ok=True)
        shutil.copy(job["msa_path"], msa_dir / Path(job["msa_path"]).name)

    reference = work_dir / "reference.csv"
    reference.write_text(
        "DMS_id,DMS_filename,target_seq,seq_len,MSA_filename,MSA_start,MSA_end\n"
        f"{job['uniprot_id']},{variant_file.name},{job['sequence']},"
        f"{len(job['sequence'])},"
        f"{Path(job['msa_path']).name if job.get('msa_path') else ''},1,"
        f"{len(job['sequence'])}\n"
    )
    output_dir = work_dir / "tranception_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-I", str(repo / "score_tranception_proteingym.py"),
        "--checkpoint", str(checkpoint),
        "--DMS_reference_file_path", str(reference),
        "--DMS_data_folder", str(work_dir),
        "--DMS_index", "0",
        "--output_scores_folder", str(output_dir),
    ] + (["--inference_time_retrieval", "--MSA_folder", str(msa_dir)] if retrieval else []) + ["--batch_size_inference", str(options.get("batch_size", 10))]
    run_command(command, cwd=repo, label="Tranception")

    produced = sorted(output_dir.rglob("*.csv"))
    if not produced:
        fail(f"Tranception wrote no scores under {output_dir}")

    lookup: dict[str, float] = {}
    with produced[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("mutant")
            value = (row.get("avg_score") or row.get("Tranception_score")
                     or row.get("score"))
            if key and value:
                try:
                    lookup[key.strip()] = float(value)
                except ValueError:
                    continue

    rows = []
    for position, wt, mut in variants:
        if mut == wt:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": 0.0})
            continue
        value = lookup.get(f"{wt}{position}{mut}")
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                     "score": "" if value is None else value})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": "Tranception_Large",
        "scoring": "autoregressive log-likelihood ratio, fused with MSA retrieval when enabled",
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
