#!/usr/bin/env python3
"""EVE: train a protein-family VAE on the alignment, then score variants.

EVE is trained per protein, not per variant, so the cost is GPU hours for
training and then a cheap scoring pass. The trained model is cached under the
weights directory and reused.

The evolutionary index is the negative of the mutant-to-wild-type
log-likelihood ratio, so a higher index means more pathogenic.
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
    identifier = job["uniprot_id"]

    if not job.get("msa_path"):
        fail("EVE needs an MSA")

    offset = 0
    if options.get("use_domain"):
        domain = job.get("domain")
        if not domain:
            fail("eve_domain was requested but no domain window was resolved")
        offset = domain[0] - 1
        variants = [(p, w, m) for p, w, m in variants if domain[0] <= p <= domain[1]]
        if not variants:
            fail("no requested variants fall inside the chosen domain window")
        identifier = f"{identifier}_{domain[0]}-{domain[1]}"

    repo = repo_path(job, "EVE", "EVE_DIR", "https://github.com/OATML-Markslab/EVE",
                     "scripts/install_eve.sh")

    alignment_dir = work_dir / "alignments"
    alignment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(job["msa_path"], alignment_dir / f"{identifier}.a2m")
    mapping = work_dir / "mapping.csv"
    mapping.write_text(
        "protein_name,msa_location,theta\n"
        f"{identifier},{identifier}.a2m,{options.get('theta', 0.2)}\n"
    )

    model_dir = Path(job["weights_dir"]) / "eve" / identifier
    model_dir.mkdir(parents=True, exist_ok=True)
    if not list(model_dir.glob("*.pt")) or options.get("retrain"):
        run_command([
            sys.executable, "-I", str(repo / "train_VAE.py"),
            "--MSA_data_folder", str(alignment_dir), "--MSA_list", str(mapping),
            "--protein_index", "0", "--MSA_weights_location", str(model_dir / "weights"),
            "--VAE_checkpoint_location", str(model_dir),
            "--model_name_suffix", "vepbench",
            "--model_parameters_location", str(repo / "EVE" / "default_model_params.json"),
            "--training_logs_location", str(work_dir / "logs"),
        ], cwd=repo, label="EVE train_VAE.py")

    variant_file = write_variant_list(work_dir / "variants.csv", variants, style="csv",
                                      offset=offset)
    output_dir = work_dir / "eve_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command([
        sys.executable, "-I", str(repo / "compute_evol_indices.py"),
        "--MSA_data_folder", str(alignment_dir), "--MSA_list", str(mapping),
        "--protein_index", "0", "--MSA_weights_location", str(model_dir / "weights"),
        "--VAE_checkpoint_location", str(model_dir), "--model_name_suffix", "vepbench",
        "--model_parameters_location", str(repo / "EVE" / "default_model_params.json"),
        "--computation_mode", "input_mutations_list",
        "--all_singles_mutations_folder", str(variant_file.parent),
        "--output_evol_indices_location", str(output_dir),
        "--num_samples_compute_evol_indices", str(options.get("num_samples", 20000)),
    ], cwd=repo, label="EVE compute_evol_indices.py")

    produced = sorted(output_dir.rglob("*.csv"))
    if not produced:
        fail(f"EVE wrote no evolutionary indices under {output_dir}")

    lookup: dict[str, float] = {}
    with produced[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("mutations") or row.get("mutant")
            value = row.get("evol_indices") or row.get("evol_index")
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
        value = lookup.get(f"{wt}{position - offset}{mut}")
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                     "score": "" if value is None else value})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": f"EVE VAE ({identifier})",
        "scoring": "evolutionary index; higher is more pathogenic",
        "domain_window": job.get("domain") if options.get("use_domain") else None,
        "checkpoint_dir": str(model_dir),
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
