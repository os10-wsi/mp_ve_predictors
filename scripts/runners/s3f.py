#!/usr/bin/env python3
"""S3F: multi-scale sequence, structure and surface fitness prediction.\n\nZhang et al., NeurIPS 2024 (arXiv:2412.01108),\ngithub.com/DeepGraphLearning/S3F. The S3F-MSA variant also consumes an\nalignment; select it with the use_msa option."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repo_driver import (
    read_score_table,
    rows_from_lookup,
    run_command,
    write_variant_list,
)
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

    if not job.get("structure_path"):
        fail("S3F needs a structure")

    repo = repo_path(job, "S3F", "S3F_DIR", "https://github.com/DeepGraphLearning/S3F", "scripts/install_s3f.sh")
    entry = repo / "script/predict.py"
    if not entry.exists():
        candidates = sorted(repo.rglob("*predict*.py"))
        if not candidates:
            fail(
                f"could not find an entry point in the S3F checkout at {repo}. "
                f"The upstream layout changes between releases; check scripts/install_s3f.sh"
            )
        entry = candidates[0]

    variant_file = write_variant_list(work_dir / "variants.csv", variants, style="csv")
    output = work_dir / "s3f_scores.csv"
    # -s rather than -I: isolated mode implies -P, which keeps the script's
    # own directory off sys.path, so an upstream entry point cannot import its
    # siblings (ProteinMPNN's protein_mpnn_utils, EVE's model code). -s still
    # ignores user site-packages, which is the isolation that matters here.
    command = [sys.executable, "-s", str(entry)] + ["--sequence", job["sequence"], "--structure", job["structure_path"], "--mutations", str(variant_file), "--output", str(output), "--model", "s3f_msa" if options.get("use_msa") else "s3f"] + (["--msa", job["msa_path"]] if options.get("use_msa") and job.get("msa_path") else [])
    run_command(command, cwd=repo, label="S3F predict")

    lookup = read_score_table(output, ["mutant", "variant", "mutation"], ["score", "prediction", "fitness"])
    rows = rows_from_lookup(variants, lookup)

    write_scores(job, rows)
    write_meta(job, {
        "model_version": "S3F",
        "scoring": "fitness log-likelihood ratio; higher is more tolerated",
        "upstream_command": " ".join(command),
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
