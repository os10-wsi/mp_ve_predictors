#!/usr/bin/env python3
"""DeepSequence scoring.

DeepSequence was published against Python 2.7 and Theano, which no longer
coexist with anything else here, so this delegates to the container built from
envs/deepsequence/Dockerfile. The mutation effect is the ELBO of the mutant
minus the ELBO of the wild type, so higher is more tolerated.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repo_driver import write_variant_list  # noqa: E402
from _runner_common import fail, parse_job, read_variants, write_meta, write_scores  # noqa: E402

IMAGE = "vepbench-deepsequence"


def main() -> int:
    job = parse_job()
    options = job.get("options", {})
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])

    if not job.get("msa_path"):
        fail("DeepSequence needs an MSA")
    if shutil.which("docker") is None:
        fail(
            "DeepSequence runs through a container here and docker is not on PATH. "
            f"Build it with: docker build -t {IMAGE} envs/deepsequence"
        )

    shared = work_dir / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copy(job["msa_path"], shared / "alignment.a2m")
    write_variant_list(shared / "variants.csv", variants, style="csv")

    command = [
        "docker", "run", "--rm", "-v", f"{shared}:/data", IMAGE,
        "--alignment", "/data/alignment.a2m", "--mutations", "/data/variants.csv",
        "--output", "/data/scores.csv", "--samples", str(options.get("samples", 2000)),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        fail(
            f"the DeepSequence container failed:\n{completed.stderr[-2000:]}\n"
            f"Build it with: docker build -t {IMAGE} envs/deepsequence"
        )

    table = shared / "scores.csv"
    if not table.exists():
        fail(f"the DeepSequence container wrote no scores to {table}")

    lookup: dict[str, float] = {}
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("mutant")
            value = row.get("elbo_difference") or row.get("score")
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
        "model_version": f"DeepSequence VAE ({options.get('samples', 2000)} ELBO samples)",
        "scoring": "ELBO(mutant) - ELBO(wild type); higher is more tolerated",
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
