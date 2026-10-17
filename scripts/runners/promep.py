#!/usr/bin/env python3
"""ProMEP scoring.\n\nCheng et al., Cell Research 2024;34:630-647, github.com/wenjiegroup/ProMEP.\nAlignment-free multimodal model over sequence and structure. The upstream\nrepository ships inference_dms.py for exactly this kind of saturation scan."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repo_driver import read_score_table, rows_from_lookup, run_command, write_variant_list  # noqa: E402
from _runner_common import describe_hardware, fail, parse_job, pick_device, read_variants, write_meta, write_scores  # noqa: E402
from _third_party import repo_path  # noqa: E402


def main() -> int:
    job = parse_job()
    options = job.get("options", {})
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])
    device = pick_device(options)

    if not job.get("structure_path"):
        fail("ProMEP needs a structure")

    repo = repo_path(job, "ProMEP", "PROMEP_DIR", "https://github.com/wenjiegroup/ProMEP", "scripts/install_promep.sh")
    entry = repo / "inference_dms.py"
    if not entry.exists():
        candidates = sorted(repo.rglob("*inference*.py"))
        if not candidates:
            fail(
                f"could not find an entry point in the ProMEP checkout at {repo}. "
                f"The upstream layout changes between releases; check scripts/install_promep.sh"
            )
        entry = candidates[0]

    variant_file = write_variant_list(work_dir / "variants.csv", variants, style="csv")
    output = work_dir / "promep_scores.csv"
    command = [sys.executable, "-I", str(entry)] + ["--pdb_file", job["structure_path"], "--mutation_file", str(variant_file), "--output_file", str(output)]
    run_command(command, cwd=repo, label="ProMEP inference_dms")

    lookup = read_score_table(output, ["mutant", "mutation", "variant"], ["score", "promep_score", "prediction"])
    rows = rows_from_lookup(variants, lookup)

    write_scores(job, rows)
    write_meta(job, {
        "model_version": "ProMEP",
        "scoring": "multimodal log-likelihood ratio; higher is more tolerated",
        "upstream_command": " ".join(command),
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
