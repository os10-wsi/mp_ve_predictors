#!/usr/bin/env python3
"""RaSP saturation-scan driver.\n\nRaSP predicts Rosetta cartesian_ddg values from a structure, so it inherits\nthe Rosetta convention where positive ddG is destabilising, and Rosetta's\nsoluble-protein training bias. It scores the whole protein in one pass, which\nis what makes it practical for saturation scans."""

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
        fail("RaSP needs a structure")

    repo = repo_path(job, "RaSP", "RASP_DIR", "https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg", "scripts/install_rasp.sh")
    entry = repo / "src/run_pipeline.py"
    if not entry.exists():
        candidates = sorted(repo.rglob("*run_pipeline*.py"))
        if not candidates:
            fail(
                f"could not find an entry point in the RaSP checkout at {repo}. "
                f"The upstream layout changes between releases; check scripts/install_rasp.sh"
            )
        entry = candidates[0]

    variant_file = write_variant_list(work_dir / "variants.csv", variants, style="csv")
    output = work_dir / "rasp_scores.csv"
    command = [sys.executable, "-I", str(entry)] + ["--pdb", job["structure_path"], "--chain", job.get("structure_chain") or "A", "--output", str(output)]
    run_command(command, cwd=repo, label="RaSP run_pipeline")

    lookup = read_score_table(output, ["variant", "mutant", "mutation"], ["score_ml", "ddG_pred", "score", "prediction"])
    rows = rows_from_lookup(variants, lookup)

    write_scores(job, rows)
    write_meta(job, {
        "model_version": "RaSP cavity model + ddG head",
        "scoring": "predicted Rosetta cartesian_ddg in kcal/mol; positive is destabilising",
        "upstream_command": " ".join(command),
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
