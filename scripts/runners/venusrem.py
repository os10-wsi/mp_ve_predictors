#!/usr/bin/env python3
"""VenusREM retrieval-enhanced scoring.\n\nTan et al., Bioinformatics 2025;41(Suppl 1):i401-i409,\ngithub.com/ai4protein/VenusREM. Combines a structure-aware language model\nwith alignment-derived residue preferences, so it needs sequence, structure\nand an alignment."""

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
        fail("VenusREM needs a structure")
    if not job.get("msa_path"):
        fail("VenusREM needs an MSA")

    repo = repo_path(job, "VenusREM", "VENUSREM_DIR", "https://github.com/ai4protein/VenusREM", "scripts/install_venusrem.sh")
    entry = repo / "src/score.py"
    if not entry.exists():
        candidates = sorted(repo.rglob("*score*.py"))
        if not candidates:
            fail(
                f"could not find an entry point in the VenusREM checkout at {repo}. "
                f"The upstream layout changes between releases; check scripts/install_venusrem.sh"
            )
        entry = candidates[0]

    variant_file = write_variant_list(work_dir / "variants.csv", variants, style="csv")
    output = work_dir / "venusrem_scores.csv"
    # -s rather than -I: isolated mode implies -P, which keeps the script's
    # own directory off sys.path, so an upstream entry point cannot import its
    # siblings (ProteinMPNN's protein_mpnn_utils, EVE's model code). -s still
    # ignores user site-packages, which is the isolation that matters here.
    command = [sys.executable, "-s", str(entry)] + ["--pdb", job["structure_path"], "--msa", job["msa_path"], "--mutant", str(variant_file), "--output", str(output)]
    run_command(command, cwd=repo, label="VenusREM score")

    lookup = read_score_table(output, ["mutant", "variant", "mutation"], ["score", "prediction"])
    rows = rows_from_lookup(variants, lookup)

    write_scores(job, rows)
    write_meta(job, {
        "model_version": "VenusREM",
        "scoring": "retrieval-enhanced log-likelihood ratio; higher is more tolerated",
        "upstream_command": " ".join(command),
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
