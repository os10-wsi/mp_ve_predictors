#!/usr/bin/env python3
"""SaProt-dG and ESM3-dG from the absolute-stability-predictor repository.\n\ngithub.com/yehlincho/absolute-stability-predictor fine-tunes SaProt 650M and\nESM3 with LoRA plus a stability head to predict absolute dG and mutational\nddG from a structure. Weights are on Hugging Face under\nYehlin/absolute-stability.\n\nThis is supervised regression, not a masked-marginal likelihood difference.\nThe likelihood readings of the same backbones are registered separately as\nsaprot_dg_masked_marginal and esm3_dg_masked_marginal, so the two are never\nconfused. The ddG sign convention is not documented in the sources used to\nbuild this wrapper, so the predictor entry marks it unverified."""

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
        fail("the dG stability models needs a structure")

    repo = repo_path(job, "absolute-stability-predictor", "STABILITY_DG_DIR", "https://github.com/yehlincho/absolute-stability-predictor", "scripts/install_stability_dg.sh")
    entry = repo / "predict.py"
    if not entry.exists():
        candidates = sorted(repo.rglob("*predict*.py"))
        if not candidates:
            fail(
                f"could not find an entry point in the absolute-stability-predictor checkout at {repo}. "
                f"The upstream layout changes between releases; check scripts/install_stability_dg.sh"
            )
        entry = candidates[0]

    variant_file = write_variant_list(work_dir / "variants.csv", variants, style="csv")
    output = work_dir / "stability_dg_scores.csv"
    # -s rather than -I: isolated mode implies -P, which keeps the script's
    # own directory off sys.path, so an upstream entry point cannot import its
    # siblings (ProteinMPNN's protein_mpnn_utils, EVE's model code). -s still
    # ignores user site-packages, which is the isolation that matters here.
    command = [sys.executable, "-s", str(entry)] + ["--pdb", job["structure_path"], "--chain", job.get("structure_chain") or "A", "--mutations", str(variant_file), "--model", options.get("backbone", "saprot"), "--ensemble", str(int(options.get("ensemble", 3))), "--output", str(output)]
    run_command(command, cwd=repo, label="absolute-stability-predictor")

    lookup = read_score_table(output, ["mutant", "mutation", "variant"], ["ddG", "predicted_ddG", "ddg", "score"])
    rows = rows_from_lookup(variants, lookup)

    write_scores(job, rows)
    write_meta(job, {
        "model_version": "absolute-stability-predictor",
        "scoring": "predicted ddG in kcal/mol; sign convention UNVERIFIED",
        "upstream_command": " ".join(command),
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
