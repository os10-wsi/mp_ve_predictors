#!/usr/bin/env python3
"""EVmutation scoring through EVcouplings.

Fits a pairwise undirected graphical model to the alignment with plmc, then
scores each substitution as the change in statistical energy,
E(mutant) - E(wild type). More negative is more deleterious, the convention
in Hopf et al. 2017.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import fail, parse_job, read_variants, write_meta, write_scores  # noqa: E402


def main() -> int:
    job = parse_job()
    options = job.get("options", {})
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])

    if not job.get("msa_path"):
        fail("EVmutation needs an MSA")

    try:
        from evcouplings.couplings import CouplingsModel
        from evcouplings.couplings.pipeline import infer_plmc
    except ImportError as exc:
        fail(
            f"evcouplings is not importable ({exc}). Install it with "
            f"python -m vepbench install-env evcouplings"
        )

    model_file = work_dir / "couplings.model"
    if not model_file.exists():
        outcome = infer_plmc(
            prefix=str(work_dir / "plmc"), alignment_file=job["msa_path"],
            focus_mode=True, focus_sequence="query",
            theta=float(options.get("theta", 0.8)),
            lambda_h=float(options.get("lambda_h", 0.01)),
            lambda_J=float(options.get("lambda_J", 0.01)),
            lambda_J_times_Lq=True, lambda_group=None, scale_clusters=None,
            cpu=int(options.get("cpu", 4)),
            iterations=int(options.get("iterations", 100)),
            alphabet="-ACDEFGHIKLMNPQRSTVWY", ignore_gaps=False, reuse_ecs=False,
        )
        Path(outcome["model_file"]).replace(model_file)

    model = CouplingsModel(str(model_file))

    rows, unscored = [], 0
    for position, wt, mut in variants:
        if mut == wt:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": 0.0})
            continue
        try:
            # delta_hamiltonian returns (full, couplings, fields)
            value = float(model.delta_hamiltonian([(position, wt, mut)])[0])
        except Exception:  # noqa: BLE001 - positions outside the focus region
            value = ""
            unscored += 1
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": value})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": f"EVmutation/plmc (theta {options.get('theta', 0.8)})",
        "scoring": "epistatic statistical energy difference E(mutant) - E(wild type)",
        "model_file": str(model_file), "unscored_variants": unscored,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
