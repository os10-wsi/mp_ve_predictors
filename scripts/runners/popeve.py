#!/usr/bin/env python3
"""popEVE scoring.

popEVE combines EVE with ESM-1v through a Gaussian process calibrated across
the human proteome. Proteome-wide precomputed scores exist and are both faster
and closer to what the authors published, so they are the default. Scores are
calibrated so that more negative is more damaging.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import fail, parse_job, read_variants, write_meta, write_scores

POPEVE_SITE = "https://pop.evemodel.org/"


def load_precomputed(job: dict) -> dict[str, float] | None:
    """Look for a downloaded popEVE table for this protein."""
    directories = []
    if os.environ.get("POPEVE_DIR"):
        directories.append(Path(os.environ["POPEVE_DIR"]))
    directories.append(Path(job["weights_dir"]) / "popeve")

    for directory in directories:
        if not directory.exists():
            continue
        for pattern in (f"{job['uniprot_id']}*.csv", f"*{job['uniprot_id']}*.csv"):
            for path in sorted(directory.glob(pattern)):
                lookup: dict[str, float] = {}
                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        key = row.get("mutant") or row.get("mutation") or row.get("variant")
                        value = (row.get("popEVE") or row.get("popeve")
                                 or row.get("score") or row.get("popEVE_score"))
                        if key and value:
                            try:
                                lookup[key.strip().upper()] = float(value)
                            except ValueError:
                                continue
                if lookup:
                    return lookup
    return None


def main() -> int:
    job = parse_job()
    options = job.get("options", {})
    variants = read_variants(job)

    if not options.get("use_precomputed", True):
        fail(
            "running popEVE locally is not wired up here. It needs EVE scores and "
            "ESM-1v scores for this protein, then the Gaussian process fit from "
            "github.com/debbiemarkslab/popEVE. Run eve_full and esm1v_full first, then "
            "follow that repository. Precomputed scores are the supported route."
        )

    lookup = load_precomputed(job)
    if lookup is None:
        fail(
            f"no precomputed popEVE table found for {job['uniprot_id']}. Download the "
            f"proteome-wide scores from {POPEVE_SITE} into "
            f"{Path(job['weights_dir']) / 'popeve'}, or set POPEVE_DIR."
        )

    rows = []
    for position, wt, mut in variants:
        if mut == wt:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": 0.0})
            continue
        value = lookup.get(f"{wt}{position}{mut}".upper())
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                     "score": "" if value is None else value})

    write_scores(job, rows)
    write_meta(job, {
        "model_version": "popEVE (precomputed proteome-wide scores)",
        "scoring": "popEVE score; more negative is more damaging",
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
