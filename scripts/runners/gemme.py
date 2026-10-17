#!/usr/bin/env python3
"""GEMME scoring.

GEMME needs an alignment and a Java runtime, since it drives JET2 internally.
Its predicted effects are negative numbers, with more negative meaning more
deleterious.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repo_driver import run_command  # noqa: E402
from _runner_common import fail, parse_job, read_variants, write_meta, write_scores  # noqa: E402
from _third_party import repo_path  # noqa: E402


def main() -> int:
    job = parse_job()
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])

    if not job.get("msa_path"):
        fail("GEMME needs an MSA")
    if shutil.which("java") is None:
        fail("GEMME needs a Java runtime on PATH (Java 8 or newer)")

    repo = repo_path(job, "GEMME", "GEMME_DIR", "https://github.com/elolaine/GEMME",
                     "scripts/install_gemme.sh")
    script = repo / "gemme.py"
    if not script.exists():
        fail(f"expected {script} in the GEMME checkout")

    alignment = work_dir / "input.fasta"
    shutil.copy(job["msa_path"], alignment)
    mutations = work_dir / "mutations.txt"
    mutations.write_text(
        "\n".join(f"{w}{p}{m}" for p, w, m in variants if m != w) + "\n"
    )

    run_command(
        [sys.executable, "-I", str(script), str(alignment), "-r", "input",
         "-f", str(alignment), "-m", str(mutations), "-o", "gemme_out"],
        cwd=work_dir, label="GEMME",
    )

    produced = sorted(work_dir.rglob("*normPred*.txt")) or sorted(work_dir.rglob("*_pred*.txt"))
    if not produced:
        fail(f"GEMME wrote no prediction file under {work_dir}")
    table = produced[0]

    lookup: dict[str, float] = {}
    for line in table.read_text().splitlines()[1:]:
        fields = line.replace('"', "").split()
        if len(fields) < 2:
            continue
        try:
            lookup[fields[0].upper()] = float(fields[1])
        except ValueError:
            continue

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
        "model_version": "GEMME with JET2",
        "scoring": "GEMME predicted effect; more negative is more deleterious",
        "upstream_table": str(table),
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
