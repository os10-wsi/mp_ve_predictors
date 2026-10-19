#!/usr/bin/env python3
"""VESPA and VESPA-light scoring.

VESPA outputs a probability in [0,1] that a substitution has a functional
effect, so higher is more deleterious. It is supervised, unlike most of the
other sequence-only methods here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repo_driver import run_command
from _runner_common import (
    describe_hardware,
    fail,
    parse_job,
    pick_device,
    read_variants,
    write_meta,
    write_scores,
)


def main() -> int:
    job = parse_job()
    options = job.get("options", {})
    sequence = job["sequence"]
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])
    device = pick_device(options)
    light = bool(options.get("light"))

    fasta = work_dir / "query.fasta"
    fasta.write_text(f">{job['uniprot_id']}\n{sequence}\n")
    mutations = work_dir / "mutations.txt"
    mutations.write_text("\n".join(f"{w}{p}{m}" for p, w, m in variants if m != w) + "\n")
    output_dir = work_dir / "vespa_out"

    # -s rather than -I: isolated mode implies -P, which keeps the script's
    # own directory off sys.path, so an upstream entry point cannot import its
    # siblings (ProteinMPNN's protein_mpnn_utils, EVE's model code). -s still
    # ignores user site-packages, which is the isolation that matters here.
    command = [sys.executable, "-s", "-m", "vespa.predict", str(fasta),
               "--output", str(output_dir)]
    if light:
        command.append("--vespal")
    run_command(command, cwd=work_dir, label="VESPA")

    produced = sorted(output_dir.rglob("*.csv"))
    if not produced:
        fail(f"VESPA wrote no CSV output under {output_dir}")

    column = "VESPAl" if light else "VESPA"
    lookup: dict[str, float] = {}
    for table in produced:
        text = table.read_text().splitlines()
        if not text:
            continue
        for separator in (";", ","):
            header = [h.strip() for h in text[0].split(separator)]
            if column in header:
                score_index = header.index(column)
                for line in text[1:]:
                    fields = line.split(separator)
                    if len(fields) <= score_index:
                        continue
                    try:
                        lookup[fields[0].strip().upper()] = float(fields[score_index])
                    except ValueError:
                        continue
                break
    if not lookup:
        fail(f"could not find a {column!r} column in the VESPA output under {output_dir}")

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
        "model_version": "VESPA-light" if light else "VESPA",
        "scoring": "probability of a functional effect in [0,1]; higher is more deleterious",
        "scored_variants": sum(1 for r in rows if r["score"] != ""),
        **describe_hardware(device),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
