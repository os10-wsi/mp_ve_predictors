#!/usr/bin/env python3
"""FoldX BuildModel driver.

FoldX is licensed software and its binary is never committed here; point
``FOLDX_BINARY`` at your own copy.

The procedure is the one FoldX documents. RepairPDB runs once per structure
and is cached, since it is slow and deterministic. BuildModel then runs per
variant against the repaired structure, and the total energy difference is
read from ``Dif_*.fxout``.

FoldX reports ddG = G(mutant) - G(wild type) in kcal/mol, so positive is
destabilising. This is the slow path: a saturation scan is 19L BuildModel
runs, and results are cached per variant so an interrupted run resumes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _runner_common import fail, parse_job, read_variants, write_meta, write_scores


def run_foldx(binary: str, arguments: list[str], cwd: Path):
    return subprocess.run([binary, *arguments], capture_output=True, text=True,
                          cwd=str(cwd), check=False)


def repair(binary: str, structure: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repaired = cache_dir / f"{structure.stem}_Repair.pdb"
    if repaired.exists():
        return repaired
    shutil.copy(structure, cache_dir / structure.name)
    completed = run_foldx(binary, ["--command=RepairPDB", f"--pdb={structure.name}"],
                          cache_dir)
    if completed.returncode != 0 or not repaired.exists():
        fail(f"FoldX RepairPDB failed:\n{completed.stdout[-1500:]}\n{completed.stderr[-1500:]}")
    return repaired


def build_model(binary: str, repaired: Path, mutation: str, work_dir: Path) -> float:
    work_dir.mkdir(parents=True, exist_ok=True)
    local_pdb = work_dir / repaired.name
    if not local_pdb.exists():
        shutil.copy(repaired, local_pdb)
    mutant_file = work_dir / "individual_list.txt"
    mutant_file.write_text(f"{mutation};\n")

    completed = run_foldx(binary, [
        "--command=BuildModel", f"--pdb={repaired.name}",
        f"--mutant-file={mutant_file.name}", "--numberOfRuns=1",
    ], work_dir)
    difference = work_dir / f"Dif_{repaired.stem}.fxout"
    if completed.returncode != 0 or not difference.exists():
        raise RuntimeError(
            f"FoldX BuildModel failed for {mutation}: "
            f"{completed.stdout[-600:]} {completed.stderr[-600:]}"
        )
    for line in difference.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) > 1 and fields[0].endswith(".pdb"):
            return float(fields[1])
    raise RuntimeError(f"could not parse a total energy from {difference}")


def main() -> int:
    job = parse_job()
    options = job.get("options", {})
    variants = read_variants(job)
    work_dir = Path(job["work_dir"])

    binary = os.environ.get("FOLDX_BINARY")
    if not binary or not Path(binary).exists():
        fail(
            "set FOLDX_BINARY to your licensed foldx executable. FoldX is not "
            "redistributable, so this repository cannot install it for you. Academic "
            "licences: https://foldxsuite.crg.eu/"
        )
    if not job.get("structure_path"):
        fail("FoldX needs a structure")

    rotabase = os.environ.get("FOLDX_ROTABASE")
    if rotabase and Path(rotabase).exists():
        shutil.copy(rotabase, work_dir / "rotabase.txt")

    structure = Path(job["structure_path"])
    chain = job.get("structure_chain") or "A"
    cache_dir = Path(job["weights_dir"]) / "foldx_repaired"
    repaired = repair(binary, structure, cache_dir) if options.get("repair", True) else structure

    cache_path = work_dir / "foldx_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    rows, failures = [], 0
    for index, (position, wt, mut) in enumerate(variants, start=1):
        if mut == wt:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": 0.0})
            continue
        mutation = f"{wt}{chain}{position}{mut}"
        if mutation in cache:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                         "score": cache[mutation]})
            continue
        try:
            value = build_model(binary, repaired, mutation, work_dir / "runs" / mutation)
        except RuntimeError as exc:
            print(f"  {mutation}: {exc}", file=sys.stderr)
            failures += 1
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": ""})
            continue
        cache[mutation] = value
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": value})
        if index % 25 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"  FoldX: {index}/{len(variants)} variants", file=sys.stderr)

    cache_path.write_text(json.dumps(cache))
    version = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             check=False)
    write_scores(job, rows)
    write_meta(job, {
        "model_version": (version.stdout.strip().splitlines() or ["FoldX"])[0][:120],
        "scoring": "BuildModel total energy difference, ddG = G(mutant) - G(wild type)",
        "repaired_structure": str(repaired), "failed_variants": failures,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
