"""Shared plumbing for runners that drive a third-party research repository.

The pattern is always the same: write the inputs the upstream code expects,
run its documented entry point, then read its output table back. Keeping that
here means each runner states only what is specific to it, and the upstream
command ends up visible in metadata.json.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def write_variant_list(path: Path, variants, *, style: str = "short", offset: int = 0) -> Path:
    """Write the variant list in whichever notation the upstream tool expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(position - offset, wt, mut) for position, wt, mut in variants]
    if style == "short":
        path.write_text("\n".join(f"{w}{p}{m}" for p, w, m in rows) + "\n")
    elif style == "csv":
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["mutant"])
            for p, w, m in rows:
                writer.writerow([f"{w}{p}{m}"])
    elif style == "columns":
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["position", "wt", "mut"])
            writer.writerows(rows)
    else:
        raise ValueError(f"unknown variant list style {style!r}")
    return path


def run_command(command: list[str], cwd: Path, label: str, env: dict | None = None) -> str:
    """Run an upstream command, failing loudly with its own error text."""
    print(f"  running {label}: {' '.join(command)}", file=sys.stderr)
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=str(cwd), env=env, check=False
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{label} failed with exit code {completed.returncode}.\n"
            f"--- stdout ---\n{completed.stdout[-2000:]}\n"
            f"--- stderr ---\n{completed.stderr[-2000:]}"
        )
    return completed.stdout


def read_score_table(path: Path, variant_keys: list[str], score_keys: list[str]) -> dict:
    """Read an upstream score table, trying several plausible column names."""
    if not path.exists():
        raise SystemExit(f"expected an output table at {path} but it was not written")
    out: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header row")
        variant_column = next((c for c in variant_keys if c in reader.fieldnames), None)
        score_column = next((c for c in score_keys if c in reader.fieldnames), None)
        if variant_column is None or score_column is None:
            raise SystemExit(
                f"{path} has columns {reader.fieldnames}, but none matching "
                f"{variant_keys} and {score_keys}. The upstream output format may "
                f"have changed."
            )
        for row in reader:
            try:
                out[row[variant_column].strip()] = float(row[score_column])
            except (TypeError, ValueError):
                continue
    if not out:
        raise SystemExit(f"{path} contained no usable scores")
    return out


def rows_from_lookup(variants, lookup: dict, offset: int = 0) -> list[dict]:
    """Match upstream scores back onto the requested variants.

    A variant the upstream tool did not score is written empty rather than
    zero, so a missing value is never mistaken for a neutral one.
    """
    rows = []
    for position, wt, mut in variants:
        if mut == wt:
            rows.append({"position": position, "wt_aa": wt, "mut_aa": mut, "score": 0.0})
            continue
        value = lookup.get(f"{wt}{position - offset}{mut}")
        rows.append({"position": position, "wt_aa": wt, "mut_aa": mut,
                     "score": "" if value is None else value})
    return rows
