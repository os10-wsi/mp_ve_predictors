"""Did the run finish, and if not, what stopped it.

A scoring run over tens of thousands of proteins fails in two different
ways, and they need opposite responses. A protein with no table at all
means its task never produced anything -- it died, was killed, or never
started. A protein whose table is missing one score means that score
failed while the rest succeeded. Reporting a single "complete" number
hides the difference, so this reports both.

Scores that were never going to run are not failures. AlphaMissense has
no yeast entries and FoldX is scored only at ClinVar positions, so on a
non-human protein both are correctly absent. Those are counted as
unavailable, separately from the ones that broke.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from vepbench.config import PATHS, Paths

log = logging.getLogger(__name__)

#: Noise that differs per protein and would split one cause into many.
_VARIABLE = re.compile(r"/\S+|\b[A-NR-Z][0-9][A-Z0-9]{3}[0-9]\b|\d+")


def normalise(message: str) -> str:
    """Reduce a failure message to its cause, so identical ones group.

    Paths, accessions and byte counts differ on every protein. Leaving
    them in turns one recurring failure into thousands of unique strings
    and buries the thing you need to read.
    """
    text = message.rsplit(" | ", maxsplit=1)[-1].strip()
    return _VARIABLE.sub("#", text)[:110] or "(no message recorded)"


@dataclass
class Report:
    """What the run produced, against what was asked for."""

    expected: list[str]
    scored: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    status_counts: dict[str, Counter] = field(default_factory=dict)
    reasons: dict[str, Counter] = field(default_factory=dict)
    variants: int = 0

    @property
    def complete(self) -> bool:
        """Whether every protein ran and nothing failed."""
        return not self.missing and not self.unreadable and not any(
            counts.get("failed", 0) for counts in self.status_counts.values())

    def lines(self) -> list[str]:
        total = len(self.expected)
        done = len(self.scored)
        share = 100 * done / total if total else 0.0
        out = [f"{done:,} of {total:,} proteins have a scores table ({share:.1f}%)"]
        if self.variants:
            out.append(f"{self.variants:,} variant rows written")
        if self.missing:
            out.append(f"{len(self.missing):,} have no table at all")
        if self.unreadable:
            out.append(f"{len(self.unreadable):,} have an unreadable status.json")

        if self.status_counts:
            states = ["ok", "cached", "unavailable", "failed"]
            width = max(len(n) for n in self.status_counts)
            out.append("")
            out.append(f"{'score':<{width}}  " + "".join(f"{s:>13}" for s in states))
            out.append("-" * (width + 2 + 13 * len(states)))
            for name in sorted(self.status_counts,
                               key=lambda n: -self.status_counts[n].get("failed", 0)):
                counts = self.status_counts[name]
                row = "".join(f"{counts.get(s, 0):>13,}" for s in states)
                out.append(f"{name:<{width}}  {row}")

        failing = {n: c for n, c in self.status_counts.items() if c.get("failed")}
        if failing:
            out.append("")
            out.append("failures:")
            width = max(len(n) for n in failing)
            for name, counts in sorted(failing.items(),
                                       key=lambda kv: -kv[1]["failed"]):
                attempted = counts.get("failed", 0) + counts.get("ok", 0) \
                    + counts.get("cached", 0)
                failed = counts["failed"]
                verdict = ("every attempt" if failed == attempted else "intermittent")
                out.append(f"  {name:<{width}}  {failed:,} of {attempted:,} "
                           f"({100 * failed / attempted:.1f}%)  {verdict}")
                for message, seen in self.reasons.get(name, Counter()).most_common(2):
                    out.append(f"  {'':<{width}}  {seen:>7,}  {message}")

        out.append("")
        out.append("run is complete" if self.complete
                   else "run is NOT complete -- see above")
        return out


def check(accessions: list[str], *, paths: Paths = PATHS) -> Report:
    """Read every protein's status and summarise the run."""
    report = Report(expected=list(accessions))
    for accession in accessions:
        directory = paths.output_dir(accession)
        if not (directory / "scores.tsv").exists():
            report.missing.append(accession)
            continue
        report.scored.append(accession)

        status_path = directory / "status.json"
        if not status_path.exists():
            report.unreadable.append(accession)
            continue
        try:
            status = json.loads(status_path.read_text())
        except ValueError:
            report.unreadable.append(accession)
            continue

        report.variants += int(status.get("n_variants") or 0)
        for entry in status.get("scores", []):
            name = entry.get("score", "?")
            state = entry.get("status", "?")
            report.status_counts.setdefault(name, Counter())[state] += 1
            if state == "failed":
                report.reasons.setdefault(name, Counter())[
                    normalise(entry.get("message") or "")] += 1
    return report


def write_missing(report: Report, path: Path) -> Path | None:
    """Write the accessions with no table, ready to resubmit."""
    if not report.missing:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report.missing) + "\n")
    return path
