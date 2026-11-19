"""Comparing predictor behaviour across protein families.

Family members are overwhelmingly unreviewed, so there are no ClinVar
labels and no ROC curves for them. The question changes accordingly: not
how well a predictor separates known variants, but how its scores differ
between families, and whether predictors that agree in one family agree
in another.

Four things come out of it. The distribution of each predictor's damage
score within each family. The agreement between every pair of predictors,
computed per family, which is where a predictor that only looks sound
because it tracks the others gets found out. The split between
transmembrane and loop positions. And, where a family happens to contain
reviewed human proteins with ClinVar variants, an AUC for those -- almost
always too few to lean on, which is why it is reported with its count
beside it.

Scores are read from the ``__damaging`` columns, so every number points
the same way regardless of which direction the method reported.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vepbench.roc import MIN_PER_CLASS, _ranks, evaluate

log = logging.getLogger(__name__)

SUFFIX = "__damaging"

#: A family needs this many scored proteins before its distribution means
#: anything. Below it the numbers are about which proteins happened to run.
MIN_PROTEINS = 10


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, ties averaged."""
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    a, b = _ranks(x[keep]), _ranks(y[keep])
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def load_scored(
    accessions: list[str], membership: dict[str, str], *, outputs: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read every scored protein's table, tagged with its family.

    Only the columns needed for the comparison are kept. A family run is
    tens of millions of rows and holding the full tables would not fit.
    """
    frames: list[pd.DataFrame] = []
    missing = 0
    for accession in accessions:
        path = Path(outputs) / accession / "scores.tsv"
        if not path.exists():
            missing += 1
            continue
        frame = pd.read_csv(path, sep="\t")
        keep = [c for c in frame.columns
                if c.endswith(SUFFIX) or c in ("position", "in_transmembrane",
                                               "label", "uniprot_id")]
        if columns:
            keep = [c for c in keep
                    if not c.endswith(SUFFIX) or c[: -len(SUFFIX)] in columns]
        frame = frame.loc[:, keep]
        frame["family"] = membership.get(accession, "")
        frames.append(frame)
    if missing:
        log.warning("%d of %d accessions had no scores table", missing,
                    len(accessions))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def predictors_in(frame: pd.DataFrame) -> list[str]:
    return sorted(c[: -len(SUFFIX)] for c in frame.columns if c.endswith(SUFFIX))


def distributions(frame: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Each predictor's score distribution within each family."""
    rows = []
    for (family, name), values in _iter_family_scores(frame, predictors):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        rows.append({
            "family": family, "predictor": name, "n_variants": int(finite.size),
            "median": float(np.median(finite)), "mean": float(finite.mean()),
            "q1": float(np.quantile(finite, 0.25)),
            "q3": float(np.quantile(finite, 0.75)),
            "sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def _iter_family_scores(frame: pd.DataFrame, predictors: list[str]):
    for family, group in frame.groupby("family"):
        for name in predictors:
            column = f"{name}{SUFFIX}"
            if column in group.columns:
                yield (family, name), pd.to_numeric(
                    group[column], errors="coerce").to_numpy(dtype=float)


def agreement(frame: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Spearman between every pair of predictors, within each family."""
    rows = []
    for family, group in frame.groupby("family"):
        available = [n for n in predictors if f"{n}{SUFFIX}" in group.columns]
        values = {n: pd.to_numeric(group[f"{n}{SUFFIX}"],
                                   errors="coerce").to_numpy(dtype=float)
                  for n in available}
        for index, first in enumerate(available):
            for second in available[index + 1:]:
                rows.append({
                    "family": family, "predictor_a": first, "predictor_b": second,
                    "spearman": spearman(values[first], values[second]),
                    "n_variants": int(np.isfinite(values[first])
                                      .__and__(np.isfinite(values[second])).sum()),
                })
    return pd.DataFrame(rows)


def topology_split(frame: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Scores inside transmembrane segments against the rest.

    Transmembrane annotations on unreviewed entries are automatic and
    sometimes absent entirely, so the count of annotated proteins is
    reported beside the numbers rather than assumed.
    """
    if "in_transmembrane" not in frame.columns:
        return pd.DataFrame()
    inside = frame["in_transmembrane"].astype("boolean")
    rows = []
    for family, group in frame.groupby("family"):
        mask = inside.loc[group.index]
        annotated = int(mask.notna().sum())
        for name in predictors:
            column = f"{name}{SUFFIX}"
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            tm = values[mask.fillna(False)].dropna()
            loop = values[(~mask.fillna(True))].dropna()
            if tm.empty or loop.empty:
                continue
            rows.append({
                "family": family, "predictor": name,
                "n_annotated_rows": annotated,
                "n_tm": int(tm.size), "n_loop": int(loop.size),
                "median_tm": float(tm.median()), "median_loop": float(loop.median()),
                "difference": float(tm.median() - loop.median()),
            })
    return pd.DataFrame(rows)


def labelled_auc(
    frame: pd.DataFrame, predictors: list[str], *, min_per_class: int = MIN_PER_CLASS
) -> pd.DataFrame:
    """AUC per family over whatever ClinVar-labelled members it contains.

    Pooled across the family's proteins, because no single member has
    enough labelled variants. The count of contributing proteins is part
    of the output: a family resting on two proteins is not evidence.
    """
    if "label" not in frame.columns:
        return pd.DataFrame()
    rows = []
    for family, group in frame.groupby("family"):
        labelled = group.loc[group["label"].notna()]
        if labelled.empty:
            continue
        oriented = pd.DataFrame({"label": labelled["label"]})
        present = []
        for name in predictors:
            column = f"{name}{SUFFIX}"
            if column in labelled.columns:
                oriented[name] = labelled[column].to_numpy()
                present.append(name)
        for result in evaluate(oriented, present, uniprot_id=str(family),
                               min_per_class=min_per_class):
            row = result.as_row()
            row["family"] = row.pop("uniprot_id")
            row["n_proteins"] = int(labelled["uniprot_id"].nunique())
            rows.append(row)
    return pd.DataFrame(rows)


def write_all(
    frame: pd.DataFrame, predictors: list[str], directory: Path
) -> dict[str, Path]:
    """Write the four tables. Returns the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "family_distributions": distributions(frame, predictors),
        "family_agreement": agreement(frame, predictors),
        "family_topology": topology_split(frame, predictors),
        "family_auc": labelled_auc(frame, predictors),
    }
    written: dict[str, Path] = {}
    for stem, table in tables.items():
        if table.empty:
            log.info("%s is empty and was not written", stem)
            continue
        path = directory / f"{stem}.tsv"
        tmp = path.with_suffix(".tsv.tmp")
        table.to_csv(tmp, sep="\t", index=False)
        tmp.replace(path)
        written[stem] = path
    return written
