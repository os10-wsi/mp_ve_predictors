"""The output tables and the ROC panel PDF.

Three tables fall out of the ROC results. The wide one has a row per
protein and a column pair per predictor, which is the shape to scan across
a thousand proteins. The long one has a row per protein and predictor,
which is the shape to plot from. The summary collapses across proteins.

The PDF is one panel per protein, every predictor drawn in it. Two things
make that survivable at a thousand proteins. Curves are decimated before
drawing, because a panel two inches wide cannot show five hundred points
and keeping them would produce a file too large to open. And the legend is
drawn once per page rather than once per panel, which is both less ink and
less to read.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vepbench.roc import RocResult, long_table, summarise, wide_table

log = logging.getLogger(__name__)

#: Validated for deuteranopia, protanopia and tritanopia separation against
#: a near-white surface. Slots past the eighth repeat a hue and change the
#: line style instead, because a ninth generated hue would not be safe.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#2b2b28"
MUTED = "#767670"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

PANELS_PER_PAGE = 6
PANEL_COLUMNS = 3
#: Points kept per curve. A panel this size resolves nothing finer, and the
#: file size is linear in this number times proteins times predictors.
CURVE_POINTS = 120


def style_for(index: int) -> tuple[str, str]:
    """A colour and line style for the nth predictor, stable across panels."""
    colour = PALETTE[index % len(PALETTE)]
    dashes = ["-", "--", ":"][index // len(PALETTE)]
    return colour, dashes


def decimate(values: np.ndarray, keep: int = CURVE_POINTS) -> np.ndarray:
    """Evenly thin an array, always keeping the first and last point."""
    if len(values) <= keep:
        return values
    index = np.unique(np.linspace(0, len(values) - 1, keep).astype(int))
    return values[index]


def write_tables(
    results: list[RocResult], predictors: list[str], directory: Path
) -> dict[str, Path]:
    """Write the wide, long and summary tables. Returns the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    long = long_table(results)
    frames = {
        "auc_by_protein": wide_table(results, predictors),
        "auc_long": long,
        "auc_summary": summarise(long),
    }
    written: dict[str, Path] = {}
    for stem, frame in frames.items():
        path = directory / f"{stem}.tsv"
        tmp = path.with_suffix(".tsv.tmp")
        frame.to_csv(tmp, sep="\t", index=False)
        tmp.replace(path)
        written[stem] = path
    return written


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib is not installed; skipping the ROC plots "
                    "(pip install 'vepbench[plots]')")
        return None
    return plt


def roc_pdf(
    results: list[RocResult], predictors: list[str], path: Path, *,
    panels_per_page: int = PANELS_PER_PAGE,
) -> Path | None:
    """One panel per protein, every predictor's curve drawn in it."""
    plt = _pyplot()
    if plt is None or not results:
        return None
    from matplotlib.backends.backend_pdf import PdfPages

    by_protein: dict[str, list[RocResult]] = {}
    for result in results:
        by_protein.setdefault(result.uniprot_id, []).append(result)
    proteins = sorted(by_protein)

    order = {name: index for index, name in enumerate(predictors)}
    rows = int(np.ceil(panels_per_page / PANEL_COLUMNS))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        for start in range(0, len(proteins), panels_per_page):
            page = proteins[start : start + panels_per_page]
            figure, axes = plt.subplots(
                rows, PANEL_COLUMNS, figsize=(3.1 * PANEL_COLUMNS, 3.3 * rows),
                squeeze=False,
            )
            flat = [axes[r][c] for r in range(rows) for c in range(PANEL_COLUMNS)]
            seen: dict[str, tuple[str, str]] = {}

            for axis, accession in zip(flat, page, strict=False):
                entries = sorted(by_protein[accession],
                                 key=lambda r: order.get(r.predictor, 99))
                for entry in entries:
                    colour, dashes = style_for(order.get(entry.predictor, 0))
                    seen[entry.predictor] = (colour, dashes)
                    axis.plot(decimate(entry.fpr), decimate(entry.tpr),
                              color=colour, linestyle=dashes, linewidth=1.3)
                axis.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8,
                          linestyle="--", zorder=0)
                counts = entries[0] if entries else None
                subtitle = (f"  n={counts.n_pathogenic}P/{counts.n_benign}B"
                            if counts else "")
                axis.set_title(f"{accession}{subtitle}", fontsize=8.5,
                               color=INK, loc="left")
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                axis.set_xticks([0, 0.5, 1])
                axis.set_yticks([0, 0.5, 1])
                axis.tick_params(labelsize=7, colors=MUTED, length=2)
                axis.grid(True, color=GRID, linewidth=0.6)
                axis.set_axisbelow(True)
                axis.spines[["top", "right"]].set_visible(False)
                axis.spines[["left", "bottom"]].set_color(GRID)

            for axis in flat[len(page):]:
                axis.set_visible(False)

            # The legend sits in reserved space at the foot of the page, and
            # the shared axis label just above it. Letting tight_layout place
            # either lands them on top of each other.
            legend_band = 0.055 + 0.028 * np.ceil(len(seen) / 5)
            figure.patch.set_facecolor(SURFACE)
            figure.tight_layout(rect=(0.03, legend_band, 1, 1))
            figure.supxlabel("false positive rate", fontsize=9, color=INK,
                             y=legend_band - 0.018)
            figure.supylabel("true positive rate", fontsize=9, color=INK)
            if seen:
                handles = [
                    plt.Line2D([], [], color=c, linestyle=d, linewidth=2.4)
                    for c, d in (seen[n] for n in predictors if n in seen)
                ]
                labels = [n for n in predictors if n in seen]
                figure.legend(handles, labels, loc="lower center", frameon=False,
                              ncol=min(5, len(labels)), fontsize=8,
                              bbox_to_anchor=(0.5, 0.0), handlelength=2.4,
                              labelcolor=INK)
            pdf.savefig(figure, facecolor=SURFACE)
            plt.close(figure)

    log.info("wrote %d ROC panels across %d pages to %s", len(proteins),
             int(np.ceil(len(proteins) / panels_per_page)), path)
    return path


def ranking_plot(summary: pd.DataFrame, path: Path) -> Path | None:
    """Median AUC per predictor across proteins, with the interquartile range."""
    plt = _pyplot()
    if plt is None or summary.empty:
        return None
    frame = summary.dropna(subset=["median_auc"]).sort_values("median_auc")
    if frame.empty:
        return None
    figure, axis = plt.subplots(figsize=(7.4, max(2.6, 0.36 * len(frame) + 1.2)))
    y = np.arange(len(frame))
    axis.barh(y, frame["median_auc"] - 0.5, left=0.5, color=PALETTE[0], height=0.62)
    low = (frame["median_auc"] - frame["q1_auc"]).clip(lower=0)
    high = (frame["q3_auc"] - frame["median_auc"]).clip(lower=0)
    axis.errorbar(frame["median_auc"], y, xerr=np.vstack([low, high]), fmt="none",
                  ecolor=INK, elinewidth=1.0, capsize=2.5, alpha=0.7)
    axis.axvline(0.5, color=MUTED, linewidth=1.0, linestyle="--")
    axis.set_yticks(y)
    axis.set_yticklabels(frame["predictor"], fontsize=9)
    axis.set_xlabel("AUC within a protein, median across proteins "
                    "(0.5 is chance; bars show the interquartile range)")
    axis.set_xlim(min(0.35, float(frame["median_auc"].min()) - 0.05), 1.0)
    for index, count in enumerate(frame["n_proteins"]):
        axis.text(1.0, index, f"  n={int(count)}", va="center", fontsize=7.5,
                  color=MUTED)
    axis.grid(True, axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return Path(path)
