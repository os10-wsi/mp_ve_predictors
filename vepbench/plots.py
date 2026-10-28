"""Figures for the proteome-wide analysis.

Every function returns the path it wrote, or None when matplotlib is absent
or the data cannot support the figure. Returning None rather than raising
keeps a missing plotting dependency from failing a run that has already
spent days of compute.

Colour is used for one job per figure and never for decoration. Predictors
calibrated on ClinVar are drawn in a second hue throughout, because the
single most misleading thing this analysis could do is rank them beside
unsupervised predictors without saying which is which. AUROC uses a
diverging scale with its neutral point at 0.5, so chance reads as absent
colour rather than as a low value.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Validated for deuteranopia, protanopia and tritanopia separation against a
# near-white surface (worst adjacent dE 24.7, protan).
UNSUPERVISED = "#2a78d6"
CLINVAR_TRAINED = "#eb6834"
PATHOGENIC = "#eb6834"
BENIGN = "#2a78d6"
INK = "#2b2b28"
MUTED = "#767670"
GRID = "#e4e3df"
NEUTRAL = "#f0efec"
SURFACE = "#fcfcfb"

CHANCE = 0.5


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib is not installed; skipping the figure "
                    "(pip install 'vepbench[proteome]')")
        return None
    return plt


def _finish(figure, axis_or_axes, path: Path):
    axes = axis_or_axes if isinstance(axis_or_axes, (list, tuple)) else [axis_or_axes]
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=9, length=3)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_color(INK)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(figure)
    log.info("wrote %s", path)
    return path


def _colours(flags) -> list[str]:
    return [CLINVAR_TRAINED if bool(f) else UNSUPERVISED for f in flags]


def _legend(axis, plt, labels=("unsupervised", "calibrated on ClinVar"),
            colours=(UNSUPERVISED, CLINVAR_TRAINED)):
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8, color=c)
               for c in colours]
    legend = axis.legend(handles, labels, frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK)


def auroc_ranking(summary: pd.DataFrame, path: Path, *, title: str | None = None):
    """Per-protein median AUROC per predictor, with the interquartile range.

    The per-protein median leads because pooled AUROC across the proteome
    rewards a predictor that merely separates genes.
    """
    plt = _pyplot()
    if plt is None or summary.empty:
        return None
    frame = summary.dropna(subset=["median_auroc"]).sort_values("median_auroc")
    if frame.empty:
        return None

    figure, axis = plt.subplots(figsize=(8.0, max(3.0, 0.34 * len(frame) + 1.4)))
    y = np.arange(len(frame))
    colours = _colours(frame.get("trained_on_clinvar", pd.Series(False, index=frame.index)))
    axis.barh(y, frame["median_auroc"] - CHANCE, left=CHANCE, color=colours, height=0.62)

    if {"q1_auroc", "q3_auroc"}.issubset(frame.columns):
        low = (frame["median_auroc"] - frame["q1_auroc"]).clip(lower=0)
        high = (frame["q3_auroc"] - frame["median_auroc"]).clip(lower=0)
        axis.errorbar(frame["median_auroc"], y, xerr=np.vstack([low, high]), fmt="none",
                      ecolor=INK, elinewidth=1.0, capsize=2.5, alpha=0.65)

    axis.axvline(CHANCE, color=MUTED, linewidth=1.0, linestyle="--")
    axis.set_yticks(y)
    axis.set_yticklabels(frame["predictor"], fontsize=9)
    axis.set_xlabel("AUROC within a protein, median across proteins "
                    "(0.5 is chance; bars show the interquartile range)")
    axis.set_title(title or "Separating pathogenic from benign ClinVar variants",
                   fontsize=11, color=INK, loc="left")
    axis.set_xlim(min(0.35, float(frame["median_auroc"].min()) - 0.05), 1.0)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    if "n_proteins" in frame.columns:
        for index, count in enumerate(frame["n_proteins"]):
            axis.text(1.0, index, f"  n={int(count)}", va="center", fontsize=7.5,
                      color=MUTED)
    _legend(axis, plt)
    return _finish(figure, axis, Path(path))


def stratified_heatmap(results: pd.DataFrame, path: Path, *, max_predictors: int = 30):
    """AUROC per predictor across every stratum, diverging about chance."""
    plt = _pyplot()
    if plt is None or results.empty:
        return None
    frame = results.copy()
    frame["label"] = frame["stratum"] + ": " + frame["group"]
    matrix = frame.pivot_table(index="predictor", columns="label", values="auroc")
    if matrix.empty:
        return None
    order = matrix.mean(axis=1).sort_values(ascending=False).index[:max_predictors]
    # Keep the strata in the order they are produced, so related groups sit
    # together; pivot_table would otherwise sort them alphabetically and
    # scatter 'buried' away from 'exposed'.
    columns = list(dict.fromkeys(frame["label"]))
    matrix = matrix.loc[order, [c for c in columns if c in matrix.columns]]

    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list(
        "auroc", ["#d03b3b", NEUTRAL, UNSUPERVISED]
    )
    # Chance stays the neutral midpoint, but the arms follow the data. A
    # symmetric span would spend half the colour range on below-chance
    # values that mostly do not occur, leaving every real difference as one
    # shade of blue. Each arm keeps a floor so a single below-chance cell is
    # still unmistakably red.
    values = matrix.to_numpy(dtype=float)
    norm = TwoSlopeNorm(
        vmin=min(float(np.nanmin(values)), CHANCE - 0.05), vcenter=CHANCE,
        vmax=max(float(np.nanmax(values)), CHANCE + 0.05),
    )

    figure, axis = plt.subplots(
        figsize=(1.1 + 0.5 * matrix.shape[1], 1.6 + 0.28 * matrix.shape[0])
    )
    image = axis.imshow(matrix.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(np.arange(matrix.shape[1]))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(np.arange(matrix.shape[0]))
    axis.set_yticklabels(matrix.index, fontsize=8)
    axis.set_title("AUROC by structural and clinical context", fontsize=11,
                   color=INK, loc="left")
    bar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    bar.set_label("AUROC (0.5 is chance)", fontsize=8, color=INK)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    axis.spines[:].set_visible(False)
    axis.tick_params(colors=MUTED, length=0)
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        label.set_color(INK)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    log.info("wrote %s", path)
    return Path(path)


def score_distributions(frame: pd.DataFrame, predictors: list[str], path: Path,
                        *, label_col: str = "label", bins: int = 60):
    """Harmonized score distributions for pathogenic and benign variants."""
    plt = _pyplot()
    if plt is None or frame.empty or not predictors:
        return None
    from vepbench.schema import HARMONIZED_SUFFIX

    predictors = predictors[:6]
    columns = 2 if len(predictors) > 1 else 1
    rows = int(np.ceil(len(predictors) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 2.7 * rows),
                                squeeze=False)
    flat = [axes[r][c] for r in range(rows) for c in range(columns)]
    labels = pd.to_numeric(frame[label_col], errors="coerce")

    drawn = []
    for axis, name in zip(flat, predictors, strict=False):
        column = f"{name}{HARMONIZED_SUFFIX}"
        if column not in frame:
            continue
        scores = pd.to_numeric(frame[column], errors="coerce")
        benign = scores[(labels == 0) & scores.notna()]
        pathogenic = scores[(labels == 1) & scores.notna()]
        if benign.empty or pathogenic.empty:
            continue
        edges = np.histogram_bin_edges(
            pd.concat([benign, pathogenic]).to_numpy(dtype=float), bins=bins
        )
        axis.hist(benign, bins=edges, color=BENIGN, alpha=0.75, density=True,
                  label=f"benign (n={len(benign):,})")
        axis.hist(pathogenic, bins=edges, color=PATHOGENIC, alpha=0.65, density=True,
                  label=f"pathogenic (n={len(pathogenic):,})")
        axis.set_title(name, fontsize=10, color=INK, loc="left")
        axis.set_yticks([])
        axis.legend(frameon=False, fontsize=8)
        drawn.append(axis)

    for axis in flat[len(drawn):]:
        axis.set_visible(False)
    if not drawn:
        plt.close(figure)
        return None
    drawn[-1].set_xlabel("harmonized score (higher is predicted more damaging)")
    return _finish(figure, drawn, Path(path))


def pooled_versus_per_protein(pooled: pd.DataFrame, per_protein: pd.DataFrame, path: Path):
    """Pooled AUROC against the within-protein median, per predictor.

    A point well below the diagonal is a predictor whose pooled number
    comes partly from separating genes rather than variants: pooled is what
    gets quoted, per-protein is what holds up. Points on the diagonal mean
    pooling cost nothing, which is a result too.
    """
    plt = _pyplot()
    if plt is None or pooled.empty or per_protein.empty:
        return None
    overall = pooled.loc[(pooled["stratum"] == "all")].set_index("predictor")
    merged = per_protein.set_index("predictor").join(
        overall["auroc"].rename("pooled_auroc"), how="inner"
    ).dropna(subset=["median_auroc", "pooled_auroc"])
    if merged.empty:
        return None

    figure, axis = plt.subplots(figsize=(6.2, 5.6))
    colours = _colours(merged.get("trained_on_clinvar", pd.Series(False, index=merged.index)))
    axis.scatter(merged["pooled_auroc"], merged["median_auroc"], s=46, c=colours,
                 edgecolors=SURFACE, linewidths=1.2, zorder=3)
    limits = [0.4, 1.0]
    axis.plot(limits, limits, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    # Label only the extremes: a name on every point is unreadable.
    gap = merged["pooled_auroc"] - merged["median_auroc"]
    for name in list(gap.nlargest(3).index) + list(merged["median_auroc"].nlargest(2).index):
        axis.annotate(name, (merged.at[name, "pooled_auroc"],
                             merged.at[name, "median_auroc"]),
                      textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK)
    axis.set_xlim(*limits)
    axis.set_ylim(*limits)
    axis.set_xlabel("AUROC pooled across the proteome")
    axis.set_ylabel("AUROC within a protein (median)")
    axis.set_title("Pooled against within-protein AUROC", fontsize=11,
                   color=INK, loc="left")
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    _legend(axis, plt)
    return _finish(figure, axis, Path(path))


def coverage_bars(coverage: pd.DataFrame, path: Path):
    """How many proteins each predictor actually produced scores for."""
    plt = _pyplot()
    if plt is None or coverage.empty:
        return None
    frame = coverage.sort_values("proteins")
    figure, axis = plt.subplots(figsize=(7.4, max(3.0, 0.3 * len(frame) + 1.2)))
    y = np.arange(len(frame))
    axis.barh(y, frame["proteins"], color=UNSUPERVISED, height=0.62)
    axis.set_yticks(y)
    axis.set_yticklabels(frame["predictor"], fontsize=9)
    axis.set_xlabel("proteins with at least one score")
    axis.set_title("Predictor coverage across the run", fontsize=11, color=INK, loc="left")
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    return _finish(figure, axis, Path(path))


def protein_track(
    predictions: pd.DataFrame, path: Path, *, uniprot_id: str,
    predictor: str, clinvar: pd.DataFrame | None = None,
    tm_segments: list[tuple[int, int]] | None = None,
):
    """One protein along its sequence: mean predicted effect, membrane, ClinVar.

    The point of the per-protein figure is to put the three layers on one
    axis, so a run of high scores can be read against where the membrane is
    and where the clinical variants fall.
    """
    plt = _pyplot()
    if plt is None or predictions.empty:
        return None
    from vepbench.schema import HARMONIZED_SUFFIX

    column = f"{predictor}{HARMONIZED_SUFFIX}"
    if column not in predictions:
        log.info("%s has no %s column; skipping its track", uniprot_id, column)
        return None
    scores = pd.to_numeric(predictions[column], errors="coerce")
    by_position = scores.groupby(predictions["position"]).mean()
    if by_position.dropna().empty:
        return None
    # Reindex over the whole sequence so a position the predictor could not
    # score stays a gap. Dropping those instead would join the two sides
    # with a straight line that reads as a real, smooth stretch of score.
    positions = predictions["position"]
    by_position = by_position.reindex(
        range(int(positions.min()), int(positions.max()) + 1)
    )

    figure, axis = plt.subplots(figsize=(10.0, 3.2))
    for start, end in tm_segments or []:
        axis.axvspan(start - 0.5, end + 0.5, color=NEUTRAL, zorder=0)
    axis.plot(by_position.index, by_position.to_numpy(), color=UNSUPERVISED,
              linewidth=1.4, zorder=2)

    if clinvar is not None and not clinvar.empty:
        labels = pd.to_numeric(clinvar["label"], errors="coerce")
        top = float(by_position.max())
        bottom = float(by_position.min())
        offset = 0.04 * (top - bottom or 1.0)
        for value, colour, marker, name in (
            (1, PATHOGENIC, "v", "pathogenic"), (0, BENIGN, "^", "benign"),
        ):
            subset = clinvar.loc[labels == value, "position"]
            if subset.empty:
                continue
            height = top + offset if value == 1 else bottom - offset
            axis.plot(subset, np.full(len(subset), height), marker, color=colour,
                      markersize=5, linestyle="", label=f"ClinVar {name}", zorder=3)
        if axis.get_legend_handles_labels()[0]:
            legend = axis.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
            for text in legend.get_texts():
                text.set_color(INK)

    axis.set_xlabel("residue position")
    axis.set_ylabel(f"{predictor}\n(mean over substitutions)", fontsize=9)
    shaded = " · shaded bands are transmembrane segments" if tm_segments else ""
    axis.set_title(f"{uniprot_id}{shaded}", fontsize=11, color=INK, loc="left")
    axis.set_xlim(0, float(by_position.index.max()) + 1)
    # The ClinVar markers sit just outside the score range; without extra
    # headroom the legend lands on top of the pathogenic row.
    low, high = axis.get_ylim()
    axis.set_ylim(low - 0.05 * (high - low), high + 0.28 * (high - low))
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    return _finish(figure, axis, Path(path))
