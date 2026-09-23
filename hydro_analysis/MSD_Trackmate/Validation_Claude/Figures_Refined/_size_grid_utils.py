"""
Shared helper for the refined eMSD/iMSD "by particle size" count-histogram
grids (items 4 and 10 of the refined-figures spec). Used by
eMSD_Refined.py and iMSD_histograms_Refined.py only -- does not read any
cache itself, pure plotting helper, no data loading/analysis logic (kept
separate from core/ per CLAUDE.md's io/analysis/visualization split, since
this is specific to the Validation_Claude figure family, not a
project-wide primitive).

Design choices, so both callers stay consistent:
  - Counts, not density: plain (unnormalised) histograms. An optional KDE
    overlay (`overlay_kde=True`) is rescaled to the Counts axis
    (density * N * bin_width) rather than plotted at raw density scale, so
    it stays meaningful next to unnormalised bars -- used for the
    additional "_kde" variant outputs, not the primary Counts-only ones.
  - Identical bin edges and identical Counts (y) upper limit across every
    size panel within one metric grid, so panels are directly comparable.
  - Explicit fixed x-range for the exponent n (-0.3 to 1.2) and track
    length (0-200 frames) axes, per spec; any values outside that range
    are counted (not silently dropped from the count) and reported both
    to the console and as a footer note on the figure. D and step size
    get a shared, data-derived x-range instead (no user-specified cutoff
    exists for these, so nothing is out-of-range by construction).
  - Axis labels only on the outer edge of the grid (label_outer); panel
    titles slightly bolder (fontweight="semibold").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


@dataclass
class GridAxisSpec:
    """Bin edges / y-limit derived for one metric, reusable across figures
    that must share axis limits (e.g. iMSD D grid + its water-only /
    hydrogel-only counterparts, item 10)."""
    bin_edges: np.ndarray
    y_max: float
    log_x: bool


def _clean(vals) -> np.ndarray:
    a = np.asarray(vals, dtype=float)
    return a[np.isfinite(a)]


def compute_axis_spec(df: pd.DataFrame, metric: str, log_x: bool = False,
                       x_range: tuple[float, float] | None = None,
                       n_bins: int = 34) -> tuple[GridAxisSpec, int]:
    """Pools ALL rows (every size, both conditions) to derive shared bin
    edges and a shared Counts upper limit. Returns (spec, n_out_of_range)."""
    vals = _clean(df[metric])
    if log_x:
        vals = vals[vals > 0]

    n_out_of_range = 0
    if x_range is not None:
        lo, hi = x_range
        n_out_of_range = int(((vals < lo) | (vals > hi)).sum())
        bin_edges = np.linspace(lo, hi, n_bins + 1)
    elif vals.size == 0:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif log_x:
        bin_edges = np.logspace(np.log10(vals.min()), np.log10(vals.max()), n_bins + 1)
    else:
        bin_edges = np.linspace(vals.min(), vals.max(), n_bins + 1)

    # Shared Counts (y) upper limit: max single-condition, single-size bin
    # height that will actually appear in any panel of the grid.
    y_max = 1.0
    if "particle_size_nm" in df.columns and "condition" in df.columns:
        for _, g in df.groupby(["particle_size_nm", "condition"]):
            v = _clean(g[metric])
            if log_x:
                v = v[v > 0]
            if v.size == 0:
                continue
            counts, _ = np.histogram(v, bins=bin_edges)
            y_max = max(y_max, float(counts.max()))
    return GridAxisSpec(bin_edges=bin_edges, y_max=y_max * 1.08, log_x=log_x), n_out_of_range


def _scaled_kde(ax, v: np.ndarray, bin_edges: np.ndarray, log_x: bool, color: str) -> None:
    """KDE curve rescaled to the Counts axis (density * N * bin_width), so
    it overlays meaningfully on unnormalised histogram bars instead of the
    near-invisible density-scale curve a raw gaussian_kde would give here.
    Needs >=2 distinct values; drawn in log10-space for log_x metrics
    (matches the convention used by the original iMSD/eMSD histogram
    scripts before Counts replaced Density)."""
    if v.size < 2 or np.ptp(v) == 0:
        return
    if log_x:
        # KDE and bin widths both evaluated in log10(x)-space (bins are
        # log-spaced here), so counts-per-bin ~ density_log10 * log_bin_width * N.
        x_grid = np.logspace(np.log10(bin_edges[0]), np.log10(bin_edges[-1]), 300)
        log_bin_width_avg = float(np.mean(np.diff(np.log10(bin_edges))))
        y = gaussian_kde(np.log10(v))(np.log10(x_grid)) * log_bin_width_avg * v.size
    else:
        x_grid = np.linspace(bin_edges[0], bin_edges[-1], 300)
        bin_width_avg = float(np.mean(np.diff(bin_edges)))
        y = gaussian_kde(v)(x_grid) * bin_width_avg * v.size
    ax.plot(x_grid, y, color=color, linewidth=1.5, zorder=4)


def _panel_hist(ax, water_vals, hydrogel_vals, spec: GridAxisSpec,
                 style_water: dict, style_hydrogel: dict, overlay_kde: bool = False) -> None:
    for vals, style in ((water_vals, style_water), (hydrogel_vals, style_hydrogel)):
        v = _clean(vals)
        if spec.log_x:
            v = v[v > 0]
        if v.size == 0:
            continue
        ax.hist(v, bins=spec.bin_edges, alpha=0.55, color=style["face"],
                edgecolor=style["edge"], linewidth=0.4, label=style["label"])
        if overlay_kde:
            _scaled_kde(ax, v, spec.bin_edges, spec.log_x, style["edge"])
    if spec.log_x:
        ax.set_xscale("log")
    ax.set_xlim(spec.bin_edges[0], spec.bin_edges[-1])
    ax.set_ylim(0.0, spec.y_max)


def plot_count_grid_by_size(
    df: pd.DataFrame, metric: str, xlabel: str, dls_labels: dict[float, int],
    spec: GridAxisSpec, condition_filter: str | None = None,
    style_water: dict | None = None, style_hydrogel: dict | None = None,
    overlay_kde: bool = False,
) -> plt.Figure:
    """One panel per particle size (up to 6), Water vs. Hydrogel overlaid
    (unless condition_filter restricts to one), identical bin edges and
    Counts y-limit across all panels (from `spec`), outer-only axis
    labels, bolder panel titles."""
    style_water = style_water or dict(face="#009900", edge="#006600", label="Water")
    style_hydrogel = style_hydrogel or dict(face="#0000da", edge="#000099", label="Hydrogel")

    sizes = sorted(df["particle_size_nm"].dropna().unique())
    n = len(sizes)
    ncols = 3 if n > 3 else max(n, 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2.2),
                              constrained_layout=True, squeeze=False, sharex=True, sharey=True)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    for idx, (ax, size_nm) in enumerate(zip(ax_flat, sizes)):
        water = df.loc[(df["condition"] == "water") & (df["particle_size_nm"] == size_nm), metric] \
            if condition_filter in (None, "water") else pd.Series(dtype=float)
        hydrogel = df.loc[(df["condition"] == "hydrogel") & (df["particle_size_nm"] == size_nm), metric] \
            if condition_filter in (None, "hydrogel") else pd.Series(dtype=float)
        _panel_hist(ax, water, hydrogel, spec, style_water, style_hydrogel, overlay_kde=overlay_kde)
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm", fontsize=9.5, fontweight="semibold")
        ax.label_outer()

    for ax in ax_flat[:n]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
    for ax in ax_flat[:n]:
        ax.label_outer()

    handles_labels = ax_flat[0].get_legend_handles_labels()
    if handles_labels[0]:
        ax_flat[0].legend(loc="upper right", fontsize=7, frameon=False)
    return fig
