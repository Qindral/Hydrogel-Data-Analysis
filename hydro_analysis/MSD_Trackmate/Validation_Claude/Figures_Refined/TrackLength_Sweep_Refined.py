"""
Minimum-track-length sweep -- D and n as continuous curves over L_min.

Replaces the fixed length-bin boxplots (Correlations_Refined.py,
B_length_bin_*.png) with one line per particle size: top panel the relative
diffusion coefficient D(L_min) / D(10), bottom panel the anomalous exponent
n(L_min), both on linear axes. D is normalised per file to that file's own
standard value (L_min = 10) BEFORE averaging, so sizes spanning several
orders of magnitude in D share one linear axis and files with many tracks
do not dominate. Each curve is the plain arithmetic mean over files -- no
uncertainty bands, by design.

A size's curve is continued only while enough of its files still
contribute (at least MIN_FILES and at least MIN_FILE_FRACTION of the files
present at L_min = 10) and stops at the first L_min where this fails.
Otherwise the mean would jump whenever single files drop out of the
average. Per file, the upstream sweep already stops below
MIN_TRACKS_PER_POINT trajectories (see TrackLength_Sweep_Compute.py).

Two separate figures (water, hydrogel). The y-limits are shared so the
magnitude of the effect stays directly comparable; the x-range is set per
figure, since hydrogel trajectories are much shorter and its curves end
at far smaller L_min.

Pure consumer of cache/track_length_sweep.pkl -- run
Validation_Claude/TrackLength_Sweep_Compute.py first. No refitting here.

Outputs (Figures_Refined/Correlations, PNG, 600 dpi):
E_track_length_sweep_water.png, E_track_length_sweep_hydrogel.png.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.Correlations_Refined import SAVE_PATH

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_SWEEP = Path(__file__).parent.parent.parent / "cache" / "track_length_sweep.pkl"

MIN_FILES = 2              # absolute minimum of contributing files per curve point
MIN_FILE_FRACTION = 0.5    # ... and at least this fraction of the size's files at L_min = 10
LINE_WIDTH = 1.8
CONDITION_TITLES = {"water": "Water", "hydrogel": "Hydrogel"}


def mean_curves(sweep: pd.DataFrame, condition: str) -> dict[float, pd.DataFrame]:
    """{size_nm: DataFrame[L_min, D_rel, exponent, n_files]}, mean over files,
    truncated at the first L_min with too few contributing files."""
    sub = sweep[sweep["condition"] == condition]
    curves = {}
    for size_nm, g in sub.groupby("particle_size_nm"):
        agg = (g.groupby("L_min")
                .agg(D_rel=("D_rel", "mean"), exponent=("exponent", "mean"), n_files=("xml_path", "nunique"))
                .reset_index().sort_values("L_min"))
        if agg.empty:
            continue
        required = max(MIN_FILES, int(np.ceil(MIN_FILE_FRACTION * agg["n_files"].iloc[0])))
        too_few = np.flatnonzero(agg["n_files"].to_numpy() < required)
        agg = agg.iloc[:too_few[0]] if too_few.size else agg
        if len(agg) >= 2:
            curves[float(size_nm)] = agg
    return curves


def _padded_limits(values: np.ndarray, include: float, pad: float = 0.06) -> tuple[float, float]:
    lo, hi = min(values.min(), include), max(values.max(), include)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def plot_sweep(curves: dict[float, pd.DataFrame], title: str, dls_labels: dict[float, int],
               xlim: tuple[float, float], d_ylim: tuple[float, float], n_ylim: tuple[float, float]) -> plt.Figure:
    fig, (ax_d, ax_n) = plt.subplots(2, 1, figsize=(7.15, 5.00), constrained_layout=True, sharex=True)
    ax_d.axhline(1.0, color="black", linewidth=1.0, linestyle=_DASH_THEORY, zorder=1)
    ax_n.axhline(1.0, color="black", linewidth=1.0, linestyle=_DASH_THEORY, zorder=1)

    handles = []
    for size_nm, agg in sorted(curves.items(), reverse=True):
        base, _ = SIZE_COLORS.get(size_nm, ("#999999", "#555555"))
        ax_d.plot(agg["L_min"], agg["D_rel"], color=base, linewidth=LINE_WIDTH, zorder=3)
        ax_n.plot(agg["L_min"], agg["exponent"], color=base, linewidth=LINE_WIDTH, zorder=3)
        handles.append(Line2D([0], [0], color=base, linewidth=LINE_WIDTH,
                              label=f"{dls_labels.get(size_nm, int(size_nm))} nm (N = {int(agg['n_files'].iloc[0])})"))

    ax_d.set_xlim(*xlim)
    ax_d.set_ylim(*d_ylim)
    ax_n.set_ylim(*n_ylim)
    ax_d.set_ylabel(r"$D(L_\mathrm{min})\,/\,D(10)$")
    ax_n.set_ylabel(r"Anomalous exponent $n$")
    ax_n.set_xlabel(r"Minimum track length $L_\mathrm{min}$ (frames)")
    ax_d.set_title(title, fontsize=10, fontweight="semibold")
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 6), frameon=False, fontsize=8)
    return fig


def main() -> None:
    if not CACHE_SWEEP.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {CACHE_SWEEP}\n"
                                "Bitte zuerst TrackLength_Sweep_Compute.py ausführen.")
    with open(CACHE_SWEEP, "rb") as f:
        sweep = pickle.load(f)["sweep"]
    dls_labels = get_dls_labels()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    all_curves = {cond: mean_curves(sweep, cond) for cond in CONDITION_TITLES}
    stacked = pd.concat([agg for curves in all_curves.values() for agg in curves.values()], ignore_index=True)
    d_ylim = _padded_limits(stacked["D_rel"].to_numpy(dtype=float), include=1.0)
    n_ylim = _padded_limits(stacked["exponent"].to_numpy(dtype=float), include=1.0)

    with plt.rc_context(_RC):
        for condition, title in CONDITION_TITLES.items():
            curves = all_curves[condition]
            if not curves:
                print(f"{title}: keine Kurve mit ausreichend Dateien.")
                continue
            for size_nm, agg in sorted(curves.items()):
                print(f"{title} {dls_labels.get(size_nm, int(size_nm))} nm: L_min 10-{int(agg['L_min'].iloc[-1])}, "
                      f"D/D(10) am Ende {agg['D_rel'].iloc[-1]:.2f}, n {agg['exponent'].iloc[0]:.2f} -> "
                      f"{agg['exponent'].iloc[-1]:.2f}, {int(agg['n_files'].iloc[0])} -> "
                      f"{int(agg['n_files'].iloc[-1])} Dateien.")
            l_min_values = pd.concat([agg["L_min"] for agg in curves.values()])
            xlim = (float(l_min_values.min()), float(l_min_values.max()))
            fig = plot_sweep(curves, title, dls_labels, xlim, d_ylim, n_ylim)
            path = SAVE_PATH / f"E_track_length_sweep_{condition}.png"
            safe_savefig(fig, path, dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot gespeichert: {path}")


if __name__ == "__main__":
    main()
