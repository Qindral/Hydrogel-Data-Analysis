"""
Track-length mean/median table and one combined histogram + KDE figure.

Consumer of cache/msd_d0_results.pkl (MSD_FromTrackmate_D0.py) and
cache/msd_20mg_files.pkl (MSD_FromTrackmate_20mg.py). Only trajectories present in each saved fit_results_MSD["imsd"] are included;
particle IDs are matched within each file before pooling. Tracks must have at
least MIN_TRACK_LENGTH (currently 10) recorded frames, as in the MSD analysis.
Length is the number of recorded positions (frames), not elapsed duration.
Upstream edge filtering/splitting is preserved; no refitting is performed.
Statistics and density normalization use all accepted tracks, including >100 frames.
The figure displays 0–100 frames with 40 equal-width bins over 10–100 frames.
KDE reflection at the minimum length prevents smoothing below the cutoff. Missing size/condition combinations are explicitly reported.

Run this file with your editor's Run button or a Python file association.
No console arguments or editable-package installation are required; select a
Python environment with the project's requirements installed. Outputs are saved
before the plot window opens. Refresh upstream caches after raw-data changes.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Resolve imports independently of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.core.analysis import MIN_TRACK_LENGTH
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────
CACHE_DIR = PROJECT_ROOT / "hydro_analysis" / "MSD_Trackmate" / "cache"
CACHE_D0 = CACHE_DIR / "msd_d0_results.pkl"
CACHE_DEFF = CACHE_DIR / "msd_20mg_files.pkl"
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
    r"\Experiments and Results - Data\Auswertungsbilder\TrackLength_Histogram"
)
SHOW_PLOT = True
N_BINS = 40
DISPLAY_MAX_FRAMES = 100  # View only; statistics and densities use all accepted tracks.
STYLES = {
    "D0": dict(face="#3B8C8C", edge="#2A6666", label="D0 (water)"),
    "Deff": dict(face="#D98C3D", edge="#A6672D", label="Deff (20 mg/mL)"),
}


def _load_cache(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _pool_track_lengths_by_size(results: dict) -> dict[float, np.ndarray]:
    """Pool lengths of saved MSD particle IDs, matched separately within each file."""
    size_groups: dict[float, list] = {}
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        tracks_df = r.get("tracks_df")
        if size_nm is None or tracks_df is None or tracks_df.empty:
            continue
        imsd = (r.get("fit_results_MSD") or {}).get("imsd")
        if imsd is None or len(imsd.columns) == 0:
            continue
        lengths = tracks_df.groupby("particle").size().reindex(imsd.columns)
        if lengths.isna().any():
            raise ValueError(f"MSD particle IDs missing from tracks_df: {r.get('xml_path')}")
        if (lengths < MIN_TRACK_LENGTH).any():
            raise ValueError(f"Cached MSD tracks violate the minimum length: {r.get('xml_path')}")
        size_groups.setdefault(float(size_nm), []).append(lengths.to_numpy())
    return {s: np.concatenate(v) for s, v in size_groups.items()}


def summarize(groups, labels):
    """One row per size/condition, including missing combinations."""
    sizes = sorted(set().union(*(set(g) for g in groups.values())))
    rows = []
    for size in sizes:
        for condition, by_size in groups.items():
            vals = by_size.get(size, np.array([]))
            rows.append(dict(
                nominal_size_nm=size, dls_size_nm=labels.get(size, size),
                condition=condition, n_tracks=len(vals),
                mean_frames=float(np.mean(vals)) if len(vals) else np.nan,
                median_frames=float(np.median(vals)) if len(vals) else np.nan,
            ))
    return pd.DataFrame(rows)


def summary_markdown(summary):
    lines = ["| Size (nm, DLS) | D0 N | D0 mean | D0 median | Deff N | Deff mean | Deff median |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for size, rows in summary.groupby("nominal_size_nm", sort=True):
        cells = [f"{rows.iloc[0]['dls_size_nm']:g}"]
        for condition in STYLES:
            row = rows.loc[rows.condition == condition].iloc[0]
            cells += [str(int(row.n_tracks)),
                      f"{row.mean_frames:.2f}" if row.n_tracks else "—",
                      f"{row.median_frames:.2f}" if row.n_tracks else "—"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot_combined(groups, labels):
    sizes = sorted(set().union(*(set(g) for g in groups.values())))
    if not sizes:
        raise ValueError("No cached track lengths available.")
    # Align the first bin to the cutoff; normalization includes the full accepted tail.
    bins = np.linspace(MIN_TRACK_LENGTH, DISPLAY_MAX_FRAMES, N_BINS + 1)
    x = np.linspace(MIN_TRACK_LENGTH, DISPLAY_MAX_FRAMES, 1200)
    ncols = min(3, len(sizes))
    nrows = int(np.ceil(len(sizes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.15, 5.00),
                             constrained_layout=True, squeeze=False,
                             sharex=True, sharey=True)
    for ax, size in zip(axes.flat, sizes):
        missing = []
        tails = []
        for condition, by_size in groups.items():
            vals = by_size.get(size, np.array([]))
            style = STYLES[condition]
            if not len(vals):
                missing.append(condition)
                continue
            tails.append(f"{condition}: {100 * np.mean(vals > DISPLAY_MAX_FRAMES):.1f}% > {DISPLAY_MAX_FRAMES} fr")
            counts, _ = np.histogram(vals, bins=bins)
            density = counts / (len(vals) * np.diff(bins))
            ax.bar(bins[:-1], density, width=np.diff(bins), align="edge",
                   color=style["face"], edgecolor=style["edge"],
                   alpha=0.30, linewidth=0.4)
            if len(vals) > 1 and np.ptp(vals) > 0:
                kde = gaussian_kde(vals)
                # Reflect at the selection boundary; draw no density below the cutoff.
                ax.plot(x, kde(x) + kde(2 * MIN_TRACK_LENGTH - x),
                        color=style["edge"], linewidth=1.8)
            else:
                ax.axvline(vals[0], color=style["edge"], linewidth=1.8)
        if tails:
            ax.text(0.97, 0.85, "\n".join(tails), ha="right", va="top",
                    transform=ax.transAxes, fontsize=6.5, color="0.35")
        if missing:
            ax.text(0.97, 0.96, ", ".join(missing) + ": no cached data",
                    ha="right", va="top", transform=ax.transAxes, fontsize=7)
        ax.set_title(f"{labels.get(size, size):g} nm", fontsize=10)
        ax.axvline(MIN_TRACK_LENGTH, color="0.5", linewidth=0.7, linestyle=":")
        ax.set_xlim(0, DISPLAY_MAX_FRAMES)
        ax.minorticks_on()
    for ax in list(axes.flat)[len(sizes):]:
        ax.set_visible(False)
    fig.supxlabel(f"Track length (recorded frames; MSD tracks ≥ {MIN_TRACK_LENGTH})", fontsize=10)
    fig.supylabel("Probability density (per frame)", fontsize=10)
    fig.legend(handles=[Patch(facecolor=s["face"], edgecolor=s["edge"],
                              alpha=0.6, label=s["label"]) for s in STYLES.values()],
               loc="outside upper center", ncol=2, frameon=False)
    return fig


def main():
    groups = {
        "D0": _pool_track_lengths_by_size(_load_cache(CACHE_D0, "MSD_FromTrackmate_D0.py")),
        "Deff": _pool_track_lengths_by_size(_load_cache(CACHE_DEFF, "MSD_FromTrackmate_20mg.py")),
    }
    for condition, group in groups.items():
        if not group:
            raise ValueError(f"No track-length data found for {condition}.")
    labels = get_dls_labels()
    summary = summarize(groups, labels)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SAVE_PATH / "track_length_summary.csv", index=False)
    table = summary_markdown(summary)
    notes = (f"Mean and median use saved MSD trajectories with at least {MIN_TRACK_LENGTH} recorded frames; each accepted trajectory has equal weight. "
             "Frames are observations, not elapsed time: acquisition rates can differ. "
             f"The plot shows 0–{DISPLAY_MAX_FRAMES} frames and annotates tail percentages; densities use all accepted tracks. "
             "DLS sizes label the nominal groups. Missing combinations are not zero-length tracks.\n\n")
    (SAVE_PATH / "track_length_summary.md").write_text(notes + table, encoding="utf-8")
    print(table)
    with plt.rc_context(_RC):
        fig = plot_combined(groups, labels)
        fig.savefig(SAVE_PATH / "track_length_d0_deff_kde.png", dpi=600)
        print(f"Saved table and figure: {SAVE_PATH}")
        if SHOW_PLOT:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
