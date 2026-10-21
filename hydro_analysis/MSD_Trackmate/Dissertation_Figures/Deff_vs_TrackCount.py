"""
Diffusion coefficient D vs. number of analysed tracks per file -- 20 mg/mL C16
hydrogel SPT measurements (companion to Abb11, individual hydrogel D values).

Pure consumer: reads msd_20mg_files.pkl (produced by MSD_FromTrackmate_20mg.py),
never raw data. One point per file:
  x = number of tracks with >= MIN_TRACK_FRAMES detections in tracks_df
      (core.analysis.MIN_TRACK_LENGTH, the approved 10-frame minimum; the same
      tracks that enter the MSD fit), log axis;
  y = fit_results_MSD["D_um2_per_s"] of that file (the value shown in Abb11), log axis.
Colour = particle size (SIZE_COLORS, dark variant because hydrogel); marker =
loading type from core.io.condition_label_from_filename() (Surface loading "o",
Injection "s"). No fit line. Spearman rho and p per size are printed and
written to deff_vs_trackcount.csv, not drawn.

Figure: Styleguide v2, width class half, PDF, saved to SAVE_PATH.
No other script reads the outputs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
from scipy.stats import spearmanr

from hydro_analysis.core.analysis import MIN_TRACK_LENGTH
from hydro_analysis.core.io import condition_label_from_filename, get_dls_labels
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).resolve().parents[1] / "cache" / "msd_20mg_files.pkl"
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\SPT"
)
FIG_NAME = "deff_vs_trackcount"
STYLE = Path(__file__).resolve().parents[2] / "thesis.mplstyle"
WIDTH_IN, RATIO = 3.07, 1.42            # Styleguide width class "half"
MIN_TRACK_FRAMES = MIN_TRACK_LENGTH     # approved minimum track length (10 frames)
SIZES_NM = (20.0, 50.0)                 # nominal sizes measured in hydrogel (DLS 35 / 50 nm)
CONDITION_MARKERS = {"Surface loading": "o", "Injection": "s"}   # decided by user, 2026-10-14
POINT_ALPHA = 0.8


def count_tracks(tracks_df: pd.DataFrame, min_frames: int) -> int:
    """Number of tracks with at least min_frames detections."""
    return int((tracks_df.groupby("particle").size() >= min_frames).sum())


def test_count_tracks_synthetic() -> None:
    """Synthetic check with a known answer: tracks of 3, 10, 12 and 9 detections."""
    lengths = {1: 3, 2: 10, 3: 12, 4: 9}
    df = pd.DataFrame([{"particle": p, "frame": f} for p, n in lengths.items() for f in range(n)])
    recovered = count_tracks(df, 10)
    print(f"Synthetic check count_tracks: true 2, recovered {recovered}")
    assert recovered == 2


def collect_rows(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        fit = r.get("fit_results_MSD") or {}
        tracks = r.get("tracks_df")
        rows.append({
            "file": r.get("base_name"),
            "particle_size_nm": r.get("particle_size_nm"),
            "loading": condition_label_from_filename(r.get("base_name", "")),
            "n_tracks_min_frames": count_tracks(tracks, MIN_TRACK_FRAMES) if tracks is not None else np.nan,
            "D_um2_per_s": fit.get("D_um2_per_s", np.nan),
        })
    return pd.DataFrame(rows)


def main() -> None:
    test_count_tracks_synthetic()
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_20mg.py ausführen.")
    with open(CACHE_FILE, "rb") as f:
        results = pickle.load(f)
    df = collect_rows(results)
    dls_labels = get_dls_labels()
    print(f"Cache geladen: {CACHE_FILE} ({len(df)} Dateien)")

    stats = []
    for size_nm in SIZES_NM:
        sub = df[(df["particle_size_nm"] == size_nm) & df["D_um2_per_s"].notna()]
        rho, p = spearmanr(sub["n_tracks_min_frames"], sub["D_um2_per_s"])
        stats.append({"label_nm": dls_labels.get(size_nm, int(size_nm)), "n_files": len(sub),
                      "spearman_rho": rho, "p_value": p})
        print(f"{dls_labels.get(size_nm, int(size_nm))} nm: {len(sub)} files, Spearman rho = {rho:.2f}, p = {p:.2f}")
    other = df[~df["particle_size_nm"].isin(SIZES_NM)]
    if len(other):
        print(f"[NOTE] {len(other)} file(s) of other sizes not shown: {sorted(other['file'])}")
    unknown = df[df["loading"].isna()]
    if len(unknown):
        print(f"[NOTE] {len(unknown)} file(s) without A/B code, drawn as '^': {sorted(unknown['file'])}")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    out = df.sort_values(["particle_size_nm", "file"])
    out.to_csv(SAVE_PATH / f"{FIG_NAME}.csv", index=False)
    pd.DataFrame(stats).to_csv(SAVE_PATH / f"{FIG_NAME}_spearman.csv", index=False)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN / RATIO))
        for size_nm in SIZES_NM:
            _base, dark = SIZE_COLORS[size_nm]
            for loading, marker in list(CONDITION_MARKERS.items()) + [(None, "^")]:
                g = df[(df["particle_size_nm"] == size_nm) & (df["loading"].isna() if loading is None
                                                               else df["loading"] == loading)]
                if g.empty:
                    continue
                ax.scatter(g["n_tracks_min_frames"], g["D_um2_per_s"], s=16, marker=marker, alpha=POINT_ALPHA,
                           facecolor=dark, edgecolor=dark, linewidth=0.6, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        fmt = FuncFormatter(lambda v, _pos: f"{v:g}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel(f"Tracks per file (≥ {MIN_TRACK_FRAMES} frames)")
        ax.set_ylabel(r"$D$ (µm$^2$ s$^{-1}$)")
        handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=4, markerfacecolor=SIZE_COLORS[s][1],
                          markeredgecolor=SIZE_COLORS[s][1], label=f"{dls_labels.get(s, int(s))} nm")
                   for s in SIZES_NM]
        handles += [Line2D([0], [0], marker=m, linestyle="none", markersize=4, markerfacecolor="0.4",
                           markeredgecolor="0.4", label=lab) for lab, m in CONDITION_MARKERS.items()]
        fig.legend(handles=handles, loc="outside upper center", ncol=4, handletextpad=0.1, columnspacing=0.6)
        fig.savefig(SAVE_PATH / f"{FIG_NAME}.pdf")
        plt.close(fig)
    print(f"Plot gespeichert: {SAVE_PATH / (FIG_NAME + '.pdf')}")


if __name__ == "__main__":
    main()
