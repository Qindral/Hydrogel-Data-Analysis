"""
TEST SCRIPT — exploratory correlations between per-file D_MSD (and its fit
error) and various per-file quality/processing metrics, for the D0 (water)
calibration measurements across all particle sizes, colored by particle
size. Companion to MSD_Diffusion_Correlations_Test_20mg.py, same metrics and
plot style, applied to the water dataset instead of the 20 mg/mL hydrogel.

Not a publication figure -- a quick diagnostic grid to check whether D or
its error is driven by acquisition/processing artifacts rather than real
physics. Unlike the hydrogel dataset, water files carry no Surface
loading/Injection condition token, so color here encodes particle size only
(all six sizes) instead of size+condition.

Reads msd_d0_results.pkl (produced by MSD_FromTrackmate_D0.py -- unlike the
20 mg/mL pipeline this is already the per-file cache; there is no separate
"_files" vs "_result" split for D0) and re-derives the same per-file metrics
as the 20 mg/mL test script:
  - num_points            = len(tracks_df)                      (rows = detections)
  - mean_track_length     = tracks_df.groupby('particle').size().mean()
  - edge_filter_removed   = detections dropped by remove_edge_artifacts()
                            (core.io.single_file_data), i.e. "processed away"
  - mean_brightness       = mean pixel value of a sample of frames from the
                            raw TIFF (tif_path), read via tifffile page-by-page
                            so full stacks don't have to be loaded into memory
  - sigma_loc_nm          = localization-uncertainty term from the per-file
                            power-law MSD fit itself (fit_results_MSD)

No TrackMate-settings correlation here (unlike the 20 mg/mL script): the
water measurements' full TrackMate project XMLs live in differently-named
analysis folders per size ("Trackmate_Analyses"/"analysis_results" for
20 nm, "Analysis" for 100/200/500 nm, none obviously for 1000 nm) rather
than the one consistent folder per size used for the hydrogel data, so that
part was left out rather than guessing the wrong mapping.

One 2x6 grid: top row = D vs each metric, bottom row = D_error vs the same
metric. Each panel prints its Pearson r in the corner as a quick eyeball
check for correlation strength.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile

from hydro_analysis.core.io import get_dls_labels

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_d0_results.pkl"
SAVE_PATH: Path | None = None  # test script -- set a Path to also save a PNG

BRIGHTNESS_MAX_FRAMES = 20   # sampled evenly across the stack

# One color per nominal particle size (Style_guide.txt palette, extended to 6)
SIZE_COLORS = {
    20.0:   ("#3B8C8C", "#2A6666"),   # teal
    50.0:   ("#D98C3D", "#A6672D"),   # orange
    100.0:  ("#5B7FBF", "#3F5C8C"),   # blue
    200.0:  ("#8C3B6E", "#66294F"),   # magenta
    500.0:  ("#6E8C3B", "#4F6629"),   # olive
    1000.0: ("#8C5B3B", "#664129"),   # brown
}
POINT_ALPHA = 0.65

METRICS = [
    ("num_tracks",        "Anzahl Trajektorien"),
    ("num_points",        "Anzahl Datenpunkte"),
    ("edge_removed",      "Prozessierte Punkte (Edge-Filter entfernt)"),
    ("mean_track_length", "Mittlere Trajektorienlänge (Frames)"),
    ("mean_brightness",   "Mittlere Kamera-Helligkeit (a.u.)"),
    ("sigma_loc_nm",      "Lokalisationsunsicherheit $\\sigma_{loc}$ (nm)"),
]


def load_results() -> dict:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_D0.py ausführen."
        )
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def _mean_brightness(tif_path: str | None, max_frames: int = BRIGHTNESS_MAX_FRAMES) -> float:
    """Mean pixel value over a handful of frames sampled evenly across the
    stack, read page-by-page so the whole (potentially huge) TIFF never has
    to be loaded into memory at once."""
    if not tif_path:
        return float("nan")
    path = Path(tif_path)
    if not path.exists():
        return float("nan")
    try:
        with tifffile.TiffFile(path) as tif:
            n_pages = len(tif.pages)
            if n_pages == 0:
                return float("nan")
            idx = np.linspace(0, n_pages - 1, num=min(max_frames, n_pages), dtype=int)
            frames = [tif.pages[i].asarray() for i in np.unique(idx)]
        return float(np.mean([f.mean() for f in frames]))
    except Exception as exc:
        print(f"  [WARN] Helligkeit konnte nicht gelesen werden: {path.name} ({exc})")
        return float("nan")


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def build_table(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        if size_nm not in SIZE_COLORS:
            continue
        fit = r.get("fit_results_MSD")
        D = fit.get("D_um2_per_s") if fit else None
        if D is None:
            continue

        tracks_df = r.get("tracks_df")
        if tracks_df is not None and not tracks_df.empty:
            num_points = len(tracks_df)
            mean_track_length = float(tracks_df.groupby("particle").size().mean())
        else:
            num_points = 0
            mean_track_length = float("nan")

        rows.append({
            "base_name":          r.get("base_name"),
            "size_nm":            float(size_nm),
            "D":                  float(D),
            "D_error":            float(fit.get("D_error", np.nan)),
            "num_tracks":         r.get("num_tracks", np.nan),
            "num_points":         num_points,
            "edge_removed":       r.get("edge_filter_removed", np.nan),
            "mean_track_length":  mean_track_length,
            "mean_brightness":    _mean_brightness(r.get("tif_path")),
            "sigma_loc_nm":       float(fit.get("sigma_loc_nm", np.nan)),
        })
    return pd.DataFrame(rows)


def plot_correlation_grid(table: pd.DataFrame, metrics: list[tuple[str, str]],
                          title: str, save_name: str) -> None:
    """One 2x len(metrics) grid: top row D vs each metric, bottom row D_error
    vs the same metric. Colored by particle size."""
    dls_labels = get_dls_labels()
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.2 * len(metrics), 7.5),
                             constrained_layout=True)

    for col, (key, label) in enumerate(metrics):
        x_all = table[key].to_numpy(dtype=float)

        for row, (y_key, y_label) in enumerate([("D", r"$D$ (µm²/s)"), ("D_error", r"$D_{err}$ (µm²/s)")]):
            ax = axes[row, col]
            y_all = table[y_key].to_numpy(dtype=float)
            r_val = _pearson_r(x_all, y_all)

            for size_nm, (face, edge) in SIZE_COLORS.items():
                sub = table[table["size_nm"] == size_nm]
                if sub.empty:
                    continue
                ax.scatter(sub[key], sub[y_key], s=32, marker="o",
                          facecolor=face, edgecolor=edge,
                          linewidth=0.6, alpha=POINT_ALPHA, zorder=3)

            ax.set_xlabel(label if row == 1 else "", fontsize=8)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=9)
            ax.set_title(label if row == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.text(0.03, 0.93, f"r = {r_val:.2f}" if np.isfinite(r_val) else "r = n/a",
                   transform=ax.transAxes, fontsize=7.5, va="top",
                   bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.8))

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=face,
              markeredgecolor=edge, markersize=7, label=f"{dls_labels.get(size_nm, int(size_nm))} nm")
        for size_nm, (face, edge) in SIZE_COLORS.items()
        if size_nm in table["size_nm"].unique()
    ]
    # Two-line suptitle -- unlike a manually positioned fig.text() at y>1,
    # constrained_layout reserves canvas space for the suptitle automatically,
    # so it's never clipped outside the figure.
    fig.suptitle(f"{title}\nr = Pearson-Korrelationskoeffizient (pro Panel, x vs. y)",
                fontsize=10)
    # "outside" location -- constrained_layout reserves canvas space for it,
    # unlike bbox_to_anchor with y<0, which gets clipped outside the figure
    # unless the plot is saved with bbox_inches="tight" (invisible otherwise).
    fig.legend(handles=legend_elements, loc="outside lower center",
              ncol=len(legend_elements), frameon=False, fontsize=9)

    plt.show()

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        png_path = SAVE_PATH / save_name
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Plot gespeichert: {png_path}")


def main() -> None:
    results = load_results()
    print(f"Cache geladen: {CACHE_FILE} ({len(results)} Dateien)")

    table = build_table(results)
    print(f"{len(table)} Dateien (D0, alle Größen) mit gültigem D-Wert.")
    if table.empty:
        print("Keine Daten -- Abbruch.")
        return

    plot_correlation_grid(
        table, METRICS,
        title="D_MSD und D_error vs. Datei-/Prozessierungs-Metriken (D0, Wasser)",
        save_name="diffusion_correlations_test_d0.png",
    )


if __name__ == "__main__":
    main()
