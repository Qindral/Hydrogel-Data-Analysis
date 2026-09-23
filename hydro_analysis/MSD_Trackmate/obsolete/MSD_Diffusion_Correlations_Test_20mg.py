"""
TEST SCRIPT — exploratory correlations between per-file D_MSD (and its fit
error) and various per-file quality/processing metrics, for the 35 nm and
50 nm particles (20 mg/mL C16 hydrogel), colored by particle size and
condition (Surface loading / Injection).

Not a publication figure -- a quick diagnostic grid to check whether D or
its error is driven by acquisition/processing artifacts rather than real
physics: does D drift with how many trajectories/data points a file has,
how much edge-filtering was applied, how long the average trajectory is, or
how bright the raw camera image was.

Reads msd_20mg_files.pkl (produced by MSD_FromTrackmate_20mg.py) for the
per-file D/D_error/num_tracks/tracks_df/edge_filter_removed fields, and
re-derives:
  - num_points            = len(tracks_df)                      (rows = detections)
  - mean_track_length     = tracks_df.groupby('particle').size().mean()
  - edge_filter_removed   = detections dropped by remove_edge_artifacts()
                            (core.io.single_file_data), i.e. "processed away"
  - mean_brightness       = mean pixel value of a sample of frames from the
                            raw TIFF (tif_path), read via tifffile page-by-page
                            so full stacks don't have to be loaded into memory
  - sigma_loc_nm          = localization-uncertainty term from the per-file
                            power-law MSD fit itself (fit_results_MSD), i.e.
                            how noisy the fit thinks each particle's position
                            estimate was -- not derived from raw pixels

Additionally reads the *full* TrackMate project XMLs (detector/tracker
settings -- not exported into the simplified "_Tracks.xml" files used above)
from the "analysis" (20 nm) / "Analysis_new" (50 nm) folders next to Tracks/
Tracks_new, matched to each file by name (base_name with "_Tracks" and
spaces stripped). These full save files can be tens/hundreds of MB, so only
the last ~1 MB is read (the <Settings> block is always near the end, after
the spot/track data) instead of loading the whole file:
  - radius, threshold                = LoG DetectorSettings
  - initial_quality                  = InitialSpotFilter QUALITY threshold
  - linking_max_distance             = LAP tracker frame-to-frame linking
  - gap_closing_max_distance,
    max_frame_gap                    = LAP tracker gap-closing

Two separate 2x6 grids are produced: one for the file/processing metrics
above, one for the TrackMate settings. Each panel prints its Pearson r in
the corner as a quick eyeball check for correlation strength.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tifffile

from hydro_analysis.core.io import condition_label_from_filename

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_files.pkl"
SAVE_PATH: Path | None = None  # test script -- set a Path to also save a PNG

TARGET_NOMINAL_SIZES_NM = {20.0}   # -> 35 nm only (DLS-measured); add 50.0 back for both sizes
BRIGHTNESS_MAX_FRAMES = 20               # sampled evenly across the stack

# Full TrackMate project XMLs (detector/tracker settings), next to Tracks/Tracks_new
ANALYSIS_DIRS = {
    20.0: Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\analysis"),
    #50.0: Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Analysis_new"),
}
SETTINGS_TAIL_BYTES = 1_000_000   # <Settings> is always near the end of the file

COLOR_SURFACE      = "#3B8C8C"   # teal -- Surface loading
COLOR_SURFACE_DARK = "#2A6666"
COLOR_INJECTION      = "#D98C3D"   # orange -- Injection
COLOR_INJECTION_DARK = "#A6672D"
SIZE_MARKER = {20.0: "s", 50.0: "o"}   # 35 nm -> square, 50 nm -> circle
SIZE_LABEL  = {20.0: "35 nm", 50.0: "50 nm"}
CONDITION_STYLE = {
    "Surface loading": {"face": COLOR_SURFACE,   "edge": COLOR_SURFACE_DARK},
    "Injection":       {"face": COLOR_INJECTION, "edge": COLOR_INJECTION_DARK},
}
POINT_ALPHA = 0.6

METRICS = [
    ("num_tracks",        "Anzahl Trajektorien"),
    ("num_points",        "Anzahl Datenpunkte"),
    ("edge_removed",      "Prozessierte Punkte (Edge-Filter entfernt)"),
    ("mean_track_length", "Mittlere Trajektorienlänge (Frames)"),
    ("mean_brightness",   "Mittlere Kamera-Helligkeit (a.u.)"),
    ("sigma_loc_nm",      "Lokalisationsunsicherheit $\\sigma_{loc}$ (nm)"),
]

METRICS_TRACKMATE = [
    ("radius",                   "Detektor RADIUS (µm)"),
    ("threshold",                "Detektor THRESHOLD"),
    ("initial_quality",          "Initial-Spot-Filter QUALITY"),
    ("linking_max_distance",     "LAP LINKING_MAX_DISTANCE (µm)"),
    ("gap_closing_max_distance", "LAP GAP_CLOSING_MAX_DISTANCE (µm)"),
    ("max_frame_gap",            "LAP MAX_FRAME_GAP (Frames)"),
]


def load_results() -> dict:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst MSD_FromTrackmate_20mg.py ausführen."
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


def _settings_path(base_name: str, size_nm: float) -> Path | None:
    """Map a "..._Tracks.xml" (or "..._processed_Tracks.xml") base_name to its
    full TrackMate project XML in the analysis/Analysis_new folder, e.g.
    "20nm_1d_A2_03_Tracks.xml" -> ".../analysis/20nm_1d_A2_03.xml" and
    "50 nm A2_2d_01_Tracks.xml" -> ".../Analysis_new/50nmA2_2d_01.xml"."""
    folder = ANALYSIS_DIRS.get(size_nm)
    if folder is None:
        return None
    name = re.sub(r"_Tracks\.xml$", ".xml", base_name).replace(" ", "")
    path = folder / name
    return path if path.exists() else None


def _parse_trackmate_settings(path: Path, tail_bytes: int = SETTINGS_TAIL_BYTES) -> dict:
    """Extract detector/tracker settings from a full TrackMate project XML.
    The <Settings> block always sits near the very end of the file (after
    the potentially huge <Model> spot/track data), so only the last
    `tail_bytes` of the file are read/decoded instead of the whole thing."""
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as f:
            if file_size > tail_bytes:
                f.seek(file_size - tail_bytes)
            data = f.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"  [WARN] TrackMate-Settings konnten nicht gelesen werden: {path.name} ({exc})")
        return {}

    m = re.search(r"<Settings>.*?</Settings>", data, re.S)
    if not m:
        return {}
    block = m.group(0)

    out: dict = {}
    det = re.search(r'<DetectorSettings[^>]*RADIUS="([\d.]+)"[^>]*THRESHOLD="([\d.eE+-]+)"', block)
    if det:
        out["radius"] = float(det.group(1))
        out["threshold"] = float(det.group(2))
    qual = re.search(r'<InitialSpotFilter feature="QUALITY" value="([\d.eE+-]+)"', block)
    if qual:
        out["initial_quality"] = float(qual.group(1))
    link = re.search(r'<Linking LINKING_MAX_DISTANCE="([\d.]+)"', block)
    if link:
        out["linking_max_distance"] = float(link.group(1))
    gap = re.search(r'<GapClosing[^>]*GAP_CLOSING_MAX_DISTANCE="([\d.]+)"[^>]*MAX_FRAME_GAP="(\d+)"', block)
    if gap:
        out["gap_closing_max_distance"] = float(gap.group(1))
        out["max_frame_gap"] = int(gap.group(2))
    return out


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def build_table(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        if size_nm not in TARGET_NOMINAL_SIZES_NM:
            continue
        fit = r.get("fit_results_MSD")
        D = fit.get("D_um2_per_s") if fit else None
        if D is None:
            continue
        condition = condition_label_from_filename(r.get("base_name", ""))
        if condition not in CONDITION_STYLE:
            continue

        tracks_df = r.get("tracks_df")
        if tracks_df is not None and not tracks_df.empty:
            num_points = len(tracks_df)
            mean_track_length = float(tracks_df.groupby("particle").size().mean())
        else:
            num_points = 0
            mean_track_length = float("nan")

        base_name = r.get("base_name", "")
        settings_path = _settings_path(base_name, float(size_nm))
        tm_settings = _parse_trackmate_settings(settings_path) if settings_path else {}

        rows.append({
            "base_name":          base_name,
            "size_nm":            float(size_nm),
            "condition":          condition,
            "D":                  float(D),
            "D_error":            float(fit.get("D_error", np.nan)),
            "num_tracks":         r.get("num_tracks", np.nan),
            "num_points":         num_points,
            "edge_removed":       r.get("edge_filter_removed", np.nan),
            "mean_track_length":  mean_track_length,
            "mean_brightness":    _mean_brightness(r.get("tif_path")),
            "sigma_loc_nm":       float(fit.get("sigma_loc_nm", np.nan)),
            "radius":                   tm_settings.get("radius", np.nan),
            "threshold":                tm_settings.get("threshold", np.nan),
            "initial_quality":          tm_settings.get("initial_quality", np.nan),
            "linking_max_distance":     tm_settings.get("linking_max_distance", np.nan),
            "gap_closing_max_distance": tm_settings.get("gap_closing_max_distance", np.nan),
            "max_frame_gap":            tm_settings.get("max_frame_gap", np.nan),
        })
    return pd.DataFrame(rows)


def plot_correlation_grid(table: pd.DataFrame, metrics: list[tuple[str, str]],
                          title: str, save_name: str) -> None:
    """One 2x len(metrics) grid: top row D vs each metric, bottom row D_error
    vs the same metric. Colored by condition, marker shape by particle size."""
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.2 * len(metrics), 7.5),
                             constrained_layout=True)

    for col, (key, label) in enumerate(metrics):
        x_all = table[key].to_numpy(dtype=float)

        for row, (y_key, y_label) in enumerate([("D", r"$D$ (µm²/s)"), ("D_error", r"$D_{err}$ (µm²/s)")]):
            ax = axes[row, col]
            y_all = table[y_key].to_numpy(dtype=float)
            r_val = _pearson_r(x_all, y_all)

            for condition, cond_style in CONDITION_STYLE.items():
                for size_nm, marker in SIZE_MARKER.items():
                    sub = table[(table["size_nm"] == size_nm) & (table["condition"] == condition)]
                    if sub.empty:
                        continue
                    ax.scatter(sub[key], sub[y_key], s=32, marker=marker,
                              facecolor=cond_style["face"], edgecolor=cond_style["edge"],
                              linewidth=0.6, alpha=POINT_ALPHA, zorder=3)

            ax.set_xlabel(label if row == 1 else "", fontsize=8)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=9)
            ax.set_title(label if row == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.text(0.03, 0.93, f"r = {r_val:.2f}" if np.isfinite(r_val) else "r = n/a",
                   transform=ax.transAxes, fontsize=7.5, va="top",
                   bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.8))

    # Only legend entries for sizes/conditions actually present in the data
    # (e.g. just "35 nm" when TARGET_NOMINAL_SIZES_NM excludes 50 nm).
    present_sizes = set(table["size_nm"].unique())
    present_conditions = set(table["condition"].unique())
    legend_elements = [
        Line2D([0], [0], marker=marker, color="w", markerfacecolor="#888888",
              markeredgecolor="#444444", markersize=7, label=SIZE_LABEL[size_nm])
        for size_nm, marker in SIZE_MARKER.items() if size_nm in present_sizes
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cond_style["face"],
              markeredgecolor=cond_style["edge"], markersize=7, label=condition)
        for condition, cond_style in CONDITION_STYLE.items() if condition in present_conditions
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
    print(f"{len(table)} Dateien (35/50 nm, Surface loading/Injection) mit gültigem D-Wert.")
    if table.empty:
        print("Keine Daten -- Abbruch.")
        return

    n_settings = table["radius"].notna().sum()
    print(f"{n_settings}/{len(table)} Dateien mit gefundenen TrackMate-Settings.")

    plot_correlation_grid(
        table, METRICS,
        title="D_MSD und D_error vs. Datei-/Prozessierungs-Metriken (35/50 nm, 20 mg/mL)",
        save_name="diffusion_correlations_test_20mg.png",
    )
    plot_correlation_grid(
        table, METRICS_TRACKMATE,
        title="D_MSD und D_error vs. TrackMate-Einstellungen (35/50 nm, 20 mg/mL)",
        save_name="diffusion_correlations_trackmate_settings_20mg.png",
    )


if __name__ == "__main__":
    main()
