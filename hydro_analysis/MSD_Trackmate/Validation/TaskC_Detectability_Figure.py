"""
Composite figure -- very good / challenging / no-particle 35 nm detection
examples, raw crop + 3D intensity landscape + LoG detection response.

Pure consumer: reads cache/detectability_35nm.pkl (TaskC_Detectability_Compute.py).
Never touches raw TIFF/XML. Example selection (identical across this script
and TaskC_Detectability_Distribution.py) is done by _detectability_selection.select_examples().

4 rows x 3 columns:
  1) Raw fluorescence crop (2D), IDENTICAL color scale across all 3 columns
     (computed once as the shared min/max of the three selected raw crops --
     never independently normalized, so relative signal strength stays
     visually honest).
  2) Raw fluorescence, 3D intensity surface, same shared z-limits as row 1.
  3) LoG detection-response crop (2D), its own shared color scale across
     the 3 columns (independent of the raw scale).
  4) LoG response, 3D surface, shared z-limits matching row 3.

No detection-circle overlay in this primary figure (raw evidence stays
unmodified) -- see SAVE_ANNOTATED_VARIANT below for an optional separate
annotated version.

The crop window is the "apparent fluorescence/PSF extent in image pixels,"
not a claim about the physical particle diameter (35 nm is far below the
diffraction limit).

Run TaskC_Detectability_Compute.py first (or after any raw-data change) to
refresh the cache.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\Detectability_Composite\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3d projection

from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation._detectability_selection import select_examples

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent.parent / "cache" / "detectability_35nm.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "Detectability_Composite"

SAVE_ANNOTATED_VARIANT = True   # optional: a second figure with a marker at the detected center

COLUMN_ORDER = ["good", "challenging", "no_particle"]
COLUMN_TITLES = {"good": "Very good", "challenging": "Challenging", "no_particle": "No particle"}


def load_df() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {CACHE_FILE}\nBitte zuerst TaskC_Detectability_Compute.py ausführen."
        )
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def _add_scalebar(ax, mpp: float, extent_px: int, length_um: float = 1.0) -> None:
    length_px = length_um / mpp
    x0 = extent_px * 0.06
    y0 = extent_px * 0.92
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", linewidth=3, solid_capstyle="butt")
    ax.text(x0 + length_px / 2, y0 - extent_px * 0.04, f"{length_um:g} µm",
            color="white", ha="center", va="bottom", fontsize=8)


def build_composite(examples: dict, annotate: bool = False) -> plt.Figure:
    fig = plt.figure(figsize=(10.5, 13.0), constrained_layout=True)
    n = len(COLUMN_ORDER)

    raw_crops = [examples[k]["core_crop"].astype(np.float64) for k in COLUMN_ORDER]
    log_crops = [examples[k]["log_crop"].astype(np.float64) for k in COLUMN_ORDER]
    raw_vmin, raw_vmax = float(np.min(raw_crops)), float(np.max(raw_crops))
    log_vmin, log_vmax = float(np.min(log_crops)), float(np.max(log_crops))

    extent_px = raw_crops[0].shape[0]
    yy, xx = np.mgrid[0:extent_px, 0:extent_px]

    for col_idx, key in enumerate(COLUMN_ORDER):
        row = examples[key]
        raw_crop = raw_crops[col_idx]
        log_crop = log_crops[col_idx]

        # Row 1: raw 2D
        ax1 = fig.add_subplot(4, n, col_idx + 1)
        ax1.imshow(raw_crop, cmap="gray", vmin=raw_vmin, vmax=raw_vmax, origin="upper")
        ax1.set_xticks([])
        ax1.set_yticks([])
        title = COLUMN_TITLES[key]
        if key != "no_particle":
            title += f"\nSNR={row['snr']:.2g}"
        else:
            title += f"\nbackground SD={row['background_sd']:.2g}"
        ax1.set_title(title, fontsize=10)
        if col_idx == 0:
            _add_scalebar(ax1, row["mpp"], extent_px)
        if annotate and key != "no_particle":
            half = extent_px // 2
            ax1.plot(half, half, marker="+", color="#ff3333", markersize=12, markeredgewidth=1.5)

        # Row 2: raw 3D
        ax2 = fig.add_subplot(4, n, n + col_idx + 1, projection="3d")
        ax2.plot_surface(xx, yy, raw_crop, cmap="gray", vmin=raw_vmin, vmax=raw_vmax,
                         linewidth=0, antialiased=True)
        ax2.set_zlim(raw_vmin, raw_vmax)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zlabel("Intensity", fontsize=7)

        # Row 3: LoG response 2D
        ax3 = fig.add_subplot(4, n, 2 * n + col_idx + 1)
        ax3.imshow(log_crop, cmap="magma", vmin=log_vmin, vmax=log_vmax, origin="upper")
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title(f"LoG response (radius={row['radius_px']:.2g} px)", fontsize=8)

        # Row 4: LoG response 3D
        ax4 = fig.add_subplot(4, n, 3 * n + col_idx + 1, projection="3d")
        ax4.plot_surface(xx, yy, log_crop, cmap="magma", vmin=log_vmin, vmax=log_vmax,
                         linewidth=0, antialiased=True)
        ax4.set_zlim(log_vmin, log_vmax)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_zlabel("LoG response", fontsize=7)

    label_nm = examples["good"]["particle_size_nm"]
    condition = examples["good"]["condition"]
    fig.suptitle(f"Particle detectability, {label_nm:.0f} nm nominal, {condition}\n"
                 f"(crop window = apparent fluorescence/PSF extent in pixels, not physical particle diameter)",
                 fontsize=11, fontweight="semibold")
    return fig


def build_selection_table(examples: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, row in examples.items():
        n_in_movie = int((df["movie"] == row["movie"]).sum())
        rows.append({
            "role": key, "movie": row["movie"], "frame": row["frame"],
            "particle_size_nm": row["particle_size_nm"], "condition": row["condition"],
            "n_detections_in_movie": n_in_movie,
            "snr": row["snr"], "background_sd": row["background_sd"],
            "log_peak": row["log_peak"], "log_snr": row["log_snr"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_df()
    print(f"Cache geladen: {CACHE_FILE} ({len(df)} Crops)")

    examples = select_examples(df)
    table = build_selection_table(examples, df)
    print(table.to_string(index=False))

    with plt.rc_context(_RC):
        fig = build_composite(examples, annotate=False)
        plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(SAVE_PATH / "detectability_composite.png", dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {SAVE_PATH / 'detectability_composite.png'}")

        if SAVE_ANNOTATED_VARIANT:
            fig_ann = build_composite(examples, annotate=True)
            plt.show()
            if SAVE_PATH is not None:
                fig_ann.savefig(SAVE_PATH / "detectability_composite_annotated.png", dpi=600, bbox_inches="tight")
                print(f"Plot gespeichert: {SAVE_PATH / 'detectability_composite_annotated.png'}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "detectability_selection_table.csv"
        table.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
