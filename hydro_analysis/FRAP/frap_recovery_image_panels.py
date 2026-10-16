"""
FRAP image panels: fluorescence recovery of FITC-dextran 150 kDa in 20 mg/mL eADF4(C16) hydrogel
(top row) and in water (bottom row), three post-bleach frames each at t = 0, 10 and 20 s.
Consumer stage: reads cache/frap_v2_phase1.pkl written by frap_analysis_v2.py (run that first) for
the measurement table only, and reloads the raw image series of the two shown measurements with
frap_analysis_v2.load_measurement. Nothing is fitted or recomputed.

t = 0 is the first post-bleach frame (definition of frap_analysis_v2.py). The frame interval does
not give frames exactly at 10 and 20 s, so the frame closest to each target time is shown and
labelled with its real time stamp; images are never interpolated in time. Images are the raw
green-channel counts, full field of view (512 x 512 px, uncropped), linear grey scale with one
contrast for all six panels (percentiles of the last pre-bleach frames of both measurements).
No overlays besides the scale bar.

Selection: per condition the QC-passed (profile) measurement with the largest bleach depth K0,
the same rule as Fig. 1 in frap_figures.py (imported from there, not copied).

Writes OUTPUT_ROOT/figures/fig_frap_recovery_images.png (full width, 1.42:1, 600 dpi) and a sidecar
JSON of the same name with measurement folders, frame indices, time stamps and contrast limits.
Style: Styleguide_Figures_Dissertation.md v2 (thesis.mplstyle, no bbox_inches="tight").
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from hydro_analysis.FRAP.frap_analysis_v2 import DATA_ROOT, OUTPUT_ROOT, load_measurement
from hydro_analysis.FRAP.frap_figures import _focus, _representative

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "frap_v2_phase1.pkl"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
FIGURE_DIR = OUTPUT_ROOT / "figures"
FIGURE_STEM = "fig_frap_recovery_images"

ROWS = [("gel", "Hydrogel"), ("water", "Water")]   # (condition in the cache, row label); decided by user, 2026-10-14
TARGET_TIMES_S = (0.0, 10.0, 20.0)                  # decided by user, 2026-10-14
SELECTION_RULE = "QC-passed (profile) measurement with the largest K0"   # PROPOSED, not approved (E029)

FIG_WIDTH_IN = 6.30                                 # style guide class "full"
RATIO = 1.42                                        # decided by user, 2026-10-14 (whole figure)
CONTRAST_PERCENTILES = (0.5, 99.5)                  # display only, as Fig. 1 in frap_figures.py
CMAP = "gray"
SCALEBAR_UM = 50.0
SCALEBAR_HEIGHT_FRACTION = 0.012                    # of the image height (style guide §9: 1-1.5 %)

# Layout in inches
MARGIN_LEFT = 0.30
MARGIN_RIGHT = 0.63                                 # colorbar, its ticks and label
GAP_X = 0.06
GAP_Y = 0.30                                        # room for time label and panel letter of the lower row
CBAR_GAP = 0.08
CBAR_WIDTH = 0.10


# ── Helpers ────────────────────────────────────────────────────────────────────

def _axes_grid(fig, n_rows: int, n_cols: int):
    W, H = fig.get_size_inches()
    side = (W - MARGIN_LEFT - MARGIN_RIGHT - (n_cols - 1) * GAP_X) / n_cols
    block_h = n_rows * side + (n_rows - 1) * GAP_Y
    bottom0 = (H - block_h) / 2 - 0.05                  # slightly below centre: labels sit above the images
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        y = bottom0 + (n_rows - 1 - r) * (side + GAP_Y)
        for c in range(n_cols):
            x = MARGIN_LEFT + c * (side + GAP_X)
            axes[r, c] = fig.add_axes((x / W, y / H, side / W, side / H))
    x_cb = MARGIN_LEFT + n_cols * side + (n_cols - 1) * GAP_X + CBAR_GAP
    cax = fig.add_axes((x_cb / W, bottom0 / H, CBAR_WIDTH / W, block_h / H))
    return axes, cax


def _scalebar(ax, n_cols_px: int, n_rows_px: int, mpp_um: float) -> None:
    bar_px = SCALEBAR_UM / mpp_um
    bar_h = SCALEBAR_HEIGHT_FRACTION * n_rows_px
    margin = 0.05 * n_cols_px
    x0, y0 = n_cols_px - margin - bar_px, n_rows_px - margin - bar_h
    ax.add_patch(Rectangle((x0, y0), bar_px, bar_h, color="white", lw=0))
    ax.text(x0 + bar_px / 2, y0 - 0.015 * n_rows_px, f"{SCALEBAR_UM:g} µm", color="white", fontsize=8,
            ha="center", va="bottom", path_effects=[pe.withStroke(linewidth=1.0, foreground="black")])


# ── Figure ─────────────────────────────────────────────────────────────────────

def make_figure(results) -> None:
    rows = []
    for condition, label in ROWS:
        rec = _representative(results, condition)
        data = load_measurement(DATA_ROOT / rec["file"])
        t = data["post_t_s"]
        idx = [int(np.argmin(np.abs(t - target))) for target in TARGET_TIMES_S]
        rows.append({"label": label, "rec": rec, "data": data, "idx": idx})

    pre_pixels = np.concatenate([r["data"]["pre"][-1].ravel() for r in rows])
    vmin, vmax = np.percentile(pre_pixels, CONTRAST_PERCENTILES)
    norm = Normalize(vmin=vmin, vmax=vmax)
    mpp = {r["data"]["mpp_um"] for r in rows}
    if len(mpp) != 1:
        raise ValueError(f"Pixel sizes differ between the shown measurements: {mpp}; one scale bar is not valid.")

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN / RATIO), layout="none")
    axes, cax = _axes_grid(fig, len(rows), len(TARGET_TIMES_S))
    meta_rows = []
    for r, row in enumerate(rows):
        data, t = row["data"], row["data"]["post_t_s"]
        n_rows_px, n_cols_px = data["post"].shape[1:]
        for c, i in enumerate(row["idx"]):
            ax = axes[r, c]
            ax.imshow(data["post"][i], cmap=CMAP, norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.minorticks_off()
            ax.set_title(f"t = {t[i]:.1f} s", fontsize=8, fontweight="normal", pad=3)
            ax.text(0.0, 1.02, "ABCDEF"[r * len(row["idx"]) + c], transform=ax.transAxes, fontsize=10,
                    fontweight="bold", ha="left", va="bottom")
            if c == len(row["idx"]) - 1:
                _scalebar(ax, n_cols_px, n_rows_px, data["mpp_um"])
        axes[r, 0].set_ylabel(row["label"], fontsize=9, labelpad=4)
        rec = row["rec"]
        meta_rows.append({"row": row["label"], "measurement_id": rec["measurement_id"],
                          "folder": str(DATA_ROOT / rec["file"]), "K0": float(rec["K0"]),
                          "frame_indices_post": row["idx"], "t_shown_s": [float(t[i]) for i in row["idx"]],
                          "frame_interval_s_first_phase": float(data["frame_interval_s"]),
                          "image_shape_px": [int(n_rows_px), int(n_cols_px)], "mpp_um": float(data["mpp_um"])})

    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, extend="both")
    cb.set_label("Intensity (counts)", fontsize=9)
    cb.ax.tick_params(labelsize=8, which="both", direction="in")
    cb.ax.minorticks_off()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{FIGURE_STEM}.png"
    fig.savefig(path, format="png", dpi=600)
    plt.close(fig)
    sidecar = {
        "figure": path.name, "created": f"{datetime.now():%Y-%m-%d %H:%M}", "cache": str(CACHE_PATH),
        "size_in": [FIG_WIDTH_IN, round(FIG_WIDTH_IN / RATIO, 3)], "selection_rule": SELECTION_RULE,
        "candidates": {cond: list(results.loc[results["condition"] == cond, "measurement_id"]) for cond, _ in ROWS},
        "time_rule": f"frame closest to each of {list(TARGET_TIMES_S)} s after the first post-bleach frame; "
                     "label = real time stamp; no temporal interpolation",
        "contrast": f"linear {CMAP}, percentiles {CONTRAST_PERCENTILES} of the pooled last pre-bleach frames of both "
                    "measurements, identical in all panels; colorbar arrows mark clipped values",
        "contrast_vmin_vmax_counts": [float(vmin), float(vmax)],
        "image_processing": "none (raw green-channel counts, full field of view, no crop, no smoothing)",
        "scalebar_um": SCALEBAR_UM, "rows": meta_rows,
    }
    (FIGURE_DIR / f"{FIGURE_STEM}.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    print(f"Written {path}")
    for m in meta_rows:
        print(f"  {m['row']}: {m['measurement_id']} ({m['folder']}), frames {m['frame_indices_post']}, "
              f"t = {[round(v, 3) for v in m['t_shown_s']]} s")
    print(f"  contrast {vmin:.1f}-{vmax:.1f} counts")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"{CACHE_PATH} not found; run frap_analysis_v2.py first.")
    with open(CACHE_PATH, "rb") as fh:
        cache = pickle.load(fh)
    results = _focus(cache["results"])
    with plt.style.context(str(STYLE_PATH)):
        make_figure(results)


if __name__ == "__main__":
    main()
