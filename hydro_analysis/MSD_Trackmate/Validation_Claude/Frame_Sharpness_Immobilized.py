"""
Per-frame focus/sharpness check for a raw immobilized-particle TIFF movie --
flags frames that are out of focus (e.g. from stage/focus drift during an
otherwise-static control measurement), which is a plausible confound for
the sigma_loc estimates in TaskA_Immobilized_Compute.py: a defocused frame
inflates the apparent positional spread of an otherwise perfectly static
particle.

Plot-only, single-file tool: reads TIF_PATH frame-by-frame via
tifffile.imread(path, key=frame_idx) (never the whole stack at once, same
convention as TaskC_Detectability_Compute.py / PSF_Analysis.py)
and does not touch or write any core/ cache. Not wired into the
TrackMate-based pipeline; run standalone against any raw movie by changing
TIF_PATH.

Method
------
Sharpness score = variance of the Laplacian (Pech-Pacheco et al., 1999),
the standard fast focus-quality measure used in microscope autofocus and
image-QC pipelines: in-focus structures have sharp intensity gradients, so
the second derivative (Laplacian) has high local magnitude and therefore
high variance across the frame; defocus blurs those gradients out and the
variance drops. Each frame is mildly Gaussian-presmoothed first
(GAUSSIAN_PRESMOOTH_SIGMA, default 1 px) -- raw camera shot noise is itself
high-frequency and would otherwise dominate the Laplacian variance,
swamping the lower-frequency defocus signal actually being measured.

A frame is flagged "blurry" when its score falls more than MAD_K robust
standard deviations (1.4826 * median absolute deviation, a normal-
consistent, outlier-resistant estimator) below the movie's own median
score. The threshold is computed per-movie/per-run, not a fixed absolute
value, since raw sharpness scores are not comparable across movies with
different exposure, gain or particle brightness.

Outputs (Auswertungsbilder\\Immobilized_Sharpness_FocusCheck\\):
  sharpness_<stem>.png  sharpness-vs-frame-index plot, flagged frames
                        marked, plus thumbnails of the sharpest and the
                        blurriest frame for a visual sanity check
  sharpness_<stem>.csv  frame, sharpness_score, is_blurry -- one row per frame
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
TIF_PATH = Path(r"E:\Daten Promotion Sicherung\Diffusion in Hydrogel Data\20mg_20nm\Inj\B3_20nm_20mg_1d_zentral_3.tif")
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Immobilized_Sharpness_FocusCheck"

GAUSSIAN_PRESMOOTH_SIGMA = 1.0   # px, suppresses shot noise before the Laplacian
MAD_K = 3.0                      # blurry threshold: median - MAD_K * (1.4826 * MAD)

COLOR_SCORE = "#0000da"
COLOR_FLAG = "#da0000"


def _sharpness_score(frame: np.ndarray, presmooth_sigma: float) -> float:
    frame = frame.astype(np.float64)
    if presmooth_sigma > 0:
        frame = cv2.GaussianBlur(frame, ksize=(0, 0), sigmaX=presmooth_sigma)
    return float(cv2.Laplacian(frame, cv2.CV_64F).var())


def compute_sharpness_series(tif_path: Path, presmooth_sigma: float = GAUSSIAN_PRESMOOTH_SIGMA) -> np.ndarray:
    with tifffile.TiffFile(tif_path) as tif:
        series = tif.series[0]
        n_frames = series.shape[0] if series.ndim == 3 else 1

    scores = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = tifffile.imread(tif_path, key=i)
        scores[i] = _sharpness_score(frame, presmooth_sigma)
    return scores


def flag_blurry_frames(scores: np.ndarray, mad_k: float = MAD_K) -> tuple[np.ndarray, float]:
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(scores))
    threshold = median - mad_k * robust_std
    return scores < threshold, threshold


def plot_sharpness(tif_path: Path, scores: np.ndarray, is_blurry: np.ndarray, threshold: float) -> plt.Figure:
    frame_idx = np.arange(len(scores))
    sharpest_i = int(np.argmax(scores))
    blurriest_i = int(np.argmin(scores))

    fig = plt.figure(figsize=(7.15, 5.00), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.0, 1.0])
    ax = fig.add_subplot(gs[:, 0:2])
    ax_sharp = fig.add_subplot(gs[0, 2])
    ax_blur = fig.add_subplot(gs[1, 2])

    ax.plot(frame_idx, scores, color=COLOR_SCORE, linewidth=0.9, zorder=2)
    ax.scatter(frame_idx[is_blurry], scores[is_blurry], s=18, color=COLOR_FLAG,
               edgecolor="none", zorder=3, label=f"blurry (N={int(is_blurry.sum())})")
    ax.axhline(threshold, color="black", linewidth=1.0, linestyle="--", zorder=1,
               label=r"threshold (median $- k \cdot$ MAD)")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Sharpness (variance of Laplacian, a.u.)")
    ax.set_title(tif_path.name, fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    for a, i, label in ((ax_sharp, sharpest_i, "sharpest"), (ax_blur, blurriest_i, "blurriest")):
        frame = tifffile.imread(tif_path, key=i)
        vmin, vmax = np.percentile(frame, (1.0, 99.5))
        a.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.6)
        a.set_title(f"{label} (frame {i})", fontsize=8)

    return fig


def main() -> None:
    if not TIF_PATH.exists():
        raise FileNotFoundError(f"TIF_PATH nicht gefunden: {TIF_PATH}")

    scores = compute_sharpness_series(TIF_PATH)
    is_blurry, threshold = flag_blurry_frames(scores)
    blurry_idx = np.flatnonzero(is_blurry)

    print(f"{TIF_PATH.name}: {len(scores)} Frames")
    print(f"  Median Schärfe = {np.median(scores):.4g}, Schwellwert = {threshold:.4g}")
    print(f"  {len(blurry_idx)} Frames als unscharf markiert")
    if len(blurry_idx):
        print(f"  Frame-Indizes: {blurry_idx.tolist()}")

    with plt.rc_context(_RC):
        fig = plot_sharpness(TIF_PATH, scores, is_blurry, threshold)
        plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / f"sharpness_{TIF_PATH.stem.replace(' ', '_')}.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "frame": np.arange(len(scores)),
            "sharpness_score": scores,
            "is_blurry": is_blurry,
        })
        csv_path = SAVE_PATH / f"sharpness_{TIF_PATH.stem.replace(' ', '_')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
