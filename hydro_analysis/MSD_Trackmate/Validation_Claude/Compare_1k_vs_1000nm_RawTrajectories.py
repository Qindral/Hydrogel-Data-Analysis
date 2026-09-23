"""
Compare_1k_vs_1000nm_RawTrajectories.py

Diagnostic, not part of the main compute/consumer pipeline. Companion to
Loc_Error_Analyse_immob_particle.py's clean_movie_tracks() investigation:
plots "1000nm.xml" and "1k.xml" (2026.01.31_immobilized\\Analysis_immob_NEW\\)
side by side using trajectories taken as close to raw as possible, to check
whether the star/cross-shaped excursions seen in the fully cleaned "1k.xml"
panel (compare_1k_vs_1000nm_trajectories.png) survive without any of
clean_movie_tracks()'s manipulation, or were introduced/masked by it.

Deliberately skips every step of that cleaning pipeline:
  - no edge-artifact filtering/splitting (core.io.remove_edge_artifacts())
  - no step-outlier splitting
  - no outlier-range trajectory removal
  - no close-pair contamination removal
  - no drift subtraction (moot here anyway -- see that script, it is
    currently commented out)
Trajectories come directly from read_trackmate_session()'s own
FilteredTracks/AllTracks edge-walk reconstruction -- the only manipulation
applied here at all.

The one filter applied: a trajectory must cover at least MIN_FRAME_FRACTION
(4/5) of the movie's total frame count (max spot frame + 1, from ALL
detected spots, not just tracked ones) to be kept. This is a track-length/
completeness filter only -- it does not look at position, step size, or
trajectory shape, so it cannot itself manufacture directional artifacts.

mpp comes from Loc_Error_Analyse_immob_particle.py's _calibration_for() --
pure calibration lookup, independent of the tracking/cleaning pipeline.

Writes one PNG: Auswertungsbilder\\LocError_Immobilized\\
compare_1k_vs_1000nm_trajectories_raw.png (600 dpi, Style_guide.txt
convention). No cache read/written.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude.Loc_Error_Analyse_immob_particle import (
    IMMOB_FOLDERS, SAVE_PATH, COLOR_TRK, _calibration_for,
)
from hydro_analysis.MSD_Trackmate.Validation_GPT.trackmate_session import read_trackmate_session

# ── Configuration ──────────────────────────────────────────────────────────────
MOVIES = ["1000nm", "1k"]
TITLES = {"1000nm": "1000nm.xml", "1k": "1k.xml"}
MIN_FRAME_FRACTION = 0.8   # keep only particles present in >= 4/5 of the movie's frames


def _raw_tracks_nm(name: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], float, int, int]:
    """Recentered (x - x[0], y - y[0]) trajectories in nm, straight from the
    raw session XML -- see module docstring for exactly what is and is not
    applied. Returns (tracks, mpp, n_frames, n_tracks_total_before_filter)."""
    folder = next(iter(IMMOB_FOLDERS.values()))
    xml_path = folder / f"{name}.xml"
    calib = _calibration_for(xml_path)
    mpp = calib["mpp"]

    session = read_trackmate_session(xml_path)
    n_frames = int(session.spots["frame"].max()) + 1
    min_len = MIN_FRAME_FRACTION * n_frames

    all_trajectory_ids = session.trajectories["trajectory_id"].nunique()
    tracks = []
    for _, g in session.trajectories.groupby("trajectory_id"):
        if len(g) < min_len:
            continue
        g = g.sort_values("frame")
        x_nm = g["x"].to_numpy(dtype=float) * mpp * 1000.0
        y_nm = g["y"].to_numpy(dtype=float) * mpp * 1000.0
        tracks.append((x_nm - x_nm[0], y_nm - y_nm[0]))
    return tracks, mpp, n_frames, all_trajectory_ids


def main() -> None:
    per_movie = {}
    for name in MOVIES:
        tracks, mpp, n_frames, n_total = _raw_tracks_nm(name)
        per_movie[name] = (tracks, mpp, n_frames)
        print(f"  {TITLES[name]}: {n_frames} frames, {len(tracks)}/{n_total} trajectories "
              f">= {MIN_FRAME_FRACTION:.0%} frame coverage ({MIN_FRAME_FRACTION * n_frames:.0f} frames)")

    max_abs = max(
        max(np.abs(x).max(), np.abs(y).max())
        for tracks, _, _ in per_movie.values() for x, y in tracks
    )
    limit = max_abs * 1.1

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.15 * 2, 5.00), constrained_layout=True)
        for ax, name in zip(axes, MOVIES):
            tracks, mpp, n_frames = per_movie[name]
            for x_nm, y_nm in tracks:
                ax.plot(x_nm, y_nm, color=COLOR_TRK, linewidth=0.6, alpha=0.5, rasterized=True)
            ax.plot(0, 0, marker="+", color="red", markersize=8, markeredgewidth=1.2, zorder=5)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$\Delta x$ (nm)")
            ax.set_ylabel(r"$\Delta y$ (nm)")
            ax.set_title(f"{TITLES[name]} (mpp={mpp:.3g}), N={len(tracks)}", fontsize=10)

        fig.suptitle(
            f"Raw trajectories (no edge/outlier/close-pair/drift cleaning), "
            f"own starting point, $\\geq${MIN_FRAME_FRACTION:.0%} frame coverage",
            fontsize=11, fontweight="semibold")

        plt.show()
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        out_path = SAVE_PATH / "compare_1k_vs_1000nm_trajectories_raw.png"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(f"Plot gespeichert: {out_path}")


if __name__ == "__main__":
    main()
