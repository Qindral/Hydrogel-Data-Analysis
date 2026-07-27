"""
Per-file publication MSD figures — water (D0) reference measurements.

Same figure layout as MSD_per_file_publication.py / MSD_per_file_20mg.py:
  Main   iMSD · eMSD · Stokes-Einstein D₀ · power-law fit
  Inset (upper-right)  normalised MSD
  Inset (lower-right)  raw trajectories, plain, scale bar only
  Annotation  d = <label> nm

Reads the per-file results already computed by MSD_FromTrackmate_D0.py
(msd_d0_results.pkl) instead of reprocessing the raw TrackMate XML files —
that script always recomputes fresh and overwrites the pickle on every run,
so run it first (or after any change to the raw data) to refresh this input.

Tracks are pre-filtered via core.io.single_file_data() -> remove_edge_artifacts():
detections within 3% of the frame border are dropped and trajectories split at
the gap, to correct spurious TrackMate linking of near-edge detections.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_analysis.core.io import (
    get_dls_reference_maps, get_dls_sizes, get_dls_labels,
    condition_label_from_filename, parse_rec_comment_metadata, parse_chamber_day_repeat,
)
from hydro_analysis.core.analysis import DEFAULT_MSD_FIT_POINTS
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import plot_msd_single_file

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent / "cache" / "msd_d0_results.pkl"

SAVE_PATH =  Path(
    rf"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung"
    rf"\PerFile_{pd.Timestamp.now().strftime('%Y%m%d')}")
MSD_FIT_POINTS   = DEFAULT_MSD_FIT_POINTS
FPS_SMALL_TARGET = 60.0
FPS_LARGE_TARGET = 20.0


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CACHE_FILE.exists():
        print(f"Kein Cache gefunden: {CACHE_FILE}")
        print("Bitte zuerst MSD_FromTrackmate_D0.py ausführen.")
        return
    with open(CACHE_FILE, "rb") as f:
        results = pickle.load(f)
    print(f"Cache geladen: {CACHE_FILE} ({len(results)} Dateien)")

    dls_sizes     = get_dls_sizes()
    dls_labels    = get_dls_labels()
    size_override = get_dls_reference_maps()["size_override_nm"]

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    n_processed = 0
    n_skipped   = 0

    ordered = sorted(
        results.values(),
        key=lambda r: (r.get("particle_size_nm") or 0.0, r.get("base_name", "")),
    )

    for rd in ordered:
        size_nm = rd.get("particle_size_nm")
        fit     = rd.get("fit_results_MSD")
        if size_nm is None or fit is None:
            n_skipped += 1
            continue

        mapped_size = dls_sizes.get(size_nm, size_override.get(size_nm, size_nm))
        label_nm    = dls_labels.get(size_nm, int(size_nm))
        D_theo      = calculate_theoretical_diffusion(particle_size_nm=mapped_size)
        target_fps  = FPS_SMALL_TARGET if size_nm < 200 else FPS_LARGE_TARGET

        base = rd.get("base_name", "")
        fps  = rd.get("fps")
        print(f"  [{int(size_nm)} nm  {fps:.0f} fps] {base}")

        tracks_df = rd.get("tracks_df")
        if tracks_df is not None and not tracks_df.empty:
            track_lengths          = tracks_df.groupby("particle").size()
            num_detections_total   = len(tracks_df)
            mean_track_length      = track_lengths.mean()
        else:
            num_detections_total   = 0
            mean_track_length      = np.nan

        duration_s   = rd["num_frames"] / fps if fps else np.nan
        rec_meta     = parse_rec_comment_metadata(rd.get("rec_path"))
        chamber_meta = parse_chamber_day_repeat(base)

        summary_rows.append({
            "file":                          base,
            "size_nm":                       int(size_nm),
            "label_nm":                      label_nm,
            "condition":                     condition_label_from_filename(base),
            "chamber":                       chamber_meta["chamber"],
            "day":                           chamber_meta["day"],
            "repeat":                        chamber_meta["repeat"],
            "fps":                           fps,
            "mpp_um_per_px":                 rd.get("mpp"),
            "D_MSD_um2_per_s":               fit.get("D_um2_per_s",    np.nan),
            "D_error":                       fit.get("D_error",         np.nan),
            "exponent":                      fit.get("exponent",        np.nan),
            "exponent_error":                fit.get("exponent_error",  np.nan),
            "sigma_loc_nm":                  fit.get("sigma_loc_nm",    np.nan),
            "D_theo_um2_per_s":              D_theo,
            "num_trajectories":              rd.get("num_tracks"),
            "num_detections_total":          num_detections_total,
            "mean_trajectory_length_frames": mean_track_length,
            "num_frames":                    rd.get("num_frames"),
            "duration_s":                    duration_s,
            "laser_power_mW":                rec_meta["laser_power_mW"],
            "depth_um":                      rec_meta["depth_um"],
            "recorded_at":                   rec_meta["recorded_at"],
            "xml_path":                      rd.get("xml_path"),
            "tif_path":                      rd.get("tif_path"),
            "rec_path":                      rd.get("rec_path"),
        })

        xlim = (0.01, 6.0) if target_fps == FPS_SMALL_TARGET else (0.04, 3.0)
        plot_msd_single_file(
            rd=rd,
            D_theo=D_theo,
            label_nm=label_nm,
            n_fit=MSD_FIT_POINTS,
            save_path=SAVE_PATH,
            filename=f"{base}_{int(size_nm)}nm",
            xlim=xlim,
            sample_label="Wasser (D0)",
        )
        n_processed += 1

    # ── per-file summary CSV ──────────────────────────────────────────────────
    # sep=";" / decimal="," so Excel (German locale) opens this directly with
    # columns already split and numbers recognized, instead of dumping everything
    # into one column. utf-8-sig BOM so Excel auto-detects the encoding.
    if summary_rows and SAVE_PATH is not None:
        csv_path = SAVE_PATH / "per_file_summary_d0.csv"
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(csv_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
        print(f"\nSummary saved: {csv_path}")

    print(f"\nDone — {n_processed} files processed, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
