"""
Per-file inventory of all SPT measurements: MSD fit results (D, n) combined
with acquisition metadata parsed from the paired .rec file, exported to Excel.

For every TrackMate XML found under `root`, one row is written with:
  - D / D_error, exponent / exponent_error, r_squared, sigma_loc_nm  (MSD fit)
  - num_trajectories, num_detections_total, mean_trajectory_length_frames
  - fps, mpp, num_frames, duration_s
  - condition ("Surface loading" for an "A<n>" token in the filename,
    "Injection" for a "B<n>" token, e.g. "...A3.tif" vs "...B3_Crack.tif")
  - laser_power_mW, depth_um (both parsed from the .rec Comment section)
  - recorded_at (Record Date/Time header of the .rec file)
  - xml_path, tif_path, rec_path (storage locations)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hydro_analysis.core.io import single_file_data, scan_xml_folder, condition_label_from_filename, parse_rec_comment_metadata
from hydro_analysis.core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS

root = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung")
# root = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")
root = Path(r"E:\PhD Data Analysis\SPT 2025 II")

OUTPUT_XLSX = Path(__file__).parent / "per_file_inventory.xlsx"


def build_row(xml_path: Path) -> dict | None:
    rd = single_file_data(xml_path)
    if rd is None:
        return None

    try:
        perform_msd_analysis(rd, fit_points=DEFAULT_MSD_FIT_POINTS)
    except Exception as e:
        print(f"  [WARN] MSD-Fit fehlgeschlagen für {rd['base_name']}: {e}")

    fit = rd.get("fit_results_MSD") or {}
    tracks_df = rd["tracks_df"]
    if tracks_df is not None and not tracks_df.empty:
        track_lengths = tracks_df.groupby("particle").size()
        num_detections_total = len(tracks_df)
        mean_track_length = track_lengths.mean()
    else:
        num_detections_total = 0
        mean_track_length = np.nan

    fps = rd.get("fps")
    duration_s = rd["num_frames"] / fps if fps else np.nan

    rec_meta = parse_rec_comment_metadata(rd.get("rec_path"))

    return {
        "file":                          rd["base_name"],
        "particle_size_nm":              rd["particle_size_nm"],
        "condition":                     condition_label_from_filename(rd["base_name"]),
        "D_um2_per_s":                   fit.get("D_um2_per_s", np.nan),
        "D_error":                       fit.get("D_error", np.nan),
        "exponent":                      fit.get("exponent", np.nan),
        "exponent_error":                fit.get("exponent_error", np.nan),
        "r_squared":                     fit.get("r_squared", np.nan),
        "sigma_loc_nm":                  fit.get("sigma_loc_nm", np.nan),
        "num_trajectories":              rd["num_tracks"],
        "num_detections_total":          num_detections_total,
        "mean_trajectory_length_frames": mean_track_length,
        "fps":                           fps,
        "mpp_um_per_px":                 rd.get("mpp"),
        "num_frames":                    rd["num_frames"],
        "duration_s":                    duration_s,
        "laser_power_mW":                rec_meta["laser_power_mW"],
        "depth_um":                      rec_meta["depth_um"],
        "recorded_at":                   rec_meta["recorded_at"],
        "xml_path":                      rd["xml_path"],
        "tif_path":                      rd["tif_path"],
        "rec_path":                      rd["rec_path"],
    }


def main() -> None:
    xml_files = scan_xml_folder(root)
    print(f"Anzahl der .xml Dateien: {len(xml_files)}")

    rows = []
    n_skipped = 0
    for xml_path in sorted(xml_files):
        row = build_row(xml_path)
        if row is None:
            n_skipped += 1
            continue
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Verarbeitet: {len(df)} Dateien, {n_skipped} übersprungen (fehlende Kalibrierung)")
    if not df.empty:
        preview_cols = ["file", "particle_size_nm", "condition", "D_um2_per_s", "exponent", "num_trajectories"]
        print(df[preview_cols].to_string(index=False))

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nExcel gespeichert: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
