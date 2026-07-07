"""
Per-file publication MSD figures — 20 mg/mL C16 hydrogel.

Same figure layout as MSD_per_file_publication.py (D₀ reference):
  Main   iMSD · eMSD · Stokes-Einstein D₀ · power-law fit
  Inset (upper-right)  normalised MSD
  Inset (lower-right)  raw trajectories, plain, scale bar only
  Annotation  d = <label> nm

Each TrackMate XML file is processed individually; outputs are named
after the source file.

Tracks are pre-filtered via core.io.single_file_data() -> remove_edge_artifacts():
detections within 3% of the frame border are dropped and trajectories split at
the gap, to correct spurious TrackMate linking of near-edge detections.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_analysis.core.io import single_file_data, get_dls_reference_maps, get_dls_sizes, get_dls_labels  # noqa: F401
from hydro_analysis.core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import plot_msd_single_file

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")

XML_FOLDERS: dict[float, list[Path]] = {
    20.0:   [ROOT_PATH / "20 nm" / "20 nm 20 mg" / "Tracks"],
    50.0:   [ROOT_PATH / "50 nm" / "50 nm 20 mg" / "Tracks_new"],
    100.0:  [ROOT_PATH / "100 nm" / "Tracks"],
    200.0:  [ROOT_PATH / "200 nm" / "Tracks"],
    500.0:  [ROOT_PATH / "500 nm" / "Tracks"],
    1000.0: [ROOT_PATH / "1000 nm" / "Tracks"],
}

SAVE_PATH = Path(
    rf"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16"
    rf"\PerFile_{pd.Timestamp.now().strftime('%Y%m%d')}"
)
MSD_FIT_POINTS   = DEFAULT_MSD_FIT_POINTS
FPS_SMALL_TARGET = 60.0
FPS_LARGE_TARGET = 20.0
FPS_TOLERANCE    = 3.0


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    dls_sizes     = get_dls_sizes()
    dls_labels    = get_dls_labels()
    size_override = get_dls_reference_maps()["size_override_nm"]

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    n_processed = 0
    n_skipped   = 0

    for size_nm, folders in XML_FOLDERS.items():
        target_fps  = FPS_SMALL_TARGET if size_nm < 200 else FPS_LARGE_TARGET
        mapped_size = dls_sizes.get(size_nm, size_override.get(size_nm, size_nm))
        label_nm    = dls_labels.get(size_nm, int(size_nm))
        D_theo      = calculate_theoretical_diffusion(particle_size_nm=mapped_size)

        for folder in folders:
            if not folder.exists():
                print(f"WARNING: folder not found: {folder}")
                continue

            xml_files = sorted(folder.glob("*.xml"))
            if not xml_files:
                print(f"  No XML files in {folder}")
                continue

            for xml_path in xml_files:
                rd = single_file_data(xml_path)
                if rd is None:
                    n_skipped += 1
                    continue

                fps = rd.get("fps")
                if fps is None or abs(float(fps) - target_fps) > FPS_TOLERANCE:
                    print(f"  [SKIP fps={fps}] {xml_path.name}")
                    n_skipped += 1
                    continue

                rd["particle_size_nm"] = size_nm
                perform_msd_analysis(rd, fit_points=MSD_FIT_POINTS)

                fit  = rd.get("fit_results_MSD") or {}
                base = rd.get("base_name", xml_path.stem)
                print(f"  [{int(size_nm)} nm  {fps:.0f} fps] {base}")

                summary_rows.append({
                    "file":             base,
                    "size_nm":          int(size_nm),
                    "label_nm":         label_nm,
                    "fps":              fps,
                    "D_MSD_um2_per_s":  fit.get("D_um2_per_s",    np.nan),
                    "D_error":          fit.get("D_error",         np.nan),
                    "exponent":         fit.get("exponent",        np.nan),
                    "exponent_error":   fit.get("exponent_error",  np.nan),
                    "sigma_loc_nm":     fit.get("sigma_loc_nm",    np.nan),
                    "D_theo_um2_per_s": D_theo,
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
                    sample_label="20 mg/mL C16",
                )
                n_processed += 1

    # ── per-file summary CSV ──────────────────────────────────────────────────
    if summary_rows and SAVE_PATH is not None:
        csv_path = SAVE_PATH / "per_file_summary_20mg.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary saved: {csv_path}")

    print(f"\nDone — {n_processed} files processed, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
