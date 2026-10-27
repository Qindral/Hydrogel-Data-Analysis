"""
MSD analysis (20 mg Hydrogel) using core modules only.

Pure compute + pickle stage: loads XML files from explicit folders per
particle size and computes MSD independently per file. Produces NO figures
and never combines raw trajectories across files — any plotting, and any
further aggregation, is done by other scripts that load these pickles.

All data handling, fit model, parameters and uncertainties come from
msd_pipeline.py and are identical to the water stage MSD_FromTrackmate_D0.py
(user, 2026-10-27, E040 A); this script only names folders, .rec assignments
and outputs. In short: one analysis per movie (E031), abort on missing
calibration (E006 A), edge-artifact removal (E001 A), 40 Hz minimum for nominal
20-100 nm (E003 A), drift subtraction at >= 30 tracks/frame (E009 B), 10-frame
filter, fit MSD with motion blur and 4 sigma^2 for any n (Savin & Doyle 2005,
Backlund et al. 2015; E046) over 6 lags with sigma of the
immobilised particles (E039 C, E004 A, E005 A; at 0.3 um/px for nominal 20-100 nm,
E042), per-size average weighted by tracks >= 10 frames (E002 A); dynamic vs.
static localisation error variance per file in the tables (E042).

Two pickles are written, per particle size:
  msd_20mg_files.pkl   ensemble PER FILE — each file gets its own iMSD/eMSD
                        and fit from its own particles only. This is the
                        source of truth; nothing here ever pools raw
                        trajectories from different files together.
  msd_20mg_result.pkl  a WEIGHTED AVERAGE of the per-file D/exponent values
                        above (not a refit on pooled/combined tracks) — see
                        core.analysis.weighted_average_per_size().

msd_20mg_files.pkl is also loaded (as CACHE_20MG) by:
  - Hydrogel_Water_Analysis.py
  - MSD_Normalize_Hydrogel_vs_Water.py
  - Auswertung_von_iMSD.py
  - MSD_per_file_20mg.py

msd_20mg_result.pkl is also loaded (as RESULT_CACHE_FILE) by:
  - MSD_AnomalousExponent_SaxtonFit.py
  - MSD_Deff_over_D0_vs_Size_20mg_WeightedAvg.py (and the Amsden version)

Always recomputes from the raw XML files (no cache-reuse) and unconditionally
overwrites both pickle files. Tables: TABLE_PATH/msd_20mg_exclusions.csv,
msd_20mg_per_file.csv, msd_20mg_weighted_average.csv.
"""

from pathlib import Path

from hydro_analysis.MSD_Trackmate.msd_pipeline import run_pipeline

# -----------------------------
# Configuration
# -----------------------------
ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")

XML_FOLDERS = {
    20.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks")],
    50.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_new")],#Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks")],
    100.0: [ROOT_PATH / "100 nm" / "Tracks"],
    200.0: [ROOT_PATH / "200 nm" / "Tracks"],
    500.0: [ROOT_PATH / "500 nm" / "Tracks"],
    1000.0: [ROOT_PATH / "1000 nm" / "Tracks"],
}

# .rec files whose name differs from the movie (raw data stay unchanged): {XML name: .rec path}.
# B2 .rec belongs to the B3 movie; decided by user, 2026-10-27.
REC_OVERRIDES = {
    "B3_inside_1d_1mmfrom injection_50_20mg_Tracks.xml":
        Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg"
             r"\B2_inside_1d_1mmfrom injection_50_20mg.rec"),
}

# Pickle output (always overwritten fresh, never read back to skip recomputation)
CACHE_FILE        = Path(__file__).parent / "cache" / "msd_20mg_files.pkl"
RESULT_CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_result.pkl"

RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
TABLE_PATH = RESULTS / "SPT" / "msd_20mg_pipeline"


def main() -> None:
    run_pipeline(XML_FOLDERS, REC_OVERRIDES, CACHE_FILE, RESULT_CACHE_FILE, TABLE_PATH, "msd_20mg")


if __name__ == "__main__":
    main()
