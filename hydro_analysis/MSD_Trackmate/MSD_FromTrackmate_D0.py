"""
MSD analysis (water) using core modules only.

Pure compute + pickle stage: loads XML files from explicit folders per
particle size and computes MSD independently per file. Produces NO figures
and never combines raw trajectories or per-file MSD curves across files —
any plotting or aggregation is done by other scripts that load these pickles.

All data handling, fit model, parameters and uncertainties come from
msd_pipeline.py and are identical to the hydrogel stage MSD_FromTrackmate_20mg.py
(user, 2026-10-27, E040 A); this script only names folders, .rec assignments
and outputs. In short: one analysis per movie (E031; byte-identical copies in
the second 20 nm folder are listed, not used), abort on missing calibration
(E006 A), edge-artifact removal (E001 A), 40 Hz minimum for nominal 20-100 nm
(E003 A), drift subtraction at >= 30 tracks/frame (E009 B), 10-frame filter,
fit MSD with motion blur and 4 sigma^2 for any n (Savin & Doyle 2005,
Backlund et al. 2015; E046) over 6 lags with sigma of the
immobilised particles (E039 C, E004 A, E005 A), taken at 0.3 um/px for nominal
20-100 nm and at 0.15 um/px above, whatever the pixel size of the file (E042),
per-size average weighted by tracks >= 10 frames (E002 A); dynamic vs. static
localisation error variance per file in the tables (E042).

Pickles:
  msd_d0_results.pkl  ensemble PER FILE (source of truth), also loaded (as
                      CACHE_D0) by:
                        - Hydrogel_Water_Analysis.py
                        - MSD_Normalize_Hydrogel_vs_Water.py
                        - Auswertung_von_iMSD.py
                        - MSD_perfile_D0.py
                        - MSD_Diffusion_vs_Size_D0.py
  msd_d0_result.pkl   weighted average per size of the per-file values
                      (core.analysis.weighted_average_per_size), same form as
                      msd_20mg_result.pkl.

Always recomputes from the raw XML files (no cache-reuse) and unconditionally
overwrites both pickles. Tables: TABLE_PATH/msd_d0_exclusions.csv,
msd_d0_per_file.csv, msd_d0_weighted_average.csv.
"""

from pathlib import Path

from hydro_analysis.MSD_Trackmate.msd_pipeline import run_pipeline

# -----------------------------
# Configuration
# -----------------------------
XML_FOLDERS = {
20: [Path(r"E:\Daten Promotion Sicherung\SPT Analysis Buffer_Hydrogel\Buffer SPT\20 nm buffer")],
50: [Path(r"E:\Daten Promotion Sicherung\SPT Analysis Buffer_Hydrogel\Buffer SPT\50 nm buffer")],
100: [Path(r"E:\Daten Promotion Sicherung\SPT Analysis Buffer_Hydrogel\Buffer SPT\100 nm buffer")],
200: [Path(r"E:\Daten Promotion Sicherung\SPT Analysis Buffer_Hydrogel\Buffer SPT\200 nm buffer")],
500: [Path(r"E:\Daten Promotion Sicherung\SPT Analysis Buffer_Hydrogel\Buffer SPT\500 nm buffer")],
1000: [Path(r"E:\Daten Promotion Sicherung\SPT Analysis Buffer_Hydrogel\Buffer SPT\1000 nm buffer")]
}

# .rec files whose name differs from the movie (raw data stay unchanged): {XML name: .rec path}.
# 1000_nm_10_Tracks_v2 is a second analysis of the movie 1000_nm_10; decided by user, 2026-10-27.
REC_OVERRIDES = {
    "1000_nm_10_Tracks_v2.xml": Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\1000_nm_10.tif.rec"),
}

# Pickle output (always overwritten fresh, never read back to skip recomputation)
CACHE_FILE = Path(__file__).parent / "cache" / "msd_d0_results.pkl"
RESULT_CACHE_FILE = Path(__file__).parent / "cache" / "msd_d0_result.pkl"

RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
TABLE_PATH = RESULTS / "SPT" / "msd_d0_pipeline"


def main() -> None:
    run_pipeline(XML_FOLDERS, REC_OVERRIDES, CACHE_FILE, RESULT_CACHE_FILE, TABLE_PATH, "msd_d0")


if __name__ == "__main__":
    main()
