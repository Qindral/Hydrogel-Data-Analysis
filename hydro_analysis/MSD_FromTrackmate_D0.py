"""
MSD analysis (water) using core modules only.

Loads XML files from explicit folders per particle size,
computes MSD per file, and shows the final theory comparison plot.
"""

from pathlib import Path
import pandas as pd

from core.io import single_file_data
from core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS
from core.visualization import plot_theory_comparison

# -----------------------------
# Configuration
# -----------------------------
XML_FOLDERS = {
    20.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm\Tracks"),
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.16\Tracks_20"),
    ],
    50.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_50"),
    ],
    100.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\100 nm\Tracks"),
    ],
    200.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\200 nm\Tracks"),
    ],
    500.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\500 nm\Tracks"),
    ],
    1000.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_1000"),
    ],
}

SAVE_PATH = None
MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS
VERBOSE = False


def main() -> None:
    results = {}

    for size_nm, folders in XML_FOLDERS.items():
        for folder in folders:
            if not folder.exists():
                print(f"WARNING: folder not found: {folder}")
                continue
            for xml_path in sorted(folder.glob("*.xml")):
                rd = single_file_data(xml_path)
                if rd is None:
                    continue
                rd["particle_size_nm"] = size_nm if rd.get("particle_size_nm") is None else rd["particle_size_nm"]
                perform_msd_analysis(rd, fit_points=MSD_FIT_POINTS)
                results[rd["xml_path"]] = rd

    if not results:
        print("No analyzable files found.")
        return

    rows = []
    for r in results.values():
        fit = r.get("fit_results_MSD")
        weight = r.get('number_of_tracks')
        if fit is None:
            continue
        rows.append({
            "xml_path": r["xml_path"],
            "base_name": r["base_name"],
            "particle_size_nm": r.get("particle_size_nm"),
            "D_MSD_um2_per_s": r.get("D_MSD"),
        })

    summary_df = pd.DataFrame(rows)

    if VERBOSE:
        print(summary_df.to_string(index=False))

    plot_theory_comparison(summary_df, SAVE_PATH)


if __name__ == "__main__":
    main()
