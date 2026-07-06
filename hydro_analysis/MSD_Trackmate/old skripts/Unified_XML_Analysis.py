"""
Unified TrackMate XML Analysis Script

Analyzes all TrackMate XML files in a folder with:
1) MSD analysis
2) Step size diffusion analysis

Uses standardized core.io/core.analysis/core.visualization functions.
"""

from pathlib import Path

import pandas as pd

from hydro_analysis.core.io import single_file_data
from hydro_analysis.core.analysis import (
    perform_msd_analysis,
    perform_stepsize_analysis,
    DEFAULT_MSD_FIT_POINTS,
    DEFAULT_STEP_INTERVAL,
)
from hydro_analysis.core.visualization import (
    plot_msd_results,
    plot_stepsize_results,
    plot_step_size_overlay,
    plot_dx_dy_distributions,
    plot_theory_comparison,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks")
DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks")
DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_1000")
DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.02.17\Tracks")
OUTPUT_DIR = DATA_DIR.parent / "analysis_results"

MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS
STEP_INTERVAL = DEFAULT_STEP_INTERVAL


def build_results_dict(xml_files: list[Path]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for xml_file in xml_files:
        print("Processing:", xml_file)
        result = single_file_data(xml_file)
        if result is None:
            continue
        results[result["xml_path"]] = result
    return results


def main() -> None:
    if not DATA_DIR.exists():
        print(f"Data folder not found: {DATA_DIR}")
        return

    xml_files = sorted(DATA_DIR.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {DATA_DIR}")
        return

    results_dict = build_results_dict(xml_files)
    if not results_dict:
        print("No valid datasets found (missing calibration).")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run analyses per file
    for result in results_dict.values():
        perform_msd_analysis(result, fit_points=MSD_FIT_POINTS)
        perform_stepsize_analysis(result, step_interval=STEP_INTERVAL)
        print(f"  • Analyzed: {result['xml_path']}")
        print(f"mpp: {result['mpp']}, fps: {result['fps']}")
    # Per-file plots
    for result in results_dict.values():
        base = Path(result["base_name"]).stem
        plot_msd_results(result, save_path=OUTPUT_DIR / f"{base}_msd.png")
        plot_stepsize_results(result, save_path=OUTPUT_DIR / f"{base}_stepsize.png")

    # Overall plots
    plot_step_size_overlay(results_dict, OUTPUT_DIR, step_interval=STEP_INTERVAL)
    plot_dx_dy_distributions(results_dict, OUTPUT_DIR, step_interval=STEP_INTERVAL)

    # Summary table + final theory comparison
    summary_rows = []
    for result in results_dict.values():
        summary_rows.append({
            "xml_path": result["xml_path"],
            "base_name": result["base_name"],
            "particle_size_nm": result.get("particle_size_nm"),
            "D_MSD_um2_per_s": result.get("D_MSD"),
            "D_stepsize_um2_per_s": result.get("D_step"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR / "summary_results.csv"
    summary_df.to_csv(summary_csv, index=False)

    theory_plot = OUTPUT_DIR / "diffusion_vs_theory.png"
    plot_theory_comparison(summary_df, theory_plot)

    print(f"\n[OK] Analysis complete. Results in: {OUTPUT_DIR}")
    print(f"  • Summary CSV: {summary_csv}")
    print(f"  • Theory plot: {theory_plot}")


if __name__ == "__main__":
    main()
