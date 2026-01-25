"""
Step size diffusion analysis (water) using core modules only.

Uses explicit XML folders per particle size, runs step size analysis for each file,
then shows only final outputs:
- Step-size statistics per particle size
- Diffusion comparison (measured vs theory + DLS)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.io import single_file_data
from core.analysis import (
    perform_stepsize_analysis,
    calculate_theoretical_diffusion,
    DEFAULT_STEP_INTERVAL,
)
from core.visualization import plot_diffusion_comparison

# -----------------------------
# Configuration
# -----------------------------
ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung")
XML_FOLDERS = {
    20.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm\Tracks"),Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.16\Tracks_20")],
    50.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_50")],
    100.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\100 nm\Tracks")],
    200.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\200 nm\Tracks")],
    500.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\500 nm\Tracks")],
    1000.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_1000")],
}

# Folder for 20mg C16 analysis --- IGNORE ---

ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")
XML_FOLDERS = {
    20.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks")],
    50.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks")],
    100.0: [],
    200.0: [],
    500.0: [],
    1000.0: [],
}


SAVE_PATH = None                 # show plots only
STEP_INTERVAL = DEFAULT_STEP_INTERVAL
VERBOSE = True


def main() -> None:
    # Collect results for all files
    results = {}
    for size_nm, folder in XML_FOLDERS.items():

        for folder in XML_FOLDERS[size_nm]:
            for xml_path in sorted(folder.glob("*.xml")):
                rd = single_file_data(xml_path)
                if rd is None:
                    continue
                rd["particle_size_nm"] = size_nm if rd.get("particle_size_nm") is None else rd["particle_size_nm"]
                perform_stepsize_analysis(rd, step_interval=STEP_INTERVAL)
                results[rd["xml_path"]] = rd

    if not results:
        print("No analyzable files found.")
        return

    # Build per-file results_df (minimal columns needed by plot_diffusion_comparison)
    rows = []
    for r in results.values():
        fit = r.get("fit_results_step")
        if not fit:
            continue
        rows.append({
            "xml_path": r["xml_path"],
            "xml_name": r["base_name"],
            "particle_size_nm": r.get("particle_size_nm"),
            "D": r.get("D_step"),
            "D_std": fit.get("D_dir_disagreement_um2_per_s", 0.0) or 0.0,
            "mode": "Unknown",
            "quality_flag": fit.get("quality_flag", "unknown"),
            "num_steps": fit.get("n_steps_x", 0),
            "mpp": r.get("mpp"),
        })
    results_df = pd.DataFrame(rows)

    # Build combined_df (per particle size)
    groups = results_df.groupby("particle_size_nm", dropna=True)
    combined_rows = []
    for size, g in groups:
        w = g["num_steps"].to_numpy(dtype=float)
        d_vals = g["D"].to_numpy(dtype=float)
        d_err = g["D_std"].to_numpy(dtype=float)
        if np.nansum(w) > 0:
            d_weighted = np.average(d_vals, weights=w)
            d_weighted_err = np.sqrt(np.nansum((w * d_err) ** 2)) / np.nansum(w)
        else:
            d_weighted = np.nan
            d_weighted_err = np.nan
        combined_rows.append({
            "particle_size_nm": size,
            "num_files": int(len(g)),
            "total_particles": int(len(g)),
            "total_steps": int(np.nansum(w)),
            "D_measured": d_weighted,
            "D_measured_std": d_weighted_err,
            "D_theoretical": calculate_theoretical_diffusion(size),
        })
    combined_df = pd.DataFrame(combined_rows)

    # Step-size statistics per particle size (mean/std of |step| in nm)
    step_stats = []
    for size, g in groups:
        steps_nm_all = []
        for _, row in g.iterrows():
            rd = results[row["xml_path"]]
            step_df = rd["fit_results_step"]["step_df"]
            mpp = row["mpp"]
            steps_nm = np.sqrt(step_df["dx"].to_numpy()**2 + step_df["dy"].to_numpy()**2) * mpp * 1000.0
            steps_nm_all.append(steps_nm)
        if steps_nm_all:
            cat = np.concatenate(steps_nm_all)
            step_stats.append({
                "particle_size_nm": size,
                "mean_step_nm": float(np.nanmean(cat)),
                "std_step_nm": float(np.nanstd(cat)),
                "n": int(cat.size),
            })
    step_df = pd.DataFrame(step_stats).sort_values("particle_size_nm")

    # Plot step statistics
    if not step_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(step_df["particle_size_nm"], step_df["mean_step_nm"], yerr=step_df["std_step_nm"], fmt="o-", capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel("Particle size [nm]")
        ax.set_ylabel("Mean step size [nm]")
        ax.set_title("Step-size statistics per particle size")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Final diffusion comparison plot
    plot_diffusion_comparison(combined_df, results_df, save_path=SAVE_PATH)
    for particle_size, group in combined_df.groupby("particle_size_nm"):
        row = group.iloc[0]
        if VERBOSE:
            print(f"Particle size: {particle_size} nm")
            print(f"  • Measured D: {row['D_measured']:.4f} ± {row['D_measured_std']:.4f} µm²/s")
            print(f"  • Theoretical D: {row['D_theoretical']:.4f} µm²/s")
            print(f"  • Number of files: {row['num_files']}")
            print(f"  • Total steps: {row['total_steps']}")
            print(f"  • Total particles: {row['total_particles']}")
            print("")

if __name__ == "__main__":
    main()
