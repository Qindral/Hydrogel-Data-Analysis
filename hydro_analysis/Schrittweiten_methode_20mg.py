"""
Step size diffusion analysis (20 mg Hydrogel) using core modules only.

Loads XML files from explicit folders per particle size, computes step size diffusion per file,
and shows the final theory comparison plot.

Supports pickle caching to speed up repeated runs.
Set NEUBERECHNEN = True to force recomputation.
"""

from pathlib import Path
import pickle
import pandas as pd

from core.io import single_file_data
from core.analysis import perform_stepsize_analysis, DEFAULT_STEP_INTERVAL
from core.visualization import plot_step_size_overlay, plot_dx_dy_distributions, plot_theory_comparison

# -----------------------------
# Configuration
# -----------------------------
ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")

XML_FOLDERS = {
    20.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks")],
    50.0: [Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_new")],
    100.0: [ROOT_PATH / "100 nm" / "Tracks"],
    200.0: [ROOT_PATH / "200 nm" / "Tracks"],
    500.0: [ROOT_PATH / "500 nm" / "Tracks"],
    1000.0: [ROOT_PATH / "1000 nm" / "Tracks"],
}

SAVE_PATH = None
SAVE_PATH = Path(f"E:\\PhD Data Analysis\\SPT 2025 II\\Hydrogel Messung\\20mg C16\\Plots_{pd.Timestamp.now().strftime('%Y%m%d')}")
STEP_INTERVAL = DEFAULT_STEP_INTERVAL
VERBOSE = False
PRINT_FILE_SUMMARY = True
PLOT_STEP_OVERLAY = True
PLOT_DX_DY_DISTS = False

# Pickle caching
NEUBERECHNEN = False  # Set to True to force recomputation
CACHE_FILE = Path(__file__).parent / "cache" / "stepsize_20mg_results.pkl"


def load_cached_results() -> dict | None:
    """Load results from pickle cache if available."""
    if NEUBERECHNEN:
        print("NEUBERECHNEN=True: Ignoriere Cache, berechne neu...")
        return None
    if not CACHE_FILE.exists():
        print(f"Kein Cache gefunden: {CACHE_FILE}")
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            results = pickle.load(f)
        print(f"Cache geladen: {CACHE_FILE} ({len(results)} Dateien)")
        return results
    except Exception as e:
        print(f"Fehler beim Laden des Cache: {e}")
        return None


def save_results_to_cache(results: dict) -> None:
    """Save results to pickle cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"Cache gespeichert: {CACHE_FILE} ({len(results)} Dateien)")


def compute_results() -> dict:
    """Compute step size analysis for all XML files."""
    results = {}
    total_files = 0
    processed = 0

    for size_nm, folders in XML_FOLDERS.items():
        for folder in folders:
            if not folder.exists():
                print(f"WARNING: folder not found: {folder}")
                continue
            xml_files = list(folder.glob("*.xml"))
            total_files += len(xml_files)
            for xml_path in sorted(xml_files):
                rd = single_file_data(xml_path)
                if rd is None:
                    continue
                rd["particle_size_nm"] = size_nm if rd.get("particle_size_nm") is None else rd["particle_size_nm"]
                perform_stepsize_analysis(rd, step_interval=STEP_INTERVAL)
                results[rd["xml_path"]] = rd
                processed += 1
                if processed % 10 == 0:
                    print(f"  Verarbeitet: {processed}/{total_files}")

    print(f"Analyse abgeschlossen: {processed} Dateien verarbeitet")
    return results


def main() -> None:
    # Try to load from cache first
    results = load_cached_results()

    if results is None:
        # Compute fresh results
        results = compute_results()
        if results:
            save_results_to_cache(results)

    if not results:
        print("No analyzable files found.")
        return

    rows = []
    for r in results.values():
        fit = r.get("fit_results_step")
        if fit is None:
            continue
        # Count unique particles from tracks_df
        tracks_df = r.get("tracks_df")
        if tracks_df is not None and "particle" in tracks_df.columns:
            n_particles = len(tracks_df["particle"].unique())
        else:
            n_particles = r.get("number_of_tracks", 0) or 0
        n_steps = fit.get("n_steps_x", 0) or 0
        rows.append({
            "xml_path": r["xml_path"],
            "base_name": r["base_name"],
            "particle_size_nm": r.get("particle_size_nm"),
            "D_stepsize_um2_per_s": r.get("D_step"),
            "fps": r.get("fps"),
            "mpp_um_per_px": r.get("mpp"),
            "n_particles": n_particles,
            "n_steps": n_steps,
            "weight": n_particles,
        })

    summary_df = pd.DataFrame(rows)

    if VERBOSE:
        print(summary_df.to_string(index=False))
    if PRINT_FILE_SUMMARY:
        cols = ["base_name", "particle_size_nm", "n_particles", "n_steps", "fps", "mpp_um_per_px", "D_stepsize_um2_per_s"]
        file_df = summary_df[cols].sort_values(["particle_size_nm", "base_name"])
        print("\nPro Datei (Step Size):")
        print(file_df.to_string(index=False))

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)

    if PLOT_STEP_OVERLAY and SAVE_PATH is not None:
        plot_step_size_overlay(results, SAVE_PATH, step_interval=STEP_INTERVAL)
    if PLOT_DX_DY_DISTS and SAVE_PATH is not None:
        plot_dx_dy_distributions(results, SAVE_PATH, step_interval=STEP_INTERVAL)

    if SAVE_PATH is not None:
        theory_plot = SAVE_PATH / "diffusion_comparison_stepsize.png"
        plot_theory_comparison(summary_df, theory_plot)


if __name__ == "__main__":
    main()
