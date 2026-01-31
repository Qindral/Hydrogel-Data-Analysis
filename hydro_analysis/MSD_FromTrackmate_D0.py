"""
MSD analysis (water) using core modules only.

Loads XML files from explicit folders per particle size,
computes MSD per file, and shows the final theory comparison plot.

Supports pickle caching to speed up repeated runs.
Set NEUBERECHNEN = True to force recomputation.
"""

from pathlib import Path
import pickle
import pandas as pd

from core.io import single_file_data
from core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS
from core.visualization import plot_theory_comparison, plot_diffusion_comparison

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
SAVE_PATH = Path(f"E:\\PhD Data Analysis\\SPT 2025 II\\D_0 Wassermessung\\Plots_{pd.Timestamp.now().strftime('%Y%m%d')}")
MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS
VERBOSE = True

# Pickle caching
NEUBERECHNEN = True  # Set to True to force recomputation
CACHE_FILE = Path(__file__).parent / "cache" / "msd_d0_results.pkl"


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
    """Compute MSD analysis for all XML files."""
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
                perform_msd_analysis(rd, fit_points=MSD_FIT_POINTS)
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
        fit = r.get("fit_results_MSD")
        if fit is None:
            continue
        # Count unique particles from tracks_df
        tracks_df = r.get('tracks_df')
        if tracks_df is not None and 'particle' in tracks_df.columns:
            n_particles = len(tracks_df['particle'].unique())
        else:
            n_particles = r.get('number_of_tracks', 0) or 0
        rows.append({
            "xml_path": r["xml_path"],
            "base_name": r["base_name"],
            "particle_size_nm": r.get("particle_size_nm"),
            "D_MSD_um2_per_s": r.get("D_MSD"),
            "weight": n_particles,
        })

    summary_df = pd.DataFrame(rows)

    if VERBOSE:
        print(summary_df.to_string(index=False))

    # plot_theory_comparison(summary_df, SAVE_PATH)
    plot_diffusion_comparison(summary_df, SAVE_PATH)


if __name__ == "__main__":
    main()
