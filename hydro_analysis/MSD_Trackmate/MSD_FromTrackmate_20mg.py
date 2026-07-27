"""
MSD analysis (20 mg Hydrogel) using core modules only.

Pure compute + pickle stage: loads XML files from explicit folders per
particle size and computes MSD independently per file. Produces NO figures
and never combines raw trajectories across files — any plotting, and any
further aggregation, is done by other scripts that load these pickles.

Two pickles are written, per particle size:
  msd_20mg_files.pkl   ensemble PER FILE — each file gets its own iMSD/eMSD
                        and power-law fit from its own particles only. This
                        is the source of truth; nothing here ever pools raw
                        trajectories from different files together.
  msd_20mg_result.pkl  a trajectory-count-WEIGHTED AVERAGE of the per-file
                        D/exponent values above (not a refit on pooled/
                        combined tracks) — see core.analysis.weighted_average_per_size().

msd_20mg_files.pkl is also loaded (as CACHE_20MG) by:
  - Hydrogel_Water_Analysis.py
  - MSD_Normalize_Hydrogel_vs_Water.py
  - Auswertung_von_iMSD.py
  - MSD_per_file_20mg.py

msd_20mg_result.pkl is also loaded (as RESULT_CACHE_FILE) by:
  - MSD_AnomalousExponent_SaxtonFit.py

Always recomputes from the raw XML files (no cache-reuse) and unconditionally
overwrites both pickle files, so downstream scripts always get fresh results.

Tracks are pre-filtered via core.io.single_file_data() -> remove_edge_artifacts():
detections within 3% of the frame border are dropped and trajectories split at
the gap, to correct spurious TrackMate linking of near-edge detections.
"""

from pathlib import Path
import pickle
import pandas as pd

from hydro_analysis.core.io import single_file_data
from hydro_analysis.core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS, weighted_average_per_size

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

MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS
VERBOSE = False
FPS_SMALL_TARGET = 60.0
FPS_LARGE_TARGET = 20.0
FPS_TOLERANCE = 3.0  # Allowed deviation in fps
FPS_SIZE_EXACT: dict[float, float] = {50.0: 60.0}  # exact fps required for these sizes (±0.5)
PRINT_FILE_SUMMARY = True

# Pickle output (always overwritten fresh, never read back to skip recomputation)
CACHE_FILE        = Path(__file__).parent / "cache" / "msd_20mg_files.pkl"
RESULT_CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_result.pkl"


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
        if size_nm in FPS_SIZE_EXACT:
            target_fps, tol = FPS_SIZE_EXACT[size_nm], 0.5
        else:
            target_fps = FPS_SMALL_TARGET if size_nm < 200 else FPS_LARGE_TARGET
            tol = FPS_TOLERANCE
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
                fps = rd.get("fps")
                if fps is None or abs(float(fps) - target_fps) > tol:
                    print(f"  [SKIP fps={fps}] {xml_path.name}")
                    continue
                rd["particle_size_nm"] = size_nm if rd.get("particle_size_nm") is None else rd["particle_size_nm"]
                perform_msd_analysis(rd, fit_points=MSD_FIT_POINTS)
                results[rd["xml_path"]] = rd
                processed += 1
                if processed % 10 == 0:
                    print(f"  Verarbeitet: {processed}/{total_files}")

    print(f"Analyse abgeschlossen: {processed} Dateien verarbeitet")
    return results


def save_result_cache(data: dict) -> None:
    """Save the per-size weighted-average cache."""
    RESULT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Result cache gespeichert: {RESULT_CACHE_FILE} ({len(data)} Größen)")


def main() -> None:
    # Always recompute from the raw XML files — no cache-reuse.
    results = compute_results()
    if not results:
        print("No analyzable files found.")
        return
    save_results_to_cache(results)

    # ── Weighted average per size, from the independent per-file fits ──────
    avg_results = weighted_average_per_size(results)
    if avg_results:
        save_result_cache(avg_results)

    print("\n── Trajectory-weighted average per particle size ──")
    avg_rows = []
    for size_nm, pr in sorted(avg_results.items()):
        fit = pr.get("fit_results_MSD") or {}
        avg_rows.append({
            "Größe (nm)":    int(size_nm),
            "D (µm²/s)":     f"{fit.get('D_um2_per_s', float('nan')):.4f}",
            "± D":           f"{fit.get('D_error', float('nan')):.4f}",
            "n":             f"{fit.get('exponent', float('nan')):.3f}",
            "± n":           f"{fit.get('exponent_error', float('nan')):.3f}",
            "σ_lok (nm)":    f"{fit.get('sigma_loc_nm', float('nan')):.1f}",
            "N Partikel":    pr.get("n_particles_pooled"),
            "N Dateien":     pr.get("n_files"),
        })
    if avg_rows:
        print(pd.DataFrame(avg_rows).set_index("Größe (nm)").to_string())

    rows = []
    for r in results.values():
        fit = r.get("fit_results_MSD")
        if fit is None:
            continue
        # Count data points (rows in tracks_df = individual particle positions)
        tracks_df = r.get('tracks_df')
        n_particles = len(tracks_df) if tracks_df is not None else 0
        rows.append({
            "xml_path": r["xml_path"],
            "base_name": r["base_name"],
            "particle_size_nm": r.get("particle_size_nm"),
            "D_MSD_um2_per_s": r.get("D_MSD"),
            "D_error": fit.get("D_error"),
            "exponent": fit.get("exponent"),
            "exponent_error": fit.get("exponent_error"),
            "sigma_loc_nm": fit.get("sigma_loc_nm"),
            "fps": r.get("fps"),
            "mpp_um_per_px": r.get("mpp"),
            "n_particles": n_particles,
            "weight": n_particles,
        })

    summary_df = pd.DataFrame(rows)

    if VERBOSE:
        print(summary_df.to_string(index=False))
    if PRINT_FILE_SUMMARY:
        cols = ["base_name", "particle_size_nm", "n_particles", "fps", "mpp_um_per_px", "D_MSD_um2_per_s"]
        file_df = summary_df[cols].sort_values(["particle_size_nm", "base_name"])
        print("\nPro Datei (MSD):")
        print(file_df.to_string(index=False))


if __name__ == "__main__":
    main()
