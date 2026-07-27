"""
MSD analysis (water) using core modules only.

Pure compute + pickle stage: loads XML files from explicit folders per
particle size and computes MSD independently per file. Produces NO figures
and never combines raw trajectories or per-file MSD curves across files —
any plotting or aggregation is done by other scripts that load this pickle.

msd_d0_results.pkl is also loaded (as CACHE_D0) by:
  - Hydrogel_Water_Analysis.py
  - MSD_Normalize_Hydrogel_vs_Water.py
  - Auswertung_von_iMSD.py
  - MSD_perfile_D0.py

Always recomputes from the raw XML files (no cache-reuse) and unconditionally
overwrites msd_d0_results.pkl, so downstream scripts always get fresh results.

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

MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS
VERBOSE = True
FPS_SMALL_TARGET = 60.0
FPS_LARGE_TARGET = 20.0
FPS_TOLERANCE = 3.0  # Allowed deviation in fps
FPS_SIZE_EXACT: dict[float, float] = {50.0: 60.0}  # exact fps required for these sizes (±0.5)
PRINT_FILE_SUMMARY = True

# Pickle output (always overwritten fresh, never read back to skip recomputation)
CACHE_FILE = Path(__file__).parent / "cache" / "msd_d0_results.pkl"


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


def main() -> None:
    # Always recompute from the raw XML files — no cache-reuse.
    results = compute_results()
    if not results:
        print("No analyzable files found.")
        return
    save_results_to_cache(results)

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

    # ── Tabelle: trajektoriengewichteter Mittelwert pro Partikelgröße ────
    # Console-only (nothing here is saved/pickled) -- averages the already-
    # independent per-file fits, weighted by each file's trajectory count,
    # same as MSD_FromTrackmate_20mg.py's msd_20mg_result.pkl.
    print("\n── Trajectory-weighted average per particle size ──")
    avg_results = weighted_average_per_size(results)
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


if __name__ == "__main__":
    main()
