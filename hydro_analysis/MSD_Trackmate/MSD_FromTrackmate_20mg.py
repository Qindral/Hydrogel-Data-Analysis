"""
MSD analysis (20 mg Hydrogel) using core modules only.

Loads XML files from explicit folders per particle size, computes MSD per file,
and shows the final theory comparison plot.

Supports pickle caching to speed up repeated runs.
Set NEUBERECHNEN = True to force recomputation.
"""

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydro_analysis.core.io import single_file_data, get_dls_reference_maps, get_dls_sizes, get_dls_labels
from hydro_analysis.core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS, fit_powerlaw_with_errors
from hydro_analysis.core.visualization import plot_theory_comparison, plot_diffusion_comparison
from hydro_analysis.MSD_Trackmate.plot_emsd_publication import plot_emsd_publication
from hydro_analysis.core.physics import calculate_theoretical_diffusion

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

SAVE_PATH = None
SAVE_PATH = Path(f"E:\\PhD Data Analysis\\SPT 2025 II\\Hydrogel Messung\\20mg C16\\Plots_{pd.Timestamp.now().strftime('%Y%m%d')}")
MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS
VERBOSE = False
MAX_IMSD_CURVES_PER_SIZE = 3900  # None to plot all curves
FPS_SMALL_TARGET = 60.0
FPS_LARGE_TARGET = 20.0
FPS_TOLERANCE = 3.0  # Allowed deviation in fps
FPS_SIZE_EXACT: dict[float, float] = {50.0: 60.0}  # exact fps required for these sizes (±0.5)
PRINT_FILE_SUMMARY = False

# Pickle caching
NEUBERECHNEN = False  # Set to True to force recomputation
CACHE_FILE     = Path(__file__).parent / "cache" / "msd_20mg_results.pkl"
PUB_CACHE_FILE = Path(__file__).parent / "cache" / "msd_20mg_publication.pkl"
DLS_CACHE_FILE = Path(__file__).parent.parent / "Litesizer" / "cache" / "dls_reference.pkl"


def load_pub_cache() -> dict | None:
    """Load publication data cache; returns None if NEUBERECHNEN=True or cache missing."""
    if NEUBERECHNEN:
        return None
    if not PUB_CACHE_FILE.exists():
        return None
    try:
        with open(PUB_CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        print(f"Pub cache geladen: {PUB_CACHE_FILE} ({len(data)} Größen)")
        return data
    except Exception as e:
        print(f"Pub cache load failed: {e}")
        return None


def save_pub_cache(pub_data: dict) -> None:
    """Save per-size publication data to pickle."""
    PUB_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PUB_CACHE_FILE, "wb") as f:
        pickle.dump(pub_data, f)
    print(f"Pub cache gespeichert: {PUB_CACHE_FILE} ({len(pub_data)} Größen)")


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


def _collect_imsd_by_size(results: dict) -> dict[float, list[pd.DataFrame]]:
    size_map: dict[float, list[pd.DataFrame]] = {}
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        imsd = fit.get("imsd") if fit else None
        fps = r.get("fps")
        if size_nm in FPS_SIZE_EXACT:
            target_fps = FPS_SIZE_EXACT[size_nm]
            tol = 0.5
        else:
            target_fps = FPS_SMALL_TARGET if size_nm is not None and size_nm < 200 else FPS_LARGE_TARGET
            tol = FPS_TOLERANCE
        if fps is None or abs(float(fps) - target_fps) > tol:
            continue
        if size_nm is None or imsd is None or imsd.empty:
            continue
        size_map.setdefault(float(size_nm), []).append(imsd)
    return size_map


def plot_imsd_overlays_by_size(
    results: dict,
    save_path: Path | None,
    max_curves_per_size: int | None = None,
    size_override: dict[float, float] | None = None,
) -> None:
    size_map = _collect_imsd_by_size(results)
    if not size_map:
        print("Keine iMSD-Daten für die Größenübersicht gefunden.")
        return

    out_dir = None
    if save_path is not None:
        out_dir = Path(save_path) / "imsd_by_size"
        out_dir.mkdir(parents=True, exist_ok=True)

    for size_nm in sorted(size_map.keys()):
        imsd_list = size_map[size_nm]
        imsd_frames = []
        for idx, imsd in enumerate(imsd_list, 1):
            im = imsd.copy()
            im.columns = [f"{idx}_{col}" for col in im.columns]
            imsd_frames.append(im)

        imsd_all = pd.concat(imsd_frames, axis=1, join="outer").sort_index()
        if imsd_all.empty:
            continue

        imsd_plot = imsd_all
        if max_curves_per_size is not None and imsd_all.shape[1] > max_curves_per_size:
            idxs = np.linspace(0, imsd_all.shape[1] - 1, max_curves_per_size).astype(int)
            imsd_plot = imsd_all.iloc[:, idxs]

        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

        for col in imsd_plot.columns:
            s = imsd_plot[col].dropna()
            s = s[s > 0]
            if s.empty:
                continue
            ax.loglog(s.index.values, s.values, color="#777777", alpha=0.08, linewidth=0.9)

        emsd = imsd_all.mean(axis=1, skipna=True)
        emsd = emsd[emsd > 0]
        if not emsd.empty:
            ax.loglog(
                emsd.index.values,
                emsd.values,
                color="#0000da",
                linewidth=2.0,
                label=f"Ensemble MSD (N={imsd_all.shape[1]})",
            )

        mapped_size = size_override.get(size_nm, size_nm) if size_override else size_nm
        lags = np.asarray(imsd_all.index.values, dtype=float)
        lags = lags[np.isfinite(lags) & (lags > 0)]
        if lags.size > 0:
            D_theory = calculate_theoretical_diffusion(particle_size_nm=mapped_size)
            x_line = np.logspace(np.log10(lags.min()), np.log10(lags.max()), 100)
            ax.loglog(
                x_line,
                4.0 * D_theory * x_line,
                color="black",
                linestyle="--",
                linewidth=2.0,
                label=f"Theory (D={D_theory:.3g} µm²/s, size={mapped_size:.1f} nm)",
            )

        ax.set_xlabel("Lag Time (s)", fontsize=10)
        ax.set_ylabel("MSD (µm²)", fontsize=10)
        ax.set_title(f"iMSD Overlay ({mapped_size:.0f} nm)", fontsize=12, fontweight="semibold")
        ax.legend(loc="best", fontsize=9, frameon=False)

        plt.show()
        if out_dir is not None:
            base = f"imsd_overlay_{int(round(size_nm))}nm"
            png_path = out_dir / f"{base}.png"
            pdf_path = out_dir / f"{base}.pdf"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  iMSD plot saved: {png_path}")


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

    # ── Tabelle: aggregiert nach Partikelgröße ────────────────────────
    print("\n── Ensemble MSD – Zusammenfassung nach Partikelgröße ──")
    agg_rows = []
    for size, grp in summary_df.groupby("particle_size_nm"):
        D_vals = grp["D_MSD_um2_per_s"].dropna()
        n_vals = grp["exponent"].dropna()
        s_vals = grp["sigma_loc_nm"].dropna()
        agg_rows.append({
            "Größe (nm)":    int(size),
            "D (µm²/s)":    f"{D_vals.mean():.4f}",
            "± D":           f"{D_vals.std():.4f}" if len(D_vals) > 1 else f"{grp['D_error'].dropna().mean():.4f}",
            "n":             f"{n_vals.mean():.3f}",
            "± n":           f"{n_vals.std():.3f}"  if len(n_vals)  > 1 else f"{grp['exponent_error'].dropna().mean():.3f}",
            "σ_lok (nm)":   f"{s_vals.mean():.1f}" if not s_vals.empty else "n/a",
            "N Datenpunkte": int(grp["n_particles"].sum()),
            "N Dateien":     len(grp),
        })
    agg_df = pd.DataFrame(agg_rows).set_index("Größe (nm)")
    print(agg_df.to_string())

    dls_maps = get_dls_reference_maps()
    size_override = dls_maps["size_override_nm"]
    size_err = dls_maps["size_err_nm"]
    dls_override = dls_maps["dls_D_um2_per_s"]
    dls_err = dls_maps["dls_D_err_um2_per_s"]

    #plot_theory_comparison(summary_df, SAVE_PATH)
    # plot_imsd_overlays_by_size(
    #     results,
    #     SAVE_PATH,
    #     max_curves_per_size=MAX_IMSD_CURVES_PER_SIZE,
    #     size_override=size_override,
    # )
    plot_diffusion_comparison(
        summary_df,
        SAVE_PATH,
        size_override=size_override,
        size_err=size_err,
        dls_override=dls_override,
        dls_err=dls_err,
    )

    # ── Publication plots (with cache) ────────────────────────────────────────
    dls_sizes  = get_dls_sizes()
    dls_labels = get_dls_labels()
    pub_data   = load_pub_cache()

    if pub_data is not None:
        # Always refresh D_theo and refit from stored emsd_agg (cache may be stale)
        for size_nm, entry in pub_data.items():
            mapped_size = dls_sizes.get(size_nm, size_override.get(size_nm, size_nm))
            entry["D_theo"] = calculate_theoretical_diffusion(particle_size_nm=mapped_size)
            fr = fit_powerlaw_with_errors(entry["emsd_agg"], points=MSD_FIT_POINTS)
            entry["fit_agg"] = {"A": float(fr["A"][0]), "exponent": float(fr["n"][0])}

    if pub_data is None:
        pub_data = {}
        size_map = _collect_imsd_by_size(results)
        for size_nm in sorted(size_map.keys()):
            imsd_list = size_map[size_nm]
            if not imsd_list:
                continue

            imsd_frames = []
            for idx, imsd in enumerate(imsd_list, 1):
                im = imsd.copy()
                im.columns = [f"{idx}_{col}" for col in im.columns]
                imsd_frames.append(im)
            imsd_all = pd.concat(imsd_frames, axis=1, join="outer").sort_index()

            emsd_agg = imsd_all.mean(axis=1, skipna=True)
            emsd_agg = emsd_agg[emsd_agg > 0]
            if emsd_agg.empty:
                continue

            fr = fit_powerlaw_with_errors(emsd_agg, points=MSD_FIT_POINTS)
            fit_agg = {"A": float(fr["A"][0]), "exponent": float(fr["n"][0])}

            mapped_size = dls_sizes.get(size_nm, size_override.get(size_nm, size_nm))
            D_th = calculate_theoretical_diffusion(particle_size_nm=mapped_size)

            pub_data[size_nm] = {
                "imsd_all": imsd_all,
                "emsd_agg": emsd_agg,
                "fit_agg":  fit_agg,
                "D_theo":   D_th,
            }

        save_pub_cache(pub_data)

    for size_nm, entry in sorted(pub_data.items()):
        label_nm = dls_labels.get(size_nm, int(size_nm))
        xlim = (0.01, 6.0) if size_nm < 200 else (0.04, 3.0)
        plot_emsd_publication(
            imsd=entry["imsd_all"],
            emsd=entry["emsd_agg"],
            D_theo=entry["D_theo"],
            n_fit=MSD_FIT_POINTS,
            fit_result=entry["fit_agg"],
            save_path=SAVE_PATH,
            filename=f"emsd_{int(size_nm)}nm",
            particle_size_nm=label_nm,
            xlim=xlim,
            sample_label="20 mg/mL C16",
        )


if __name__ == "__main__":
    main()
