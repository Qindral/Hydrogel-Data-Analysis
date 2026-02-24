"""
Track Folder Comparison Script

Compares MSD analysis results from multiple track folders that contain
different tracking results for the same source files.

Compares:
1. Diffusion coefficient D across folders
2. Per-file comparison: D, particle count, track length
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hydro_analysis.core.io import single_file_data
from hydro_analysis.core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS

# -----------------------------
# Configuration
# -----------------------------

# Track folders to compare (same source files, different tracking methods)
TRACK_FOLDERS = {
    "Tracks": Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks"),
    "Tracks_new": Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_new"),
    "Tracks_old": Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_old"),
    "Tracks_test": Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_test"),
}

TRACK_FOLDERS = {
    "Tracks_new" : Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.02.17\Tracks"),
    "Tracks_old" : Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\500 nm\Tracks"),
}

PARTICLE_SIZE_NM = 50.0  # Nominal particle size

SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\Comparison")
MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS

# Caching
NEUBERECHNEN = True
CACHE_FILE = Path(__file__).parent / "cache" / "track_comparison_results.pkl"


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
        print(f"Cache geladen: {CACHE_FILE}")
        return results
    except Exception as e:
        print(f"Fehler beim Laden des Cache: {e}")
        return None


def save_results_to_cache(results: dict) -> None:
    """Save results to pickle cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)
    print(f"Cache gespeichert: {CACHE_FILE}")


def extract_base_name(xml_path: Path) -> str:
    """Extract the base name that identifies the source file."""
    stem = xml_path.stem
    # Remove common suffixes to get the original file identifier
    stem = stem.replace('_Tracks', '')
    stem = stem.replace('_processed', '')
    stem = stem.replace('filtered_', '')
    stem = stem.replace('Resultof', '')
    stem = stem.replace('_var', '')
    return stem


def compute_folder_results(folder_name: str, folder_path: Path) -> dict:
    """Compute MSD analysis for all XML files in a folder."""
    results = {}

    if not folder_path.exists():
        print(f"WARNING: Ordner nicht gefunden: {folder_path}")
        return results

    xml_files = list(folder_path.glob("*.xml"))
    print(f"\n{folder_name}: {len(xml_files)} XML-Dateien gefunden")

    for xml_path in sorted(xml_files):
        rd = single_file_data(xml_path)
        if rd is None:
            continue

        rd["particle_size_nm"] = PARTICLE_SIZE_NM
        perform_msd_analysis(rd, fit_points=MSD_FIT_POINTS)

        base_name = extract_base_name(xml_path)

        # Extract key metrics
        tracks_df = rd.get('tracks_df')
        n_particles = 0
        avg_track_length = 0

        if tracks_df is not None and 'particle' in tracks_df.columns:
            n_particles = len(tracks_df['particle'].unique())
            # Calculate average track length
            track_lengths = tracks_df.groupby('particle').size()
            avg_track_length = track_lengths.mean() if len(track_lengths) > 0 else 0

        results[base_name] = {
            "xml_path": str(xml_path),
            "base_name": base_name,
            "D_MSD": rd.get("D_MSD"),
            "n_particles": n_particles,
            "avg_track_length": avg_track_length,
            "fit_results": rd.get("fit_results_MSD"),
        }

    return results


def compute_all_results() -> dict:
    """Compute results for all folders."""
    all_results = {}

    for folder_name, folder_path in TRACK_FOLDERS.items():
        all_results[folder_name] = compute_folder_results(folder_name, folder_path)

    return all_results


def create_comparison_dataframe(all_results: dict) -> pd.DataFrame:
    """Create a comparison DataFrame with all files and folders."""
    # Collect all unique base names
    all_base_names = set()
    for folder_results in all_results.values():
        all_base_names.update(folder_results.keys())

    rows = []
    for base_name in sorted(all_base_names):
        row = {"base_name": base_name}
        for folder_name, folder_results in all_results.items():
            if base_name in folder_results:
                data = folder_results[base_name]
                row[f"D_{folder_name}"] = data["D_MSD"]
                row[f"n_particles_{folder_name}"] = data["n_particles"]
                row[f"track_length_{folder_name}"] = data["avg_track_length"]
            else:
                row[f"D_{folder_name}"] = np.nan
                row[f"n_particles_{folder_name}"] = np.nan
                row[f"track_length_{folder_name}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def plot_folder_d_comparison(all_results: dict, save_path: Path | None = None):
    """Plot D distribution comparison across folders."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot comparison
    ax1 = axes[0]
    d_values = []
    labels = []
    for folder_name, folder_results in all_results.items():
        d_list = [r["D_MSD"] for r in folder_results.values() if r["D_MSD"] is not None and np.isfinite(r["D_MSD"])]
        if d_list:
            d_values.append(d_list)
            labels.append(folder_name)

    if d_values:
        bp = ax1.boxplot(d_values, tick_labels=labels, patch_artist=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax1.set_ylabel("D [µm²/s]")
    ax1.set_title("Diffusionskoeffizient pro Ordner")
    ax1.tick_params(axis='x', rotation=45)

    # Histogram overlay
    ax2 = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for (folder_name, folder_results), color in zip(all_results.items(), colors):
        d_list = [r["D_MSD"] for r in folder_results.values() if r["D_MSD"] is not None and np.isfinite(r["D_MSD"])]
        if d_list:
            ax2.hist(d_list, bins=20, alpha=0.5, label=f"{folder_name} (n={len(d_list)}, µ={np.mean(d_list):.3f})", color=color)

    ax2.set_xlabel("D [µm²/s]")
    ax2.set_ylabel("Anzahl Dateien")
    ax2.set_title("D-Verteilung pro Ordner")
    ax2.legend()

    plt.tight_layout()

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "folder_d_comparison.png", dpi=150)
        print(f"Gespeichert: {save_path / 'folder_d_comparison.png'}")

    plt.show()


def plot_file_comparison(comparison_df: pd.DataFrame, save_path: Path | None = None):
    """Plot per-file comparison of D, particle count, and track length."""
    folder_names = list(TRACK_FOLDERS.keys())
    n_folders = len(folder_names)

    if n_folders < 2:
        print("Mindestens 2 Ordner für Vergleich benötigt.")
        return

    # Create pairwise scatter plots for D
    n_pairs = n_folders * (n_folders - 1) // 2
    fig, axes = plt.subplots(3, n_pairs, figsize=(5 * n_pairs, 12))
    if n_pairs == 1:
        axes = axes.reshape(3, 1)

    pair_idx = 0
    for i, folder1 in enumerate(folder_names):
        for j, folder2 in enumerate(folder_names):
            if i >= j:
                continue

            # D comparison
            ax_d = axes[0, pair_idx]
            x = comparison_df[f"D_{folder1}"]
            y = comparison_df[f"D_{folder2}"]
            mask = x.notna() & y.notna()
            ax_d.scatter(x[mask], y[mask], alpha=0.6, s=30)

            # Fit line
            if mask.sum() > 2:
                slope, intercept = np.polyfit(x[mask], y[mask], 1)
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                xlim = ax_d.get_xlim()
                ax_d.plot(xlim, [slope * xlim[0] + intercept, slope * xlim[1] + intercept],
                         'r--', alpha=0.7, label=f"r={corr:.3f}")
                ax_d.legend()

            # Identity line
            all_vals = np.concatenate([x[mask], y[mask]])
            if len(all_vals) > 0:
                lim = [all_vals.min() * 0.9, all_vals.max() * 1.1]
                ax_d.plot(lim, lim, 'k:', alpha=0.5)
                ax_d.set_xlim(lim)
                ax_d.set_ylim(lim)

            ax_d.set_xlabel(f"D [{folder1}] (µm²/s)")
            ax_d.set_ylabel(f"D [{folder2}] (µm²/s)")
            ax_d.set_title(f"D: {folder1} vs {folder2}")

            # Particle count comparison
            ax_n = axes[1, pair_idx]
            x = comparison_df[f"n_particles_{folder1}"]
            y = comparison_df[f"n_particles_{folder2}"]
            mask = x.notna() & y.notna()
            ax_n.scatter(x[mask], y[mask], alpha=0.6, s=30, color='green')

            if mask.sum() > 2:
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                ax_n.set_title(f"Partikelzahl: {folder1} vs {folder2} (r={corr:.3f})")
            else:
                ax_n.set_title(f"Partikelzahl: {folder1} vs {folder2}")

            all_vals = np.concatenate([x[mask], y[mask]])
            if len(all_vals) > 0:
                lim = [0, all_vals.max() * 1.1]
                ax_n.plot(lim, lim, 'k:', alpha=0.5)
                ax_n.set_xlim(lim)
                ax_n.set_ylim(lim)

            ax_n.set_xlabel(f"n_particles [{folder1}]")
            ax_n.set_ylabel(f"n_particles [{folder2}]")

            # Track length comparison
            ax_t = axes[2, pair_idx]
            x = comparison_df[f"track_length_{folder1}"]
            y = comparison_df[f"track_length_{folder2}"]
            mask = x.notna() & y.notna()
            ax_t.scatter(x[mask], y[mask], alpha=0.6, s=30, color='orange')

            if mask.sum() > 2:
                corr = np.corrcoef(x[mask], y[mask])[0, 1]
                ax_t.set_title(f"Track-Länge: {folder1} vs {folder2} (r={corr:.3f})")
            else:
                ax_t.set_title(f"Track-Länge: {folder1} vs {folder2}")

            all_vals = np.concatenate([x[mask], y[mask]])
            if len(all_vals) > 0:
                lim = [0, all_vals.max() * 1.1]
                ax_t.plot(lim, lim, 'k:', alpha=0.5)
                ax_t.set_xlim(lim)
                ax_t.set_ylim(lim)

            ax_t.set_xlabel(f"avg_track_length [{folder1}]")
            ax_t.set_ylabel(f"avg_track_length [{folder2}]")

            pair_idx += 1

    plt.tight_layout()

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "file_comparison.png", dpi=150)
        print(f"Gespeichert: {save_path / 'file_comparison.png'}")

    plt.show()


def create_matched_files_comparison(all_results: dict, save_path: Path | None = None) -> pd.DataFrame:
    """
    Create detailed comparison for files that exist in multiple folders.
    Returns a DataFrame with side-by-side comparison.
    """
    folder_names = list(all_results.keys())

    # Find files that exist in at least 2 folders
    all_base_names = {}
    for folder_name, folder_results in all_results.items():
        for base_name in folder_results.keys():
            if base_name not in all_base_names:
                all_base_names[base_name] = []
            all_base_names[base_name].append(folder_name)

    # Filter to only matched files (in 2+ folders)
    matched_files = {k: v for k, v in all_base_names.items() if len(v) >= 2}

    if not matched_files:
        print("Keine gematchten Dateien gefunden (Dateien die in mehreren Ordnern vorkommen).")
        return pd.DataFrame()

    print("\n" + "=" * 70)
    print(f"GEMATCHTE DATEIEN VERGLEICH ({len(matched_files)} Dateien in mehreren Ordnern)")
    print("=" * 70)

    rows = []
    for base_name in sorted(matched_files.keys()):
        row = {"base_name": base_name, "in_folders": len(matched_files[base_name])}

        d_values = []
        n_values = []
        len_values = []

        for folder_name in folder_names:
            if folder_name in all_results and base_name in all_results[folder_name]:
                data = all_results[folder_name][base_name]
                row[f"D_{folder_name}"] = data["D_MSD"]
                row[f"n_{folder_name}"] = data["n_particles"]
                row[f"len_{folder_name}"] = data["avg_track_length"]

                if data["D_MSD"] is not None and np.isfinite(data["D_MSD"]):
                    d_values.append(data["D_MSD"])
                if data["n_particles"] > 0:
                    n_values.append(data["n_particles"])
                if data["avg_track_length"] > 0:
                    len_values.append(data["avg_track_length"])
            else:
                row[f"D_{folder_name}"] = np.nan
                row[f"n_{folder_name}"] = np.nan
                row[f"len_{folder_name}"] = np.nan

        # Calculate variation metrics
        if len(d_values) >= 2:
            row["D_std"] = np.std(d_values)
            row["D_range"] = max(d_values) - min(d_values)
            row["D_cv"] = np.std(d_values) / np.mean(d_values) * 100  # Coefficient of variation %
        if len(n_values) >= 2:
            row["n_std"] = np.std(n_values)
            row["n_range"] = max(n_values) - min(n_values)
        if len(len_values) >= 2:
            row["len_std"] = np.std(len_values)

        rows.append(row)

    matched_df = pd.DataFrame(rows)

    # Print summary table
    print(f"\nDatei                          | ", end="")
    for fn in folder_names:
        print(f"D_{fn[:8]:8s} | ", end="")
    print("D_CV%  | Δn_max")
    print("-" * (35 + 14 * len(folder_names) + 20))

    for _, row in matched_df.iterrows():
        print(f"{row['base_name'][:30]:30s} | ", end="")
        for fn in folder_names:
            d_val = row.get(f"D_{fn}", np.nan)
            if pd.notna(d_val):
                print(f"{d_val:10.4f} | ", end="")
            else:
                print(f"{'---':>10s} | ", end="")

        cv = row.get("D_cv", np.nan)
        n_range = row.get("n_range", np.nan)
        cv_str = f"{cv:.1f}" if pd.notna(cv) else "---"
        n_str = f"{int(n_range)}" if pd.notna(n_range) else "---"
        print(f"{cv_str:>6s} | {n_str:>6s}")

    # Save matched comparison
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        csv_path = save_path / "matched_files_comparison.csv"
        matched_df.to_csv(csv_path, index=False)
        print(f"\nGematchte Dateien CSV gespeichert: {csv_path}")

    return matched_df


def plot_matched_files_detail(matched_df: pd.DataFrame, save_path: Path | None = None):
    """Plot detailed comparison for matched files."""
    if matched_df.empty:
        return

    folder_names = [col.replace("D_", "") for col in matched_df.columns if col.startswith("D_") and not col.endswith(("_std", "_range", "_cv"))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. D values per file (grouped bar chart)
    ax1 = axes[0, 0]
    x = np.arange(len(matched_df))
    width = 0.8 / len(folder_names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(folder_names)))

    for i, (fn, color) in enumerate(zip(folder_names, colors)):
        d_col = f"D_{fn}"
        if d_col in matched_df.columns:
            values = matched_df[d_col].fillna(0)
            ax1.bar(x + i * width, values, width, label=fn, color=color, alpha=0.8)

    ax1.set_xlabel("Datei")
    ax1.set_ylabel("D [µm²/s]")
    ax1.set_title("Diffusionskoeffizient pro Datei und Ordner")
    ax1.set_xticks(x + width * (len(folder_names) - 1) / 2)
    ax1.set_xticklabels([n[:15] for n in matched_df["base_name"]], rotation=45, ha="right", fontsize=8)
    ax1.legend()

    # 2. D coefficient of variation
    ax2 = axes[0, 1]
    if "D_cv" in matched_df.columns:
        cv_values = matched_df["D_cv"].dropna()
        ax2.hist(cv_values, bins=15, edgecolor='black', alpha=0.7)
        ax2.axvline(cv_values.mean(), color='red', linestyle='--', label=f'Mittel: {cv_values.mean():.1f}%')
        ax2.axvline(cv_values.median(), color='orange', linestyle='--', label=f'Median: {cv_values.median():.1f}%')
    ax2.set_xlabel("Variationskoeffizient D [%]")
    ax2.set_ylabel("Anzahl Dateien")
    ax2.set_title("Variation von D zwischen Ordnern (pro Datei)")
    ax2.legend()

    # 3. Particle count comparison
    ax3 = axes[1, 0]
    for i, (fn, color) in enumerate(zip(folder_names, colors)):
        n_col = f"n_{fn}"
        if n_col in matched_df.columns:
            values = matched_df[n_col].fillna(0)
            ax3.bar(x + i * width, values, width, label=fn, color=color, alpha=0.8)

    ax3.set_xlabel("Datei")
    ax3.set_ylabel("Anzahl Partikel")
    ax3.set_title("Partikelzahl pro Datei und Ordner")
    ax3.set_xticks(x + width * (len(folder_names) - 1) / 2)
    ax3.set_xticklabels([n[:15] for n in matched_df["base_name"]], rotation=45, ha="right", fontsize=8)
    ax3.legend()

    # 4. Track length comparison
    ax4 = axes[1, 1]
    for i, (fn, color) in enumerate(zip(folder_names, colors)):
        len_col = f"len_{fn}"
        if len_col in matched_df.columns:
            values = matched_df[len_col].fillna(0)
            ax4.bar(x + i * width, values, width, label=fn, color=color, alpha=0.8)

    ax4.set_xlabel("Datei")
    ax4.set_ylabel("Mittlere Track-Länge [Frames]")
    ax4.set_title("Track-Länge pro Datei und Ordner")
    ax4.set_xticks(x + width * (len(folder_names) - 1) / 2)
    ax4.set_xticklabels([n[:15] for n in matched_df["base_name"]], rotation=45, ha="right", fontsize=8)
    ax4.legend()

    plt.tight_layout()

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "matched_files_detail.png", dpi=150)
        print(f"Gespeichert: {save_path / 'matched_files_detail.png'}")

    plt.show()


def print_summary_statistics(all_results: dict, comparison_df: pd.DataFrame):
    """Print summary statistics for each folder."""
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    for folder_name, folder_results in all_results.items():
        d_values = [r["D_MSD"] for r in folder_results.values() if r["D_MSD"] is not None and np.isfinite(r["D_MSD"])]
        n_particles = [r["n_particles"] for r in folder_results.values()]
        track_lengths = [r["avg_track_length"] for r in folder_results.values() if r["avg_track_length"] > 0]

        print(f"\n{folder_name}:")
        print(f"  Dateien: {len(folder_results)}")
        if d_values:
            print(f"  D: {np.mean(d_values):.4f} ± {np.std(d_values):.4f} µm²/s (median: {np.median(d_values):.4f})")
        if n_particles:
            print(f"  Partikel gesamt: {sum(n_particles)}, Mittel pro Datei: {np.mean(n_particles):.1f}")
        if track_lengths:
            print(f"  Track-Länge: {np.mean(track_lengths):.1f} ± {np.std(track_lengths):.1f} Frames")

    # Pairwise D comparison
    print("\n" + "-" * 70)
    print("Paarweise D-Differenzen (gleiche Dateien):")
    folder_names = list(TRACK_FOLDERS.keys())
    for i, folder1 in enumerate(folder_names):
        for j, folder2 in enumerate(folder_names):
            if i >= j:
                continue
            d1 = comparison_df[f"D_{folder1}"]
            d2 = comparison_df[f"D_{folder2}"]
            mask = d1.notna() & d2.notna()
            if mask.sum() > 0:
                diff = (d1[mask] - d2[mask]).values
                rel_diff = ((d1[mask] - d2[mask]) / d1[mask] * 100).values
                print(f"  {folder1} - {folder2}:")
                print(f"    Abs. Diff: {np.mean(diff):.4f} ± {np.std(diff):.4f} µm²/s")
                print(f"    Rel. Diff: {np.mean(rel_diff):.1f} ± {np.std(rel_diff):.1f} %")
                print(f"    Gemeinsame Dateien: {mask.sum()}")


def main():
    # Try to load from cache
    all_results = load_cached_results()

    if all_results is None:
        all_results = compute_all_results()
        if all_results:
            save_results_to_cache(all_results)

    # Check if we have results
    total_files = sum(len(r) for r in all_results.values())
    if total_files == 0:
        print("Keine analysierbaren Dateien gefunden.")
        return

    # Create comparison DataFrame
    comparison_df = create_comparison_dataframe(all_results)

    # Print statistics
    print_summary_statistics(all_results, comparison_df)

    # Create matched files comparison (files in multiple folders)
    matched_df = create_matched_files_comparison(all_results, SAVE_PATH)

    # Save comparison CSV
    if SAVE_PATH:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "track_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nVergleichs-CSV gespeichert: {csv_path}")

    # Create plots
    plot_folder_d_comparison(all_results, SAVE_PATH)
    plot_file_comparison(comparison_df, SAVE_PATH)

    # Plot matched files detail
    if not matched_df.empty:
        plot_matched_files_detail(matched_df, SAVE_PATH)


if __name__ == "__main__":
    main()
