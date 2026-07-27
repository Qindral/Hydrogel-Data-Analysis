"""LiteSizer data visualization following the style guide."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydro_analysis.Litesizer.litesizer_parser import LiteSizerData, MeasurementData, load_litesizer_xlsx

# Style guide colors (Jet-based palette)
COLORS = {
    "base": ["#0000da", "#004cff", "#00c4ff", "#49ffad", "#adff49", "#ffd700", "#ff6800", "#da0000"],
    "dark": ["#000099", "#0035b2", "#0089b2", "#33b279", "#79b233", "#b29600", "#b24900", "#990000"],
    "bright": ["#1f1fde", "#1f61ff", "#1fcbff", "#5bffb8", "#b8ff5b", "#ffde1f", "#ff751f", "#de1f1f"],
}


def setup_style():
    """Apply style guide settings."""
    mpl.rcParams.update({
        "figure.figsize": (7.15, 5.00),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "legend.fontsize": 9,
        "legend.frameon": False,
    })


def calculate_statistics(values: List[float]) -> Tuple[float, float, float]:
    """Calculate mean, standard deviation, and standard error.

    Returns
    -------
    mean, std, sem : tuple of floats
    """
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # Sample std
    sem = std / np.sqrt(len(arr))  # Standard error of mean
    return mean, std, sem


def filter_measurements_by_size(
    measurements: List[MeasurementData],
    size_nm: int,
) -> List[MeasurementData]:
    """Filter measurements by nominal particle size."""
    return [m for m in measurements if f"{size_nm} nm" in m.name]


def plot_size_distributions(
    measurements: List[MeasurementData],
    title: str = "Size Distribution",
    distribution_type: str = "volume",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot size distributions for multiple measurements.

    Parameters
    ----------
    measurements : list of MeasurementData
        Measurements to plot.
    title : str
        Plot title.
    distribution_type : str
        One of "volume", "intensity", "number".
    ax : Axes, optional
        Matplotlib axes to plot on.

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    for i, m in enumerate(measurements):
        # Get distribution data
        if distribution_type == "volume":
            df = m.size_distribution_volume
        elif distribution_type == "intensity":
            df = m.size_distribution_intensity
        else:
            df = m.size_distribution_number

        if df.empty:
            continue

        color = COLORS["base"][i % len(COLORS["base"])]
        ax.plot(
            df["diameter_nm"],
            df["frequency_pct"],
            color=color,
            linewidth=1.8,
            label=m.name,
        )

    ax.set_xlabel("Particle diameter (nm)")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xscale("log")

    # Minor ticks
    ax.minorticks_on()

    return ax


def plot_particle_analysis(
    data: LiteSizerData,
    size_nm: int,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, dict]:
    """Create analysis plot for a specific particle size.

    Parameters
    ----------
    data : LiteSizerData
        Loaded LiteSizer data.
    size_nm : int
        Nominal particle size in nm.
    output_dir : Path, optional
        Directory to save figures.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    stats : dict
        Statistics dictionary.
    """
    setup_style()

    # Filter measurements
    measurements = filter_measurements_by_size(data.measurements, size_nm)
    if not measurements:
        raise ValueError(f"No measurements found for {size_nm} nm particles")

    # Exclude high-PDI measurements
    MAX_PDI = 42.0
    filtered, n_excluded = [], 0
    for m in measurements:
        if m.polydispersity_pct is not None and m.polydispersity_pct > MAX_PDI:
            print(f"  [EXCLUDED PDI={m.polydispersity_pct:.1f}%] {m.name}")
            n_excluded += 1
        else:
            filtered.append(m)
    if n_excluded:
        print(f"  → {n_excluded} measurement(s) excluded (PDI > {MAX_PDI:.0f}%)")
    measurements = filtered
    if not measurements:
        raise ValueError(f"No measurements remain for {size_nm} nm after PDI filter")

    # Calculate statistics
    diameters = [m.hydrodynamic_diameter_nm for m in measurements if m.hydrodynamic_diameter_nm]
    diffusions = [m.diffusion_coefficient_um2s for m in measurements if m.diffusion_coefficient_um2s]
    pdis = [m.polydispersity_pct for m in measurements if m.polydispersity_pct]

    d_mean, d_std, d_sem = calculate_statistics(diameters)
    diff_mean, diff_std, diff_sem = calculate_statistics(diffusions)
    pdi_mean, pdi_std, pdi_sem = calculate_statistics(pdis)

    stats = {
        "diameter_nm": {"mean": d_mean, "std": d_std, "sem": d_sem, "n": len(diameters)},
        "diffusion_um2s": {"mean": diff_mean, "std": diff_std, "sem": diff_sem, "n": len(diffusions)},
        "pdi_pct": {"mean": pdi_mean, "std": pdi_std, "sem": pdi_sem, "n": len(pdis)},
    }

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 5.0), constrained_layout=True)

    # Left: Size distribution
    plot_size_distributions(
        measurements,
        title=f"Size Distribution ({size_nm} nm nominal)",
        distribution_type="volume",
        ax=axes[0],
    )

    # Add mean line
    axes[0].axvline(d_mean, color="#da0000", linestyle="--", linewidth=1.5, alpha=0.8,
                    label=f"Mean: {d_mean:.1f} nm")
    axes[0].legend(loc="upper right")

    # Right: Statistics summary
    ax_stats = axes[1]
    ax_stats.axis("off")

    # Create statistics text
    stats_text = f"""
    Particle Size Analysis: {size_nm} nm (nominal)
    {'='*50}

    Number of measurements: {len(measurements)}

    Hydrodynamic Diameter:
        Mean:  {d_mean:.2f} nm
        Std:   {d_std:.2f} nm
        SEM:   {d_sem:.2f} nm
        Result: {d_mean:.1f} ± {d_sem:.1f} nm (SEM)
                {d_mean:.1f} ± {d_std:.1f} nm (SD)

    Diffusion Coefficient:
        Mean:  {diff_mean:.4f} µm²/s
        Std:   {diff_std:.4f} µm²/s
        SEM:   {diff_sem:.4f} µm²/s
        Result: {diff_mean:.3f} ± {diff_sem:.3f} µm²/s (SEM)

    Polydispersity Index (PDI):
        Mean:  {pdi_mean:.2f}%
        Std:   {pdi_std:.2f}%
        Result: {pdi_mean:.1f} ± {pdi_std:.1f}%

    Individual Measurements:
    """

    for m in measurements:
        stats_text += f"\n      {m.name}: {m.hydrodynamic_diameter_nm:.2f} nm, D = {m.diffusion_coefficient_um2s:.4f} µm²/s"

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, family="monospace", verticalalignment="top")

    # Save figure
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"particle_analysis_{size_nm}nm.png", dpi=600)

    return fig, stats


def print_statistics(stats: dict, size_nm: int) -> None:
    """Print statistics to console."""
    d = stats["diameter_nm"]
    diff = stats["diffusion_um2s"]
    pdi = stats["pdi_pct"]

    print(f"\n{'='*60}")
    print(f"  Statistics for {size_nm} nm particles (n={d['n']})")
    print(f"{'='*60}")
    print(f"\n  Hydrodynamic Diameter:")
    print(f"    {d['mean']:.2f} ± {d['sem']:.2f} nm (SEM)")
    print(f"    {d['mean']:.2f} ± {d['std']:.2f} nm (SD)")
    print(f"\n  Diffusion Coefficient:")
    print(f"    {diff['mean']:.4f} ± {diff['sem']:.4f} µm²/s (SEM)")
    print(f"    {diff['mean']:.4f} ± {diff['std']:.4f} µm²/s (SD)")
    print(f"\n  Polydispersity Index:")
    print(f"    {pdi['mean']:.2f} ± {pdi['std']:.2f}%")
    print(f"{'='*60}\n")


def analyze_all_sizes(
    data: LiteSizerData,
    sizes: List[int],
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Analyze all particle sizes and create summary.

    Parameters
    ----------
    data : LiteSizerData
        Loaded data.
    sizes : list of int
        Particle sizes to analyze.
    output_dir : Path, optional
        Output directory for figures.

    Returns
    -------
    summary_df : DataFrame
        Summary statistics for all sizes.
    """
    setup_style()

    all_stats = []

    for size in sizes:
        try:
            fig, stats = plot_particle_analysis(data, size_nm=size, output_dir=output_dir)
            print_statistics(stats, size_nm=size)
            plt.close(fig)

            # Add to summary
            all_stats.append({
                "nominal_size_nm": size,
                "measured_diameter_nm": stats["diameter_nm"]["mean"],
                "diameter_std": stats["diameter_nm"]["std"],
                "diameter_sem": stats["diameter_nm"]["sem"],
                "diffusion_um2s": stats["diffusion_um2s"]["mean"],
                "diffusion_std": stats["diffusion_um2s"]["std"],
                "diffusion_sem": stats["diffusion_um2s"]["sem"],
                "pdi_pct": stats["pdi_pct"]["mean"],
                "pdi_std": stats["pdi_pct"]["std"],
                "n_measurements": stats["diameter_nm"]["n"],
            })
        except ValueError as e:
            print(f"Skipping {size} nm: {e}")

    return pd.DataFrame(all_stats)


def plot_comparison_summary(
    summary_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """Create comparison plot for all particle sizes.

    Parameters
    ----------
    summary_df : DataFrame
        Summary statistics from analyze_all_sizes.
    output_dir : Path, optional
        Output directory.

    Returns
    -------
    fig : Figure
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(14.3, 10.0), constrained_layout=True)

    # Color for each size
    n_sizes = len(summary_df)
    colors = [COLORS["base"][i % len(COLORS["base"])] for i in range(n_sizes)]

    # 1. Measured vs Nominal diameter
    ax = axes[0, 0]
    ax.errorbar(
        summary_df["nominal_size_nm"],
        summary_df["measured_diameter_nm"],
        yerr=summary_df["diameter_sem"],
        fmt="o",
        markersize=8,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        color=COLORS["base"][0],
        markerfacecolor=COLORS["base"][0],
        markeredgecolor=COLORS["dark"][0],
        markeredgewidth=0.8,
    )
    # Add 1:1 line
    max_size = summary_df["measured_diameter_nm"].max() * 1.1
    ax.plot([0, max_size], [0, max_size], "--", color="#888888", linewidth=1.5, label="1:1")
    ax.set_xlabel("Nominal diameter (nm)")
    ax.set_ylabel("Measured diameter (nm)")
    ax.set_title("Measured vs. Nominal Particle Size")
    ax.legend(loc="upper left")
    ax.set_xlim(0, max_size)
    ax.set_ylim(0, max_size)
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # 2. Diffusion coefficient vs size
    ax = axes[0, 1]
    ax.errorbar(
        summary_df["measured_diameter_nm"],
        summary_df["diffusion_um2s"],
        xerr=summary_df["diameter_sem"],
        yerr=summary_df["diffusion_sem"],
        fmt="s",
        markersize=8,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        color=COLORS["base"][1],
        markerfacecolor=COLORS["base"][1],
        markeredgecolor=COLORS["dark"][1],
        markeredgewidth=0.8,
    )
    ax.set_xlabel("Measured diameter (nm)")
    ax.set_ylabel("Diffusion coefficient (µm²/s)")
    ax.set_title("Diffusion Coefficient vs. Particle Size")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 3. PDI vs size
    ax = axes[1, 0]
    ax.errorbar(
        summary_df["nominal_size_nm"],
        summary_df["pdi_pct"],
        yerr=summary_df["pdi_std"],
        fmt="^",
        markersize=8,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        color=COLORS["base"][2],
        markerfacecolor=COLORS["base"][2],
        markeredgecolor=COLORS["dark"][2],
        markeredgewidth=0.8,
    )
    ax.set_xlabel("Nominal diameter (nm)")
    ax.set_ylabel("Polydispersity Index (%)")
    ax.set_title("PDI vs. Particle Size")
    ax.axhline(10, color="#888888", linestyle="--", linewidth=1, alpha=0.7, label="PDI = 10%")
    ax.legend(loc="upper right")

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis("off")

    table_text = "Summary Statistics (Mean ± SEM)\n" + "=" * 70 + "\n\n"
    table_text += f"{'Size (nm)':<12} {'Diameter (nm)':<20} {'D (µm²/s)':<18} {'PDI (%)':<12} {'n':<5}\n"
    table_text += "-" * 70 + "\n"

    for _, row in summary_df.iterrows():
        table_text += (
            f"{int(row['nominal_size_nm']):<12} "
            f"{row['measured_diameter_nm']:.1f} ± {row['diameter_sem']:.1f}      "
            f"{row['diffusion_um2s']:.3f} ± {row['diffusion_sem']:.3f}   "
            f"{row['pdi_pct']:.1f} ± {row['pdi_std']:.1f}    "
            f"{int(row['n_measurements']):<5}\n"
        )

    ax.text(0.05, 0.95, table_text, transform=ax.transAxes,
            fontsize=10, family="monospace", verticalalignment="top")

    fig.suptitle("LiteSizer Particle Size Analysis - All Sizes", fontsize=14, fontweight="semibold")

    if output_dir:
        output_dir = Path(output_dir)
        fig.savefig(output_dir / "particle_analysis_comparison.png", dpi=600)

    return fig


def main():
    """Analyze all particle sizes."""
    path = Path(r"H:\Daten Promotion Sicherung\Lite Sizer Particle Measurements\Size_repitition_All Sizes.xlsx")
    output_dir = Path(r"H:\Daten Promotion Sicherung\Lite Sizer Particle Measurements\analysis_output")

    print(f"Loading data from: {path}")
    data = load_litesizer_xlsx(path)

    # Available sizes
    sizes = [20, 50, 100, 200, 500, 1000]

    print("\n" + "=" * 70)
    print("  ANALYZING ALL PARTICLE SIZES")
    print("=" * 70)

    # Analyze all sizes
    summary_df = analyze_all_sizes(data, sizes, output_dir=output_dir)

    # Create comparison plot
    fig = plot_comparison_summary(summary_df, output_dir=output_dir)

    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print("\nAll results (Mean ± SEM):")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        print(f"  {int(row['nominal_size_nm']):>4} nm: "
              f"d = {row['measured_diameter_nm']:.1f} ± {row['diameter_sem']:.1f} nm, "
              f"D = {row['diffusion_um2s']:.3f} ± {row['diffusion_sem']:.3f} µm²/s, "
              f"PDI = {row['pdi_pct']:.1f}% (n={int(row['n_measurements'])})")
    print("-" * 70)

    print(f"\nFigures saved to: {output_dir}")

    plt.show()

    return data, summary_df


if __name__ == "__main__":
    main()
