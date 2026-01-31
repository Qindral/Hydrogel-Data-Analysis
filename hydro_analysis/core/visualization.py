"""Visualization functions that return figures.
Includes MSD plots and step size histograms.
Here wont be stored any data loading or analysis methods.
"""

from pathlib import Path
from typing import List, Dict, Any, Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .physics import calculate_theoretical_diffusion


def _normalize_results(results: Dict[str, Dict[str, Any]] | Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(results, pd.DataFrame):
        raise ValueError("Expected standardized result_dict entries (with tracks_df). Pass a dict or list of result_dicts.")
    if isinstance(results, dict):
        return [r for r in results.values() if r is not None]
    return [r for r in results if r is not None]


# ============================================================================
# VISUALIZATION MSD
# ============================================================================

def plot_msd_results(result_dict: Dict[str, Any], save_path: Path = None):
    """
    Create MSD plot with individual and ensemble curves from result_dict.
    
    Args:
        result_dict: Standardized result dictionary with fit_results_MSD
        save_path: Path to save PNG
    """
    if result_dict['fit_results_MSD'] is None:
        print("No MSD results to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    msd_result = result_dict['fit_results_MSD']
    imsd = msd_result['imsd']
    emsd = msd_result['emsd']
    A = msd_result['A']
    n = msd_result['exponent']
    particle_size = result_dict['particle_size_nm']

    # Plot individual MSDs (lighter)
    for col in imsd.columns[:50]:  # Limit to 50 for clarity
        ax.loglog(imsd.index, imsd[col], alpha=0.1, color='gray')
    
    # Calculate standard error for ensemble MSD
    sem = imsd.std(axis=1) / np.sqrt(imsd.shape[1])
    
    # Plot ensemble MSD with error bars
    ax.errorbar(emsd.index, emsd.values, yerr=sem.values,
                fmt='o-', label=f'Ensemble MSD (N={result_dict["num_tracks"]})', 
                linewidth=2, markersize=4, color='blue',
                capsize=3, capthick=1, elinewidth=1, alpha=0.8)
    
    # Plot fit with uncertainty band
    x_fit = np.logspace(np.log10(emsd.index[0]), np.log10(emsd.index[5]), 100)
    y_fit = A * x_fit**n
    
    # Calculate uncertainty band from parameter errors
    A_upper = (A + msd_result['A_err'])
    A_lower = (A - msd_result['A_err'])
    n_upper = (n + msd_result['exponent_error'])
    n_lower = (n - msd_result['exponent_error'])
    
    y_upper = A_upper * x_fit**n_upper
    y_lower = A_lower * x_fit**n_lower
    
    # Plot fit line
    ax.loglog(x_fit, y_fit, '--', 
             label=f'Fit: A={A:.3f}±{msd_result["A_err"]:.3f}, n={n:.3f}±{msd_result["exponent_error"]:.3f}',
             linewidth=2, color='red')
    
    # Add uncertainty band
    if msd_result['A_err'] > 1:
        ax.fill_between(x_fit, y_lower, y_upper, 
                        color='red', alpha=0.2, label='Fit Uncertainty')
    
    D = msd_result['D_um2_per_s']
    D_err = msd_result['D_error']
    n = msd_result['exponent']
    n_err = msd_result['exponent_error']
    
    if particle_size is not None:
        D_thr = calculate_theoretical_diffusion(particle_size_nm=particle_size)
        ax.loglog(x_fit, (2 * D_thr) * x_fit, 'g:', 
                 label=f'Theoretical MSD (Particle Size: {particle_size} nm)', linewidth=2)


    ax.set_xlabel('Lag Time (s)', fontsize=12)
    ax.set_ylabel('MSD (µm²)', fontsize=12)
    ax.set_title(f'MSD Analysis\nD = {D:.4f} ± {D_err:.4f} µm²/s\n'
                f'Exponent n = {n:.3f} ± {n_err:.3f}', fontsize=14)
    ax.legend(fontsize=10, loc='best')    
    fig.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  MSD plot saved: {save_path}")

def plot_trajectories(result_dict: Dict[str, Any], save_path: Path, max_tracks: int = 100, fading: bool = True):
    """
    Plot particle trajectories from result_dict.
    
    Args:
        result_dict: Standardized result dictionary with tracks_df
        save_path: Path to save PNG
        max_tracks: Maximum number of tracks to plot
        fading: Whether to apply fading effect to trajectories
    """
    tracks = result_dict['tracks_df']
    if tracks is None or tracks.empty:
        print("No tracks to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_particles = tracks['particle'].unique()
    n_plot = min(len(unique_particles), max_tracks)
    if not fading:
        for pid in unique_particles[:n_plot]:
            track = tracks[tracks['particle'] == pid].sort_values('frame')
            ax.plot(track['x'], track['y'], alpha=0.5, linewidth=1)
    else:
        for pid in unique_particles[:n_plot]:
            track = tracks[tracks['particle'] == pid].sort_values('frame')
            n_points = len(track)
            for i in range(n_points - 1):
                alpha = (i + 1) / n_points  # Fading effect
                ax.plot(track['x'].values[i:i+2], track['y'].values[i:i+2], 
                        color='blue', alpha=alpha, linewidth=3)
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'Particle Trajectories (N={result_dict["num_tracks"]})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    fig.tight_layout()
    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Trajectory plot saved: {save_path}")


# ============================================================================
# VISUALIZATION StepSize
# ============================================================================

def plot_stepsize_results(result_dict: Dict[str, Any], bins: int = 50, save_path: Path = None):
    """
    Create step size histogram with Gaussian fit from result_dict.
    
    Args:
        result_dict: Standardized result dictionary with fit_results_step
        bins: Number of histogram bins
        save_path: Path to save PNG
    """
    if result_dict['fit_results_step'] is None:
        print("No step size results to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    step_result = result_dict['fit_results_step']
    fit_x = step_result['fit_x']
    fit_y = step_result['fit_y']
    step_df = step_result['step_df']
    mpp = result_dict['mpp']
    
    displacements_x = step_df['dx'].values * mpp
    displacements_y = step_df['dy'].values * mpp
    
    # Histogram
    n, bins, patches_x = ax.hist(displacements_x, bins=bins, density=True, 
                                alpha=0.6, color='blue', edgecolor='black',
                                label='X Displacements')
    n, bins, patches_y = ax.hist(displacements_y, bins=bins, density=True, 
                                alpha=0.6, color='green', edgecolor='black',
                                label='Y Displacements')
    
    # Gaussian fit parameters with errors
    mu_x = fit_x['mu_nm'] / 1000.0
    sigma_x = fit_x['sigma_nm'] / 1000.0
    mu_y = fit_y['mu_nm'] / 1000.0
    sigma_y = fit_y['sigma_nm'] / 1000.0
    
    mu_err_x = 0.0
    sigma_err_x = 0.0
    mu_err_y = 0.0
    sigma_err_y = 0.0
    
    x = np.linspace(min(displacements_x.min(),displacements_y.min()), max(displacements_x.max(), displacements_y.max()), 200)
    
    # Plot main Gaussian fit
    gaussian_x = stats.norm.pdf(x, mu_x, sigma_x)
    gaussian_y = stats.norm.pdf(x, mu_y, sigma_y)
    ax.plot(x, gaussian_x, 'b-', linewidth=2.5, 
           label=f'Gaussian Fit X\nμ = {mu_x:.4f} ± {mu_err_x:.4f} µm\nσ = {sigma_x:.4f} ± {sigma_err_x:.4f} µm')
    ax.plot(x, gaussian_y, 'g-', linewidth=2.5, 
           label=f'Gaussian Fit Y\nμ = {mu_y:.4f} ± {mu_err_y:.4f} µm\nσ = {sigma_y:.4f} ± {sigma_err_y:.4f} µm')
    # Add uncertainty bands (±1 standard error)
    gaussian_upper_mu_x = stats.norm.pdf(x, mu_x + mu_err_x, sigma_x)
    gaussian_lower_mu_x = stats.norm.pdf(x, mu_x - mu_err_x, sigma_x)
    gaussian_upper_sigma_x = stats.norm.pdf(x, mu_x, sigma_x + sigma_err_x)
    gaussian_lower_sigma_x = stats.norm.pdf(x, mu_x, sigma_x - sigma_err_x)
    
    # Combined uncertainty (approximate)
    gaussian_upper_x = np.maximum(gaussian_upper_mu_x, gaussian_upper_sigma_x)
    gaussian_lower_x = np.minimum(gaussian_lower_mu_x, gaussian_lower_sigma_x)
    
    ax.fill_between(x, gaussian_lower_x, gaussian_upper_x, 
                    color='red', alpha=0.2, label='Fit Uncertainty (±1σ)')
    
     # Add uncertainty bands (±1 standard error)
    gaussian_upper_mu_y = stats.norm.pdf(x, mu_y + mu_err_y, sigma_y)
    gaussian_lower_mu_y = stats.norm.pdf(x, mu_y - mu_err_y, sigma_y)
    gaussian_upper_sigma_y = stats.norm.pdf(x, mu_y, sigma_y + sigma_err_y)
    gaussian_lower_sigma_y = stats.norm.pdf(x, mu_y, sigma_y - sigma_err_y)
    
    # Combined uncertainty (approximate)
    gaussian_upper_y = np.maximum(gaussian_upper_mu_y, gaussian_upper_sigma_y)
    gaussian_lower_y = np.minimum(gaussian_lower_mu_y, gaussian_lower_sigma_y)
    
    ax.fill_between(x, gaussian_lower_y, gaussian_upper_y, 
                    color='red', alpha=0.2, label='Fit Uncertainty (±1σ)')
    
    D_x = step_result['D_x_um2_per_s']
    dt_x = step_result['dt_ms'] / 1000.0

    D_y = step_result['D_y_um2_per_s']
    
    ax.set_xlabel('Displacement (µm)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Step Size Analysis (Δt = {dt_x:.3f} s)\\n'
                f'D_x = {D_x:.4f} µm²/s, D_y = {D_y:.4f} µm²/s\\n'
                f'N = {step_result["n_steps_x"]} steps', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add quality indicator as text
    # quality = stepsize_result_x['quality']
    # quality_color = 'green' if quality == 'PASS' else 'orange'
    # ax.text(0.02, 0.98, f'Quality: {quality}', 
    #        transform=ax.transAxes, fontsize=10,
    #        verticalalignment='top', 
    #        bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.3))
    
    fig.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Step size plot saved: {save_path}")


def plot_step_size_overlay(
    results: Dict[str, Dict[str, Any]] | Iterable[Dict[str, Any]],
    save_path: Path,
    step_interval: int = 1,
) -> None:
    """
    Create overlay histogram showing step size distributions for each individual file.
    
    Args:
        results: Iterable or dict of standardized result_dict entries
        save_path: Directory to save plot
        step_interval: Unused when step_df already exists in fit_results_step
    """
    print("\nCreating step size overlay plot (individual files)...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color map for different files
    results_list = _normalize_results(results)
    n_files = len(results_list)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_files, 20)))
    
    for idx, result in enumerate(results_list):
        mpp = result.get('mpp')
        particle_size = result.get('particle_size_nm')
        base_name = result.get('base_name', 'unknown')

        step_result = result.get('fit_results_step')
        step_df = step_result.get('step_df') if step_result else None
        if step_df is None or step_df.empty or mpp is None:
            continue
            
        # Calculate Euclidean step sizes from dx and dy for visualization
        dx_nm = step_df['dx'].values * mpp * 1000.0  # Convert to nm
        dy_nm = step_df['dy'].values * mpp * 1000.0  # Convert to nm
        steps_nm = np.sqrt(dx_nm**2 + dy_nm**2)
        
        # Plot histogram with transparency
        color = colors[idx % len(colors)]
        size_label = f"{particle_size:.0f} nm" if particle_size is not None else "unknown size"
        label = f"{size_label} - {base_name[:30]} (N={len(steps_nm)}, mean={np.mean(steps_nm):.1f} nm)"
        ax.hist(steps_nm, bins=60, alpha=0.4, color=color, edgecolor='black', linewidth=0.3,
               label=label)
    
    ax.set_xlabel('Step size [nm]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Step Size Distribution - Individual Files', fontsize=16, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'water_step_size_overlay_individual_files.png', dpi=300)
    plt.close(fig)
    
    print(f"[OK] Step size overlay plot saved to {save_path}")

def plot_dx_dy_distributions(
    results: Dict[str, Dict[str, Any]] | Iterable[Dict[str, Any]],
    save_path: Path = None,
    step_interval: int = 1,
) -> None:
    """
    Create histograms of dx and dy step component distributions for each individual file.
    
    This shows the directional components of particle motion, which should be
    centered around zero for isotropic Brownian motion.
    
    Args:
        results: Iterable or dict of standardized result_dict entries
        save_path: Directory to save plots
        step_interval: Unused when step_df already exists in fit_results_step
    """
    print("\nCreating individual file dx/dy distribution plots...")
    
    results_list = _normalize_results(results)
    for result in results_list:
        mpp = result.get('mpp')
        particle_size = result.get('particle_size_nm')
        xml_name = result.get('base_name', 'unknown')

        step_result = result.get('fit_results_step')
        step_df = step_result.get('step_df') if step_result else None
        if step_df is None or step_df.empty or mpp is None:
            continue
            
        all_dx = step_df['dx'].values * mpp * 1000.0  # Convert to nm
        all_dy = step_df['dy'].values * mpp * 1000.0  # Convert to nm
        
        # Create figure with 3 subplots: dx, dy, and 2D histogram
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot dx distribution
        ax1 = axes[0, 0]
        ax1.hist(all_dx, bins=50, alpha=0.7, color='cornflowerblue', edgecolor='black')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.axvline(np.mean(all_dx), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(all_dx):.1f} nm')
        ax1.set_xlabel('dx [nm]', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('X-Direction Displacement Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot dy distribution
        ax2 = axes[0, 1]
        ax2.hist(all_dy, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvline(np.mean(all_dy), color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(all_dy):.1f} nm')
        ax2.set_xlabel('dy [nm]', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Y-Direction Displacement Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 2D histogram (dx vs dy)
        ax3 = axes[1, 0]
        h = ax3.hist2d(all_dx, all_dy, bins=50, cmap='Blues', cmin=1)
        plt.colorbar(h[3], ax=ax3, label='Counts')
        ax3.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
        ax3.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
        ax3.set_xlabel('dx [nm]', fontsize=12)
        ax3.set_ylabel('dy [nm]', fontsize=12)
        ax3.set_title('2D Displacement Distribution', fontsize=12, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # Plot scatter with transparency to show density
        ax4 = axes[1, 1]
        # Subsample if too many points
        n_points = len(all_dx)
        if n_points > 5000:
            indices = np.random.choice(n_points, 5000, replace=False)
            dx_plot = np.array(all_dx)[indices]
            dy_plot = np.array(all_dy)[indices]
            alpha_val = 0.1
        else:
            dx_plot = all_dx
            dy_plot = all_dy
            alpha_val = 0.3
        
        ax4.scatter(dx_plot, dy_plot, s=10, alpha=alpha_val, c='navy')
        ax4.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
        ax4.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
        
        # Add circle showing mean step size
        mean_r = np.mean(np.sqrt(np.array(all_dx)**2 + np.array(all_dy)**2))
        circle = plt.Circle((0, 0), mean_r, color='red', fill=False, 
                           linestyle='--', linewidth=2, label=f'Mean r = {mean_r:.1f} nm')
        ax4.add_patch(circle)
        
        ax4.set_xlabel('dx [nm]', fontsize=12)
        ax4.set_ylabel('dy [nm]', fontsize=12)
        ax4.set_title('Displacement Scatter Plot', fontsize=12, fontweight='bold')
        ax4.set_aspect('equal')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Overall title
        size_label = f"{particle_size:.0f} nm" if particle_size is not None else "unknown size"
        fig.suptitle(f'Displacement Component Analysis - {size_label}\n{xml_name}\n'
                    f'N = {len(all_dx)} steps, σ(dx) = {np.std(all_dx):.1f} nm, '
                    f'σ(dy) = {np.std(all_dy):.1f} nm',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        # Create safe filename from xml_name
        safe_filename = xml_name.replace('.xml', '').replace(' ', '_').replace('/', '_')
        if save_path is not None:
            size_slug = f"{particle_size:.0f}nm" if particle_size is not None else "unknown"
            plt.savefig(save_path / f'water_dx_dy_dist_{size_slug}_{safe_filename}.png', dpi=300)
            plt.close(fig)
    
    print(f"[OK] Individual file dx/dy distribution plots saved to {save_path}")


def plot_1d_steps_and_gaussian_fit(
    steps_px,
    fit_1d: dict,
    mpp_um_per_px: float,
    bins: int = 60,
    ax=None,
):
    steps_px = np.asarray(steps_px, dtype=float)
    steps_nm = steps_px * mpp_um_per_px * 1000.0

    mu = float(fit_1d.get("mu_nm", np.nan))
    sigma = float(fit_1d.get("sigma_nm", np.nan))
    D = float(fit_1d.get("D_um2_per_s", np.nan))
    axis = str(fit_1d.get("axis", ""))

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(steps_nm, bins=bins, density=True, alpha=0.6)

    if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
        x = np.linspace(np.nanmin(steps_nm), np.nanmax(steps_nm), 400)
        ax.plot(x, stats.norm.pdf(x, loc=mu, scale=sigma))

    ax.set_xlabel(f"step size {axis} (nm)")
    ax.set_ylabel("density")
    ax.set_title(f"{axis}: D = {D:.4g} µm²/s")

    return ax


def plot_overlaid_xy_hist_and_fits(
    steps_x_px,
    fit_x: dict,
    steps_y_px,
    fit_y: dict,
    mpp_um_per_px: float,
    bins: int = 60,
    ax=None,
):
    steps_x_px = np.asarray(steps_x_px, dtype=float)
    steps_y_px = np.asarray(steps_y_px, dtype=float)
    steps_x_nm = steps_x_px * mpp_um_per_px * 1000.0
    steps_y_nm = steps_y_px * mpp_um_per_px * 1000.0

    mux, sx = float(fit_x.get("mu_nm", np.nan)), float(fit_x.get("sigma_nm", np.nan))
    muy, sy = float(fit_y.get("mu_nm", np.nan)), float(fit_y.get("sigma_nm", np.nan))
    Dx, Dy = float(fit_x.get("D_um2_per_s", np.nan)), float(fit_y.get("D_um2_per_s", np.nan))

    if ax is None:
        fig, ax = plt.subplots()

    lo = np.nanmin([np.nanmin(steps_x_nm), np.nanmin(steps_y_nm)])
    hi = np.nanmax([np.nanmax(steps_x_nm), np.nanmax(steps_y_nm)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1.0, 1.0

    ax.hist(steps_x_nm, bins=bins, range=(lo, hi), density=True, alpha=0.45, label=f"x (D={Dx:.4g})")
    ax.hist(steps_y_nm, bins=bins, range=(lo, hi), density=True, alpha=0.45, label=f"y (D={Dy:.4g})")

    x = np.linspace(lo, hi, 500)
    if np.isfinite(mux) and np.isfinite(sx) and sx > 0:
        ax.plot(x, stats.norm.pdf(x, loc=mux, scale=sx))
    if np.isfinite(muy) and np.isfinite(sy) and sy > 0:
        ax.plot(x, stats.norm.pdf(x, loc=muy, scale=sy))

    ax.set_xlabel("step size (nm)")
    ax.set_ylabel("density")
    ax.legend()
    return ax




# ============================================================================
# VISUALIZATION Final Plots
# ============================================================================



# ============================================================================
# VISUALIZATION Comparision MSD Vs StepSize 
# ============================================================================


def plot_diffusion_comparison(summary_df: pd.DataFrame, save_path: Path = None) -> None:
    """
    Create comparison plot of measured vs theoretical diffusion coefficients.
    Shows individual measurements and averages with error bars.
    Follows the project style guide with logarithmic axes.

    Uses the same input format as plot_theory_comparison for consistency.

    Args:
        summary_df: DataFrame with columns: particle_size_nm, D_MSD_um2_per_s, weight
        save_path: Directory to save plot (optional)
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter

    # Style guide colors (Jet-derived palette)
    COLOR_MEASURED_BASE = '#0000da'   # Index 1 base (blue)
    COLOR_MEASURED_DARK = '#000099'   # Index 1 dark
    COLOR_DLS = '#da00bd'             # Markierung A base (magenta)
    COLOR_DLS_DARK = '#9b5191'        # Markierung A bright
    COLOR_THEORY = 'black'

    # DLS measurements in µm²/s
    DLS_MEASUREMENTS = {
        20: 12.38750325,
        50: 8.201969711,
        100: 4.139082033,
        200: 1.745323167,
        500: 0.621773811,
        1000: 0.356862091,
    }

    df = summary_df.dropna(subset=['particle_size_nm']).copy()

    if df.empty:
        print("Cannot create diffusion comparison: no particle size data")
        return

    # Style guide: figure size (7.15, 5.00) inch, ratio 1.43:1
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

    # Apply style guide tick settings (slightly larger)
    ax.tick_params(axis='both', which='major', direction='in', length=5.0,
                   width=1.2, top=True, right=True, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=2.5,
                   width=1.0, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    # Determine data and axis range
    all_sizes = df['particle_size_nm'].unique()
    x_min, x_max = 15, 1500

    # Plot theoretical curve (dashed line)
    sizes_range = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    D_theory_range = [calculate_theoretical_diffusion(s) for s in sizes_range]
    ax.plot(sizes_range, D_theory_range, color=COLOR_THEORY, linewidth=2.2,
           linestyle='--', zorder=5)

    # Plot theoretical points at standard sizes
    theory_sizes = [20, 50, 100, 200, 500, 1000]
    theory_D = [calculate_theoretical_diffusion(s) for s in theory_sizes]
    ax.scatter(theory_sizes, theory_D, s=80, marker='x', c=COLOR_THEORY,
              linewidths=2.0, zorder=6)

    # Horizontal offset factor for log scale
    offset_factor = 0.93

    # Collect statistics for averages
    msd_stats = []  # (size, mean, std, n_particles)

    # Plot MSD individual measurements and collect stats
    if 'D_MSD_um2_per_s' in df.columns:
        for size in sorted(df['particle_size_nm'].unique()):
            subset = df[df['particle_size_nm'] == size]
            # Get rows with valid D values
            valid_subset = subset.dropna(subset=['D_MSD_um2_per_s'])
            msd_vals = valid_subset['D_MSD_um2_per_s'].values
            if len(msd_vals) > 0:
                x_offset = size * offset_factor
                # Individual: larger markers, alpha 0.6
                ax.scatter([x_offset] * len(msd_vals), msd_vals,
                          s=50, alpha=0.6, marker='o',
                          facecolors=COLOR_MEASURED_BASE, edgecolors='none',
                          zorder=2)
                # Sum particle counts (weights) - use valid_subset to match D values
                if 'weight' in valid_subset.columns:
                    weights = valid_subset['weight'].fillna(0).values
                    n_particles = int(np.sum(weights))
                else:
                    n_particles = len(msd_vals)
                msd_stats.append((size, np.mean(msd_vals), np.std(msd_vals), n_particles))

    # Plot MSD averages with error bars
    if msd_stats:
        sizes_msd = [s[0] * offset_factor for s in msd_stats]
        means_msd = [s[1] for s in msd_stats]
        stds_msd = [s[2] for s in msd_stats]
        ax.errorbar(sizes_msd, means_msd, yerr=stds_msd,
                   fmt='o', markersize=12, markerfacecolor=COLOR_MEASURED_BASE,
                   markeredgecolor=COLOR_MEASURED_DARK, markeredgewidth=1.2,
                   ecolor=COLOR_MEASURED_DARK, elinewidth=1.8, capsize=5.0, capthick=1.8,
                   zorder=10, label='_nolegend_')

    # Plot DLS reference values
    dls_sizes = [s for s in all_sizes if s in DLS_MEASUREMENTS]
    dls_D = [DLS_MEASUREMENTS[s] for s in dls_sizes]
    if dls_sizes:
        ax.scatter(dls_sizes, dls_D, s=150, marker='*',
                  facecolors=COLOR_DLS, edgecolors=COLOR_DLS_DARK,
                  linewidths=1.0, zorder=8)

    # Create legend handles
    legend_elements = [
        Line2D([0], [0], color=COLOR_THEORY, linewidth=2.2, linestyle='--',
               label='Stokes-Einstein Theory'),
        Line2D([0], [0], marker='x', color=COLOR_THEORY, markersize=10,
               markeredgewidth=2.0, label='Theoretical D', linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLOR_DLS,
               markeredgecolor=COLOR_DLS_DARK, markersize=14, markeredgewidth=1.0,
               label='DLS Reference', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MEASURED_BASE,
               markeredgecolor=COLOR_MEASURED_DARK, markersize=12, markeredgewidth=1.2,
               label='MSD Mean (± SD)', linestyle='None'),
        Line2D([0], [0], marker='o', color=COLOR_MEASURED_BASE, markersize=8, alpha=0.6,
               label='MSD Individual', linestyle='None'),
    ]

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)

    # Custom x-axis ticks at particle sizes
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(['20', '50', '100', '200', '500', '1000']))
    ax.xaxis.set_minor_locator(FixedLocator([]))  # No minor ticks

    # Axis labels (larger: 12pt)
    ax.set_xlabel('Particle Size (nm)', fontsize=12)
    ax.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)

    # Title (style guide: 12pt, semibold)
    ax.set_title('Diffusion Coefficients: MSD Method vs Theory',
                fontsize=13, fontweight='semibold')

    # Legend
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=False)

    # Add sample size annotations (n = total trajectories)
    for size, mean, std, n in msd_stats:
        y_pos = mean / (1 + std / mean + 0.4) if mean > 0 and std > 0 else mean * 0.6
        ax.annotate(f'n={n}', xy=(size * offset_factor, y_pos),
                   fontsize=9, ha='center', va='top', color=COLOR_MEASURED_DARK)

    plt.show()

    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
        # Save as PNG (style guide: 600 dpi)
        png_path = save_path / 'diffusion_comparison_msd.png'
        fig.savefig(png_path, dpi=600, bbox_inches='tight')

        # Also save as PDF
        pdf_path = save_path / 'diffusion_comparison_msd.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')

        print(f"Diffusion comparison plot saved: {png_path}")
        print(f"Diffusion comparison plot saved: {pdf_path}")

        plt.close(fig)


def plot_theory_comparison(summary_df: pd.DataFrame, save_path: Path):
    """
    Plot measured vs theoretical diffusion coefficients.

    Shows individual measurements as small transparent markers and
    averages with error bars (standard deviation) as larger prominent markers.
    Includes DLS reference measurements and Stokes-Einstein theory curve.
    Follows the project style guide with logarithmic axes.

    Args:
        summary_df: DataFrame with results (columns: particle_size_nm, D_MSD_um2_per_s, weight)
        save_path: Path to save plot
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter

    # Style guide colors (Jet-derived palette)
    COLOR_MSD_BASE = '#0000da'       # Index 1 base (blue)
    COLOR_MSD_DARK = '#000099'       # Index 1 dark
    COLOR_STEP_BASE = '#da0000'      # Index 8 base (red)
    COLOR_STEP_DARK = '#990000'      # Index 8 dark
    COLOR_DLS = '#da00bd'            # Markierung A base (magenta)
    COLOR_DLS_DARK = '#9b5191'       # Markierung A bright
    COLOR_THEORY = 'black'

    # DLS measurements in µm²/s (reference values)
    DLS_MEASUREMENTS = {
        20: 12.38750325,
        50: 8.201969711,
        100: 4.139082033,
        200: 1.745323167,
        500: 0.621773811,
        1000: 0.356862091,
    }

    df = summary_df.dropna(subset=['particle_size_nm']).copy()

    if df.empty:
        print("Cannot create theory comparison: no particle size data")
        return

    # Calculate theoretical D for each particle size
    df['D_theory'] = df['particle_size_nm'].apply(calculate_theoretical_diffusion)

    # Style guide: figure size (7.15, 5.00) inch, ratio 1.43:1
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

    # Apply style guide tick settings (slightly larger)
    ax.tick_params(axis='both', which='major', direction='in', length=5.0,
                   width=1.2, top=True, right=True, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=2.5,
                   width=1.0, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    # Axis range
    x_min, x_max = 15, 1500

    # Plot theoretical curve (dashed line)
    sizes_range = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    D_theory_range = [calculate_theoretical_diffusion(s) for s in sizes_range]
    ax.plot(sizes_range, D_theory_range, color=COLOR_THEORY, linewidth=2.2,
           linestyle='--', label='Stokes-Einstein Theory', zorder=5)

    # Horizontal offset factor for better visibility (multiplicative for log scale)
    offset_factor = 0.90  # MSD slightly left
    offset_factor_right = 1.10  # Step size slightly right

    # Collect statistics for averages
    msd_stats = []  # (size, mean, std, n_particles)
    step_stats = []

    # Plot MSD individual measurements and collect stats
    if 'D_MSD_um2_per_s' in df.columns:
        for size in sorted(df['particle_size_nm'].unique()):
            subset = df[df['particle_size_nm'] == size]
            valid_subset = subset.dropna(subset=['D_MSD_um2_per_s'])
            msd_vals = valid_subset['D_MSD_um2_per_s'].values
            if len(msd_vals) > 0:
                x_offset = size * offset_factor
                # Individual: larger markers, alpha 0.6
                ax.scatter([x_offset] * len(msd_vals), msd_vals,
                          s=50, alpha=0.6, marker='o',
                          facecolors=COLOR_MSD_BASE, edgecolors='none',
                          zorder=2)
                # Sum particle counts (weights) - use fillna to handle NaN
                if 'weight' in valid_subset.columns:
                    n_particles = int(valid_subset['weight'].fillna(0).sum())
                else:
                    n_particles = len(msd_vals)
                msd_stats.append((size, np.mean(msd_vals), np.std(msd_vals), n_particles))

    # Plot step size individual measurements and collect stats
    if 'D_stepsize_um2_per_s' in df.columns:
        for size in sorted(df['particle_size_nm'].unique()):
            subset = df[df['particle_size_nm'] == size]
            valid_subset = subset.dropna(subset=['D_stepsize_um2_per_s'])
            step_vals = valid_subset['D_stepsize_um2_per_s'].values
            if len(step_vals) > 0:
                x_offset = size * offset_factor_right
                # Individual: larger markers, alpha 0.6
                ax.scatter([x_offset] * len(step_vals), step_vals,
                          s=50, alpha=0.6, marker='s',
                          facecolors=COLOR_STEP_BASE, edgecolors='none',
                          zorder=2)
                # Sum particle counts (weights) - use fillna to handle NaN
                if 'weight' in valid_subset.columns:
                    n_particles = int(valid_subset['weight'].fillna(0).sum())
                else:
                    n_particles = len(step_vals)
                step_stats.append((size, np.mean(step_vals), np.std(step_vals), n_particles))

    # Plot MSD averages with error bars
    if msd_stats:
        sizes_msd = [s[0] * offset_factor for s in msd_stats]
        means_msd = [s[1] for s in msd_stats]
        stds_msd = [s[2] for s in msd_stats]
        ax.errorbar(sizes_msd, means_msd, yerr=stds_msd,
                   fmt='o', markersize=12, markerfacecolor=COLOR_MSD_BASE,
                   markeredgecolor=COLOR_MSD_DARK, markeredgewidth=1.2,
                   ecolor=COLOR_MSD_DARK, elinewidth=1.8, capsize=5.0, capthick=1.8,
                   zorder=10, label='_nolegend_')

    # Plot step size averages with error bars
    if step_stats:
        sizes_step = [s[0] * offset_factor_right for s in step_stats]
        means_step = [s[1] for s in step_stats]
        stds_step = [s[2] for s in step_stats]
        ax.errorbar(sizes_step, means_step, yerr=stds_step,
                   fmt='s', markersize=12, markerfacecolor=COLOR_STEP_BASE,
                   markeredgecolor=COLOR_STEP_DARK, markeredgewidth=1.2,
                   ecolor=COLOR_STEP_DARK, elinewidth=1.8, capsize=5.0, capthick=1.8,
                   zorder=10, label='_nolegend_')

    # Plot DLS reference values for ALL standard sizes
    standard_sizes = [20, 50, 100, 200, 500, 1000]
    dls_sizes = [s for s in standard_sizes if s in DLS_MEASUREMENTS]
    dls_D = [DLS_MEASUREMENTS[s] for s in dls_sizes]
    if dls_sizes:
        ax.scatter(dls_sizes, dls_D, s=150, marker='*',
                  facecolors=COLOR_DLS, edgecolors=COLOR_DLS_DARK,
                  linewidths=1.0, zorder=8)

    # Plot theoretical points at ALL standard sizes
    theory_sizes = standard_sizes
    theory_D = [calculate_theoretical_diffusion(s) for s in theory_sizes]
    ax.scatter(theory_sizes, theory_D, s=80, marker='x', c=COLOR_THEORY,
              linewidths=2.0, zorder=6)

    # Create legend handles (larger markers to match plot)
    legend_elements = [
        Line2D([0], [0], color=COLOR_THEORY, linewidth=2.2, linestyle='--',
               label='Stokes-Einstein Theory'),
        Line2D([0], [0], marker='x', color=COLOR_THEORY, markersize=10,
               markeredgewidth=2.0, label='Theoretical D', linestyle='None'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLOR_DLS,
               markeredgecolor=COLOR_DLS_DARK, markersize=14, markeredgewidth=1.0,
               label='DLS Reference', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MSD_BASE,
               markeredgecolor=COLOR_MSD_DARK, markersize=12, markeredgewidth=1.2,
               label='MSD Method (Mean ± SD)', linestyle='None'),
        Line2D([0], [0], marker='o', color=COLOR_MSD_BASE, markersize=8, alpha=0.6,
               label='MSD Individual', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_STEP_BASE,
               markeredgecolor=COLOR_STEP_DARK, markersize=12, markeredgewidth=1.2,
               label='Step Size Method (Mean ± SD)', linestyle='None'),
        Line2D([0], [0], marker='s', color=COLOR_STEP_BASE, markersize=8, alpha=0.6,
               label='Step Size Individual', linestyle='None'),
    ]

    # Logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)

    # Custom x-axis ticks at particle sizes
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(['20', '50', '100', '200', '500', '1000']))
    ax.xaxis.set_minor_locator(FixedLocator([]))  # No minor ticks

    # Axis labels (larger: 12pt)
    ax.set_xlabel('Particle Size (nm)', fontsize=12)
    ax.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)

    # Title (style guide: 13pt, semibold)
    ax.set_title('Measured vs Theoretical Diffusion Coefficients',
                fontsize=13, fontweight='semibold')

    # Legend
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=False)

    # Add sample size annotations (n = total trajectories)
    for size, mean, std, n in msd_stats:
        y_pos = mean / (1 + std / mean + 0.4) if mean > 0 and std > 0 else mean * 0.6
        ax.annotate(f'n={n}', xy=(size * offset_factor, y_pos),
                   fontsize=9, ha='center', va='top', color=COLOR_MSD_DARK)
    for size, mean, std, n in step_stats:
        y_pos = mean / (1 + std / mean + 0.4) if mean > 0 and std > 0 else mean * 0.6
        ax.annotate(f'n={n}', xy=(size * offset_factor_right, y_pos),
                   fontsize=9, ha='center', va='top', color=COLOR_STEP_DARK)

    plt.show()

    # Save as PNG (style guide: 600 dpi)
    fig.savefig(save_path, dpi=600, bbox_inches='tight')

    # Also save as PDF (vector format for publication)
    pdf_path = save_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')

    plt.close(fig)
    print(f"Theory comparison plot saved: {save_path}")
    print(f"Theory comparison plot saved: {pdf_path}")


# ============================================================================
# VISUALIZATION All Particles Sizes
# ============================================================================
