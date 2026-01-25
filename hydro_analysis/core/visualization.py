"""Visualization functions that return figures instead of saving."""

from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from core.analysis import calculate_theoretical_diffusion 


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_msd_results(msd_result: Dict,particle_size: float, save_path: Path = None):
    """
    Create MSD plot with individual and ensemble curves.
    
    Args:
        msd_result: Results from perform_msd_analysis()
        save_path: Path to save PNG
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    imsd = msd_result['imsd']
    emsd = msd_result['emsd']    
    fit_result = msd_result['fit_result']
    A = fit_result["A"][0]
    n = fit_result["n"][0]

    # Plot individual MSDs (lighter)
    for col in imsd.columns[:50]:  # Limit to 50 for clarity
        ax.loglog(imsd.index, imsd[col], alpha=0.1, color='gray')
    
    # Calculate standard error for ensemble MSD
    sem = imsd.std(axis=1) / np.sqrt(imsd.shape[1])
    
    # Plot ensemble MSD with error bars
    ax.errorbar(emsd.index, emsd.values, yerr=sem.values,
                fmt='o-', label=f'Ensemble MSD (N={msd_result["n_particles"]})', 
                linewidth=2, markersize=4, color='blue',
                capsize=3, capthick=1, elinewidth=1, alpha=0.8)
    
    # Plot fit with uncertainty band
    x_fit = np.logspace(np.log10(emsd.index[0]), np.log10(emsd.index[5]), 100)
    y_fit = A * x_fit**n
    
    # Calculate uncertainty band from parameter errors
    A_upper = (A + fit_result["A_err"][0])
    A_lower = (A - fit_result["A_err"][0])
    n_upper = (n + fit_result["n_err"][0])
    n_lower = (n - fit_result["n_err"][0])
    
    y_upper = A_upper * x_fit**n_upper
    y_lower = A_lower * x_fit**n_lower
    
    # Plot fit line
    ax.loglog(x_fit, y_fit, '--', 
             label=f'Fit: A={A:.3f}±{fit_result["A_err"][0]:.3f}, n={n:.3f}±{fit_result["n_err"][0]:.3f}',
             linewidth=2, color='red')
    
    # Add uncertainty band
    if fit_result["A_err"][0] > 1:
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


def plot_stepsize_results(stepsize_result_x: Dict,stepsize_result_y: Dict,bins: int = 50, save_path: Path = None):
    """
    Create step size histogram with Gaussian fit.
    
    {'dx_all': dx_all,
        'D_um2_per_s': D,
        'D_error': D_error,
        'sigma_x': sigma_fit,
        'mean_x': x0_fit,
        'sigma_err_x': perr[2],
        'mean_err_x': perr[1],
        'n_steps': len(dx_all),
        'quality_ok': len(quality_issues) == 0,
        'quality_issues': quality_issues if quality_issues else ['OK']
    }

    Args:
        stepsize_result_x: Results from perform_stepsize_analysis() for x displacements
        stepsize_result_y: Results from perform_stepsize_analysis() for y displacements
        save_path: Path to save PNG
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    displacements_x = stepsize_result_x['dx_all']
    displacements_y = stepsize_result_y['dx_all']
    
    # Histogram
    n, bins, patches_x = ax.hist(displacements_x, bins=50, density=True, 
                                alpha=0.6, color='blue', edgecolor='black',
                                label='Observed Displacements')
    n, bins, patches_y = ax.hist(displacements_y, bins=50, density=True, 
                                alpha=0.6, color='green', edgecolor='black',
                                label='Observed Displacements')
    
    # Gaussian fit parameters with errors
    mu_x = stepsize_result_x['mean_x']
    sigma_x = stepsize_result_x['sigma_x']
    mu_y = stepsize_result_y['mean_x']
    sigma_y = stepsize_result_y['sigma_x']
    
    mu_err_x = stepsize_result_x['mean_err_x']
    sigma_err_x = stepsize_result_x['sigma_err_x']
    mu_err_y = stepsize_result_y['mean_err_x']
    sigma_err_y = stepsize_result_y['sigma_err_x']
    
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
    
    D_x = stepsize_result_x['D_um2_per_s']
    D_err_x = stepsize_result_x['D_error']
    dt_x = stepsize_result_x['dt']

    D_y = stepsize_result_y['D_um2_per_s']
    D_err_y = stepsize_result_y['D_error']
    dt_y = stepsize_result_y['dt']
    
    ax.set_xlabel('Displacement (µm)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Step Size Analysis (Δt = {dt_x:.3f} s)\n'
                f'D = {D_x:.4f} ± {D_err_x:.4f} µm²/s\n'
                f'N = {stepsize_result_x["n_steps"]} steps', fontsize=14)
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


def plot_step_size_overlay(results_df: pd.DataFrame, save_path: Path) -> None:
    """
    Create overlay histogram showing step size distributions for each individual file.
    
    Args:
        results_df: DataFrame from analyze_all_files()
        file_records.append({
                'particle_size_nm': particle_size,
                'xml_path': str(xml_file),
                'xml_name': xml_file.name,
                'x_max': x_max,
                'y_max': y_max,
                'exposure_ms': exposure_ms,
                'delay_ms': delay_ms,
                'fps': fps,
                'mpp': mpp,
                'mode': mode,
            })

        save_path: Directory to save plot
    """
    print("\nCreating step size overlay plot (individual files)...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color map for different files
    n_files = len(results_df)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_files, 20)))
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        xml_path = Path(row['xml_path'])
        particle_size = row['particle_size_nm']
        
        # Load and calculate step sizes for this file
        df = read_trackmate_xml(xml_path)
        if df is None:
            continue
            
        step_df = calculate_step_sizes(df, step_interval=STEP_INTERVAL)
        if step_df.empty:
            continue
            
        # Calculate Euclidean step sizes from dx and dy for visualization
        dx_nm = step_df['dx'].values * row['mpp'] * 1000.0  # Convert to nm
        dy_nm = step_df['dy'].values * row['mpp'] * 1000.0  # Convert to nm
        steps_nm = np.sqrt(dx_nm**2 + dy_nm**2)
        
        # Plot histogram with transparency
        color = colors[idx % len(colors)]
        label = f"{particle_size:.0f} nm - {row['xml_name'][:30]} (N={len(steps_nm)}, mean={np.mean(steps_nm):.1f} nm)"
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



def plot_trajectories(tracks: pd.DataFrame, save_path: Path, max_tracks: int = 100, fading: bool = True):
    """
    Plot particle trajectories.
    
    Args:
        tracks: Track DataFrame
        save_path: Path to save PNG
        max_tracks: Maximum number of tracks to plot
        fading: Whether to apply fading effect to trajectories
    """
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
    ax.set_title(f'Particle Trajectories (N={tracks["particle"].nunique()})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    fig.tight_layout()
    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Trajectory plot saved: {save_path}")

def plot_theory_comparison(summary_df: pd.DataFrame, save_path: Path):
    """
    Plot measured vs theoretical diffusion coefficients.
    
    Args:
        summary_df: DataFrame with results
        save_path: Path to save plot
    """
    df = summary_df.dropna(subset=['particle_size_nm']).copy()
    
    if df.empty:
        print("Cannot create theory comparison: no particle size data")
        return
    
    # Calculate theoretical D for each particle size
    df['D_theory'] = df['particle_size_nm'].apply(calculate_theoretical_diffusion)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot theoretical curve
    sizes_range = np.logspace(np.log10(df['particle_size_nm'].min()), 
                             np.log10(df['particle_size_nm'].max()), 100)
    D_theory_range = [calculate_theoretical_diffusion(s) for s in sizes_range]
    ax.plot(sizes_range, D_theory_range, 'k-', linewidth=2.5, 
           label='Stokes-Einstein Theory', zorder=10)
    
    # Plot MSD measurements
    if 'D_MSD_um2_per_s' in df.columns:
        for size in df['particle_size_nm'].unique():
            subset = df[df['particle_size_nm'] == size]
            msd_vals = subset['D_MSD_um2_per_s'].dropna()
            if len(msd_vals) > 0:
                ax.scatter([size] * len(msd_vals), msd_vals, 
                          s=150, alpha=0.6, marker='o', 
                          edgecolors='blue', linewidths=2,
                          facecolors='lightblue')
    
    # Plot step size measurements
    if 'D_stepsize_um2_per_s' in df.columns:
        for size in df['particle_size_nm'].unique():
            subset = df[df['particle_size_nm'] == size]
            step_vals = subset['D_stepsize_um2_per_s'].dropna()
            if len(step_vals) > 0:
                ax.scatter([size] * len(step_vals), step_vals,
                          s=150, alpha=0.6, marker='^',
                          edgecolors='red', linewidths=2,
                          facecolors='lightcoral')
    
    # Create legend handles manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=2.5, label='Stokes-Einstein Theory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
               markeredgecolor='blue', markersize=12, markeredgewidth=2, 
               label='MSD Method', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='lightcoral',
               markeredgecolor='red', markersize=12, markeredgewidth=2,
               label='Step Size Method', linestyle='None')
    ]
    
    ax.set_xlabel('Particle Size (nm)', fontsize=14)
    ax.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=14)
    ax.set_title('Measured vs Theoretical Diffusion Coefficients', 
                fontsize=16, fontweight='bold')
    ax.legend(handles=legend_elements, loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    fig.tight_layout()
    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Theory comparison plot saved: {save_path}")

def plot_diffusion_comparison(summary_df: pd.DataFrame, save_path: Path):
    """
    Create boxplot comparing MSD and Step Size methods by particle size.
    
    Args:
        summary_df: DataFrame with results from multiple files
        save_path: Path to save comparison plot
    """
    if summary_df.empty or 'particle_size_nm' not in summary_df.columns:
        print("Cannot create comparison plot: insufficient data")
        return
    
    # Remove entries without particle size
    df = summary_df.dropna(subset=['particle_size_nm']).copy()
    
    if df.empty:
        print("Cannot create comparison plot: no particle size information found")
        return
    
    # Get unique particle sizes
    particle_sizes = sorted(df['particle_size_nm'].unique())
    
    # Prepare data for boxplot
    msd_data = []
    stepsize_data = []
    labels = []
    
    for size in particle_sizes:
        subset = df[df['particle_size_nm'] == size]
        
        # MSD data
        if 'D_MSD_um2_per_s' in subset.columns:
            msd_values = subset['D_MSD_um2_per_s'].dropna().values
            if len(msd_values) > 0:
                msd_data.append(msd_values)
            else:
                msd_data.append([])
        else:
            msd_data.append([])
        
        # Step size data
        if 'D_stepsize_um2_per_s' in subset.columns:
            step_values = subset['D_stepsize_um2_per_s'].dropna().values
            if len(step_values) > 0:
                stepsize_data.append(step_values)
            else:
                stepsize_data.append([])
        else:
            stepsize_data.append([])
        
        labels.append(f'{int(size)} nm')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Boxplots side by side
    x_pos = np.arange(len(particle_sizes))
    width = 0.35
    
    # MSD boxplots
    bp1 = ax1.boxplot(msd_data, positions=x_pos - width/2, widths=width*0.8,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='darkblue', linewidth=2),
                      showfliers=True)
    
    # Step size boxplots
    bp2 = ax1.boxplot(stepsize_data, positions=x_pos + width/2, widths=width*0.8,
                      patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', alpha=0.7),
                      medianprops=dict(color='darkred', linewidth=2),
                      showfliers=True)
    
    # Set x-axis tick labels
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    
    # Plot theoretical values as black X markers
    theoretical_D = [calculate_theoretical_diffusion(size) for size in particle_sizes]
    ax1.scatter(x_pos, theoretical_D, marker='x', s=200, linewidths=3,
               color='black', zorder=10, label='Stokes-Einstein Theory')
    
    ax1.set_xlabel('Particle Size', fontsize=12)
    ax1.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax1.set_title('Diffusion Coefficient Comparison by Particle Size', fontsize=14, fontweight='bold')
    
    # Update legend to include theory
    from matplotlib.lines import Line2D
    legend_elements = [
        bp1["boxes"][0],
        bp2["boxes"][0],
        Line2D([0], [0], marker='x', color='black', linestyle='None', 
               markersize=12, markeredgewidth=3, label='Theory')
    ]
    ax1.legend(legend_elements, ['MSD Method', 'Step Size Method', 'Stokes-Einstein Theory'], 
              loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Plot 2: Method agreement scatter plot
    for size in particle_sizes:
        subset = df[df['particle_size_nm'] == size]
        
        if 'D_MSD_um2_per_s' in subset.columns and 'D_stepsize_um2_per_s' in subset.columns:
            msd_vals = subset['D_MSD_um2_per_s'].values
            step_vals = subset['D_stepsize_um2_per_s'].values
            
            # Only plot where both values exist
            mask = ~np.isnan(msd_vals) & ~np.isnan(step_vals)
            if mask.sum() > 0:
                ax2.scatter(msd_vals[mask], step_vals[mask], 
                          s=100, alpha=0.6, label=f'{int(size)} nm',
                          edgecolors='black', linewidths=1)
    
    # Add diagonal line (perfect agreement)
    all_D = []
    if 'D_MSD_um2_per_s' in df.columns:
        all_D.extend(df['D_MSD_um2_per_s'].dropna().values)
    if 'D_stepsize_um2_per_s' in df.columns:
        all_D.extend(df['D_stepsize_um2_per_s'].dropna().values)
    
    if all_D:
        lim_min = min(all_D) * 0.5
        lim_max = max(all_D) * 2
        ax2.plot([lim_min, lim_max], [lim_min, lim_max], 
                'k--', linewidth=1.5, alpha=0.5, label='Perfect Agreement')
        ax2.set_xlim(lim_min, lim_max)
        ax2.set_ylim(lim_min, lim_max)
    
    ax2.set_xlabel('D from MSD (µm²/s)', fontsize=12)
    ax2.set_ylabel('D from Step Size (µm²/s)', fontsize=12)
    ax2.set_title('Method Agreement', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_aspect('equal')
    
    fig.tight_layout()
    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot saved: {save_path}")


def plot_dx_dy_distributions(results_df: pd.DataFrame, save_path: Path = None) -> None:
    """
    Create histograms of dx and dy step component distributions for each individual file.
    
    This shows the directional components of particle motion, which should be
    centered around zero for isotropic Brownian motion.
    
    Args:
        results_df: DataFrame from analyze_all_files()
        save_path: Directory to save plots
    """
    print("\nCreating individual file dx/dy distribution plots...")
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        xml_path = Path(row['xml_path'])
        particle_size = row['particle_size_nm']
        xml_name = row['xml_name']
        
        # Load and calculate step components for this file
        df = read_trackmate_xml(xml_path)
        if df is None:
            continue
            
        step_df = calculate_step_sizes(df, step_interval=STEP_INTERVAL)
        if step_df.empty:
            continue
            
        all_dx = step_df['dx'].values * row['mpp'] * 1000.0  # Convert to nm
        all_dy = step_df['dy'].values * row['mpp'] * 1000.0  # Convert to nm
        
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
        fig.suptitle(f'Displacement Component Analysis - {particle_size:.0f} nm\n{xml_name}\n'
                    f'N = {len(all_dx)} steps, σ(dx) = {np.std(all_dx):.1f} nm, '
                    f'σ(dy) = {np.std(all_dy):.1f} nm',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        # Create safe filename from xml_name
        safe_filename = xml_name.replace('.xml', '').replace(' ', '_').replace('/', '_')
        if save_path is not None:
            plt.savefig(save_path / f'water_dx_dy_dist_{particle_size:.0f}nm_{safe_filename}.png', dpi=300)
            plt.close(fig)
    
    print(f"[OK] Individual file dx/dy distribution plots saved to {save_path}")

def plot_diffusion_comparison(combined_df: pd.DataFrame, results_df: pd.DataFrame, save_path: Path = None) -> None:
    """
    Create comparison plot of measured vs theoretical diffusion coefficients.
    Shows individual files discretely with their uncertainties.
    
    Args:
        combined_df: DataFrame from combine_by_particle_size()
        results_df: DataFrame from analyze_all_files() with individual file results
        save_path: Directory to save plot
    """
    # DLS measurements in µm²/s (converted from original nm²/ms values)
    DLS_MEASUREMENTS = {
        20: 12.38750325 * 1e3 / 1000.0,   # nm²/ms -> µm²/s
        50: 8.201969711 * 1e3 / 1000.0,
        100: 4.139082033 * 1e3 / 1000.0,
        200: 1.745323167 * 1e3 / 1000.0,
        500: 0.621773811 * 1e3 / 1000.0,
        1000: 0.356862091 * 1e3 / 1000.0
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    particle_sizes = combined_df['particle_size_nm'].values
    measured_D = combined_df['D_measured'].values
    measured_D_err = combined_df['D_measured_std'].values
    theoretical_D = combined_df['D_theoretical'].values
    
    # Color palette for different particle sizes
    colors_by_size = plt.cm.tab10(np.linspace(0, 1, len(particle_sizes)))
    size_to_color = dict(zip(particle_sizes, colors_by_size))
    
    # Plot individual files with small horizontal offset for visibility
    # Use different markers for different FPS modes and gray out bad quality fits
    offset_scale = 0.15  # Adjust this to control horizontal spread
    for idx, file_data in results_df.iterrows():
        size = file_data['particle_size_nm']
        D = file_data['D']
        D_err = file_data['D_std']
        mode = file_data.get('mode', 'Unknown')
        quality = file_data.get('quality_flag', 'unknown')
        
        # Different markers for different FPS modes
        if mode == '60 FPS':
            marker = '^'  # Triangle up for 60 FPS
            base_alpha = 0.8
        elif mode == '20 FPS':
            marker = 's'  # Square for 20 FPS
            base_alpha = 0.7
        else:
            marker = 'o'  # Circle for unknown
            base_alpha = 0.6
        
        # Gray out bad quality fits
        if quality != 'good':
            color = 'gray'
            alpha = 0.3
            linewidth = 1.0
        else:
            color = size_to_color[size]
            alpha = base_alpha
            linewidth = 1.5
        
        # Calculate how many files we have for this size and create offset
        size_files = results_df[results_df['particle_size_nm'] == size]
        file_index = list(size_files.index).index(idx)
        num_files = len(size_files)
        
        # Center the offsets around the nominal size
        if num_files > 1:
            offset = (file_index - (num_files - 1) / 2) * (size * offset_scale / num_files)
        else:
            offset = 0
        
        x_pos = size + offset
        
        # Plot individual file with error bar
        ax.errorbar(x_pos, D, yerr=D_err, fmt=marker, 
                   markersize=7, color=color, 
                   ecolor=color, elinewidth=linewidth, 
                   capsize=3, capthick=linewidth, alpha=alpha)
    
    # Add custom legend entries for different FPS modes and quality
    ax.errorbar([], [], [], fmt='^', markersize=7, color='gray', 
               ecolor='gray', elinewidth=1.5, capsize=3, capthick=1.5,
               label='60 FPS (good quality)', alpha=0.8)
    ax.errorbar([], [], [], fmt='s', markersize=7, color='gray', 
               ecolor='gray', elinewidth=1.5, capsize=3, capthick=1.5,
               label='20 FPS (good quality)', alpha=0.7)
    ax.errorbar([], [], [], fmt='o', markersize=7, color='gray', 
               ecolor='gray', elinewidth=1.0, capsize=3, capthick=1.0,
               label='Bad quality (grayed out)', alpha=0.3)
    
    # Plot mean values for each size (larger markers)
    ax.errorbar(particle_sizes, measured_D, yerr=measured_D_err, fmt='D', 
               markersize=10, color='blue', ecolor='black', elinewidth=2, 
               capsize=5, capthick=2, label='Mean D per Size ± Std', zorder=5)
    
    # Plot theoretical values
    ax.scatter(particle_sizes, theoretical_D, s=150, color='black', 
              marker='x', linewidths=3, label='Theoretical D (Stokes-Einstein)', zorder=6)
    
    # Plot DLS values if available
    dls_sizes = [s for s in particle_sizes if s in DLS_MEASUREMENTS]
    dls_D = [DLS_MEASUREMENTS[s] for s in dls_sizes]
    if dls_sizes:
        ax.scatter(dls_sizes, dls_D, s=150, color='red', marker='*', 
                  linewidths=2, edgecolors='darkred', label='D from DLS', zorder=6)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Particle size [nm]', fontsize=12)
    ax.set_ylabel('Diffusion coefficient D [µm²/s]', fontsize=12)
    ax.set_title('Diffusion Coefficients in water: Comparing two analysis methods\nStep Size Method vs Theory', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path / 'diffusion_comparison_stepsize_individual.png', dpi=300)
        print(f"\n✓ Comparison plot saved to: {save_path / 'diffusion_comparison_stepsize_individual.png'}")
        
        plt.close(fig)
