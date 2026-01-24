"""Visualization functions that return figures instead of saving."""

from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.analysis import calculate_theoretical_diffusion


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_msd_results(msd_result: Dict,particle_size: float, save_path: Path):
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


def plot_stepsize_results(stepsize_result: Dict, save_path: Path):
    """
    Create step size histogram with Gaussian fit.
    
    Args:
        stepsize_result: Results from perform_stepsize_analysis()
        save_path: Path to save PNG
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    displacements = stepsize_result['displacements']
    
    # Histogram
    n, bins, patches = ax.hist(displacements, bins=50, density=True, 
                                alpha=0.6, color='blue', edgecolor='black',
                                label='Observed Displacements')
    
    # Gaussian fit parameters with errors
    mu = stepsize_result['mu']
    sigma = stepsize_result['sigma']
    
    # Get errors from fit_gaussian_to_displacements
    fit_stats = fit_gaussian_to_displacements(displacements)
    mu_err = fit_stats['mu_err']
    sigma_err = fit_stats['sigma_err']
    
    x = np.linspace(displacements.min(), displacements.max(), 200)
    
    # Plot main Gaussian fit
    gaussian = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, gaussian, 'r-', linewidth=2.5, 
           label=f'Gaussian Fit\nμ = {mu:.4f} ± {mu_err:.4f} µm\nσ = {sigma:.4f} ± {sigma_err:.4f} µm')
    
    # Add uncertainty bands (±1 standard error)
    gaussian_upper_mu = stats.norm.pdf(x, mu + mu_err, sigma)
    gaussian_lower_mu = stats.norm.pdf(x, mu - mu_err, sigma)
    gaussian_upper_sigma = stats.norm.pdf(x, mu, sigma + sigma_err)
    gaussian_lower_sigma = stats.norm.pdf(x, mu, sigma - sigma_err)
    
    # Combined uncertainty (approximate)
    gaussian_upper = np.maximum(gaussian_upper_mu, gaussian_upper_sigma)
    gaussian_lower = np.minimum(gaussian_lower_mu, gaussian_lower_sigma)
    
    ax.fill_between(x, gaussian_lower, gaussian_upper, 
                    color='red', alpha=0.2, label='Fit Uncertainty (±1σ)')
    
    D = stepsize_result['D_um2_per_s']
    D_err = stepsize_result['D_error']
    dt = stepsize_result['dt']
    
    ax.set_xlabel('Displacement (µm)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Step Size Analysis (Δt = {dt:.3f} s)\n'
                f'D = {D:.4f} ± {D_err:.4f} µm²/s\n'
                f'N = {stepsize_result["n_steps"]} steps', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add quality indicator as text
    quality = stepsize_result['quality']
    quality_color = 'green' if quality == 'PASS' else 'orange'
    ax.text(0.02, 0.98, f'Quality: {quality}', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor=quality_color, alpha=0.3))
    
    fig.tight_layout()
    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Step size plot saved: {save_path}")


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

