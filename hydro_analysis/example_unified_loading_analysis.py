"""Example: Complete MSD Analysis using Unified Data Loading

This script demonstrates how to use the new unified data loading functions
to perform MSD analysis on multiple datasets organized by particle size.

This replaces the old pattern of manually parsing XMLs and looping through files.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Import unified loading functions
from hydro_analysis.core import (
    group_datasets_by_particle_size,
    load_all_datasets_from_folder,
)

# Import analysis functions (when implemented in core.analysis)
# For now, we'll use trackpy directly
import trackpy as tp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def compute_msd_for_tracks(tracks: pd.DataFrame, max_lagtime: int = 100) -> pd.DataFrame:
    """Compute ensemble MSD for tracks.
    
    Args:
        tracks: Track DataFrame with 'particle', 'frame', 'x', 'y'
        max_lagtime: Maximum lag time for MSD calculation
    
    Returns:
        MSD DataFrame with columns for each particle
    """
    msd = tp.imsd(
        tracks,
        mpp=tracks.attrs.get('mpp', 1.0),
        fps=tracks.attrs.get('fps', 1.0),
        max_lagtime=max_lagtime
    )
    return msd


def fit_diffusion_coefficient(msd: pd.DataFrame, n_points: int = 6) -> dict:
    """Extract diffusion coefficient from MSD.
    
    MSD(t) = 4 * D * t^n
    D = A / 4 where A is the prefactor
    
    Args:
        msd: MSD DataFrame
        n_points: Number of initial points to fit
    
    Returns:
        Dictionary with D, D_err, n, n_err
    """
    import numpy as np
    from scipy.optimize import curve_fit
    
    # Ensemble average
    em = msd.mean(axis=1)
    
    # Fit power law: log(MSD) = log(A) + n*log(t)
    t = em.iloc[:n_points].index.values
    y = em.iloc[:n_points].values
    
    # Remove any NaN or zero values
    mask = np.isfinite(t) & np.isfinite(y) & (t > 0) & (y > 0)
    if mask.sum() < 3:
        return {'D': None, 'D_err': None, 'n': None, 'n_err': None}
    
    log_t = np.log(t[mask])
    log_y = np.log(y[mask])
    
    # Linear fit in log-log space
    coeffs, cov = np.polyfit(log_t, log_y, 1, cov=True)
    n_fit, log_A = coeffs
    
    # Extract errors
    errors = np.sqrt(np.diag(cov))
    n_err = errors[0]
    
    # Convert back to linear space
    A = np.exp(log_A)
    A_err = A * errors[1]
    
    # D = A / 4 for 2D diffusion
    D = A / 4.0
    D_err = A_err / 4.0
    
    return {
        'D': D,
        'D_err': D_err,
        'n': n_fit,
        'n_err': n_err,
        'A': A,
        'A_err': A_err
    }


def theoretical_diffusion(diameter_nm: float, temp_K: float = 293.15, 
                         viscosity_Pa_s: float = 0.001002) -> float:
    """Calculate theoretical diffusion coefficient using Stokes-Einstein.
    
    D = k_B * T / (3 * π * η * d)
    
    Args:
        diameter_nm: Particle diameter in nanometers
        temp_K: Temperature in Kelvin (default: 20°C)
        viscosity_Pa_s: Dynamic viscosity in Pa·s (default: water at 20°C)
    
    Returns:
        Diffusion coefficient in µm²/s
    """
    k_B = 1.380649e-23  # Boltzmann constant [J/K]
    
    # Convert diameter to meters
    diameter_m = diameter_nm * 1e-9
    
    # Calculate D in m²/s
    D_m2_s = k_B * temp_K / (3 * 3.14159 * viscosity_Pa_s * diameter_m)
    
    # Convert to µm²/s
    D_um2_s = D_m2_s * 1e12
    
    return D_um2_s


# ============================================================================
# EXAMPLE 1: Analyze by Particle Size
# ============================================================================

def example_1_analyze_by_particle_size():
    """Example: Load data grouped by particle size and analyze."""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Analyze by Particle Size")
    print("="*80 + "\n")
    
    # Define paths
    root_path = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung")
    
    if not root_path.exists():
        print(f"⚠️  Path not found: {root_path}")
        print("Adjust the path in the script to match your data location.")
        return
    
    # Load data grouped by particle size (ONE LINE!)
    size_groups = group_datasets_by_particle_size(
        root_path=root_path,
        min_length=30  # Filter short tracks
    )
    
    if not size_groups:
        print("No particle sizes detected in folder structure.")
        return
    
    # Analyze each size
    results = []
    
    for size_nm, tracks in sorted(size_groups.items()):
        print(f"\n🔬 Analyzing {size_nm} nm particles...")
        print(f"   Particles: {tracks['particle'].nunique()}")
        print(f"   Detections: {len(tracks)}")
        
        # Compute MSD
        msd = compute_msd_for_tracks(tracks, max_lagtime=100)
        
        # Fit diffusion coefficient
        fit_result = fit_diffusion_coefficient(msd, n_points=6)
        
        # Calculate theoretical value
        D_theory = theoretical_diffusion(size_nm)
        
        # Store results
        results.append({
            'particle_size_nm': size_nm,
            'n_particles': tracks['particle'].nunique(),
            'n_detections': len(tracks),
            'D_measured': fit_result['D'],
            'D_error': fit_result['D_err'],
            'D_theory': D_theory,
            'exponent_n': fit_result['n'],
            'exponent_n_err': fit_result['n_err'],
            'mpp': tracks.attrs.get('mpp', None),
            'fps': tracks.attrs.get('fps', None),
        })
        
        print(f"   D_measured = {fit_result['D']:.4f} ± {fit_result['D_err']:.4f} µm²/s")
        print(f"   D_theory   = {D_theory:.4f} µm²/s")
        print(f"   Ratio      = {fit_result['D']/D_theory:.2f}")
        print(f"   Exponent n = {fit_result['n']:.3f} ± {fit_result['n_err']:.3f}")
    
    # Create summary DataFrame
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df_results.to_string(index=False))
    
    # Save results
    output_path = Path("unified_loading_results.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: D vs particle size
    ax1.errorbar(
        df_results['particle_size_nm'],
        df_results['D_measured'],
        yerr=df_results['D_error'],
        fmt='o',
        markersize=8,
        label='Measured',
        capsize=5
    )
    ax1.plot(
        df_results['particle_size_nm'],
        df_results['D_theory'],
        'k--',
        linewidth=2,
        label='Theory (Stokes-Einstein)'
    )
    ax1.set_xlabel('Particle Size (nm)', fontsize=12)
    ax1.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Diffusion vs Particle Size')
    
    # Plot 2: Measured/Theory ratio
    ratio = df_results['D_measured'] / df_results['D_theory']
    ax2.plot(
        df_results['particle_size_nm'],
        ratio,
        'o-',
        markersize=8
    )
    ax2.axhline(y=1.0, color='k', linestyle='--', label='Perfect agreement')
    ax2.set_xlabel('Particle Size (nm)', fontsize=12)
    ax2.set_ylabel('D_measured / D_theory', fontsize=12)
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Agreement with Theory')
    
    plt.tight_layout()
    
    output_fig = Path("unified_loading_comparison.png")
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to: {output_fig}")
    
    plt.show()


# ============================================================================
# EXAMPLE 2: Load All Datasets from Folder (No Grouping)
# ============================================================================

def example_2_load_all_datasets():
    """Example: Load all datasets individually."""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Load All Datasets")
    print("="*80 + "\n")
    
    root_path = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung")
    
    if not root_path.exists():
        print(f"⚠️  Path not found: {root_path}")
        return
    
    # Load all datasets as dictionary
    datasets = load_all_datasets_from_folder(
        root_path=root_path,
        min_length=30,
        combine_per_dataset=True  # Combine XMLs within each dataset
    )
    
    print(f"\n✅ Loaded {len(datasets)} datasets:\n")
    
    for name, tracks in datasets.items():
        print(f"📁 {name}:")
        print(f"   Particles: {tracks['particle'].nunique()}")
        print(f"   Detections: {len(tracks)}")
        print(f"   MPP: {tracks.attrs.get('mpp', 'N/A')} µm/px")
        print(f"   FPS: {tracks.attrs.get('fps', 'N/A')}")
        print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("UNIFIED DATA LOADING - PRACTICAL EXAMPLES")
    print("="*80)
    
    # Run examples
    try:
        example_1_analyze_by_particle_size()
    except Exception as e:
        print(f"\n❌ Example 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_2_load_all_datasets()
    except Exception as e:
        print(f"\n❌ Example 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80)
