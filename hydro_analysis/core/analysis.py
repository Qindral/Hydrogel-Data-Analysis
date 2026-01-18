"""Analysis functions for diffusion calculations."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_step_size_diffusion(
    tracks: pd.DataFrame,
    step_interval: int = 1,
    max_sigma_ratio: float = 1.5,
    max_mean_sigma_ratio: float = 0.3
) -> dict:
    """Compute diffusion coefficient from step size distributions.
    
    Method: For each track, calculate frame-to-frame displacements:
        dx_i = x_{i+1} - x_i
        dy_i = y_{i+1} - y_i
    
    Fit Gaussian distributions to dx and dy to extract variance σ².
    For 2D Brownian motion: σ² = 2*D*dt, therefore:
        D = σ² / (2 * dt)
    
    Args:
        tracks: DataFrame with columns ['particle', 'frame', 'x', 'y']
                and attrs {'mpp', 'fps'}
        step_interval: Use every nth step (1=all, 6=every 6th)
        max_sigma_ratio: Max ratio between sigma_x and sigma_y for isotropy
        max_mean_sigma_ratio: Max |mean|/sigma for centered distribution
    
    Returns:
        Dictionary with keys:
        - D_um2_per_s: Diffusion coefficient in µm²/s
        - D_error: Error estimate
        - sigma_x, sigma_y: Standard deviations in µm
        - mean_x, mean_y: Mean displacements in µm (should be ~0)
        - n_steps: Number of steps used
        - quality_ok: Boolean indicating if quality criteria met
        - quality_issues: List of quality warnings
    """
    mpp = tracks.attrs.get('mpp', 0.15)
    fps = tracks.attrs.get('fps', 20.0)
    dt = step_interval / fps  # Time between analyzed steps
    
    # Collect displacements from all tracks
    dx_all, dy_all = [], []
    
    for pid in tracks['particle'].unique():
        track = tracks[tracks['particle'] == pid].sort_values('frame')
        
        # Use every nth step
        track_subset = track.iloc[::step_interval].copy()
        
        if len(track_subset) < 2:
            continue
        
        # Calculate displacements in pixels
        dx_px = np.diff(track_subset['x'].values)
        dy_px = np.diff(track_subset['y'].values)
        
        # Convert to micrometers
        dx_all.extend(dx_px * mpp)
        dy_all.extend(dy_px * mpp)
    
    if len(dx_all) < 10:
        return {
            'D_um2_per_s': np.nan,
            'D_error': np.nan,
            'sigma_x': np.nan,
            'sigma_y': np.nan,
            'mean_x': np.nan,
            'mean_y': np.nan,
            'n_steps': len(dx_all),
            'quality_ok': False,
            'quality_issues': ['Insufficient steps (<10)']
        }
    
    dx_arr = np.array(dx_all)
    dy_arr = np.array(dy_all)
    
    # Fit Gaussians to get mean and sigma
    mean_x, sigma_x = dx_arr.mean(), dx_arr.std(ddof=1)
    mean_y, sigma_y = dy_arr.mean(), dy_arr.std(ddof=1)
    
    # Combined variance (average of x and y)
    sigma_sq = (sigma_x**2 + sigma_y**2) / 2.0
    
    # Diffusion coefficient: D = σ² / (2 * dt)
    D = sigma_sq / (2.0 * dt)
    
    # Error estimate (propagating std error)
    se_x = sigma_x / np.sqrt(len(dx_arr))
    se_y = sigma_y / np.sqrt(len(dy_arr))
    D_error = np.sqrt(se_x**2 + se_y**2) / (2.0 * dt)
    
    # Quality checks
    quality_issues = []
    
    # Check isotropy (sigma_x ≈ sigma_y)
    sigma_ratio = max(sigma_x, sigma_y) / (min(sigma_x, sigma_y) + 1e-10)
    if sigma_ratio > max_sigma_ratio:
        quality_issues.append(f'Anisotropic (σ_x/σ_y = {sigma_ratio:.2f})')
    
    # Check for drift (mean should be close to 0)
    mean_sigma_x = abs(mean_x) / (sigma_x + 1e-10)
    mean_sigma_y = abs(mean_y) / (sigma_y + 1e-10)
    
    if mean_sigma_x > max_mean_sigma_ratio:
        quality_issues.append(f'Drift in X (|µ|/σ = {mean_sigma_x:.2f})')
    if mean_sigma_y > max_mean_sigma_ratio:
        quality_issues.append(f'Drift in Y (|µ|/σ = {mean_sigma_y:.2f})')
    
    return {
        'D_um2_per_s': D,
        'D_error': D_error,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'sigma_ratio': sigma_ratio,
        'n_steps': len(dx_all),
        'quality_ok': len(quality_issues) == 0,
        'quality_issues': quality_issues if quality_issues else ['OK']
    }


def compute_theoretical_diffusion(particle_size_nm: float, 
                                  temperature_K: float = 293.15,
                                  viscosity_pa_s: float = 0.001002) -> float:
    """Calculate theoretical diffusion coefficient using Stokes-Einstein.
    
    D = k_B * T / (6 * π * η * r)
    
    Args:
        particle_size_nm: Particle diameter in nanometers
        temperature_K: Temperature in Kelvin (default: 20°C = 293.15 K)
        viscosity_pa_s: Dynamic viscosity in Pa·s (default: water at 20°C)
    
    Returns:
        Diffusion coefficient in µm²/s
    """
    k_B = 1.380649e-23  # J/K
    
    # Convert diameter to radius in meters
    R = (particle_size_nm / 2.0) * 1e-9
    
    # Stokes-Einstein
    D_m2_s = k_B * temperature_K / (6 * np.pi * viscosity_pa_s * R)
    
    # Convert m²/s to µm²/s
    D_um2_s = D_m2_s * 1e12
    
    return D_um2_s
