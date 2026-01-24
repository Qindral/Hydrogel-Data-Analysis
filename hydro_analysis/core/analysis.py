"""Analysis functions for diffusion calculations."""

from typing import Tuple, Optional, Dict
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import stats
import trackpy as tp


# Physical constants
TEMPERATURE_K = 293.15
VISCOSITY_PA_S = 0.001002
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

def compute_step_size_diffusion(
    tracks: pd.DataFrame,
    mpp: float,
    fps: float,
    step_interval: int = 1,
    max_sigma_ratio: float = 1.5,
    max_mean_sigma_ratio: float = 0.3
) -> dict:
    """Compute diffusion coefficient from step size distributions.
    
    Method: For each track, calculate frame-to-frame displacements.
    Fit Gaussian distributions to dx and dy to extract variance σ².
    For 2D Brownian motion: σ² = 2*D*dt, therefore: D = σ² / (2 * dt)
    
    Args:
        tracks: DataFrame with columns ['particle', 'frame', 'x', 'y']
        mpp: Micrometers per pixel
        fps: Frames per second
        step_interval: Use every nth step (1=all, 6=every 6th)
        max_sigma_ratio: Max ratio between sigma_x and sigma_y for isotropy
        max_mean_sigma_ratio: Max |mean|/sigma for centered distribution
    
    Returns:
        Dictionary with diffusion results and quality metrics
    """
    dt = step_interval / fps
    
    dx_all, dy_all = [], []
    
    for pid in tracks['particle'].unique():
        track = tracks[tracks['particle'] == pid].sort_values('frame')
        track_subset = track.iloc[::step_interval].copy()
        
        if len(track_subset) < step_interval + 1:
            continue
        
        frames = track_subset['frame'].values
        x_vals = track_subset['x'].values
        y_vals = track_subset['y'].values
        
        dx_px = []
        dy_px = []
        
        for i in range(len(frames) - 1):
            if frames[i+1] - frames[i] == step_interval:
                dx_px.append(x_vals[i+1] - x_vals[i])
                dy_px.append(y_vals[i+1] - y_vals[i])
        
        dx_all.extend(np.array(dx_px) * mpp)
        dy_all.extend(np.array(dy_px) * mpp)
    
    dx_all = np.array(dx_all)
    dy_all = np.array(dy_all)
    
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

def fit_gaussian_diffusion_stepsize(step_array) -> dict:
    dx_arr = np.array(step_array)
    
    # Fit Gaussians to get mean and sigma
    mean_x, sigma_x = dx_arr.mean(), dx_arr.std(ddof=1)
    
    
    # Diffusion coefficient: D = σ² / (2 * dt)
    D = sigma_x**2 / (2.0 * dt)
    
    # Error estimate (propagating std error)
    se_x = sigma_x / np.sqrt(len(dx_arr))
    D_error = np.sqrt(se_x**2) / (2.0 * dt)
    
    # Quality checks
    quality_issues = []
    
    # Check for drift (mean should be close to 0)
    mean_sigma_x = abs(mean_x) / (sigma_x + 1e-10)
    
    if mean_sigma_x > max_mean_sigma_ratio:
        quality_issues.append(f'Drift in X (|µ|/σ = {mean_sigma_x:.2f})')
    if mean_sigma_y > max_mean_sigma_ratio:
        quality_issues.append(f'Drift in Y (|µ|/σ = {mean_sigma_y:.2f})')
    
    return {
        'D_um2_per_s': D,
        'D_error': D_error,
        'sigma_x': sigma_x,
        'mean_x': mean_x,
        'sigma_ratio': sigma_ratio,
        'n_steps': len(dx_all),
        'quality_ok': len(quality_issues) == 0,
        'quality_issues': quality_issues if quality_issues else ['OK']
    }




def fit_powerlaw_with_errors(em_series: pd.Series, points: int = 10, 
                            ax=None, plot: bool = False) -> SimpleNamespace:
    """Fit power-law model y = A * x^n to ensemble MSD data.
    
    The data must be already calibrated with fps and mpp -> units of time [s] and µm²

    Performs linear regression in log-space to estimate parameters and
    their standard errors.
    
    
    Args:
        em_series: Ensemble MSD pandas Series (index=lag time, values=MSD)
        points: Number of initial points to use for fitting
        ax: Optional matplotlib axis for plotting
        
    Returns:
       Powerlaw Fit Data with fitted parameters:
        - A: Prefactor (array)
        - n: Exponent (array)
        - A_err, n_err: Standard errors
        - logA, logA_err: Log-space values
        - cov: Covariance matrix
    """
    xs = em_series.iloc[0:points].index.values.astype(float)
    ys = em_series.iloc[0:points].values.astype(float)
    
    
    lx = np.log(xs)
    ly = np.log(ys)
    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
    
    n_fit = float(coeffs[0])
    logA_fit = float(coeffs[1])
    
    se = np.sqrt(np.diag(cov))
    se_n = float(se[0])
    se_logA = float(se[1])
    
    A_fit = float(np.exp(logA_fit))
    se_A = A_fit * se_logA
    
    return {
        "A" : np.array([A_fit]),
        "n": np.array([n_fit]),
        "A_err": np.array([se_A]),
        "n_err": np.array([se_n]),
        "logA": np.array([logA_fit]),
        "logA_err": np.array([se_logA]),
        "cov": cov
    }


def perform_msd_analysis(tracks: pd.DataFrame,mpp: float, fps: float, fit_points: int = 6) -> Dict[str, any]:
    """
    Perform MSD analysis on tracks.
    
    Args:
        tracks: Track DataFrame with mpp and fps in attrs
        fit_points: Number of points for power-law fitting
        
    Returns:
        Dictionary with MSD results
    """
  
    # Compute MSD
    imsd = tp.imsd(tracks, mpp=mpp, fps=fps)
    emsd = tp.emsd(tracks, mpp=mpp, fps=fps)
    
    # Fit power-law
    fit_result = fit_powerlaw_with_errors(emsd, points=fit_points)
    
    # Extract diffusion coefficient (D = A/4 for 2D)
    D = fit_result["A"][0] / 4.0
    D_err = fit_result["A_err"][0] / 4.0
    n = fit_result["n"][0]
    n_err = fit_result["n_err"][0]
    
    return {
        'method': 'MSD',
        'D_um2_per_s': D,
        'D_error': D_err,
        'exponent': n,
        'exponent_error': n_err,
        'n_particles': tracks['particle'].nunique(),
        'n_detections': len(tracks),
        'imsd': imsd,
        'emsd': emsd,
        'fit_result': fit_result
    }



def calculate_theoretical_diffusion(particle_size_nm: float, 
                                   temperature: float = TEMPERATURE_K,
                                   viscosity: float = VISCOSITY_PA_S) -> float:
    """
    Calculate theoretical diffusion coefficient using Stokes-Einstein equation.
    
    D = k_B * T / (6 * pi * eta * r)
    
    Args:
        particle_size_nm: Particle diameter in nanometers
        temperature: Temperature in Kelvin
        viscosity: Dynamic viscosity in Pa·s
        
    Returns:
        Diffusion coefficient in µm²/s
    """
    radius_m = (particle_size_nm / 2) * 1e-9  # Convert nm to m
    D_m2_per_s = BOLTZMANN_CONSTANT * temperature / (6 * np.pi * viscosity * radius_m)
    D_um2_per_s = D_m2_per_s * 1e12  # Convert m²/s to µm²/s
    return D_um2_per_s

