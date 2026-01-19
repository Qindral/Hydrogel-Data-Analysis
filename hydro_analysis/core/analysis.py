"""Analysis functions for diffusion calculations."""

from typing import Tuple, Optional, Dict
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy import stats
import trackpy as tp


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


def fit_powerlaw_with_errors(em_series: pd.Series, points: int = 10, 
                            ax=None, plot: bool = False) -> SimpleNamespace:
    """Fit power-law model y = A * x^n to ensemble MSD data.
    
    Performs linear regression in log-space to estimate parameters and
    their standard errors.
    
    From MSD_FromTrackmate_D0.py
    
    Args:
        em_series: Ensemble MSD pandas Series (index=lag time, values=MSD)
        points: Number of initial points to use for fitting
        ax: Optional matplotlib axis for plotting
        plot: Whether to create a plot
        
    Returns:
        SimpleNamespace with fitted parameters:
        - A: Prefactor (array)
        - n: Exponent (array)
        - A_err, n_err: Standard errors
        - logA, logA_err: Log-space values
        - cov: Covariance matrix
    """
    xs = em_series.iloc[0:points].index.values.astype(float)
    ys = em_series.iloc[0:points].values.astype(float)
    
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    
    if mask.sum() < 2:
        return tp.utils.fit_powerlaw(em_series.iloc[0:points], plot=plot, ax=ax)
    
    lx = np.log(xs[mask])
    ly = np.log(ys[mask])
    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
    
    n_fit = float(coeffs[0])
    logA_fit = float(coeffs[1])
    
    se = np.sqrt(np.diag(cov))
    se_n = float(se[0])
    se_logA = float(se[1])
    
    A_fit = float(np.exp(logA_fit))
    se_A = A_fit * se_logA
    
    return SimpleNamespace(
        A=np.array([A_fit]),
        n=np.array([n_fit]),
        A_err=np.array([se_A]),
        n_err=np.array([se_n]),
        logA=np.array([logA_fit]),
        logA_err=np.array([se_logA]),
        cov=cov
    )


def read_trackmate_xml(xml_file_path) -> Optional[pd.DataFrame]:
    """Parse TrackMate XML file and convert to pandas DataFrame.
    
    From MSD_FromTrackmate_D0.py
    
    The TrackMate XML format contains 'particle' elements with nested
    'detection' elements for each time point.
    
    Args:
        xml_file_path: Path to TrackMate XML file
        
    Returns:
        DataFrame with columns ['frame', 'particle', 'x', 'y'], or None on error
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path
    
    xml_file_path = Path(xml_file_path)
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        data_rows = []
        
        for particle_id, particle in enumerate(root.findall('particle')):
            for detection in particle.findall('detection'):
                t_raw = detection.get('t')
                x_raw = detection.get('x')
                y_raw = detection.get('y')
                
                row = {
                    'frame': int(float(t_raw)),
                    'particle': particle_id + 1,
                    'x': float(x_raw),
                    'y': float(y_raw)
                }
                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        if not df.empty:
            df = df[['frame', 'particle', 'x', 'y']]
            df = df.sort_values(by=['frame', 'particle']).reset_index(drop=True)
            
        return df

    except Exception as e:
        print(f"Error parsing {xml_file_path}: {e}")
        return None


def calculate_step_sizes(df: pd.DataFrame, step_interval: int = 1) -> pd.DataFrame:
    """Calculate frame-to-frame displacements (dx, dy) for each particle.
    
    From Schrittweiten_methode_D0.py
    
    For each particle trajectory, computes displacement components between
    consecutive positions:
        dx_i = x_{i+step_interval} - x_i
        dy_i = y_{i+step_interval} - y_i
    
    Args:
        df: DataFrame with columns ['frame', 'particle', 'x', 'y']
        step_interval: Use every n-th frame (1=consecutive, 6=every 6th)
        
    Returns:
        DataFrame with columns ['particle', 'frame', 'dx', 'dy', 'frame_interval']
    """
    step_data = []
    skipped_particles = []
    
    for particle_id in df['particle'].unique():
        particle_df = df[df['particle'] == particle_id].sort_values('frame')
        
        if len(particle_df) < step_interval + 1:
            skipped_particles.append((particle_id, len(particle_df)))
            continue
        
        x_vals = particle_df['x'].values
        y_vals = particle_df['y'].values
        frames = particle_df['frame'].values
        
        for i in range(0, len(x_vals) - step_interval, step_interval):
            dx = x_vals[i + step_interval] - x_vals[i]
            dy = y_vals[i + step_interval] - y_vals[i]
            
            step_data.append({
                'particle': particle_id,
                'frame': frames[i],
                'dx': dx,
                'dy': dy,
                'frame_interval': step_interval
            })
    
    if skipped_particles:
        print(f"Note: {len(skipped_particles)} particles with <{step_interval+1} points skipped")
    
    return pd.DataFrame(step_data)


def calculate_diffusion_from_steps(step_df: pd.DataFrame, mpp: float, fps: float,
                                   max_sigma_ratio: float = 1.5,
                                   max_mean_sigma_ratio: float = 0.3) -> Dict:
    """Calculate diffusion coefficient from displacement distributions using Gaussian fits.
    
    From Schrittweiten_methode_D0.py
    
    For 2D Brownian motion, dx and dy are independently Gaussian distributed with
    variance σ² = 2*D*dt. Fit Gaussian distributions to both dx and dy,
    extract variances, and average:
        D = <σ²> / (2*dt)
    
    Quality checks:
    - Isotropy: σ_x and σ_y should be similar (ratio < max_sigma_ratio)
    - No drift: μ_x and μ_y should be near zero (|μ|/σ < max_mean_sigma_ratio)
    
    Args:
        step_df: DataFrame with 'dx' and 'dy' columns (in pixels)
        mpp: Micrometers per pixel calibration
        fps: Frames per second
        max_sigma_ratio: Max ratio between sigma_x and sigma_y for isotropy
        max_mean_sigma_ratio: Max |mean|/sigma for centered distribution
        
    Returns:
        Dictionary with keys:
        - D: Diffusion coefficient in µm²/s (averaged from dx and dy)
        - D_std: Standard error estimate in µm²/s
        - sigma_x, sigma_y: Std dev from Gaussian fit (µm)
        - mu_x, mu_y: Mean of Gaussian fits (µm)
        - D_x, D_y: Diffusion coefficients from each direction (µm²/s)
        - sigma_ratio: Ratio of sigmas
        - mean_x_ratio, mean_y_ratio: |mean|/sigma ratios
        - is_isotropic: Boolean
        - is_centered: Boolean (no drift)
        - quality_flag: 'good', 'anisotropic', 'drift', or 'both'
        - num_steps: Number of steps analyzed
        - num_particles: Number of unique particles
    """
    dx_um = step_df['dx'].values * mpp
    dy_um = step_df['dy'].values * mpp
    
    frame_interval = step_df['frame_interval'].iloc[0] if 'frame_interval' in step_df.columns else 1
    
    mu_x, sigma_x = stats.norm.fit(dx_um)
    mu_y, sigma_y = stats.norm.fit(dy_um)
    
    dt = frame_interval / fps  # seconds
    
    D_x = (sigma_x**2) / (2.0 * dt)  # µm²/s
    D_y = (sigma_y**2) / (2.0 * dt)  # µm²/s
    
    D = (D_x + D_y) / 2.0
    D_std = np.abs(D_x - D_y) / 2.0
    
    sigma_ratio = max(sigma_x, sigma_y) / (min(sigma_x, sigma_y) + 1e-10)
    is_isotropic = sigma_ratio <= max_sigma_ratio
    
    mean_x_ratio = abs(mu_x) / (sigma_x + 1e-10)
    mean_y_ratio = abs(mu_y) / (sigma_y + 1e-10)
    is_centered = (mean_x_ratio <= max_mean_sigma_ratio and 
                   mean_y_ratio <= max_mean_sigma_ratio)
    
    if is_isotropic and is_centered:
        quality_flag = 'good'
    elif not is_isotropic and not is_centered:
        quality_flag = 'both'
    elif not is_isotropic:
        quality_flag = 'anisotropic'
    else:
        quality_flag = 'drift'
    
    return {
        'D': D,
        'D_std': D_std,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'mu_x': mu_x,
        'mu_y': mu_y,
        'D_x': D_x,
        'D_y': D_y,
        'sigma_ratio': sigma_ratio,
        'mean_x_ratio': mean_x_ratio,
        'mean_y_ratio': mean_y_ratio,
        'is_isotropic': is_isotropic,
        'is_centered': is_centered,
        'quality_flag': quality_flag,
        'num_steps': len(dx_um),
        'num_particles': step_df['particle'].nunique()
    }

