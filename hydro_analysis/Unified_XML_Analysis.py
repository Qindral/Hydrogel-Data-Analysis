"""
Unified TrackMate XML Analysis Script

This standalone script processes TrackMate XML files and performs both:
1. MSD (Mean Squared Displacement) analysis
2. Step size (displacement) diffusion analysis

Features:
- Takes XML file path as input
- Automatically searches for TIFF or .rec files in the same directory
- Extracts calibration parameters (mpp, fps) from files or uses auto-detection
- Performs both MSD and step size analysis on the same data
- Saves results as CSV and plots as PNG

Usage:
    python Unified_XML_Analysis.py path/to/Tracks.xml
    
Or run interactively and provide paths when prompted.

Author: Jonas
Date: 2026-01-19
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
from scipy import stats


# ============================================================================
# CONFIGURATION
# ============================================================================

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
TEMPERATURE = 293.15  # K (20°C)
WATER_VISCOSITY = 0.001002  # Pa·s (at 25°C)

# Default calibration parameters
DEFAULT_MPP = 0.15  # micrometers per pixel
DEFAULT_FPS = 20  # frames per second

# Analysis parameters
MIN_TRACK_LENGTH = 30  # minimum frames for MSD analysis
MIN_TRACK_LENGTH_STEPSIZE = 10  # minimum frames for step size analysis
MSD_FIT_POINTS = 6  # number of points for power-law fitting
STEP_INTERVAL = 6  # use every n-th step (reduces correlation)

# Quality criteria
MIN_EXPONENT = 0.85  # minimum MSD exponent for free diffusion
MAX_SIGMA_RATIO = 1.5  # maximum ratio sigma_x/sigma_y for isotropic motion
MAX_MEAN_SIGMA_RATIO = 0.3  # maximum |mean|/sigma ratio (no drift)


# ============================================================================
# CALIBRATION & FILE DETECTION
# ============================================================================

def parse_rec_file(rec_path: Path) -> Dict[str, any]:
    """
    Parse .rec file to extract acquisition parameters.
    
    Looks for:
    - Exposure / Delay line to calculate FPS
    - Picture Size line to get image dimensions
    
    Args:
        rec_path: Path to .rec file
        
    Returns:
        Dictionary with exposure_ms, delay_ms, fps, size_x, size_y
    """
    result = {
        'exposure_ms': None,
        'delay_ms': None,
        'fps': None,
        'size_x': None,
        'size_y': None
    }
    
    try:
        # Try UTF-16 first (PCO CamWare standard), then UTF-8
        try:
            content = rec_path.read_text(encoding='utf-16', errors='replace')
        except:
            content = rec_path.read_text(encoding='utf-8', errors='replace')
            
        # Search for exposure/delay
        match = re.search(r'Exposure\s*/\s*Delay\s*:\s*([\d.]+)\s*ms\s*/\s*([\d.]+)\s*ms', 
                         content, re.IGNORECASE)
        if match:
            exposure = float(match.group(1))
            delay = float(match.group(2))
            result['exposure_ms'] = exposure
            result['delay_ms'] = delay
            
            total_time_ms = exposure + delay
            if total_time_ms > 0:
                result['fps'] = 1000.0 / total_time_ms
        
        # Search for image size
        size_match = re.search(r'Picture\s+Size\s+horz\.?/vert\.?\s*:\s*(\d+)\s*/\s*(\d+)', 
                              content, re.IGNORECASE)
        if size_match:
            result['size_x'] = int(size_match.group(1))
            result['size_y'] = int(size_match.group(2))
                
    except Exception as e:
        print(f"  Warning: Could not parse {rec_path.name}: {e}")
    
    return result


def extract_image_dimensions_from_xml(xml_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract maximum x and y coordinates from XML detections.
    
    Args:
        xml_path: Path to TrackMate XML file
        
    Returns:
        (x_max, y_max) in pixels, or (None, None) if failed
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        x_coords = []
        y_coords = []
        
        for particle in root.findall('particle'):
            for detection in particle.findall('detection'):
                x_raw = detection.get('x')
                y_raw = detection.get('y')
                if x_raw and y_raw:
                    x_coords.append(float(x_raw))
                    y_coords.append(float(y_raw))
        
        if x_coords and y_coords:
            x_max = int(np.ceil(max(x_coords)))
            y_max = int(np.ceil(max(y_coords)))
            return x_max, y_max
        
        return None, None
    
    except Exception as e:
        print(f"  Warning: Could not extract dimensions from XML: {e}")
        return None, None


def get_mpp_from_fps_and_size(fps: Optional[float] = None, 
                              x_max: Optional[int] = None, 
                              y_max: Optional[int] = None) -> Tuple[float, float, str]:
    """
    Determine mpp and fps based on image size or frame rate.
    
    Mode detection:
    - 60 FPS mode: 200×150 px → 0.30 µm/px, 60 FPS
    - 20 FPS mode: larger → 0.15 µm/px, 20 FPS
    
    Args:
        fps: Frames per second (optional)
        x_max: Image width in pixels (optional)
        y_max: Image height in pixels (optional)
        
    Returns:
        (mpp, fps, mode) tuple
    """
    # Image size-based detection (most reliable)
    if x_max is not None and y_max is not None:
        if x_max <= 250 and y_max <= 200:
            return 0.3, 60.0, '60 FPS'
        else:
            return 0.15, 20.0, '20 FPS'
    
    # FPS-based detection
    if fps is not None:
        if 50 <= fps <= 70:
            return 0.3, fps, '60 FPS'
        elif 15 <= fps <= 30:
            return 0.15, fps, '20 FPS'
    
    # Default fallback
    print(f"  Warning: Could not determine mode. Using defaults (mpp={DEFAULT_MPP}, fps={DEFAULT_FPS})")
    return DEFAULT_MPP, DEFAULT_FPS, 'Unknown'


def find_calibration_files(xml_path: Path) -> Dict[str, any]:
    """
    Search for TIFF or .rec files in the same directory as XML.
    
    Args:
        xml_path: Path to TrackMate XML file
        
    Returns:
        Dictionary with calibration info: fps, mpp, mode, files_found
    """
    xml_dir = xml_path.parent
    
    result = {
        'fps': None,
        'mpp': None,
        'mode': 'Unknown',
        'tiff_files': [],
        'rec_files': [],
        'x_max': None,
        'y_max': None
    }
    
    # Search for TIFF files
    tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    for pattern in tiff_patterns:
        result['tiff_files'].extend(list(xml_dir.glob(pattern)))
    
    # Search for .rec files
    result['rec_files'] = list(xml_dir.glob('*.rec'))
    
    # Try to get dimensions from XML
    x_max, y_max = extract_image_dimensions_from_xml(xml_path)
    result['x_max'] = x_max
    result['y_max'] = y_max
    
    # Try to parse .rec file for FPS
    fps_from_rec = None
    if result['rec_files']:
        rec_info = parse_rec_file(result['rec_files'][0])
        fps_from_rec = rec_info.get('fps')
        
        # Use image size from .rec if available
        if x_max is None and rec_info.get('size_x') is not None:
            x_max = rec_info['size_x']
            y_max = rec_info['size_y']
            result['x_max'] = x_max
            result['y_max'] = y_max
    
    # Determine calibration
    mpp, fps, mode = get_mpp_from_fps_and_size(fps_from_rec, x_max, y_max)
    result['mpp'] = mpp
    result['fps'] = fps
    result['mode'] = mode
    
    return result


# ============================================================================
# XML LOADING
# ============================================================================

def load_tracks_xml(xml_path: Path, mpp: float = 0.15, fps: float = 20.0) -> pd.DataFrame:
    """
    Load TrackMate XML and convert to trackpy-compatible DataFrame.
    
    Args:
        xml_path: Path to TrackMate XML file
        mpp: Micrometers per pixel
        fps: Frames per second
        
    Returns:
        DataFrame with columns ['particle', 'frame', 'x', 'y'] and attrs
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    data = []
    for pid, particle in enumerate(root.findall('particle')):
        for detection in particle.findall('detection'):
            data.append({
                'particle': pid,
                'frame': int(float(detection.get('t'))),
                'x': float(detection.get('x')),
                'y': float(detection.get('y')),
            })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df[['particle', 'frame', 'x', 'y']]
        df = df.sort_values(['particle', 'frame']).reset_index(drop=True)
    
    # Set trackpy attributes
    df.attrs['mpp'] = mpp
    df.attrs['fps'] = fps
    
    return df


def filter_tracks(df: pd.DataFrame, min_length: int = 30) -> pd.DataFrame:
    """
    Filter tracks by minimum length.
    
    Args:
        df: Track DataFrame
        min_length: Minimum number of detections
        
    Returns:
        Filtered DataFrame
    """
    counts = df.groupby('particle').size()
    valid = counts[counts >= min_length].index
    filtered = df[df['particle'].isin(valid)].reset_index(drop=True)
    
    # Preserve attributes
    filtered.attrs['mpp'] = df.attrs.get('mpp', DEFAULT_MPP)
    filtered.attrs['fps'] = df.attrs.get('fps', DEFAULT_FPS)
    
    return filtered


# ============================================================================
# MSD ANALYSIS
# ============================================================================

def fit_powerlaw_with_errors(em_series: pd.Series, points: int = 10) -> SimpleNamespace:
    """
    Fit power-law y = A * x^n to MSD data with error estimates.
    
    Args:
        em_series: Ensemble MSD Series (index=lag, values=MSD)
        points: Number of initial points to fit
        
    Returns:
        SimpleNamespace with A, n, A_err, n_err, logA, logA_err, cov
    """
    xs = em_series.iloc[0:points].index.values.astype(float)
    ys = em_series.iloc[0:points].values.astype(float)
    
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    
    if mask.sum() < 2:
        # Fallback
        return tp.utils.fit_powerlaw(em_series.iloc[0:points])
    
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


def perform_msd_analysis(tracks: pd.DataFrame, fit_points: int = 6) -> Dict[str, any]:
    """
    Perform MSD analysis on tracks.
    
    Args:
        tracks: Track DataFrame with mpp and fps in attrs
        fit_points: Number of points for power-law fitting
        
    Returns:
        Dictionary with MSD results
    """
    mpp = tracks.attrs.get('mpp', DEFAULT_MPP)
    fps = tracks.attrs.get('fps', DEFAULT_FPS)
    
    # Compute MSD
    imsd = tp.imsd(tracks, mpp=mpp, fps=fps)
    emsd = imsd.mean(axis=1)
    
    # Fit power-law
    fit_result = fit_powerlaw_with_errors(emsd, points=fit_points)
    
    # Extract diffusion coefficient (D = A/4 for 2D)
    D = fit_result.A[0] / 4.0
    D_err = fit_result.A_err[0] / 4.0
    n = fit_result.n[0]
    n_err = fit_result.n_err[0]
    
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


# ============================================================================
# STEP SIZE ANALYSIS
# ============================================================================

def calculate_displacements(tracks: pd.DataFrame, step_interval: int = 1) -> np.ndarray:
    """
    Calculate frame-to-frame displacements for all tracks.
    
    Args:
        tracks: Track DataFrame with mpp in attrs
        step_interval: Use every n-th step (reduces correlation)
        
    Returns:
        Array of displacements in micrometers
    """
    mpp = tracks.attrs.get('mpp', DEFAULT_MPP)
    
    displacements = []
    
    for pid in tracks['particle'].unique():
        track = tracks[tracks['particle'] == pid].sort_values('frame')
        
        if len(track) < step_interval + 1:
            continue
        
        # Calculate displacements with given interval
        for i in range(0, len(track) - step_interval, step_interval):
            dx = (track['x'].iloc[i + step_interval] - track['x'].iloc[i]) * mpp
            dy = (track['y'].iloc[i + step_interval] - track['y'].iloc[i]) * mpp
            displacements.extend([dx, dy])
    
    return np.array(displacements)


def fit_gaussian_to_displacements(displacements: np.ndarray) -> Dict[str, float]:
    """
    Fit Gaussian to displacement distribution.
    
    Args:
        displacements: Array of displacements
        
    Returns:
        Dictionary with mu, sigma, and their errors
    """
    if len(displacements) < 10:
        return {
            'mu': np.nan,
            'sigma': np.nan,
            'mu_err': np.nan,
            'sigma_err': np.nan
        }
    
    # Fit Gaussian
    mu, sigma = stats.norm.fit(displacements)
    
    # Standard errors (approximate)
    n = len(displacements)
    mu_err = sigma / np.sqrt(n)
    sigma_err = sigma / np.sqrt(2 * n)
    
    return {
        'mu': mu,
        'sigma': sigma,
        'mu_err': mu_err,
        'sigma_err': sigma_err
    }


def perform_stepsize_analysis(tracks: pd.DataFrame, step_interval: int = 6) -> Dict[str, any]:
    """
    Perform step size diffusion analysis.
    
    Method: D = σ² / (2 * dt) where σ is std of displacement distribution
    
    Args:
        tracks: Track DataFrame with mpp and fps in attrs
        step_interval: Use every n-th step
        
    Returns:
        Dictionary with step size analysis results
    """
    fps = tracks.attrs.get('fps', DEFAULT_FPS)
    dt = step_interval / fps  # Time interval for steps
    
    # Calculate displacements
    displacements = calculate_displacements(tracks, step_interval)
    
    if len(displacements) < 10:
        return {
            'method': 'Step Size',
            'D_um2_per_s': np.nan,
            'D_error': np.nan,
            'n_steps': 0,
            'sigma': np.nan,
            'mu': np.nan,
            'quality': 'FAILED'
        }
    
    # Fit Gaussian
    fit_stats = fit_gaussian_to_displacements(displacements)
    sigma = fit_stats['sigma']
    sigma_err = fit_stats['sigma_err']
    
    # Calculate diffusion coefficient
    D = sigma**2 / (2 * dt)
    D_err = 2 * sigma * sigma_err / (2 * dt)
    
    # Quality checks
    quality = 'PASS'
    if abs(fit_stats['mu']) / sigma > MAX_MEAN_SIGMA_RATIO:
        quality = 'WARNING: Non-zero mean (drift detected)'
    
    return {
        'method': 'Step Size',
        'D_um2_per_s': D,
        'D_error': D_err,
        'n_steps': len(displacements) // 2,
        'sigma': sigma,
        'mu': fit_stats['mu'],
        'dt': dt,
        'step_interval': step_interval,
        'quality': quality,
        'displacements': displacements
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_msd_results(msd_result: Dict, save_path: Path):
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
    y_fit = fit_result.A[0] * x_fit**fit_result.n[0]
    
    # Calculate uncertainty band from parameter errors
    A_upper = (fit_result.A[0] + fit_result.A_err[0])
    A_lower = (fit_result.A[0] - fit_result.A_err[0])
    n_upper = (fit_result.n[0] + fit_result.n_err[0])
    n_lower = (fit_result.n[0] - fit_result.n_err[0])
    
    y_upper = A_upper * x_fit**n_upper
    y_lower = A_lower * x_fit**n_lower
    
    # Plot fit line
    ax.loglog(x_fit, y_fit, '--', 
             label=f'Fit: A={fit_result.A[0]:.3f}±{fit_result.A_err[0]:.3f}, n={fit_result.n[0]:.3f}±{fit_result.n_err[0]:.3f}',
             linewidth=2, color='red')
    
    # Add uncertainty band
    ax.fill_between(x_fit, y_lower, y_upper, 
                    color='red', alpha=0.2, label='Fit Uncertainty')
    
    D = msd_result['D_um2_per_s']
    D_err = msd_result['D_error']
    n = msd_result['exponent']
    n_err = msd_result['exponent_error']
    
    ax.set_xlabel('Lag Time (s)', fontsize=12)
    ax.set_ylabel('MSD (µm²)', fontsize=12)
    ax.set_title(f'MSD Analysis\nD = {D:.4f} ± {D_err:.4f} µm²/s\n'
                f'Exponent n = {n:.3f} ± {n_err:.3f}', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    fig.tight_layout()
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
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Step size plot saved: {save_path}")


def plot_trajectories(tracks: pd.DataFrame, save_path: Path, max_tracks: int = 100):
    """
    Plot particle trajectories.
    
    Args:
        tracks: Track DataFrame
        save_path: Path to save PNG
        max_tracks: Maximum number of tracks to plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_particles = tracks['particle'].unique()
    n_plot = min(len(unique_particles), max_tracks)
    
    for pid in unique_particles[:n_plot]:
        track = tracks[tracks['particle'] == pid].sort_values('frame')
        ax.plot(track['x'], track['y'], alpha=0.5, linewidth=1)
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'Particle Trajectories (N={tracks["particle"].nunique()})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Trajectory plot saved: {save_path}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_xml_file(xml_path: Path, output_dir: Optional[Path] = None):
    """
    Perform complete analysis on a single TrackMate XML file.
    
    Args:
        xml_path: Path to TrackMate XML file
        output_dir: Directory to save results (defaults to XML directory)
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {xml_path.name}")
    print(f"{'='*80}\n")
    
    # Set output directory
    if output_dir is None:
        output_dir = xml_path.parent / 'analysis_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find calibration files
    print("Step 1: Searching for calibration files...")
    calib_info = find_calibration_files(xml_path)
    print(f"  TIFF files found: {len(calib_info['tiff_files'])}")
    print(f"  .rec files found: {len(calib_info['rec_files'])}")
    print(f"  Image dimensions: {calib_info['x_max']} × {calib_info['y_max']} px")
    print(f"  Mode: {calib_info['mode']}")
    print(f"  Calibration: mpp={calib_info['mpp']} µm/px, fps={calib_info['fps']} Hz")
    
    # Load tracks
    print("\nStep 2: Loading tracks from XML...")
    tracks = load_tracks_xml(xml_path, calib_info['mpp'], calib_info['fps'])
    print(f"  Total detections: {len(tracks)}")
    print(f"  Total particles: {tracks['particle'].nunique()}")
    
    # Filter tracks for MSD
    print(f"\nStep 3: Filtering tracks (min length = {MIN_TRACK_LENGTH})...")
    tracks_msd = filter_tracks(tracks, MIN_TRACK_LENGTH)
    print(f"  Particles for MSD analysis: {tracks_msd['particle'].nunique()}")
    
    # Filter tracks for step size (less strict)
    tracks_step = filter_tracks(tracks, MIN_TRACK_LENGTH_STEPSIZE)
    print(f"  Particles for step size analysis: {tracks_step['particle'].nunique()}")
    
    # Perform MSD analysis
    print("\nStep 4: Performing MSD analysis...")
    if tracks_msd['particle'].nunique() > 0:
        msd_result = perform_msd_analysis(tracks_msd, MSD_FIT_POINTS)
        print(f"  D_MSD = {msd_result['D_um2_per_s']:.4f} ± {msd_result['D_error']:.4f} µm²/s")
        print(f"  Exponent n = {msd_result['exponent']:.3f} ± {msd_result['exponent_error']:.3f}")
        
        # Plot MSD
        msd_plot_path = output_dir / f"{xml_path.stem}_MSD.png"
        plot_msd_results(msd_result, msd_plot_path)
    else:
        print("  WARNING: No tracks long enough for MSD analysis")
        msd_result = None
    
    # Perform step size analysis
    print("\nStep 5: Performing step size analysis...")
    if tracks_step['particle'].nunique() > 0:
        stepsize_result = perform_stepsize_analysis(tracks_step, STEP_INTERVAL)
        print(f"  D_stepsize = {stepsize_result['D_um2_per_s']:.4f} ± {stepsize_result['D_error']:.4f} µm²/s")
        print(f"  Quality: {stepsize_result['quality']}")
        
        # Plot step size
        stepsize_plot_path = output_dir / f"{xml_path.stem}_StepSize.png"
        plot_stepsize_results(stepsize_result, stepsize_plot_path)
    else:
        print("  WARNING: No tracks for step size analysis")
        stepsize_result = None
    
    # Plot trajectories
    print("\nStep 6: Plotting trajectories...")
    traj_plot_path = output_dir / f"{xml_path.stem}_Trajectories.png"
    plot_trajectories(tracks, traj_plot_path)
    
    # Save summary CSV
    print("\nStep 7: Saving summary...")
    summary_data = {
        'xml_file': xml_path.name,
        'total_particles': tracks['particle'].nunique(),
        'total_detections': len(tracks),
        'mpp': calib_info['mpp'],
        'fps': calib_info['fps'],
        'mode': calib_info['mode'],
        'image_width': calib_info['x_max'],
        'image_height': calib_info['y_max'],
    }
    
    if msd_result:
        summary_data.update({
            'D_MSD_um2_per_s': msd_result['D_um2_per_s'],
            'D_MSD_error': msd_result['D_error'],
            'MSD_exponent': msd_result['exponent'],
            'MSD_exponent_error': msd_result['exponent_error'],
            'MSD_n_particles': msd_result['n_particles'],
        })
    
    if stepsize_result:
        summary_data.update({
            'D_stepsize_um2_per_s': stepsize_result['D_um2_per_s'],
            'D_stepsize_error': stepsize_result['D_error'],
            'stepsize_n_steps': stepsize_result['n_steps'],
            'stepsize_quality': stepsize_result['quality'],
        })
    
    summary_df = pd.DataFrame([summary_data])
    summary_path = output_dir / f"{xml_path.stem}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary saved: {summary_path}")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    return summary_data


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for script execution."""
    
    # Check for command line argument
    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
    else:
        # Interactive mode
        print("Unified TrackMate XML Analysis")
        print("=" * 80)
        xml_input = input("\nEnter path to TrackMate XML file: ").strip('"').strip("'")
        xml_path = Path(xml_input)
    
    # Validate input
    if not xml_path.exists():
        print(f"\nERROR: File not found: {xml_path}")
        sys.exit(1)
    
    if not xml_path.suffix.lower() == '.xml':
        print(f"\nERROR: File must be XML format: {xml_path}")
        sys.exit(1)
    
    # Run analysis
    try:
        analyze_xml_file(xml_path)
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
