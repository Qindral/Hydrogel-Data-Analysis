"""
TrackMate XML Particle Counter and Step Size Diffusion Analysis

This script processes TrackMate-generated XML files to count particles and calculate
diffusion coefficients using the step size (displacement) method for particles in water.

Main features:
- Scans directory structure for particle size folders with Tracks subfolders
- Counts total particles per particle size from TrackMate XML files
- Calculates diffusion coefficients from step size distributions
- Exports summary statistics and comparisons with theory to CSV

Method: Diffusion coefficient from displacement distributions
For each particle track, calculate frame-to-frame displacements in x and y:
    dx_i = x_{i+1} - x_i
    dy_i = y_{i+1} - y_i
Fit Gaussian distributions to dx and dy separately to extract variance σ².
For 2D Brownian motion: σ² = 2*D*dt, therefore:
    D = σ² / (2 * dt)
where dt is the time interval between frames.

Author: Jonas
Date: 2026-01-12
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory paths
ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung")
SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\trackmate_MSD_results")

# Create output directory if it doesn't exist
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
TEMPERATURE = 293.15  # K (20°C)
WATER_VISCOSITY = 0.001002  # Pa·s (at 25°C)

# Default calibration parameters
DEFAULT_MPP = 0.15  # micrometers per pixel
DEFAULT_FPS = 20  # frames per second

# Analysis parameters
STEP_INTERVAL = 6  # Use every n-th step (1=all steps, 3=every 3rd step, etc.)
                   # Larger intervals reduce correlation but decrease statistics

# Track filtering
MIN_TRACK_LENGTH = 10  # Minimum number of detections per track to be included

# Quality criteria for Gaussian fits
MAX_SIGMA_RATIO = 1.5  # Maximum ratio between sigma_x and sigma_y for isotropic motion
MAX_MEAN_SIGMA_RATIO = 0.3  # Maximum ratio |mean|/sigma for centered distribution (no drift)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_particle_size_from_path(folder_path: Path) -> Optional[float]:
    """
    Extract particle size (in nm) from folder name.
    
    Supports various naming formats:
    - "50nm"
    - "100_nm" 
    - "200 nm"
    
    Args:
        folder_path: Path object of the folder
        
    Returns:
        Particle size in nanometers, or None if not found
    """
    folder_name = folder_path.name
    match = re.search(r'(\d+)\s*nm', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def parse_rec_file(rec_path: Path) -> Dict[str, any]:
    """
    Parse .rec file to extract acquisition parameters.
    
    Looks for exposure time, delay, and image size to calculate FPS and MPP.
    Format: "Exposure / Delay        : 50.000000 ms / 0.000000 ms"
    Format: "Picture Size horz./vert.: 200/150"
    
    Args:
        rec_path: Path to .rec file
        
    Returns:
        Dictionary with keys:
        - exposure_ms: Exposure time in milliseconds
        - delay_ms: Delay time in milliseconds
        - fps: Calculated frames per second (1000 / (exposure + delay))
        - size_x: Image width in pixels
        - size_y: Image height in pixels
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
            
        # Search for exposure/delay line
        match = re.search(r'Exposure\s*/\s*Delay\s*:\s*([\d.]+)\s*ms\s*/\s*([\d.]+)\s*ms', 
                         content, re.IGNORECASE)
        
        if match:
            exposure = float(match.group(1))
            delay = float(match.group(2))
            
            result['exposure_ms'] = exposure
            result['delay_ms'] = delay
            
            # Calculate FPS: 1000 ms/s / (exposure + delay in ms)
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
        print(f"    Warning: Could not parse {rec_path.name}: {e}")
    
    return result


def get_mpp_from_fps_and_size(fps: Optional[float] = None, 
                              x_max: Optional[int] = None, 
                              y_max: Optional[int] = None) -> tuple[float, float, str]:
    """
    Determine micrometers per pixel (mpp) and FPS based on image size or FPS.
    
    Mode detection:
    - 60 FPS mode: 200×150 px window → 0.30 µm/px, 60 FPS
    - 20 FPS mode: larger window → 0.15 µm/px, 20 FPS
    
    Args:
        fps: Frames per second (optional)
        x_max: Maximum x coordinate (image width in pixels, optional)
        y_max: Maximum y coordinate (image height in pixels, optional)
        
    Returns:
        Tuple of (mpp, fps, mode) where mode is '60 FPS', '20 FPS', or 'Unknown'
    """
    # First try to detect mode by image size (most reliable)
    if x_max is not None and y_max is not None:
        # 60 FPS mode has small window (200×150 px)
        if x_max <= 250 and y_max <= 200:
            return 0.3, 60.0, '60 FPS'
        # 20 FPS mode has larger window
        else:
            return 0.15, 20.0, '20 FPS'
    
    # Fallback to FPS-based detection
    if fps is not None:
        if 50 <= fps <= 70:
            return 0.3, fps, '60 FPS'
        elif 15 <= fps <= 30:
            return 0.15, fps, '20 FPS'
    
    # Default fallback
    print(f"  Warning: Could not determine mode (fps={fps}, size={x_max}×{y_max}). Using defaults.")
    return DEFAULT_MPP, DEFAULT_FPS, 'Unknown'


def get_mpp_from_fps(fps: float) -> float:
    """
    Determine micrometers per pixel (mpp) based on frame rate.
    
    Different acquisition rates use different magnifications:
    - ~60 FPS (50-70 FPS): lower magnification, mpp = 0.3 µm/px
    - ~20 FPS (15-30 FPS): higher magnification, mpp = 0.15 µm/px
    
    Args:
        fps: Frames per second
        
    Returns:
        Micrometers per pixel calibration value
    """
    if fps >= 50 and fps <= 70:  # ~60 FPS - widened range
        return 0.3
    elif fps >= 15 and fps <= 30:  # ~20 FPS - widened range
        return 0.15
    else:
        # Default to 20 FPS calibration for other rates
        print(f"Warning: Unusual FPS {fps:.2f}, using default mpp=0.15")
        return 0.15


def get_mpp_from_size(width: int, height: int) -> float:
    """
    Determine micrometers per pixel based on image dimensions.
    Based on known camera configurations.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Micrometers per pixel calibration value
    """
    # Known dimension to mpp mappings
    dimension_map = {
        (200, 150): 0.3,
        (400, 300): 0.15,
        (696, 520): 0.149,
    }
    
    if (width, height) in dimension_map:
        return dimension_map[(width, height)]
    
    # Fallback: assume smaller images have higher magnification
    if width <= 250 and height <= 200:
        return 0.3
    elif width <= 450 and height <= 350:
        return 0.15
    else:
        return 0.149


# ============================================================================
# XML PARSING FUNCTIONS
# ============================================================================

def extract_image_dimensions_from_xml(xml_file_path: Path) -> tuple[Optional[int], Optional[int]]:
    """
    Extract image dimensions (x_max, y_max) from TrackMate XML file.
    
    Args:
        xml_file_path: Path to TrackMate XML file
        
    Returns:
        Tuple of (x_max, y_max) in pixels, or (None, None) if extraction fails
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        x_coords = []
        y_coords = []
        
        # Collect all x and y coordinates
        for particle in root.findall('particle'):
            for detection in particle.findall('detection'):
                x_raw = detection.get('x')
                y_raw = detection.get('y')
                if x_raw and y_raw:
                    x_coords.append(float(x_raw))
                    y_coords.append(float(y_raw))
        
        if x_coords and y_coords:
            # Round up to nearest integer for image dimensions
            x_max = int(np.ceil(max(x_coords)))
            y_max = int(np.ceil(max(y_coords)))
            return x_max, y_max
        
        return None, None
    
    except Exception as e:
        print(f"  Warning: Could not extract dimensions from {xml_file_path.name}: {e}")
        return None, None


def read_trackmate_xml(xml_file_path: Path) -> Optional[pd.DataFrame]:
    """
    Parse TrackMate XML file and convert to pandas DataFrame.
    
    The TrackMate XML format contains 'particle' elements with nested
    'detection' elements for each time point.
    
    Tracks with fewer than MIN_TRACK_LENGTH points are filtered out.
    
    Args:
        xml_file_path: Path to TrackMate XML file
        
    Returns:
        DataFrame with columns ['frame', 'particle', 'x', 'y'], or None on error
    """
    try:
        # Parse XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        data_rows = []
        
        # Iterate through all particle tracks
        for particle_id, particle in enumerate(root.findall('particle')):
            # Extract detections (position at each time point)
            for detection in particle.findall('detection'):
                # Get attributes
                t_raw = detection.get('t')
                x_raw = detection.get('x')
                y_raw = detection.get('y')
                
                # Convert to appropriate types and store
                row = {
                    'frame': int(float(t_raw)),  # Handle "40.0" format
                    'particle': particle_id + 1,  # 1-indexed particle IDs
                    'x': float(x_raw),
                    'y': float(y_raw)
                }
                data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        if not df.empty:
            # Filter tracks by minimum length
            track_lengths = df.groupby('particle').size()
            valid_particles = track_lengths[track_lengths >= MIN_TRACK_LENGTH].index
            df = df[df['particle'].isin(valid_particles)].copy()
            
            if df.empty:
                print(f"  Warning: No tracks with >={MIN_TRACK_LENGTH} points found")
                return None
            
            # Renumber particles to be consecutive after filtering
            unique_particles = sorted(df['particle'].unique())
            particle_map = {old_id: new_id for new_id, old_id in enumerate(unique_particles, 1)}
            df['particle'] = df['particle'].map(particle_map)
            
            # Ensure correct column order
            df = df[['frame', 'particle', 'x', 'y']]
            
            # Sort for better readability
            df = df.sort_values(by=['frame', 'particle']).reset_index(drop=True)
            
        return df

    except Exception as e:
        print(f"Error parsing {xml_file_path}: {e}")
        return None


def collect_all_files_by_particle_size(root_path: Path) -> pd.DataFrame:
    """
    Scan directory structure and collect XML track files with associated FPS data.
    
    This function focuses on XML files (which contain the track data) and retrieves
    FPS information from .rec files in the same particle size folder.
    
    Expected structure:
        root_path/
            50nm/
                *.rec (for FPS extraction)
                Tracks/
                    *.xml (primary data)
            100nm/
                *.rec
                Tracks/
                    *.xml
    
    Args:
        root_path: Root directory to scan
        
    Returns:
        DataFrame with columns:
        - particle_size_nm: Particle size in nanometers
        - xml_path: Path to XML track file
        - xml_name: Basename of XML file
        - exposure_ms: Exposure time from matched .rec file
        - delay_ms: Delay time from matched .rec file
        - fps: Calculated frames per second from matched .rec file
        - mpp: Micrometers per pixel calibration
        - mode: Detection mode description
    """
    file_records = []
    
    print(f"\nDEBUG: Scanning folders in {root_path}")
    print("=" * 70)
    
    # Walk through subdirectories
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Extract particle size from folder name
        particle_size = extract_particle_size_from_path(subfolder)
        
        print(f"\nFolder: {subfolder.name}")
        print(f"   Particle size: {particle_size} nm" if particle_size else "   Warning: No particle size detected")
        
        if particle_size is None:
            continue
        
        # Find all REC files in main folder and parse them
        rec_files = sorted(list(subfolder.glob("*.rec")))
        print(f"   Found {len(rec_files)} .rec files")
        
        # Parse all REC files and store by basename for individual matching
        rec_info_by_basename = {}
        
        for rec_file in rec_files:
            rec_info = parse_rec_file(rec_file)
            if rec_info['fps'] is not None:
                basename = rec_file.stem
                rec_info_by_basename[basename] = rec_info
                print(f"     • {rec_file.name}: {rec_info['fps']:.2f} fps, size={rec_info['size_x']}x{rec_info['size_y']} px" 
                      if rec_info['size_x'] else f"     • {rec_file.name}: {rec_info['fps']:.2f} fps")
        
        if not rec_info_by_basename:
            print(f"   Warning: No valid FPS data from .rec files")
        
        # Find all XML files in Tracks subfolder
        tracks_folder = subfolder / "Tracks"
        xml_files = []
        
        if tracks_folder.exists() and tracks_folder.is_dir():
            xml_files = sorted(list(tracks_folder.glob("*.xml")))
            print(f"   Found {len(xml_files)} XML track files in Tracks/")
        else:
            print(f"   Warning: No Tracks/ subfolder found")
        
        if len(xml_files) == 0:
            print(f"   Skipping - no XML files found")
            continue
        
        # Create one record per XML file with individual .rec matching
        for xml_file in xml_files:
            # Extract image dimensions from XML
            x_max, y_max = extract_image_dimensions_from_xml(xml_file)
            
            # Match XML to .rec file based on filename
            xml_basename = xml_file.stem.replace('_Tracks', '').replace(' Tracks', '')
            matched_rec_info = None
            
            for rec_basename, rec_info in rec_info_by_basename.items():
                if rec_basename in xml_basename or xml_basename in rec_basename:
                    matched_rec_info = rec_info
                    print(f"     Matched {xml_file.name} -> {rec_basename}.rec")
                    break
            
            # Use matched .rec parameters if available
            if matched_rec_info:
                rec_fps = matched_rec_info['fps']
                exposure_ms = matched_rec_info['exposure_ms']
                delay_ms = matched_rec_info['delay_ms']
                if matched_rec_info['size_x'] is not None:
                    x_max = matched_rec_info['size_x']
                    y_max = matched_rec_info['size_y']
            else:
                rec_fps = None
                exposure_ms = None
                delay_ms = None
            
            # Determine mpp based on image size (prioritize), then fps
            if x_max is not None and y_max is not None:
                mpp = get_mpp_from_size(x_max, y_max)
                fps = rec_fps if rec_fps else DEFAULT_FPS
                mode = f'{x_max}x{y_max}px'
            else:
                mpp, fps, mode = get_mpp_from_fps_and_size(
                    fps=rec_fps,
                    x_max=x_max,
                    y_max=y_max
                )
            
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
            
            size_info = f"{x_max}x{y_max}" if x_max and y_max else "unknown size"
            print(f"     [OK] {xml_file.name} [{size_info} -> {mode}, {mpp} um/px]")
        
        print(f"   -> Created {len(xml_files)} records")
    
    print(f"\n" + "=" * 70)
    print(f"[OK] Total XML files found: {len(file_records)}")
    print("=" * 70 + "\n")
    
    df = pd.DataFrame(file_records)
    
    return df


def find_tracks_in_particle_folders(root_path: Path) -> Dict[float, List[Path]]:
    """
    Scan directory structure for TrackMate XML files organized by particle size.
    
    Expected structure:
        root_path/
            50nm/
                Tracks/
                    *.xml
            100nm/
                Tracks/
                    *.xml
    
    Args:
        root_path: Root directory to scan
        
    Returns:
        Dictionary mapping particle_size_nm -> list of XML file paths
    """
    tracks_by_size = defaultdict(list)
    
    # Walk through subdirectories
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Extract particle size from folder name
        particle_size = extract_particle_size_from_path(subfolder)
        if particle_size is None:
            continue
        
        # Look for 'Tracks' subfolder
        tracks_folder = subfolder / "Tracks"
        if not tracks_folder.exists() or not tracks_folder.is_dir():
            continue
        
        # Find all XML files in Tracks folder
        xml_files = list(tracks_folder.glob("*.xml"))
        if xml_files:
            tracks_by_size[particle_size].extend(xml_files)
    
    return tracks_by_size


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_step_sizes(df: pd.DataFrame, step_interval: int = STEP_INTERVAL) -> pd.DataFrame:
    """
    Calculate frame-to-frame displacements (dx, dy) for each particle.
    
    For each particle trajectory, computes displacement components between
    consecutive positions:
        dx_i = x_{i+step_interval} - x_i
        dy_i = y_{i+step_interval} - y_i
    
    These will be used to fit Gaussian distributions and extract variance.
    
    Note: Particles with only 1 detection point are included in total counts
    but cannot contribute step measurements (need 2+ points for displacement).
    Using step_interval > 1 reduces temporal correlation but decreases statistics.
    
    Args:
        df: DataFrame with columns ['frame', 'particle', 'x', 'y']
        step_interval: Use every n-th frame (1=consecutive frames, 3=every 3rd frame)
        
    Returns:
        DataFrame with columns ['particle', 'frame', 'dx', 'dy', 'frame_interval'] containing
        all calculated displacement components for each particle
    """
    step_data = []
    skipped_particles = []
    
    # Group by particle
    for particle_id in df['particle'].unique():
        particle_df = df[df['particle'] == particle_id].sort_values('frame')
        
        # Need at least (step_interval + 1) points to calculate displacement
        if len(particle_df) < step_interval + 1:
            skipped_particles.append((particle_id, len(particle_df)))
            continue
        
        # Calculate differences
        x_vals = particle_df['x'].values
        y_vals = particle_df['y'].values
        frames = particle_df['frame'].values
        
        # Displacements with specified interval
        # dx[i] = x[i+step_interval] - x[i]
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
    
    # Report skipped particles if any
    if skipped_particles:
        print(f"    Note: {len(skipped_particles)} particles with <2 points cannot contribute steps")
    
    return pd.DataFrame(step_data)


def calculate_diffusion_from_steps(step_df: pd.DataFrame, mpp: float, fps: float) -> Dict:
    """
    Calculate diffusion coefficient from displacement distributions using Gaussian fits.
    
    For 2D Brownian motion, dx and dy are independently Gaussian distributed with
    variance σ² = 2*D*dt. We fit Gaussian distributions to both dx and dy,
    extract their variances, and average them:
        D = <σ²> / (2*dt)
    
    Quality checks:
    - Isotropy: σ_x and σ_y should be similar (ratio < MAX_SIGMA_RATIO)
    - No drift: μ_x and μ_y should be near zero (|μ|/σ < MAX_MEAN_SIGMA_RATIO)
    
    Args:
        step_df: DataFrame with 'dx' and 'dy' columns (in pixels)
        mpp: Micrometers per pixel calibration
        fps: Frames per second (determines time interval)
        
    Returns:
        Dictionary containing:
        - D: Diffusion coefficient in nm²/ms (averaged from dx and dy)
        - D_std: Standard error of D in nm²/ms
        - sigma_x: Std dev from Gaussian fit to dx (nm)
        - sigma_y: Std dev from Gaussian fit to dy (nm)
        - mu_x, mu_y: Mean of Gaussian fits (nm)
        - is_isotropic: Boolean, whether motion is isotropic
        - is_centered: Boolean, whether distribution is centered (no drift)
        - quality_flag: 'good', 'anisotropic', 'drift', or 'both'
        - num_steps: Number of steps analyzed
        - num_particles: Number of unique particles
    """
    # Convert displacements from pixels to nanometers (mpp is in µm, so * 1000 for nm)
    dx_nm = step_df['dx'].values * mpp * 1000.0
    dy_nm = step_df['dy'].values * mpp * 1000.0
    
    # Get frame interval from step_df (all steps have same interval)
    frame_interval = step_df['frame_interval'].iloc[0] if 'frame_interval' in step_df.columns else 1
    
    # Fit Gaussian distributions to dx and dy
    # stats.norm.fit returns (mean, std)
    mu_x, sigma_x = stats.norm.fit(dx_nm)
    mu_y, sigma_y = stats.norm.fit(dy_nm)
    
    # Time interval between steps in milliseconds
    # IMPORTANT: dt must account for frame_interval (e.g., every 3rd frame)
    dt = (1000.0 / fps) * frame_interval  # milliseconds
    
    # Calculate D from each direction: σ² = 2*D*dt  =>  D = σ² / (2*dt)
    D_x = (sigma_x**2) / (2.0 * dt)  # nm²/ms
    D_y = (sigma_y**2) / (2.0 * dt)  # nm²/ms
    
    # Average D from both directions and convert to µm²/s
    D = (D_x + D_y) / 2.0 / 1000.0  # Convert nm²/ms to µm²/s
    
    # Estimate uncertainty: use standard deviation of the two D values
    D_std = np.abs(D_x - D_y) / 2.0 / 1000.0  # Convert to µm²/s
    
    # Quality checks
    # 1. Isotropy check: σ_x and σ_y should be similar
    sigma_ratio = max(sigma_x, sigma_y) / min(sigma_x, sigma_y)
    is_isotropic = sigma_ratio <= MAX_SIGMA_RATIO
    
    # 2. Centering check: means should be near zero (no systematic drift)
    mean_x_ratio = abs(mu_x) / sigma_x if sigma_x > 0 else 0
    mean_y_ratio = abs(mu_y) / sigma_y if sigma_y > 0 else 0
    is_centered = (mean_x_ratio <= MAX_MEAN_SIGMA_RATIO and 
                   mean_y_ratio <= MAX_MEAN_SIGMA_RATIO)
    
    # Overall quality flag
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
        'D_x': D_x / 1000.0,  # Convert to µm²/s
        'D_y': D_y / 1000.0,  # Convert to µm²/s
        'sigma_ratio': sigma_ratio,
        'mean_x_ratio': mean_x_ratio,
        'mean_y_ratio': mean_y_ratio,
        'is_isotropic': is_isotropic,
        'is_centered': is_centered,
        'quality_flag': quality_flag,
        'num_steps': len(dx_nm),
        'num_particles': step_df['particle'].nunique()
    }


def analyze_single_file(xml_path: Path, mpp: float, fps: float) -> Optional[Dict]:
    """
    Analyze a single XML file to calculate diffusion coefficient from step sizes.
    
    Args:
        xml_path: Path to TrackMate XML file
        mpp: Micrometers per pixel
        fps: Frames per second
        
    Returns:
        Dictionary with analysis results, or None if analysis fails
    """
    # Load track data
    df = read_trackmate_xml(xml_path)
    if df is None or df.empty:
        return None
    
    # Calculate step sizes
    step_df = calculate_step_sizes(df, step_interval=STEP_INTERVAL)
    if step_df.empty:
        return None
    
    # Calculate diffusion coefficient
    results = calculate_diffusion_from_steps(step_df, mpp, fps)
    
    # Add metadata
    results['xml_path'] = str(xml_path)
    results['xml_name'] = xml_path.name
    results['mpp'] = mpp
    results['fps'] = fps
    
    return results


def analyze_all_files(files_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze all XML files to calculate diffusion coefficients.
    
    Args:
        files_df: DataFrame from collect_all_files_by_particle_size()
        
    Returns:
        DataFrame with analysis results for each file
    """
    results_list = []
    
    print("\nAnalyzing files...")
    print("=" * 70)
    
    for idx, row in files_df.iterrows():
        xml_path = Path(row['xml_path'])
        particle_size = row['particle_size_nm']
        mpp = row['mpp']
        fps = row['fps']
        mode = row['mode']
        
        print(f"\n{particle_size:.0f} nm - {xml_path.name}")
        print(f"  Mode: {mode}, mpp={mpp} µm/px, fps={fps:.2f}")
        
        # Analyze file
        results = analyze_single_file(xml_path, mpp, fps)
        
        if results is None:
            print(f"  [FAILED] Analysis failed")
            continue
        
        # Add particle size
        results['particle_size_nm'] = particle_size
        results['mode'] = mode
        results['x_max'] = row.get('x_max')
        results['y_max'] = row.get('y_max')
        
        print(f"  [OK] Particles: {results['num_particles']}, Steps: {results['num_steps']}")
        print(f"  [OK] D = {results['D']:.4e} ± {results['D_std']:.4e} nm²/ms")
        
        # Display quality information
        quality_icon = "✓" if results['quality_flag'] == 'good' else "⚠"
        quality_msg = {
            'good': 'Good fit (isotropic, centered)',
            'anisotropic': 'WARNING: Anisotropic (σ_x/σ_y = {:.2f})'.format(results['sigma_ratio']),
            'drift': 'WARNING: Drift detected (|μ|/σ too large)',
            'both': 'WARNING: Anisotropic + Drift'
        }
        print(f"  Quality: [{results['quality_flag'].upper()}] {quality_msg[results['quality_flag']]}")
        print(f"    σ_x={results['sigma_x']:.2f} nm, σ_y={results['sigma_y']:.2f} nm (ratio={results['sigma_ratio']:.2f})")
        print(f"    μ_x={results['mu_x']:.2f} nm, μ_y={results['mu_y']:.2f} nm")
        
        results_list.append(results)
    
    print("\n" + "=" * 70)
    print(f"[OK] Successfully analyzed {len(results_list)}/{len(files_df)} files")
    print("=" * 70)
    
    return pd.DataFrame(results_list)


def combine_by_particle_size(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine results across multiple files for each particle size.
    
    Calculates weighted average diffusion coefficients and aggregated statistics.
    
    Args:
        results_df: DataFrame from analyze_all_files()
        
    Returns:
        DataFrame with one row per particle size containing combined statistics
    """
    combined_results = []
    
    for particle_size in sorted(results_df['particle_size_nm'].unique()):
        size_data = results_df[results_df['particle_size_nm'] == particle_size]
        
        # Weighted average of D (weight by number of steps)
        weights = size_data['num_steps'].values
        D_values = size_data['D'].values
        D_weighted = np.average(D_values, weights=weights)
        
        # Propagate errors (weighted)
        D_errors = size_data['D_std'].values
        D_weighted_err = np.sqrt(np.sum((weights * D_errors)**2)) / np.sum(weights)
        
        # Calculate theoretical D and convert to µm²/s
        D_theory = calculate_theoretical_D(particle_size) / 1000.0
        
        combined_results.append({
            'particle_size_nm': particle_size,
            'num_files': len(size_data),
            'total_particles': size_data['num_particles'].sum(),
            'total_steps': size_data['num_steps'].sum(),
            'D_measured': D_weighted,
            'D_measured_std': D_weighted_err,
            'D_theoretical': D_theory,
            'mean_sigma': np.average((size_data['sigma_x'].values + size_data['sigma_y'].values) / 2, weights=weights),
            'modes': ', '.join(size_data['mode'].unique())
        })
    
    return pd.DataFrame(combined_results)


def count_particles_per_size(tracks_by_size: Dict[float, List[Path]]) -> pd.DataFrame:
    """
    Count total particles per particle size from TrackMate XML files.
    
    Args:
        tracks_by_size: Dictionary mapping particle_size_nm -> list of XML paths
        
    Returns:
        DataFrame with columns:
        - particle_size_nm: Particle size in nanometers
        - num_files: Number of XML files processed
        - total_particles: Total unique particles found
    """
    results = []
    
    for particle_size, xml_files in sorted(tracks_by_size.items()):
        total_particles = 0
        
        # Process each XML file
        for xml_file in xml_files:
            df = read_trackmate_xml(xml_file)
            if df is not None and not df.empty:
                # Count unique particles in this file
                num_particles = df['particle'].nunique()
                total_particles += num_particles
                print(f"  {xml_file.name}: {num_particles} particles")
        
        # Store results for this particle size
        results.append({
            'particle_size_nm': particle_size,
            'num_files': len(xml_files),
            'total_particles': total_particles
        })
        
        print(f"\n{particle_size} nm: {total_particles} total particles "
              f"from {len(xml_files)} files\n")
    
    return pd.DataFrame(results)


def plot_step_size_distributions(results_df: pd.DataFrame, save_path: Path) -> None:
    """
    Create histograms of step size distributions for each individual file.
    
    Args:
        results_df: DataFrame from analyze_all_files()
        save_path: Directory to save plots
    """
    print("\nCreating individual file step size distribution plots...")
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        xml_path = Path(row['xml_path'])
        particle_size = row['particle_size_nm']
        xml_name = row['xml_name']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        
        # Plot histogram
        ax.hist(steps_nm, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(steps_nm), color='red', linestyle='--', linewidth=2,
                  label=f'Mean = {np.mean(steps_nm):.1f} nm')
        
        ax.set_xlabel('Step size [nm]', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Step Size Distribution - {particle_size:.0f} nm\n{xml_name}\n'
                    f'N = {len(steps_nm)} steps, {row["num_particles"]} particles', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Create safe filename from xml_name
        safe_filename = xml_name.replace('.xml', '').replace(' ', '_').replace('/', '_')
        plt.savefig(save_path / f'water_step_dist_{particle_size:.0f}nm_{safe_filename}.png', dpi=300)
        plt.close(fig)
    
    print(f"[OK] Individual file step size distribution plots saved to {save_path}")


def plot_step_size_overlay(results_df: pd.DataFrame, save_path: Path) -> None:
    """
    Create overlay histogram showing step size distributions for each individual file.
    
    Args:
        results_df: DataFrame from analyze_all_files()
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


def plot_dx_dy_distributions(results_df: pd.DataFrame, save_path: Path) -> None:
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
        plt.savefig(save_path / f'water_dx_dy_dist_{particle_size:.0f}nm_{safe_filename}.png', dpi=300)
        plt.close(fig)
    
    print(f"[OK] Individual file dx/dy distribution plots saved to {save_path}")
def plot_diffusion_comparison(combined_df: pd.DataFrame, results_df: pd.DataFrame, save_path: Path) -> None:
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
    plt.savefig(save_path / 'diffusion_comparison_stepsize_individual.png', dpi=300)
    print(f"\n✓ Comparison plot saved to: {save_path / 'diffusion_comparison_stepsize_individual.png'}")
    plt.show()
    plt.close(fig)


def print_comparison_table(combined_df: pd.DataFrame) -> None:
    """
    Print formatted table comparing measured and theoretical diffusion coefficients.
    
    Args:
        combined_df: DataFrame from combine_by_particle_size()
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
    
    print("\n" + "=" * 90)
    print("DIFFUSION COEFFICIENT COMPARISON")
    print("=" * 90)
    print(f"{'Size':>8} | {'Files':>5} | {'Steps':>8} | {'D Measured':>15} | {'D Theory':>12} | {'D DLS':>12} | {'Ratio':>6}")
    print(f"{'(nm)':>8} | {'':>5} | {'':>8} | {'(µm²/s)':>15} | {'(µm²/s)':>12} | {'(µm²/s)':>12} | {'(M/T)':>6}")
    print("-" * 90)
    
    for _, row in combined_df.iterrows():
        size = row['particle_size_nm']
        num_files = row['num_files']
        total_steps = row['total_steps']
        D_meas = row['D_measured']
        D_err = row['D_measured_std']
        D_theo = row['D_theoretical']
        ratio = D_meas / D_theo
        
        dls_str = f"{DLS_MEASUREMENTS[size]:.4e}" if size in DLS_MEASUREMENTS else "N/A"
        
        print(f"{size:8.0f} | {num_files:5d} | {total_steps:8d} | {D_meas:.4e}±{D_err:.2e} | "
              f"{D_theo:.4e} | {dls_str:>12} | {ratio:6.2f}")
    
    print("=" * 90)


def calculate_theoretical_D(particle_size_nm: float) -> float:
    """
    Calculate theoretical diffusion coefficient using Stokes-Einstein equation.
    
    D = kB*T / (6*pi*eta*R)
    
    Args:
        particle_size_nm: Particle diameter in nanometers
        
    Returns:
        Diffusion coefficient in nm²/ms
    """
    d_m = particle_size_nm * 1e-9  # Convert nm to m
    R = d_m / 2.0  # Radius in m
    
    # Stokes-Einstein equation
    D_m2_s = BOLTZMANN_CONSTANT * TEMPERATURE / (6 * np.pi * WATER_VISCOSITY * R)
    
    # Convert m²/s to nm²/ms
    # m²/s * (1e9 nm/m)² * (1s/1000ms) = m²/s * 1e18 / 1000 = m²/s * 1e15
    D_nm2_ms = D_m2_s * 1e15
    
    return D_nm2_ms


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for step size diffusion analysis.
    
    This function:
    1. Scans for XML track files and extracts FPS from .rec files
    2. Counts particles per particle size
    3. Calculates diffusion coefficients using step size method
    4. Creates plots and exports results
    """
    print("=" * 70)
    print("TrackMate Step Size Diffusion Analysis")
    print("=" * 70)
    print(f"\nRoot directory: {ROOT_PATH}")
    print(f"Save directory: {SAVE_PATH}")
    print(f"Step interval: {STEP_INTERVAL} (using every {STEP_INTERVAL}{'st' if STEP_INTERVAL == 1 else 'rd' if STEP_INTERVAL == 3 else 'th'} step)")
    print(f"Quality thresholds: sigma_ratio <= {MAX_SIGMA_RATIO}, mean/sigma <= {MAX_MEAN_SIGMA_RATIO}\n")
    
    # Step 1: Collect XML files with FPS data
    print("=" * 70)
    print("Step 1: Scanning for XML track files and calibration data...")
    print("=" * 70)
    
    files_df = collect_all_files_by_particle_size(ROOT_PATH)
    
    print(f"\n[OK] Found {len(files_df)} XML files across {files_df['particle_size_nm'].nunique()} particle sizes")
    
    # Save XML file listing
    output_files_csv = SAVE_PATH / "xml_file_associations.csv"
    files_df.to_csv(output_files_csv, index=False)
    print(f"[OK] XML file associations saved to: {output_files_csv}")
    
    # Display mode statistics
    xml_with_fps = files_df[files_df['fps'].notna()]
    
    if not xml_with_fps.empty:
        print("\n" + "=" * 70)
        print("MODE DETECTION STATISTICS")
        print("=" * 70)
        
        # Group by mode
        mode_20 = xml_with_fps[xml_with_fps['mode'] == '20 FPS']
        mode_60 = xml_with_fps[xml_with_fps['mode'] == '60 FPS']
        mode_unknown = xml_with_fps[xml_with_fps['mode'] == 'Unknown']
        
        print(f"\nTotal XML files: {len(xml_with_fps)}")
        print(f"  • 20 FPS mode: {len(mode_20)} files ({len(mode_20)/len(xml_with_fps)*100:.1f}%)")
        print(f"  • 60 FPS mode: {len(mode_60)} files ({len(mode_60)/len(xml_with_fps)*100:.1f}%)")
        if len(mode_unknown) > 0:
            print(f"  • Unknown mode: {len(mode_unknown)} files ({len(mode_unknown)/len(xml_with_fps)*100:.1f}%)")
        
        # Detailed breakdown by particle size
        print("\n" + "-" * 70)
        print("BREAKDOWN BY PARTICLE SIZE")
        print("-" * 70)
        
        for particle_size in sorted(xml_with_fps['particle_size_nm'].unique()):
            size_files = xml_with_fps[xml_with_fps['particle_size_nm'] == particle_size]
            print(f"\n{particle_size:.0f} nm ({len(size_files)} XML files):")
            for _, row in size_files.iterrows():
                size_str = f"{row['x_max']}×{row['y_max']}" if pd.notna(row['x_max']) else "unknown"
                print(f"  • {row['xml_name']}: {row['mode']} ({size_str}, {row['mpp']} µm/px)")
    else:
        print("\nWarning: No calibration data found - will use default values")
    
    # Step 2: Count particles
    print("\n" + "=" * 70)
    print("Step 2: Counting particles from XML files...")
    print("=" * 70 + "\n")
    
    # Count particles per size
    particle_counts = []
    for particle_size in sorted(files_df['particle_size_nm'].unique()):
        size_xmls = files_df[files_df['particle_size_nm'] == particle_size]
        total_particles = 0
        
        print(f"{particle_size:.0f} nm:")
        for _, row in size_xmls.iterrows():
            xml_path = Path(row['xml_path'])
            df = read_trackmate_xml(xml_path)
            if df is not None and not df.empty:
                num_particles = df['particle'].nunique()
                total_particles += num_particles
                print(f"  • {row['xml_name']}: {num_particles} particles")
        
        particle_counts.append({
            'particle_size_nm': particle_size,
            'num_xml_files': len(size_xmls),
            'total_particles': total_particles
        })
        
        print(f"  → Total: {total_particles} particles from {len(size_xmls)} files\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(particle_counts)
    
    # Print summary table
    print("=" * 70)
    print("PARTICLE COUNT SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    # Save particle count summary
    output_summary_csv = SAVE_PATH / "particle_count_summary.csv"
    summary_df.to_csv(output_summary_csv, index=False)
    print(f"\n[OK] Particle count summary saved to: {output_summary_csv}")
    
    # Step 3: Calculate diffusion coefficients from step sizes
    print("\n" + "=" * 70)
    print("Step 3: Calculating diffusion coefficients from step sizes...")
    print("=" * 70)
    
    results_df = analyze_all_files(files_df)
    
    if not results_df.empty:
        # Save individual file results
        output_results_csv = SAVE_PATH / "water_stepsize_analysis_results.csv"
        results_df.to_csv(output_results_csv, index=False)
        print(f"\n[OK] Individual file results saved to: {output_results_csv}")
        
        # Step 4: Combine results by particle size
        print("\n" + "=" * 70)
        print("Step 4: Combining results by particle size...")
        print("=" * 70)
        
        combined_df = combine_by_particle_size(results_df)
        
        # Save combined results
        output_combined_csv = SAVE_PATH / "water_diffusion_coefficients_stepsize.csv"
        combined_df.to_csv(output_combined_csv, index=False)
        print(f"\n[OK] Combined results saved to: {output_combined_csv}")
        
        # Print comparison table
        print_comparison_table(combined_df)
        
        # Step 5: Create visualizations
        print("\n" + "=" * 70)
        print("Step 5: Creating visualizations...")
        print("=" * 70)
        
        plot_step_size_distributions(results_df, SAVE_PATH)
        plot_step_size_overlay(results_df, SAVE_PATH)
        plot_dx_dy_distributions(results_df, SAVE_PATH)
        plot_diffusion_comparison(combined_df, results_df, SAVE_PATH)
    
    print(f"\n[OK] Analysis complete!\n")
    print(f"All results saved to: {SAVE_PATH}\n")


if __name__ == "__main__":
    main()

