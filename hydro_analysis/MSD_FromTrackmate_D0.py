"""
TrackMate XML Particle Counter and MSD Analysis

This script processes TrackMate-generated XML files to count particles and perform
Mean Squared Displacement (MSD) analysis for different particle sizes in water.

Main features:
- Scans directory structure for particle size folders with Tracks subfolders
- Counts total particles per particle size from TrackMate XML files
- Provides utility functions for MSD analysis and power-law fitting
- Exports summary statistics to CSV

Author: Jonas
Date: 2026-01-04
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from types import SimpleNamespace
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt


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
DEFAULT_POINTS = 6  # number of points for power-law fitting

# Filter parameters
MIN_TRACK_LENGTH = 30  # minimum number of frames for a valid track
MIN_EXPONENT = 0.85  # minimum power-law exponent (n) for free diffusion


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
    
    Looks for exposure time and delay to calculate FPS.
    Format: "Exposure / Delay        : 50.000000 ms / 0.000000 ms"
    
    Args:
        rec_path: Path to .rec file
        
    Returns:
        Dictionary with keys:
        - exposure_ms: Exposure time in milliseconds
        - delay_ms: Delay time in milliseconds
        - fps: Calculated frames per second (1000 / (exposure + delay))
    """
    result = {
        'exposure_ms': None,
        'delay_ms': None,
        'fps': None
    }
    
    try:
        with open(rec_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Search for exposure/delay line
        # Pattern: "Exposure / Delay        : 50.000000 ms / 0.000000 ms"
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
                print(f"    DEBUG REC: {rec_path.name} -> Exp={exposure:.2f}ms, Delay={delay:.2f}ms, FPS={result['fps']:.2f}")
        else:
            print(f"    WARNING: Could not find Exposure/Delay in {rec_path.name}")
                
    except Exception as e:
        print(f"    ERROR: Could not parse {rec_path.name}: {e}")
    
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


def fit_powerlaw_with_errors(em_series: pd.Series, points: int = 10, 
                            ax=None, plot: bool = False) -> SimpleNamespace:
    """
    Fit power-law model y = A * x^n to ensemble MSD data.
    
    Performs linear regression in log-space to estimate parameters and
    their standard errors.
    
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
    # Extract data for fitting
    xs = em_series.iloc[0:points].index.values.astype(float)
    ys = em_series.iloc[0:points].values.astype(float)
    
    # Filter out invalid data points
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    
    if mask.sum() < 2:
        # Fallback to trackpy's built-in fitter if insufficient data
        return tp.utils.fit_powerlaw(em_series.iloc[0:points], plot=plot, ax=ax)
    
    # Perform linear regression in log-space: log(y) = n*log(x) + log(A)
    lx = np.log(xs[mask])
    ly = np.log(ys[mask])
    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
    
    # Extract fitted parameters
    n_fit = float(coeffs[0])
    logA_fit = float(coeffs[1])
    
    # Calculate standard errors
    se = np.sqrt(np.diag(cov))
    se_n = float(se[0])
    se_logA = float(se[1])
    
    # Transform back to linear space
    A_fit = float(np.exp(logA_fit))
    se_A = A_fit * se_logA  # Error propagation
    
    return SimpleNamespace(
        A=np.array([A_fit]),
        n=np.array([n_fit]),
        A_err=np.array([se_A]),
        n_err=np.array([se_n]),
        logA=np.array([logA_fit]),
        logA_err=np.array([se_logA]),
        cov=cov
    )


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
        - exposure_ms: Average exposure time from .rec files in this folder
        - delay_ms: Average delay time from .rec files
        - fps: Calculated frames per second (average from all .rec files)
        - fps_category: Classification ('~20 FPS', '~60 FPS', 'Other', 'No Data')
        - num_rec_files: Number of .rec files used for averaging
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
        
        print(f"\n📁 Folder: {subfolder.name}")
        print(f"   Particle size: {particle_size} nm" if particle_size else "   ⚠ No particle size detected")
        
        if particle_size is None:
            continue
        
        # Find all REC files in main folder and parse them
        rec_files = sorted(list(subfolder.glob("*.rec")))
        print(f"   Found {len(rec_files)} .rec files")
        
        # Parse all REC files to get FPS information for this particle size
        fps_info_list = []
        for rec_file in rec_files:
            rec_info = parse_rec_file(rec_file)
            if rec_info['fps'] is not None:
                fps_info_list.append(rec_info)
                print(f"     • {rec_file.name}: {rec_info['fps']:.2f} fps")
        
        # Calculate average FPS for this particle size
        avg_fps_info = {'exposure_ms': None, 'delay_ms': None, 'fps': None}
        num_valid_rec = 0
        
        if fps_info_list:
            avg_fps_info = {
                'exposure_ms': np.mean([x['exposure_ms'] for x in fps_info_list]),
                'delay_ms': np.mean([x['delay_ms'] for x in fps_info_list]),
                'fps': np.mean([x['fps'] for x in fps_info_list])
            }
            num_valid_rec = len(fps_info_list)
            print(f"   → Average FPS: {avg_fps_info['fps']:.2f} (from {num_valid_rec} files)")
        else:
            print(f"   ⚠ No valid FPS data from .rec files")
        
        # Find all XML files in Tracks subfolder
        tracks_folder = subfolder / "Tracks"
        xml_files = []
        
        if tracks_folder.exists() and tracks_folder.is_dir():
            xml_files = sorted(list(tracks_folder.glob("*.xml")))
            print(f"   Found {len(xml_files)} XML track files in Tracks/")
        else:
            print(f"   ⚠ No Tracks/ subfolder found")
        
        if len(xml_files) == 0:
            print(f"   ⚠ Skipping - no XML files found")
            continue
        
        # Create one record per XML file
        for xml_file in xml_files:
            # Extract image dimensions from XML
            x_max, y_max = extract_image_dimensions_from_xml(xml_file)
            
            # Determine mpp and fps based on image size and/or .rec data
            mpp, fps, mode = get_mpp_from_fps_and_size(
                fps=avg_fps_info['fps'],
                x_max=x_max,
                y_max=y_max
            )
            
            file_records.append({
                'particle_size_nm': particle_size,
                'xml_path': str(xml_file),
                'xml_name': xml_file.name,
                'x_max': x_max,
                'y_max': y_max,
                'exposure_ms': avg_fps_info['exposure_ms'],
                'delay_ms': avg_fps_info['delay_ms'],
                'fps_from_rec': avg_fps_info['fps'],
                'fps': fps,
                'mpp': mpp,
                'mode': mode,
                'num_rec_files': num_valid_rec,
            })
            
            size_info = f"{x_max}×{y_max}" if x_max and y_max else "unknown size"
            print(f"     ✓ {xml_file.name} [{size_info} → {mode}, {mpp} µm/px]")
        
        print(f"   → Created {len(xml_files)} records")
    
    print(f"\n" + "=" * 70)
    print(f"✓ Total XML files found: {len(file_records)}")
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

def calculate_imsd_for_file(xml_path: Path, mpp: float, fps: float) -> Optional[pd.DataFrame]:
    """
    Calculate individual MSD (iMSD) for all particles in a single XML file.
    
    Tracks are already complete from TrackMate and do not need linking.
    Drift is computed and subtracted before MSD calculation.
    
    Args:
        xml_path: Path to TrackMate XML file
        mpp: Micrometers per pixel
        fps: Frames per second
        
    Returns:
        DataFrame with individual MSDs (wide format: columns=particle IDs, index=lag time)
        or None if calculation fails
    """
    try:
        # Load tracks from XML (already complete, no linking needed)
        df = read_trackmate_xml(xml_path)
        if df is None or df.empty:
            return None
        
        # Ensure correct data types
        df['frame'] = df['frame'].astype(int)
        df['particle'] = df['particle'].astype(int)
        
        # Suppress trackpy warnings
        tp.quiet()
        df_corrected = df.copy()
        # # Compute and subtract drift
        drift = tp.compute_drift(df)
        # Calculate per-frame drift (differences between consecutive frames)
        drift_per_frame_x = drift['x'].diff().dropna()
        drift_per_frame_y = drift['y'].diff().dropna()
        
        # Convert to physical units
        drift_per_frame_x_um = drift_per_frame_x * mpp
        drift_per_frame_y_um = drift_per_frame_y * mpp
        
        # Calculate statistics
        mean_drift_x_um = drift_per_frame_x_um.mean()
        mean_drift_y_um = drift_per_frame_y_um.mean()
        mean_drift_x_um_per_s = mean_drift_x_um * fps
        mean_drift_y_um_per_s = mean_drift_y_um * fps
        
        print(f"  Drift per frame: x={mean_drift_x_um:.4f} µm ({mean_drift_x_um_per_s:.4f} µm/s), "
              f"y={mean_drift_y_um:.4f} µm ({mean_drift_y_um_per_s:.4f} µm/s)")
        
        # Subtract drift correction
        df_corrected = tp.subtract_drift(df.copy(), drift)
        
        # subtract_drift() uses set_index(..., drop=False) internally,
        # which creates a MultiIndex ['frame', 'particle'] but also keeps
        # them as columns. This causes "ambiguous" error in tp.imsd().
        # Solution: Drop the MultiIndex, keep only the column versions.
        if isinstance(df_corrected.index, pd.MultiIndex):
            df_corrected = df_corrected.reset_index(drop=True)
        
        # Calculate iMSD with drift-corrected positions
        imsd = tp.imsd(df_corrected, mpp, fps)
        
        return imsd
        
    except Exception as e:
        print(f"    Error calculating iMSD for {xml_path.name}: {e}")
        return None


def analyze_all_msds(files_df: pd.DataFrame, points: int = DEFAULT_POINTS) -> pd.DataFrame:
    """
    Calculate iMSD for all files with both XML and REC data.
    
    Filters files based on particle size and acquisition mode:
    - 20, 50, 100 nm: Only use 60 FPS files
    - All other sizes: Only use 20 FPS files
    
    Args:
        files_df: DataFrame from collect_all_files_by_particle_size()
        points: Number of points for power-law fitting (stored in results)
        
    Returns:
        DataFrame with columns:
        - particle_size_nm: Particle size
        - xml_name: XML filename
        - fps: Frames per second from .rec file
        - fps_category: FPS classification
        - mpp: Micrometers per pixel (based on fps)
        - num_particles: Number of tracked particles
        - num_lag_times: Number of lag times in MSD
        - points: Number of points used for power-law fitting
        - msd_data: iMSD DataFrame (wide format)
    """
    results = []
    
    # Debug: Show what we're filtering
    print(files_df.head())

    print(f"\nDEBUG: Total files in dataframe: {len(files_df)}")
    print(f"DEBUG: Files with xml_path: {files_df['xml_path'].notna().sum()}")
    print(f"DEBUG: Files with fps data: {files_df['fps'].notna().sum()}")
    
    # Filter for files that have both XML and FPS data
    valid_files = files_df[(files_df['xml_path'].notna()) & (files_df['fps'].notna())].copy()
    
    print(f"DEBUG: Valid files (XML + FPS): {len(valid_files)}")
    
    if valid_files.empty:
        print("⚠ No files with both XML tracks and FPS data found!")
        print("\nPossible issues:")
        print("  - XML files and REC files might have different base names")
        print("  - REC files might not be parsed correctly")
        return pd.DataFrame()
    
    # No FPS filtering - use all files (consistent with step size method)
    print("\n" + "=" * 70)
    print(f"Using all {len(valid_files)} files for analysis (no FPS filtering)")
    print("=" * 70)
    
    print(f"\nCalculating iMSD for {len(valid_files)} files...")
    print("-" * 70)
    
    for idx, row in valid_files.iterrows():
        xml_path = Path(row['xml_path'])
        fps = row['fps']
        mpp = row['mpp']
        mode = row['mode']
        particle_size = row['particle_size_nm']
        
        size_info = f"{row['x_max']}×{row['y_max']}" if pd.notna(row['x_max']) else "unknown"
        print(f"\n{particle_size:.0f} nm - {xml_path.name}")
        print(f"  Mode: {mode} | Size: {size_info} | FPS: {fps:.2f} | mpp: {mpp} µm/px")
        
        # Calculate iMSD
        imsd = calculate_imsd_for_file(xml_path, mpp, fps)
        
        if imsd is not None and not imsd.empty:
            num_particles = len(imsd.columns)
            num_lag_times = len(imsd)
            
            results.append({
                'particle_size_nm': particle_size,
                'xml_name': row['xml_name'],
                'mode': mode,
                'x_max': row['x_max'],
                'y_max': row['y_max'],
                'fps': fps,
                'mpp': mpp,
                'num_particles': num_particles,
                'num_lag_times': num_lag_times,
                'points': points,
                'msd_data': imsd
            })
            
            print(f"  ✓ {num_particles} particles tracked over {num_lag_times} lag times")
        else:
            print(f"  ✗ Failed to calculate MSD")
    
    return pd.DataFrame(results)


def plot_msd_by_particle_size(msd_results_df: pd.DataFrame, save_path: Path, points: int = DEFAULT_POINTS):
    """
    Create MSD plots for each particle size, showing individual file results separately.
    
    Args:
        msd_results_df: DataFrame from analyze_all_msds()
        save_path: Directory to save plots
        points: Number of points for power-law fitting
    """
    if msd_results_df.empty:
        return
    
    print("\n" + "=" * 70)
    print("Creating MSD plots by particle size...")
    print("-" * 70)
    
    # Color palette for different files
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for particle_size in sorted(msd_results_df['particle_size_nm'].unique()):
        size_data = msd_results_df[msd_results_df['particle_size_nm'] == particle_size]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each file separately with its own ensemble MSD and fit
        for file_idx, (idx, row) in enumerate(size_data.iterrows()):
            imsd = row['msd_data']
            if imsd is None or imsd.empty:
                continue
            
            color = colors[file_idx % len(colors)]
            file_label = f"{row['xml_name'].replace('.xml', '')} ({row['fps']:.1f} FPS)"
            
            # Plot individual particle MSDs with low alpha
            for col in imsd.columns:
                ax.plot(imsd.index, imsd[col], color=color, alpha=0.1, linewidth=0.5)
            
            # Calculate ensemble MSD for this file
            emsd_file = imsd.mean(axis=1)
            
            # Plot ensemble MSD for this file
            ax.plot(emsd_file.index, emsd_file, 'o-', color=color, linewidth=2, 
                   markersize=5, label=f'{file_label} (n={len(imsd.columns)})', alpha=0.8)
            
            # Fit power-law to this file's ensemble MSD
            try:
                params = fit_powerlaw_with_errors(emsd_file, points=points, plot=False)
                A = float(params.A[0])
                A_err = float(params.A_err[0])
                n = float(params.n[0])
                n_err = float(params.n_err[0])
                D = A / 4.0
                D_err = A_err / 4.0
                
                # Plot fit
                fit_x = emsd_file.iloc[0:points].index
                fit_y = A * np.array(fit_x) ** n
                ax.plot(fit_x, fit_y, '--', color=color, linewidth=2, alpha=0.7)
                
                print(f"  {file_label}: D = {D:.4e} ± {D_err:.4e} µm²/s, n = {n:.3f} ± {n_err:.3f}")
            except Exception as e:
                print(f"  Warning: Could not fit power-law for {file_label}: {e}")
        
        # Format plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Lag time [frames]', fontsize=12)
        ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [µm²]', fontsize=12)
        ax.set_title(f'Mean Squared Displacement - {particle_size:.0f} nm particles (Individual Files)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='best', ncol=1 if len(size_data) <= 4 else 2)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = save_path / f'water_MSD_{particle_size:.0f}nm_individual_files.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {plot_filename.name}")
        plt.close(fig)


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


def combine_and_analyze(paths_list: List[Path], save_path: Path = SAVE_PATH, 
                       mpp: float = DEFAULT_MPP, fps: float = DEFAULT_FPS, 
                       points: int = 10, min_track_length: int = MIN_TRACK_LENGTH) -> tuple[Dict[int, float], Dict[int, float]]:
    """
    Combine particle tracks from multiple files and perform MSD analysis.
    
    This function:
    1. Groups tracks by particle size (extracted from filename)
    2. Combines tracks from multiple files with unique particle IDs
    3. Computes ensemble MSD and fits power-law
    4. Plots results and compares to theoretical predictions
    
    Args:
        paths_list: List of XML file paths to process
        save_path: Directory for saving plots
        mpp: Micrometers per pixel calibration
        fps: Frames per second
        points: Number of points for power-law fitting
        
    Returns:
        Tuple of (combined_D, combined_D_err) dictionaries keyed by particle size
    """
    # Collect DataFrames grouped by particle size
    dfs_by_size = defaultdict(list)
    # Parse each file and group by particle size
    for p in paths_list:
        name = os.path.basename(p)
        
        # Try to extract particle size from filename
        m = re.search(r'(\d+(?:\.\d+)?)\s*nm', name, re.I)
        if m:
            size_nm = int(float(m.group(1)))
        else:
            # Fallback: look for numbers in reasonable range (20-1000 nm)
            nums = [int(x) for x in re.findall(r'(\d+)', name)]
            size_nm = next((n for n in nums if 20 <= n <= 1000), None)
        
        if size_nm is None:
            continue
            
        # Load track data
        df = read_trackmate_xml(p)
        if df is None or df.empty:
            continue
            
        dfs_by_size[size_nm].append(df)

    combined_D = {}
    combined_D_err = {}

    # Process each particle size
    for size_nm, dflist in dfs_by_size.items():
        if not dflist:
            continue
            
        # Combine tracks from multiple files with unique particle IDs
        # This prevents particle ID collisions when merging datasets
        offset = 0
        reindexed = []
        for df in dflist:
            df = df.copy()
            df['particle'] = df['particle'].astype(int) + offset
            max_id = df['particle'].max()
            offset = max(offset, max_id + 1)  # Next offset after highest ID
            reindexed.append(df)
        
        # Concatenate all DataFrames
        df_combined = pd.concat(reindexed, ignore_index=True)

        # Ensure correct data types
        df_combined['frame'] = df_combined['frame'].astype(int)
        df_combined['particle'] = df_combined['particle'].astype(int)

        # Compute Mean Squared Displacements
        # Tracks are already complete from TrackMate, no linking needed
        tp.quiet()  # Suppress trackpy warnings
        df_filtered = tp.filter_stubs(df_combined, threshold=min_track_length)  # Remove very short tracks
        # Individual and ensemble MSDs with correct mpp and fps
        im = tp.imsd(df_filtered, mpp, fps)  # Individual particle MSDs
        em = tp.emsd(df_filtered, mpp, fps)  # Ensemble average MSD

        # Fit power-law to ensemble MSD
        params = fit_powerlaw_with_errors(em[:5], points=points, plot=False)
        A = float(params.A[0])
        A_err = float(params.A_err[0]) if hasattr(params, 'A_err') else np.nan
        
        # Calculate diffusion coefficient: MSD = 4*D*t for 2D diffusion
        D = A / 4.0
        D_err = A_err / 4.0

        combined_D[size_nm] = D
        combined_D_err[size_nm] = D_err

        # Create MSD plot for this particle size
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot individual particle MSDs (gray lines in background)
        cols = list(im.columns)
        if cols:
            ax.plot(im.index, im[cols[0]], 'k-', alpha=0.2, 
                   label='Individual MSDs')
            for c in cols[1:]:
                ax.plot(im.index, im[c], 'k-', alpha=0.08)
        
        # Plot ensemble MSD
        ax.plot(em.index, em, 'o', markersize=6, color='blue', 
               label='Ensemble MSD (combined)')
        
        # Highlight fitting range
        ax.plot(em.iloc[0:points].index, em.iloc[0:points], 'o', 
               markersize=4, color='red', label='Fitting range')
        
        # Plot power-law fit
        fit_x = em.iloc[0:points].index
        fit_y = A * np.array(fit_x) ** float(params.n[0])
        ax.plot(fit_x, fit_y, 'g--', linewidth=2, alpha=0.8, 
               label=f'Fit: A={A:.2e}, n={float(params.n[0]):.2f}')
        
        # Format plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Lag time [frames]', fontsize=12)
        ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', fontsize=12)
        ax.set_title(f'Combined ensemble MSD for {size_nm} nm '
                    f'(N={len(dflist)} files)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / f'water_combined_MSD_{size_nm}nm.png', dpi=300)
        plt.show()
        #plt.close(fig)

    return combined_D, combined_D_err


def compare_diffusion_coefficients(pickle_path: Path, save_path: Path, 
                                  points: int = 10, min_exponent: float = MIN_EXPONENT) -> None:
    """
    Load MSD data from pickle, calculate D values, and compare with theory and DLS.
    
    Args:
        pickle_path: Path to pickled MSD results DataFrame
        save_path: Directory to save comparison plots
        points: Number of points for power-law fitting
    """
    print("\nLoading MSD data from pickle...")
    
    # Load MSD results DataFrame
    msd_results_df = pd.read_pickle(pickle_path)
    
    if msd_results_df.empty:
        print("✗ No data found in pickle file!")
        return
    
    print(f"✓ Loaded MSD data for {len(msd_results_df)} files")
    
    # DLS measurements (reference values)
    DLS_MEASUREMENTS = {
        20: 12.38750325,
        50: 8.201969711,
        100: 4.139082033,
        200: 1.745323167,
        500: 0.621773811,
        1000: 0.356862091
    }
    
    # Calculate D values for each particle size
    D_values = {}
    D_errors = {}
    
    print("\n" + "=" * 70)
    print("CALCULATING DIFFUSION COEFFICIENTS")
    print("=" * 70)
    
    # Store D values for each file separately
    file_D_values = []  # List of dicts for per-file analysis
    
    for particle_size in sorted(msd_results_df['particle_size_nm'].unique()):
        size_data = msd_results_df[msd_results_df['particle_size_nm'] == particle_size]
        
        print(f"\n{particle_size:.0f} nm particles ({len(size_data)} files):")
        
        # Analyze each file separately
        size_D_values = []
        for file_idx, (_, row) in enumerate(size_data.iterrows()):
            imsd = row['msd_data']
            if imsd is None or imsd.empty:
                print(f"  ✗ {row['xml_name']}: No valid MSD data")
                continue
            
            file_label = row['xml_name'].replace('.xml', '')
            print(f"\n  File: {file_label}")
            print(f"    Particles: {len(imsd.columns)}, Lag times: {len(imsd)}")
            
            # Filter particles by exponent (n >= min_exponent for free diffusion)
            valid_particles = []
            particle_exponents = []
            
            for particle_id in imsd.columns:
                particle_msd = imsd[particle_id].dropna()
                
                if len(particle_msd) < points:
                    continue
                
                try:
                    particle_params = fit_powerlaw_with_errors(particle_msd, points=points, plot=False)
                    n_particle = float(particle_params.n[0])
                    
                    if n_particle >= min_exponent:
                        valid_particles.append(particle_id)
                        particle_exponents.append(n_particle)
                except Exception:
                    continue
            
            if not valid_particles:
                print(f"    ✗ No valid particles after filtering (n >= {min_exponent:.2f})")
                continue
            
            print(f"    Valid particles: {len(valid_particles)}/{len(imsd.columns)} (n >= {min_exponent:.2f})")
            print(f"    Mean exponent: {np.mean(particle_exponents):.3f} ± {np.std(particle_exponents):.3f}")
            
            # Calculate ensemble MSD for valid particles
            filtered_imsd = imsd[valid_particles]
            ensemble_msd = filtered_imsd.mean(axis=1)
            
            # Fit power-law to ensemble MSD
            try:
                tp.quiet()
                params = fit_powerlaw_with_errors(ensemble_msd, points=points, plot=False)
                A = float(params.A[0])
                A_err = float(params.A_err[0])
                n = float(params.n[0])
                n_err = float(params.n_err[0])
                
                # Calculate diffusion coefficient: MSD = 4*D*t for 2D
                D = A / 4.0
                D_err = A_err / 4.0
                
                size_D_values.append(D)
                file_D_values.append({
                    'particle_size_nm': particle_size,
                    'file_name': file_label,
                    'D': D,
                    'D_err': D_err,
                    'A': A,
                    'A_err': A_err,
                    'n': n,
                    'n_err': n_err,
                    'num_particles': len(valid_particles)
                })
                
                print(f"    Power-law fit: A = {A:.4e} ± {A_err:.4e}, n = {n:.3f} ± {n_err:.3f}")
                print(f"    Diffusion coefficient: D = {D:.4e} ± {D_err:.4e} µm²/s")
                
                # Calculate theoretical diameter from measured D
                R_measured = (BOLTZMANN_CONSTANT * TEMPERATURE) / (6 * np.pi * WATER_VISCOSITY * D * 1e-12)
                d_measured = 2 * R_measured * 1e9
                print(f"    Calculated diameter from D: {d_measured:.1f} nm")
                
                # Create individual MSD plot for this file
                fig, ax = plt.subplots(figsize=(10, 7))
                
                # Plot individual particle MSDs
                cols = list(filtered_imsd.columns)
                if cols:
                    ax.plot(filtered_imsd.index, filtered_imsd[cols[0]], 'k-', 
                        alpha=0.2, linewidth=0.5, label='Individual MSDs')
                    for c in cols[1:]:
                        ax.plot(filtered_imsd.index, filtered_imsd[c], 'k-', 
                            alpha=0.05, linewidth=0.5)
                
                # Plot ensemble MSD
                ax.plot(ensemble_msd.index, ensemble_msd, 'o', markersize=6, 
                    color='blue', label='Ensemble MSD')
                
                # Plot fitting range
                ax.plot(ensemble_msd.iloc[0:points].index, 
                    ensemble_msd.iloc[0:points], 'o', markersize=4, 
                    color='red', label='Fitting range')
                
                # Plot power-law fit
                fit_x = ensemble_msd.iloc[0:points].index
                fit_y = A * np.array(fit_x) ** n
                ax.plot(fit_x, fit_y, 'g--', linewidth=3, alpha=0.8, 
                    label=f'Fit: A={A:.2e}, n={n:.2f}')
                
                # Plot theoretical prediction (convert to µm²/s)
                D_theory = calculate_theoretical_D(particle_size) / 1000.0
                theory_y = 4 * D_theory * np.array(fit_x)
                ax.plot(fit_x, theory_y, 'p--', linewidth=2, alpha=0.8, 
                    color='purple', label=f'Theory: D={D_theory:.2e}')
                
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Lag time [s]', fontsize=12)
                ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [µm²]', fontsize=12)
                ax.set_title(f'MSD Analysis - {particle_size:.0f} nm - {file_label}', 
                            fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, which='both')
                
                plt.tight_layout()
                # Save with unique filename per file
                safe_filename = file_label.replace(' ', '_').replace('/', '_')
                plt.savefig(save_path / f'water_MSD_fit_{particle_size:.0f}nm_{safe_filename}.png', dpi=300)
                print(f"    ✓ Saved: water_MSD_fit_{particle_size:.0f}nm_{safe_filename}.png")
                plt.close(fig)
                
            except Exception as e:
                print(f"    ✗ Error fitting power-law: {e}")
        
        # Calculate mean D for this particle size (across files)
        if size_D_values:
            D_mean = np.mean(size_D_values)
            D_std = np.std(size_D_values)
            D_values[particle_size] = D_mean
            D_errors[particle_size] = D_std
            print(f"\n  Overall {particle_size:.0f} nm: D = {D_mean:.4e} ± {D_std:.4e} µm²/s (mean ± std across {len(size_D_values)} files)")
        else:
            print(f"\n  No valid files for {particle_size:.0f} nm particles")
        
        print("-" * 70)
    
    # Save per-file D values to CSV
    if file_D_values:
        file_D_df = pd.DataFrame(file_D_values)
        file_D_csv = save_path / 'water_diffusion_coefficients_per_file.csv'
        file_D_df.to_csv(file_D_csv, index=False)
        print(f"\n✓ Per-file diffusion coefficients saved to: {file_D_csv}")
    
    # Create comparison plot
    if not D_values:
        print("\n✗ No diffusion coefficients calculated!")
        return
    
    print("\n" + "=" * 70)
    print("COMPARISON WITH THEORY AND DLS")
    print("=" * 70)
    
    particle_sizes = sorted(D_values.keys())
    measured_D = [D_values[s] for s in particle_sizes]
    measured_D_err = [D_errors[s] for s in particle_sizes]
    
    # Calculate theoretical D for all sizes
    theoretical_D = [calculate_theoretical_D(s) for s in particle_sizes]
    
    # Get DLS measurements (only for matching sizes)
    dls_sizes = [s for s in particle_sizes if s in DLS_MEASUREMENTS]
    dls_D = [DLS_MEASUREMENTS[s] for s in dls_sizes]
    
    # Print comparison table
    print("\nParticle | Measured D      | Theoretical D   | DLS D          ")
    print("Size (nm)| (µm²/s)         | (µm²/s)         | (µm²/s)        ")
    print("-" * 70)
    for i, size in enumerate(particle_sizes):
        dls_str = f"{DLS_MEASUREMENTS[size]:.4e}" if size in DLS_MEASUREMENTS else "N/A"
        print(f"{size:8.0f} | {measured_D[i]:.4e} ± {measured_D_err[i]:.2e} | "
              f"{theoretical_D[i]:.4e} | {dls_str}")
    
    # Create comparison plot with individual files shown discretely
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color palette for different particle sizes
    colors_by_size = plt.cm.tab10(np.linspace(0, 1, len(particle_sizes)))
    size_to_color = dict(zip(particle_sizes, colors_by_size))
    
    # Plot individual files with small horizontal offset for visibility
    # Use different markers for different FPS modes
    offset_scale = 0.15  # Adjust this to control horizontal spread
    for file_data in file_D_values:
        size = file_data['particle_size_nm']
        D = file_data['D']
        D_err = file_data['D_err']
        
        # Get corresponding row from msd_results_df to find mode
        matching_row = msd_results_df[
            (msd_results_df['particle_size_nm'] == size) & 
            (msd_results_df['xml_name'] == file_data['file_name'] + '.xml')
        ]
        
        if not matching_row.empty:
            mode = matching_row.iloc[0]['mode']
        else:
            mode = 'Unknown'
        
        # Different markers for different FPS modes
        if mode == '60 FPS':
            marker = '^'  # Triangle up for 60 FPS
            alpha = 0.8
        elif mode == '20 FPS':
            marker = 's'  # Square for 20 FPS
            alpha = 0.7
        else:
            marker = 'o'  # Circle for unknown
            alpha = 0.6
        
        # Calculate how many files we have for this size and create offset
        size_files = [f for f in file_D_values if f['particle_size_nm'] == size]
        file_index = size_files.index(file_data)
        num_files = len(size_files)
        
        # Center the offsets around the nominal size
        if num_files > 1:
            offset = (file_index - (num_files - 1) / 2) * (size * offset_scale / num_files)
        else:
            offset = 0
        
        x_pos = size + offset
        
        # Plot individual file with error bar
        ax.errorbar(x_pos, D, yerr=D_err, fmt=marker, 
                   markersize=7, color=size_to_color[size], 
                   ecolor=size_to_color[size], elinewidth=1.5, 
                   capsize=3, capthick=1.5, alpha=alpha)
    
    # Add custom legend entries for different FPS modes
    ax.errorbar([], [], [], fmt='^', markersize=7, color='gray', 
               ecolor='gray', elinewidth=1.5, capsize=3, capthick=1.5,
               label='Individual Files (60 FPS)', alpha=0.8)
    ax.errorbar([], [], [], fmt='s', markersize=7, color='gray', 
               ecolor='gray', elinewidth=1.5, capsize=3, capthick=1.5,
               label='Individual Files (20 FPS)', alpha=0.7)
    
    # Plot mean values for each size (larger markers)
    ax.errorbar(particle_sizes, measured_D, yerr=measured_D_err, fmt='D', 
               markersize=10, color='blue', ecolor='black', elinewidth=2, 
               capsize=5, capthick=2, label='Mean D per Size ± Std', zorder=5)
    
    # Plot theoretical values
    ax.scatter(particle_sizes, theoretical_D, s=150, color='black', 
              marker='x', linewidths=3, label='Theoretical D (Stokes-Einstein)', zorder=6)
    
    # Plot DLS values
    if dls_sizes:
        ax.scatter(dls_sizes, dls_D, s=150, color='red', marker='*', 
                  linewidths=2, edgecolors='darkred', label='D from DLS', zorder=6)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Particle size [nm]', fontsize=12)
    ax.set_ylabel('Diffusion coefficient D [µm²/s]', fontsize=12)
    ax.set_title('Overview of Diffusion Coefficients', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_path / 'water_Diffusionskoeffizienten_Übersicht.png', dpi=300)
    print(f"\n✓ Comparison plot saved to: {save_path / 'water_Diffusionskoeffizienten_Übersicht.png'}")
    plt.show()
    plt.close(fig)


def calculate_theoretical_D(particle_size_nm: float) -> float:
    """
    Calculate theoretical diffusion coefficient using Stokes-Einstein equation.
    
    D = kB*T / (6*pi*eta*R)
    
    Args:
        particle_size_nm: Particle diameter in nanometers
        
    Returns:
        Diffusion coefficient in µm²/s
    """
    d_m = particle_size_nm * 1e-9  # Convert nm to m
    R = d_m / 2.0  # Radius in m
    
    # Stokes-Einstein equation
    D_m2_s = BOLTZMANN_CONSTANT * TEMPERATURE / (6 * np.pi * WATER_VISCOSITY * R)
    
    # Convert m²/s to µm²/s
    D_um2_s = D_m2_s * 1e12
    
    return D_um2_s


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for XML-based MSD analysis.
    
    This function:
    1. Scans for XML track files and extracts FPS from .rec files
    2. Counts particles per particle size
    3. Calculates individual MSD (iMSD) for each file
    4. Creates plots and exports results
    """
    print("=" * 70)
    print("TrackMate MSD Analysis - XML-Focused Pipeline")
    print("=" * 70)
    print(f"\nRoot directory: {ROOT_PATH}")
    print(f"Save directory: {SAVE_PATH}\n")
    
    # Step 1: Collect XML files with FPS data
    print("=" * 70)
    print("Step 1: Scanning for XML track files and FPS data...")
    print("=" * 70)
    
    files_df = collect_all_files_by_particle_size(ROOT_PATH)
    

    
    print(f"\n✓ Found {len(files_df)} XML files across {files_df['particle_size_nm'].nunique()} particle sizes")
    
    # Save XML file listing
    output_files_csv = SAVE_PATH / "xml_file_associations.csv"
    files_df.to_csv(output_files_csv, index=False)
    print(f"✓ XML file associations saved to: {output_files_csv}")
    
    # Display FPS statistics
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
        print("\n⚠ Warning: No data found - will use default values")
    
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
    print(f"\n✓ Particle count summary saved to: {output_summary_csv}")
    
    # Step 3: Calculate iMSD for all files
    print("\n" + "=" * 70)
    print(f"Step 3: Calculating individual MSD (iMSD) for each file (using {DEFAULT_POINTS} points for fitting)...")
    print("=" * 70)
    
    msd_results_df = analyze_all_msds(files_df, points=DEFAULT_POINTS)
    
    if not msd_results_df.empty:
        # Summary by particle size
        print("\n" + "=" * 70)
        print("iMSD SUMMARY BY PARTICLE SIZE")
        print("=" * 70)
        
        for particle_size in sorted(msd_results_df['particle_size_nm'].unique()):
            size_data = msd_results_df[msd_results_df['particle_size_nm'] == particle_size]
            total_particles = size_data['num_particles'].sum()
            num_files = len(size_data)
            
            print(f"\n{particle_size:.0f} nm:")
            print(f"  Files analyzed: {num_files}")
            print(f"  Total particles: {total_particles}")
            print(f"  Files:")
            
            for _, row in size_data.iterrows():
                size_str = f"{row['x_max']}×{row['y_max']}" if pd.notna(row.get('x_max')) else "unknown"
                print(f"    • {row['xml_name']}: {row['num_particles']} particles "
                      f"({row['mode']}, {size_str}, {row['fps']:.2f} FPS, mpp={row['mpp']} µm/px, points={row['points']})")
        
        # Save MSD summary (without the actual MSD data arrays)
        msd_summary = msd_results_df.drop(columns=['msd_data'])
        output_msd_csv = SAVE_PATH / "msd_analysis_summary.csv"
        msd_summary.to_csv(output_msd_csv, index=False)
        print(f"\n✓ MSD analysis summary saved to: {output_msd_csv}")
        
        # Save individual MSD data as pickle for later analysis
        output_msd_pickle = SAVE_PATH / "msd_data_full.pkl"
        msd_results_df.to_pickle(output_msd_pickle)
        print(f"✓ Full MSD data (with arrays) saved to: {output_msd_pickle}")
        
        # Step 4: Create plots
        print("\n" + "=" * 70)
        print("Step 4: Creating MSD plots...")
        print("=" * 70)
        plot_msd_by_particle_size(msd_results_df, SAVE_PATH, points=DEFAULT_POINTS)
        
        # Step 5: Load saved MSD data and compare with theory
        print("\n" + "=" * 70)
        print(f"Step 5: Comparing measured D with theoretical and DLS values (using {DEFAULT_POINTS} points for fitting)...")
        print(f"Filter settings: min_exponent={MIN_EXPONENT:.2f}")
        print("=" * 70)
        compare_diffusion_coefficients(output_msd_pickle, SAVE_PATH, points=DEFAULT_POINTS, min_exponent=MIN_EXPONENT)
    
    print(f"\n✓ Analysis complete!\n")


if __name__ == "__main__":
    main()

