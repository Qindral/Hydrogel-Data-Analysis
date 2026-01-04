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
                
    except Exception as e:
        print(f"Warning: Could not parse {rec_path.name}: {e}")
    
    return result


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
    Scan directory structure and collect all .tif, .rec, and .xml files 
    organized by particle size.
    
    Expected structure:
        root_path/
            50nm/
                *.tif
                *.rec
                Tracks/
                    *.xml
            100nm/
                *.tif
                *.rec
                Tracks/
                    *.xml
    
    Args:
        root_path: Root directory to scan
        
    Returns:
        DataFrame with columns:
        - particle_size_nm: Particle size in nanometers
        - tif_path: Path to TIFF file (or None)
        - rec_path: Path to REC file (or None)
        - xml_path: Path to XML track file (or None)
        - tif_name: Basename of TIFF file
        - rec_name: Basename of REC file
        - xml_name: Basename of XML file
        - exposure_ms: Exposure time from .rec file
        - delay_ms: Delay time from .rec file
        - fps: Calculated frames per second
        - fps_category: Classification ('~20 FPS', '~60 FPS', 'Other', 'No Data')
    """
    file_records = []
    
    # Walk through subdirectories
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Extract particle size from folder name
        particle_size = extract_particle_size_from_path(subfolder)
        if particle_size is None:
            continue
        
        # Find all TIFF files in main folder
        tif_files = sorted(list(subfolder.glob("*.tif")) + list(subfolder.glob("*.tiff")))
        
        # Find all REC files in main folder
        rec_files = sorted(list(subfolder.glob("*.rec")))
        
        # Find all XML files in Tracks subfolder
        xml_files = []
        tracks_folder = subfolder / "Tracks"
        if tracks_folder.exists() and tracks_folder.is_dir():
            xml_files = sorted(list(tracks_folder.glob("*.xml")))
        
        # Create records for each combination
        # Match files by base name (without extension)
        tif_dict = {f.stem: f for f in tif_files}
        rec_dict = {f.stem: f for f in rec_files}
        xml_dict = {f.stem: f for f in xml_files}
        
        # Get all unique base names
        all_basenames = set(tif_dict.keys()) | set(rec_dict.keys()) | set(xml_dict.keys())
        
        for basename in sorted(all_basenames):
            tif_path = tif_dict.get(basename)
            rec_path = rec_dict.get(basename)
            xml_path = xml_dict.get(basename)
            
            # Parse .rec file if available
            rec_info = {'exposure_ms': None, 'delay_ms': None, 'fps': None}
            if rec_path:
                rec_info = parse_rec_file(rec_path)
            
            file_records.append({
                'particle_size_nm': particle_size,
                'tif_path': str(tif_path) if tif_path else None,
                'rec_path': str(rec_path) if rec_path else None,
                'xml_path': str(xml_path) if xml_path else None,
                'tif_name': tif_path.name if tif_path else None,
                'rec_name': rec_path.name if rec_path else None,
                'xml_name': xml_path.name if xml_path else None,
                'exposure_ms': rec_info['exposure_ms'],
                'delay_ms': rec_info['delay_ms'],
                'fps': rec_info['fps'],
            })
    
    df = pd.DataFrame(file_records)
    
    # Add fps_category column
    if not df.empty and 'fps' in df.columns:
        df['fps_category'] = df['fps'].apply(lambda x: 
            '~20 FPS' if pd.notna(x) and 18 <= x <= 22 else
            '~60 FPS' if pd.notna(x) and 55 <= x <= 65 else
            'Other' if pd.notna(x) else
            'No Data'
        )
    
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
                       points: int = 10) -> tuple[Dict[int, float], Dict[int, float]]:
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
        tp.quiet()  # Suppress trackpy warnings
        
        try:
            # Attempt linking (may be unnecessary if tracks already labeled)
            tp.link(df_combined, 12, memory=8)
        except Exception:
            pass

        # Individual and ensemble MSDs
        im = tp.imsd(df_combined, mpp, fps)  # Individual particle MSDs
        em = tp.emsd(df_combined, mpp, fps)  # Ensemble average MSD

        # Fit power-law to ensemble MSD
        params = fit_powerlaw_with_errors(em, points=points, plot=False)
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
        plt.savefig(save_path / f'combined_MSD_{size_nm}nm.png', dpi=300)
        plt.close(fig)

    # Create comparison plot: measured vs theoretical diffusion coefficients
    sizes = sorted(combined_D.keys())
    measured_mean = [combined_D[s] for s in sizes]
    measured_err = [combined_D_err[s] for s in sizes]
    
    # Calculate theoretical D using Stokes-Einstein equation
    # D = kB*T / (6*pi*eta*R) where R is particle radius
    theory_aligned = []
    for s in sizes:
        if s in [20, 50, 200, 500, 1000]:  # Canonical sizes
            d_m = s * 1e-9  # Convert nm to m
            R = d_m / 2.0  # Radius
            D_m2_s = BOLTZMANN_CONSTANT * TEMPERATURE / (6 * np.pi * WATER_VISCOSITY * R)
            D_um2_s = D_m2_s * 1e12  # Convert m²/s to µm²/s
            theory_aligned.append(D_um2_s)
        else:
            theory_aligned.append(np.nan)

    # Create comparison plot
    if sizes:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot measured values with error bars
        ax.errorbar(sizes, measured_mean, yerr=measured_err, fmt='o', 
                   color='tab:blue', capsize=5, capthick=2,
                   label='Measured D (combined)')
        
        # Plot theoretical predictions
        ax.plot(sizes, theory_aligned, 'x', markersize=10, markeredgewidth=2,
               color='black', label='Theoretical D (Stokes-Einstein)')
        
        # Format plot
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Particle size [nm]', fontsize=12)
        ax.set_ylabel('Diffusion coefficient D [µm²/s]', fontsize=12)
        ax.set_title('Measured vs Theoretical Diffusion Coefficients', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'combined_D_measured_vs_theoretical.png', dpi=300)
        plt.close(fig)

    return combined_D, combined_D_err


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for file collection and particle counting analysis.
    
    This function:
    1. Scans the root directory for particle size folders
    2. Collects all .tif, .rec, and .xml files
    3. Creates a comprehensive file mapping
    4. Counts particles per particle size from XML files
    5. Generates summary reports and CSV exports
    """
    print("=" * 70)
    print("TrackMate File Scanner & Particle Counter")
    print("=" * 70)
    print(f"\nScanning directory: {ROOT_PATH}\n")
    
    # Step 1: Collect all files and their associations
    print("Step 1: Collecting all .tif, .rec, and .xml files...")
    print("-" * 70)
    
    files_df = collect_all_files_by_particle_size(ROOT_PATH)
    
    if files_df.empty:
        print("✗ No files found in particle size subdirectories!")
        print("\nExpected directory structure:")
        print("  root/")
        print("    └── 50nm/")
        print("        ├── file1.tif")
        print("        ├── file1.rec")
        print("        └── Tracks/")
        print("            └── file1.xml")
        return
    
    print(f"✓ Found {len(files_df)} file records across {files_df['particle_size_nm'].nunique()} particle sizes\n")
    
    # Display file associations
    print("=" * 70)
    print("FILE ASSOCIATIONS")
    print("=" * 70)
    
    # Group by particle size for display
    for particle_size in sorted(files_df['particle_size_nm'].unique()):
        size_files = files_df[files_df['particle_size_nm'] == particle_size]
        print(f"\n{particle_size} nm ({len(size_files)} records):")
        print("-" * 70)
        
        for idx, row in size_files.iterrows():
            print(f"  Record {idx + 1}:")
            if row['tif_name']:
                print(f"    TIF:  {row['tif_name']}")
            if row['rec_name']:
                fps_info = ""
                if pd.notna(row['fps']):
                    fps_info = f" [{row['fps_category']}: {row['fps']:.2f} FPS, Exp: {row['exposure_ms']:.2f} ms]"
                print(f"    REC:  {row['rec_name']}{fps_info}")
            if row['xml_name']:
                print(f"    XML:  {row['xml_name']}")
            if not (row['tif_name'] or row['rec_name'] or row['xml_name']):
                print(f"    (empty record)")
    
    # Save complete file listing
    output_files_csv = SAVE_PATH / "file_associations.csv"
    files_df.to_csv(output_files_csv, index=False)
    print(f"\n✓ File associations saved to: {output_files_csv}")
    
    # Display FPS statistics
    rec_with_fps = files_df[files_df['fps'].notna()]
    if not rec_with_fps.empty:
        print("\n" + "=" * 70)
        print("FPS ANALYSIS")
        print("=" * 70)
        fps_values = rec_with_fps['fps'].values
        print(f"\nTotal .rec files with FPS data: {len(rec_with_fps)}")
        print(f"FPS range: {fps_values.min():.2f} - {fps_values.max():.2f}")
        print(f"Mean FPS: {fps_values.mean():.2f}")
        
        # Group by approximate FPS (20 or 60)
        fps_20 = rec_with_fps[rec_with_fps['fps'].between(18, 22)]
        fps_60 = rec_with_fps[rec_with_fps['fps'].between(55, 65)]
        
        print(f"\nFiles with ~20 FPS: {len(fps_20)} ({len(fps_20)/len(rec_with_fps)*100:.1f}%)")
        print(f"Files with ~60 FPS: {len(fps_60)} ({len(fps_60)/len(rec_with_fps)*100:.1f}%)")
        
        if len(fps_20) > 0:
            print(f"  → Exposure times: {fps_20['exposure_ms'].min():.2f} - {fps_20['exposure_ms'].max():.2f} ms")
        if len(fps_60) > 0:
            print(f"  → Exposure times: {fps_60['exposure_ms'].min():.2f} - {fps_60['exposure_ms'].max():.2f} ms")
        
        # Detailed listing by FPS category
        print("\n" + "-" * 70)
        print("DETAILED FPS BREAKDOWN BY FILE")
        print("-" * 70)
        
        if len(fps_20) > 0:
            print(f"\n~20 FPS Files ({len(fps_20)}):")
            for idx, row in fps_20.iterrows():
                print(f"  • {row['particle_size_nm']:.0f} nm: {row['rec_name']} "
                      f"(Exp: {row['exposure_ms']:.2f} ms, FPS: {row['fps']:.2f})")
        
        if len(fps_60) > 0:
            print(f"\n~60 FPS Files ({len(fps_60)}):")
            for idx, row in fps_60.iterrows():
                print(f"  • {row['particle_size_nm']:.0f} nm: {row['rec_name']} "
                      f"(Exp: {row['exposure_ms']:.2f} ms, FPS: {row['fps']:.2f})")
        
        # Other FPS values (not 20 or 60)
        other_fps = rec_with_fps[~rec_with_fps.index.isin(fps_20.index) & 
                                 ~rec_with_fps.index.isin(fps_60.index)]
        if len(other_fps) > 0:
            print(f"\nOther FPS Files ({len(other_fps)}):")
            for idx, row in other_fps.iterrows():
                print(f"  • {row['particle_size_nm']:.0f} nm: {row['rec_name']} "
                      f"(Exp: {row['exposure_ms']:.2f} ms, FPS: {row['fps']:.2f})")
    
    # Step 2: Count particles from XML files
    print("\n" + "=" * 70)
    print("Step 2: Counting particles from XML track files...")
    print("-" * 70 + "\n")
    
    # Filter only records with XML files
    xml_records = files_df[files_df['xml_path'].notna()]
    
    if xml_records.empty:
        print("✗ No XML track files found!")
        return
    
    # Count particles per size
    particle_counts = []
    for particle_size in sorted(xml_records['particle_size_nm'].unique()):
        size_xmls = xml_records[xml_records['particle_size_nm'] == particle_size]
        total_particles = 0
        
        print(f"{particle_size} nm:")
        for _, row in size_xmls.iterrows():
            xml_path = Path(row['xml_path'])
            df = read_trackmate_xml(xml_path)
            if df is not None and not df.empty:
                num_particles = df['particle'].nunique()
                total_particles += num_particles
                print(f"  {xml_path.name}: {num_particles} particles")
        
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
    print(f"✓ Analysis complete!\n")


if __name__ == "__main__":
    main()

