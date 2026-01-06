"""
Script to load a single track file and display both trajectories and individual MSD curves.
Similar to D0Script but focused on single-file analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import trackpy as tp
import pims
import xml.etree.ElementTree as ET
from scipy.optimize import curve_fit

def read_trackmate_xml(xml_file_path):
    """
    Liest eine XML-Datei (TrackMate Format) und konvertiert sie in ein Pandas DataFrame.
    Extrahiert nur frame, particle, x und y.
    """
    try:
        # 1. XML Datei parsen
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        data_rows = []
        
        # 2. Durch alle 'particle' Elemente iterieren
        # Wir nutzen enumerate, um eine ID für das Partikel zu erzeugen (0, 1, 2...)
        for particle_id, particle in enumerate(root.findall('particle')):
            
            # 3. Innerhalb jedes Partikels durch alle 'detection' Elemente iterieren
            for detection in particle.findall('detection'):
                # Attribute extrahieren
                t_raw = detection.get('t')
                x_raw = detection.get('x')
                y_raw = detection.get('y')
                
                # In Dictionary speichern und Typen umwandeln
                row = {
                    'frame': int(float(t_raw)),  # float->int, falls t als "40.0" gespeichert ist
                    'particle': particle_id + 1, # +1 damit Partikel bei 1 starten (optional)
                    'x': float(x_raw),
                    'y': float(y_raw)
                }
                data_rows.append(row)
        
        # 4. DataFrame erstellen
        df = pd.DataFrame(data_rows)
        
        # Sicherstellen, dass die Spalten in der gewünschten Reihenfolge sind
        if not df.empty:
            df = df[['frame', 'particle', 'x', 'y']]
            
            # Optional: Sortieren nach Frame und Partikel für bessere Lesbarkeit
            df = df.sort_values(by=['frame', 'particle']).reset_index(drop=True)
            
        return df

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None

def detect_calibration(tracks: pd.DataFrame) -> Tuple[float, float]:
    """
    Auto-detect calibration parameters based on track extent.
    
    If max x or y > 1000, assumes large field of view:
        mpp = 0.3 µm/px, fps = 60 Hz
    Otherwise assumes smaller field of view:
        mpp = 0.15 µm/px, fps = 20 Hz
    
    Parameters
    ----------
    tracks : pd.DataFrame
        Track data with columns ['x', 'y']
    
    Returns
    -------
    Tuple[float, float]
        (mpp, fps) calibration parameters
    """
    max_x = tracks['x'].max()
    max_y = tracks['y'].max()
    
    if max_x < 200 and max_y < 150:
        mpp = 0.3
        fps = 60.0
        print(f"Auto-detected: Large FOV (max x={max_x:.1f}, max y={max_y:.1f})")
    else:
        mpp = 0.15
        fps = 20.0
        print(f"Auto-detected: Small FOV (max x={max_x:.1f}, max y={max_y:.1f})")
    
    print(f"Using: mpp = {mpp} µm/px, fps = {fps} Hz")
    return mpp, fps


def load_tracks(file_path: Path) -> pd.DataFrame:
    """
    Load tracks from CSV or XML file.
    
    Parameters
    ----------
    file_path : Path
        Path to track file (CSV or XML format)
    
    Returns
    -------
    pd.DataFrame
        Tracks with columns ['particle', 'frame', 'x', 'y']
    """
    if file_path.suffix.lower() == '.xml':
        tracks = read_trackmate_xml(str(file_path))
    elif file_path.suffix.lower() == '.csv':
        tracks = pd.read_csv(file_path)
        # Ensure standard column names
        if 't' in tracks.columns and 'frame' not in tracks.columns:
            tracks.rename(columns={'t': 'frame'}, inplace=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return tracks


def fit_diffusion_coefficient(msd_df: pd.DataFrame, fps: float) -> Tuple[float, float, float]:
    """
    Fit MSD using trackpy's power law fitting.
    For purely diffusive motion: MSD = 4*D*tau^alpha, where alpha ~ 1
    
    Parameters
    ----------
    msd_df : pd.DataFrame
        MSD DataFrame with 'msd' column
    fps : float
        Frames per second
    
    Returns
    -------
    Tuple[float, float, float]
        (D, alpha, D_error) Diffusion coefficient in µm²/s, power law exponent, and uncertainty
    """
    try:
        from trackpy import utils
        
        # Use first 25% of data points for fitting
        n_fit = max(3, len(msd_df) // 4)
        msd_fit = msd_df.iloc[:n_fit]
        
        # Get lag times in seconds
        lag_times_s = msd_fit.index.values / fps
        msd_values = msd_fit['msd'].values
        
        # Fit power law: MSD = A * tau^alpha
        # For diffusion: A = 4*D, alpha ~ 1
        log_lag = np.log(lag_times_s)
        log_msd = np.log(msd_values)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_lag, log_msd, 1)
        alpha = coeffs[0]  # Power law exponent
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # Extract D from A = 4*D
        D = A / 4.0
        
        # Estimate error from residuals
        log_msd_fit = np.polyval(coeffs, log_lag)
        residuals = log_msd - log_msd_fit
        rmse = np.sqrt(np.mean(residuals**2))
        D_error = D * rmse  # Rough error estimate
        
        return D, alpha, D_error
    except Exception as e:
        return np.nan, np.nan, np.nan


def compute_individual_msd(tracks: pd.DataFrame, mpp: float, fps: float, 
                          max_lagtime: Optional[int] = None,
                          min_track_length: int = 5) -> Tuple[dict, dict]:
    """
    Compute MSD for each individual particle and fit diffusion coefficients.
    
    Parameters
    ----------
    tracks : pd.DataFrame
        Track data with columns ['particle', 'frame', 'x', 'y']
    mpp : float
        Microns per pixel
    fps : float
        Frames per second
    max_lagtime : Optional[int]
        Maximum lag time in frames
    min_track_length : int
        Minimum number of frames required for a track
    
    Returns
    -------
    Tuple[dict, dict]
        (imsd_dict, diff_coeff_dict) - MSD DataFrames and diffusion coefficients (D, alpha, D_err)
    """
    imsd_dict = {}
    diff_coeff_dict = {}
    
    for particle_id in tracks['particle'].unique():
        particle_tracks = tracks[tracks['particle'] == particle_id].copy()
        
        if len(particle_tracks) < min_track_length:  # Skip short tracks
            continue
        
        try:
            # Compute MSD for this single particle trajectory using tp.msd
            if max_lagtime is not None:
                msd_result = tp.msd(particle_tracks, mpp=mpp, fps=fps, max_lagtime=max_lagtime)
            else:
                msd_result = tp.msd(particle_tracks, mpp=mpp, fps=fps)
            
            imsd_dict[particle_id] = msd_result
            
            # Fit diffusion coefficient
            D, alpha, D_err = fit_diffusion_coefficient(msd_result, fps)
            diff_coeff_dict[particle_id] = (D, alpha, D_err)
        except Exception as e:
            print(f"Warning: Failed to compute MSD for particle {particle_id}: {e}")
            continue
    
    return imsd_dict, diff_coeff_dict


# Define reference diffusion coefficients for different particle sizes
DSL_MEASUREMENTS = {
    20: 12.38750325,
    50: 8.201969711,
    100: 4.139082033,
    200: 1.745323167,
    500: 0.621773811,
    1000: 0.356862091
}

def plot_reference_lines(ax, fps: float, alpha: float = 1.0):
    """
    Plot reference MSD lines for different particle sizes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    fps : float
        Frames per second
    alpha : float
        Power law exponent (default: 1.0 for pure diffusion)
    """
    lag_times = np.logspace(-2, 2, 100)  # 0.01 to 100 seconds
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(DSL_MEASUREMENTS)))
    
    for (size, D), color in zip(DSL_MEASUREMENTS.items(), colors):
        # MSD = 4*D*tau^alpha
        msd_ref = 4 * D * (lag_times ** alpha)
        ax.loglog(lag_times, msd_ref, '--', linewidth=1.5, alpha=0.7, 
                 color=color, label=f'{size} nm (D={D:.2f})')

def plot_tracks_and_imsd(tracks: pd.DataFrame, imsd_dict: dict, diff_coeff_dict: dict,
                        mpp: float, fps: float, save_path: Optional[Path] = None):
    """
    Create figure with trajectory plot and individual MSD curves with diffusion coefficients.
    
    Parameters
    ----------
    tracks : pd.DataFrame
        Track data
    imsd_dict : dict
        Dictionary of individual MSD DataFrames
    diff_coeff_dict : dict
        Dictionary of diffusion coefficients (D, D_err) for each particle
    mpp : float
        Microns per pixel
    fps : float
        Frames per second
    save_path : Optional[Path]
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Trajectories
    for particle_id in tracks['particle'].unique():
        particle_data = tracks[tracks['particle'] == particle_id]
        ax1.plot(particle_data['x'] * mpp, particle_data['y'] * mpp, 
                alpha=0.7, linewidth=1, label=f'Particle {particle_id}')
    
    ax1.set_xlabel('X Position (µm)')
    ax1.set_ylabel('Y Position (µm)')
    ax1.set_title(f'Particle Trajectories (N={len(tracks["particle"].unique())})')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual MSD with diffusion coefficients
    for particle_id, imsd in imsd_dict.items():
        lag_times = imsd.index.values / fps  # Convert to seconds
        msd_values = imsd['msd'].values
        
        # Get diffusion coefficient
        D, alpha, D_err = diff_coeff_dict.get(particle_id, (np.nan, np.nan, np.nan))
        print(f'P{particle_id}: D={D:.3f}±{D_err:.3f} µm²/s, α={alpha:.2f}')
        # if not np.isnan(D):
        #     label = f'P{particle_id}: D={D:.3f}±{D_err:.3f} µm²/s, α={alpha:.2f}'
        # else:
        #     label = f'P{particle_id}: D=N/A'
        
        ax2.loglog(lag_times, msd_values, alpha=0.6, linewidth=1.5)#, label=label)
    
    ax2.set_xlabel('Lag Time τ (s)')
    ax2.set_ylabel('MSD (µm²)')
    ax2.set_title('Individual MSD Curves with Diffusion Coefficients')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add reference lines for different particle sizes
    plot_reference_lines(ax2, fps)
    
    ax2.legend(fontsize=7, ncol=1, loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main(track_file: str, mpp: Optional[float] = None, fps: Optional[float] = None, 
         max_lagtime: Optional[int] = None, save_fig: bool = False,
         min_track_length: int = 5, alpha_min: float = 0.0, alpha_max: float = 2.0,
         subtract_drift: bool = True):
    """
    Main analysis function with auto-detection of calibration parameters.
    
    Parameters
    ----------
    track_file : str
        Path to track file (CSV or XML)
    mpp : Optional[float]
        Microns per pixel (if None, auto-detect based on track extent)
    fps : Optional[float]
        Frames per second (if None, auto-detect based on track extent)
    max_lagtime : Optional[int]
        Maximum lag time in frames
    save_fig : bool
        Whether to save the figure
    min_track_length : int
        Minimum number of frames required for a track (default: 5)
    alpha_min : float
        Minimum alpha exponent for filtering (default: 0.0, no filtering)
    alpha_max : float
        Maximum alpha exponent for filtering (default: 2.0, no filtering)
    subtract_drift : bool
        Whether to subtract drift from trajectories (default: True)
    """
    file_path = Path(track_file)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Track file not found: {file_path}")
    
    print(f"Loading tracks from {file_path.name}...")
    tracks = load_tracks(file_path)
    
    print(f"Found {len(tracks['particle'].unique())} particles")
    print(f"Total {len(tracks)} positions")
    
    # Subtract drift if requested
    if subtract_drift:
        print("Subtracting drift from trajectories...")
        drift = tp.compute_drift(tracks)
        tracks = tp.subtract_drift(tracks, drift)
        print(f"  Drift correction applied (max drift: {drift[['x', 'y']].abs().max().max():.2f} pixels)")
    
    # Auto-detect calibration if not provided
    if mpp is None or fps is None:
        mpp_auto, fps_auto = detect_calibration(tracks)
        if mpp is None:
            mpp = mpp_auto
        if fps is None:
            fps = fps_auto
    else:
        print(f"Using provided calibration: mpp = {mpp} µm/px, fps = {fps} Hz")
    
    # Compute individual MSD and diffusion coefficients
    print("Computing individual MSD curves and fitting diffusion coefficients...")
    imsd_dict, diff_coeff_dict = compute_individual_msd(tracks, mpp, fps, max_lagtime, min_track_length)
    print(f"Computed MSD for {len(imsd_dict)} particles")
    
    # Filter by alpha if specified
    if alpha_min > 0.0 or alpha_max < 2.0:
        filtered_ids = []
        for particle_id, (D, alpha, D_err) in diff_coeff_dict.items():
            if not np.isnan(alpha) and alpha_min <= alpha <= alpha_max:
                filtered_ids.append(particle_id)
        
        # Filter dictionaries
        imsd_dict = {pid: imsd_dict[pid] for pid in filtered_ids if pid in imsd_dict}
        diff_coeff_dict = {pid: diff_coeff_dict[pid] for pid in filtered_ids if pid in diff_coeff_dict}
        
        # Filter tracks
        tracks = tracks[tracks['particle'].isin(filtered_ids)].copy()
        
        print(f"After alpha filtering [{alpha_min:.2f}, {alpha_max:.2f}]: {len(imsd_dict)} particles remain")
    
    # Print diffusion coefficients
    print("\nDiffusion Coefficients:")
    for particle_id, (D, alpha, D_err) in diff_coeff_dict.items():
        if not np.isnan(D):
            print(f"  Particle {particle_id}: D = {D:.4f} ± {D_err:.4f} µm²/s, α = {alpha:.3f}")
        else:
            print(f"  Particle {particle_id}: D = N/A (fit failed)")
    
    # Plot
    save_path = file_path.with_suffix('.png') if save_fig else None
    plot_tracks_and_imsd(tracks, imsd_dict, diff_coeff_dict, mpp, fps, save_path)
    
    return tracks, imsd_dict, diff_coeff_dict
if __name__ == '__main__':
    # Example usage
    track_file = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm\Tracks\20nm_processed_Tracks.xml"  # Or .xml file
    track_file = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm\Tracks\20nm_60fps_2_processed_Tracks.xml"
    # Auto-detect calibration or manually specify
    # Filter tracks before analysis
    min_track_length = 10  # Minimum number of frames
    alpha_min = 0.8
    alpha_max = 50
    
    tracks, imsd_dict, diff_coeff_dict = main(
        track_file=track_file,
        mpp=None,  # Auto-detect (or set to 0.3 or 0.15)
        fps=None,  # Auto-detect (or set to 60.0 or 20.0)
        max_lagtime=None,
        save_fig=False,
        min_track_length=min_track_length,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        subtract_drift=False  # Enable drift correction
    )