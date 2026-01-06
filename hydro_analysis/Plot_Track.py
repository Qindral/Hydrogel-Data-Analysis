
"""Script to plot particles with automatic frame selection and contrast optimization.
Creates clean visualization showing particles on dark background.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
try:
    from matplotlib_scalebar.scalebar import ScaleBar
    SCALEBAR_AVAILABLE = True
except ImportError:
    SCALEBAR_AVAILABLE = False
    print("Warning: matplotlib-scalebar not available. Scalebar will be disabled.")
import imageio.v3 as iio
import xml.etree.ElementTree as ET


def read_trackmate_xml(xml_file_path: str | Path) -> pd.DataFrame:
    """
    Read TrackMate XML file and convert to pandas DataFrame.
    
    Parameters
    ----------
    xml_file_path : str | Path
        Path to TrackMate XML file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['frame', 'particle', 'x', 'y']
    """
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
        print(f"Error reading TrackMate XML: {e}")
        return None


def load_tracks(file_path: str | Path) -> pd.DataFrame:
    """
    Load tracks from CSV or XML file.
    
    Parameters
    ----------
    file_path : str | Path
        Path to track file (CSV or XML format)
    
    Returns
    -------
    pd.DataFrame
        Tracks with columns ['particle', 'frame', 'x', 'y']
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.xml':
        tracks = read_trackmate_xml(file_path)
    elif file_path.suffix.lower() == '.csv':
        tracks = pd.read_csv(file_path)
        # Ensure standard column names
        if 't' in tracks.columns and 'frame' not in tracks.columns:
            tracks.rename(columns={'t': 'frame'}, inplace=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .csv or .xml")
    
    if tracks is None or tracks.empty:
        raise ValueError(f"Failed to load tracks from {file_path}")
    
    return tracks


def plot_particles_only(
    tif_path: str | Path,
    tracks_csv: str | Path,
    output_path: str | Path = None,
    dpi: int = 300,
    mpp: float = 0.065,
    particle_size: float = None,
    contrast_percentile: tuple = (5, 99.5),
    marker_size: float = 50
):
    """
    Create clean particle visualization with automatic frame selection.
    
    - Automatically selects frame where most particles are visible
    - Dark background with bright particles for optimal contrast
    - No tracks, only particle positions shown
    - Minimal design: title and scalebar only
    
    Parameters
    ----------
    tif_path : str | Path
        Path to TIFF file
    tracks_csv : str | Path
        Path to track file (CSV or XML)
    output_path : str | Path, optional
        Save path for figure
    dpi : int
        Output resolution (default: 300)
    mpp : float
        Microns per pixel (default: 0.065)
    particle_size : float, optional
        Particle size in nm for title (default: None)
    contrast_percentile : tuple
        Percentile range for contrast (default: (5, 99.5))
    marker_size : float
        Size of particle markers (default: 50)
    """
    # Load TIFF data
    img_data = iio.imread(tif_path)
    
    # Load tracks
    tracks = load_tracks(tracks_csv)
    
    # Find frame where most particles are present
    frame_counts = tracks.groupby('frame').size()
    best_frame = frame_counts.idxmax()
    
    print(f"Selected frame {best_frame} with {frame_counts[best_frame]} particles")
    
    # Extract frame
    if img_data.ndim == 4:
        img = img_data[0, best_frame]
    elif img_data.ndim == 3:
        img = img_data[best_frame]
    else:
        img = img_data
    
    # Adjust contrast: make background dark, particles bright
    vmin, vmax = np.percentile(img, contrast_percentile)
    img_display = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    
    # Enhance contrast further by power-law transformation
    img_display = np.power(img_display, 0.5)  # Gamma correction for darker background
    
    # Get particles visible in this frame
    particles_in_frame = tracks[tracks['frame'] == best_frame]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi, facecolor='black')
    ax.set_facecolor('black')
    
    # Display image with dark background
    ax.imshow(img_display, cmap='gray', origin='lower', vmin=0, vmax=0.6)
    
    # Plot particles as bright markers
    ax.scatter(
        particles_in_frame['x'],
        particles_in_frame['y'],
        s=marker_size,
        c='cyan',
        marker='o',
        alpha=0.8,
        edgecolors='white',
        linewidths=1.5
    )
    
    # Add scalebar
    if SCALEBAR_AVAILABLE:
        scalebar = ScaleBar(
            mpp, 'um',
            length_fraction=0.2,
            location='lower right',
            box_alpha=0.8,
            color='white',
            frameon=True,
            font_properties={'size': 12, 'weight': 'bold'}
        )
        ax.add_artist(scalebar)
    
    # Add title
    if particle_size:
        title = f'Particles of {particle_size:.0f} nm in water'
    else:
        title = 'Particle tracking in water'
    
    ax.text(0.5, 0.98, title, transform=ax.transAxes, 
           fontsize=16, fontweight='bold', color='white',
           ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, 
                    edgecolor='white', linewidth=1.5))
    
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        print(f"✓ Particle visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()




if __name__ == "__main__":
    # Example usage - particle visualization without tracks
    tif_path = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\1000 nm_4.tif"
    tracks_csv = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\Tracks\1000 nm_4_Tracks.xml"
    
    print("\n" + "="*60)
    print("Particle Visualization with Auto Frame Selection")
    print("="*60 + "\n")
    
    # Create visualization showing only particles
    plot_particles_only(
        tif_path=tif_path,
        tracks_csv=tracks_csv,
        output_path=None,  # Set to filename to save
        dpi=300,
        mpp=0.3,  # Adjust based on your calibration
        particle_size=1000,  # nm
        contrast_percentile=(5, 99.5),
        marker_size=50
    )
    
    print("\n" + "="*60)
    print("✓ Visualization complete!")
    print("="*60)
