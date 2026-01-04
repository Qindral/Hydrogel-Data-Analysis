import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import tifffile
from scipy.ndimage import gaussian_filter
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

''' Here the MSD will be calculated using the tiff files only. A general Trackpy Location and Linking will be performeds as well as automatic preprocessing depending on a) Paticle size, b) FPS and c) MPP.
The code is adapted from pytrackmate_MSD_XML.py'''

paths = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung"

save_path = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\trackpy_Linking"

import xml.etree.ElementTree as ET
def load_tiff_stack(tif_path: Path) -> np.ndarray:
    """
    Load TIFF file as numpy array.
    
    Args:
        tif_path: Path to TIFF file
        
    Returns:
        Image stack as numpy array (T, Y, X) or (Y, X)
    """
    with tifffile.TiffFile(tif_path) as tif:
        image_stack = tif.asarray()
    
    # Ensure at least 3D (add time axis if single frame)
    if image_stack.ndim == 2:
        image_stack = image_stack[np.newaxis, ...]
    # If 4D, assume CTYX or CZYX and take first channel/Z
    elif image_stack.ndim == 4:
        image_stack = image_stack[0, ...]
    
    return image_stack
def extract_rec_metadata(rec_path: Path) -> Dict[str, float]:
    """
    Extract metadata from .rec file (PCO camera format).
    
    Parses pco.camware Recorder Comment Files for:
    - Frame rate (calculated from exposure time)
    - Pixel size (from ROI dimensions)
    - ROI dimensions
    
    Resolution mapping:
    - 696×520 ROI → 0.15 µm/px
    - 200×150 ROI → 0.3 µm/px
    
    Returns:
        Dictionary with 'fps', 'mpp', 'exposure_ms', 'binning', 'roi'
    """
    metadata = {}
    
    try:
        with open(rec_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Extract exposure time (ms)
            exposure_match = re.search(r'Exposure.*?:\s*(\d+\.?\d*)\s*ms', content, re.IGNORECASE)
            if exposure_match:
                exposure_ms = float(exposure_match.group(1))
                metadata['exposure_ms'] = exposure_ms
                # Calculate FPS from exposure (assuming no dead time)
                metadata['fps'] = 1000.0 / exposure_ms if exposure_ms > 0 else None
            
            # Extract binning (typically "x4/x4" format)
            binning_match = re.search(r'Binning.*?:\s*x(\d+)/x(\d+)', content, re.IGNORECASE)
            if binning_match:
                binning_x = int(binning_match.group(1))
                binning_y = int(binning_match.group(2))
                metadata['binning'] = (binning_x, binning_y)
            
            # Extract ROI dimensions
            roi_match = re.search(r'ROI.*?:\s*(\d+)-(\d+)/(\d+)-(\d+)', content, re.IGNORECASE)
            if roi_match:
                roi_x1, roi_x2 = int(roi_match.group(1)), int(roi_match.group(2))
                roi_y1, roi_y2 = int(roi_match.group(3)), int(roi_match.group(4))
                roi_width = roi_x2 - roi_x1 + 1
                roi_height = roi_y2 - roi_y1 + 1
                metadata['roi'] = (roi_x1, roi_x2, roi_y1, roi_y2)
                
                # Determine pixel size from ROI dimensions
                if roi_width == 696 and roi_height == 520:
                    metadata['mpp'] = 0.15
                elif roi_width == 200 and roi_height == 150:
                    metadata['mpp'] = 0.3
                else:
                    # Fallback to binning-based calculation if ROI doesn't match known sizes
                    if 'binning' in metadata:
                        metadata['mpp'] = 6.45 * metadata['binning'][0] / 1000
    
    except Exception as e:
        print(f"Warning: Could not parse {rec_path.name}: {e}")
    
    return metadata


def extract_particle_size_from_path(folder_path: Path) -> Optional[float]:
    """
    Extract particle size (in nm) from folder name.
    
    Expects folder names like "50nm", "100_nm", "200 nm", etc.
    """
    folder_name = folder_path.name
    match = re.search(r'(\d+)\s*nm', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def collect_datasets(root_path: Path) -> List[Dict]:
    """
    Scan root directory for TIFF files organized by particle size folders.
    
    Returns:
        List of dicts with keys: 'tif_path', 'rec_path', 'particle_size_nm', 'fps', 'mpp'
    """
    datasets = []
    
    # Walk through subdirectories
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Extract particle size from folder name
        particle_size = extract_particle_size_from_path(subfolder)
        
        # Find all .tif files in this folder
        tif_files = list(subfolder.glob("*.tif")) + list(subfolder.glob("*.tiff"))
        
        for tif_path in tif_files:
            # Look for corresponding .rec file
            rec_path = tif_path.with_suffix('.rec')
            
            dataset_info = {
                'tif_path': tif_path,
                'rec_path': rec_path if rec_path.exists() else None,
                'particle_size_nm': particle_size,
                'fps': None,
                'mpp': None
            }
            
            # Extract metadata from .rec file if available
            if rec_path and rec_path.exists():
                rec_metadata = extract_rec_metadata(rec_path)
                dataset_info.update(rec_metadata)
            
            datasets.append(dataset_info)
    
    return datasets


import matplotlib.pyplot as plt

# Main execution
root_path = Path(paths)
save_path = Path(save_path)
save_path.mkdir(parents=True, exist_ok=True)

# Collect all datasets
print("Scanning for TIFF files and metadata...")
datasets = collect_datasets(root_path)

print(f"\nFound {len(datasets)} datasets:")
for i, ds in enumerate(datasets, 1):
    print(f"\n{i}. {ds['tif_path'].name}")
    print(f"   Particle size: {ds['particle_size_nm']} nm")
    print(f"   FPS: {ds['fps']}")
    print(f"   MPP: {ds['mpp']} µm/px")
    print(f"   Has .rec file: {ds['rec_path'] is not None}")

# Store dataset info for next steps
datasets_df = pd.DataFrame(datasets)
datasets_df.to_csv(save_path / "datasets_inventory.csv", index=False)
print(f"\nDataset inventory saved to {save_path / 'datasets_inventory.csv'}")

# Group datasets by particle size
datasets_by_size = datasets_df.groupby('particle_size_nm')

# Interactive preprocessing and visualization
def fast_rolling_ball(image, radius):
    if radius < 1: return image
    k_size = int(radius * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(image, background)


def preprocess_stack(image_stack: np.ndarray, smooth: bool = True, radius: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply preprocessing: average subtraction + rolling ball background subtraction.
    
    Args:
        image_stack: 3D array (T, Y, X)
        smooth: Apply Gaussian smoothing before background subtraction
        radius: Rolling ball radius for background subtraction
        
    Returns:
        (preprocessed_stack, average_image)
    """
    # Compute temporal average
    avg_image = np.mean(image_stack, axis=0)
    
    # Subtract average from each frame
    avg_subtracted = image_stack - avg_image
    
    # Rolling ball background subtraction on each frame
    preprocessed = np.zeros_like(avg_subtracted)
    
    for i, frame in enumerate(avg_subtracted):
        if smooth:
            frame = gaussian_filter(frame, sigma=1.0)
        
        # Rolling ball: dilate with circular kernel
        preprocessed[i] = fast_rolling_ball(frame, radius)
    
    return preprocessed, avg_image


def show_preprocessing_comparison(image_stack: np.ndarray, preprocessed: np.ndarray, 
                                   avg_image: np.ndarray, title: str, mpp: Optional[float] = None):
    """
    Display side-by-side comparison using tkinter + matplotlib.
    
    Args:
        image_stack: Raw stack (T, Y, X)
        preprocessed: Preprocessed stack (T, Y, X)
        avg_image: Average image (Y, X)
        title: Window title
        mpp: Microns per pixel for scale bar
    """
    root = tk.Tk()
    root.title(title)
    
    # Frame slider and controls
    control_frame = ttk.Frame(root)
    control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    # Frame index tracker
    current_frame = tk.IntVar(value=0)
    
    # Create matplotlib figure with 3 subplots
    fig = Figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    # Initial images
    im1 = ax1.imshow(image_stack[0], cmap='gray', vmin=image_stack.min(), vmax=image_stack.max())
    ax1.set_title("Raw")
    ax1.axis('off')
    fig.colorbar(im1, ax=ax1, fraction=0.046)
    
    im2 = ax2.imshow(avg_image, cmap='gray')
    ax2.set_title("Average (subtracted)")
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    
    im3 = ax3.imshow(preprocessed[0], cmap='viridis', vmin=preprocessed.min(), vmax=preprocessed.max())
    ax3.set_title("Preprocessed")
    ax3.axis('off')
    fig.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Embed matplotlib in tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Update function for slider
    def update_frame(val):
        idx = int(val)
        im1.set_data(image_stack[idx])
        im3.set_data(preprocessed[idx])
        frame_label.config(text=f"Frame: {idx}/{len(image_stack)-1}")
        canvas.draw()
    
    # Frame slider
    ttk.Label(control_frame, text="Frame:").pack(side=tk.LEFT)
    slider = ttk.Scale(control_frame, from_=0, to=len(image_stack)-1, 
                      variable=current_frame, orient=tk.HORIZONTAL, 
                      command=update_frame, length=400)
    slider.pack(side=tk.LEFT, padx=10)
    
    frame_label = ttk.Label(control_frame, text=f"Frame: 0/{len(image_stack)-1}")
    frame_label.pack(side=tk.LEFT)
    
    # Metadata display
    if mpp:
        info_text = f"Scale: {mpp:.3f} µm/px"
        ttk.Label(control_frame, text=info_text).pack(side=tk.RIGHT, padx=10)
    
    # Close button
    ttk.Button(control_frame, text="Close", command=root.destroy).pack(side=tk.RIGHT)
    
    root.mainloop()


# Select one representative file per particle size
print("\n" + "="*60)
print("PREPROCESSING VISUALIZATION")
print("="*60)

for particle_size, group in datasets_by_size:
    if pd.isna(particle_size):
        print(f"\nSkipping datasets without particle size label")
        continue
    
    # Select first dataset for this particle size
    dataset = group.iloc[0]
    tif_path = Path(dataset['tif_path'])
    mpp = dataset.get('mpp')
    fps = dataset.get('fps')
    
    print(f"\nProcessing {particle_size} nm particles: {tif_path.name}")
    
    # Load dataset
    try:
        # Load TIFF stack directly
        image_stack = load_tiff_stack(tif_path)
        
        print(f"   Stack shape: {image_stack.shape}")
        print(f"   Metadata: {mpp} µm/px, {1.0/fps if fps else 'N/A'} s/frame")
        
        # Apply preprocessing
        preprocessed, avg_image = preprocess_stack(
            image_stack, 
            smooth=True, 
            radius=50
        )
        
        # Show comparison in tkinter
        title = f"Preprocessing: {particle_size} nm - {tif_path.name}"
        try:
            show_preprocessing_comparison(
                image_stack, 
                preprocessed, 
                avg_image, 
                title, 
                mpp=mpp
            )
        except tk.TclError as e:
            print(f"   WARNING: Could not display viewer (Tkinter error): {e}")
        except Exception as e:
            print(f"   ERROR displaying viewer: {e}")
        
        print(f"   → Viewer closed. Continuing to next particle size.")
        
    except Exception as e:
        print(f"   ERROR loading {tif_path.name}: {e}")
        continue

print("\nPreprocessing visualization complete.")
