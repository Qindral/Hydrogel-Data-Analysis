"""
MSD Analysis from TIFF Files

Load TIFF stacks, apply preprocessing (background subtraction), and prepare for tracking.
Uses standardized core modules for metadata extraction.

Author: Jonas
Date: 2026-01-13
"""

# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter
import cv2
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import from core modules
from hydro_analysis.core.io import parse_rec_file, extract_particle_size_from_path


# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung")
SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\trackpy_Linking")


# ============================================================================
# TIFF LOADING FUNCTIONS
# ============================================================================

class _TiffLogCatcher(logging.Handler):
    """Capture tifffile error logs that indicate corrupted or truncated files."""

    def __init__(self) -> None:
        super().__init__()
        self.invalid_page_offsets: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "invalid page offset" in message:
            self.invalid_page_offsets.append(message)


def load_tiff_stack(tif_path: Path) -> np.ndarray:
    """
    Load TIFF file as numpy array.

    Args:
        tif_path: Path to TIFF file

    Returns:
        Image stack as numpy array (T, Y, X)
    """
    tiff_logger = logging.getLogger("tifffile")
    handler = _TiffLogCatcher()
    previous_level = tiff_logger.level
    tiff_logger.addHandler(handler)
    tiff_logger.setLevel(logging.ERROR)
    try:
        with tifffile.TiffFile(tif_path) as tif:
            image_stack = tif.asarray()
    finally:
        tiff_logger.removeHandler(handler)
        tiff_logger.setLevel(previous_level)

    if handler.invalid_page_offsets:
        raise tifffile.TiffFileError(
            f"Invalid page offset detected in {tif_path.name}. "
            "TIFF appears truncated or corrupted."
        )

    # Ensure at least 3D (add time axis if single frame)
    if image_stack.ndim == 2:
        image_stack = image_stack[np.newaxis, ...]
    # If 4D, assume CTYX or CZYX and take first channel/Z
    elif image_stack.ndim == 4:
        image_stack = image_stack[0, ...]

    return image_stack


def collect_tiff_datasets(root_path: Path) -> List[Dict]:
    """
    Scan root directory for TIFF files organized by particle size folders.
    Uses core.io functions for metadata extraction.

    Args:
        root_path: Root directory to scan

    Returns:
        List of dicts with keys: 'tif_path', 'rec_path', 'particle_size_nm', 'fps', 'mpp'
    """
    datasets = []

    print(f"\nScanning for TIFF files in {root_path}")
    print("=" * 70)

    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue

        # Extract particle size using core function
        particle_size = extract_particle_size_from_path(subfolder)

        if particle_size:
            print(f"\nFolder: {subfolder.name} ({particle_size} nm)")
        else:
            print(f"\nFolder: {subfolder.name} (no particle size detected)")

        # Find all .tif files in this folder
        tif_files = list(subfolder.glob("*.tif")) + list(subfolder.glob("*.tiff"))
        print(f"   Found {len(tif_files)} TIFF files")

        for tif_path in tif_files:
            # Look for corresponding .rec file
            rec_path = tif_path.with_suffix('.tif.rec')
            if not rec_path.exists():
                rec_path = tif_path.with_suffix('.rec')

            dataset_info = {
                'tif_path': str(tif_path),
                'rec_path': str(rec_path) if rec_path.exists() else None,
                'particle_size_nm': particle_size,
                'fps': None,
                'mpp': None,
                'exposure_ms': None
            }

            # Extract metadata from .rec file using core function
            if rec_path.exists():
                rec_metadata = parse_rec_file(rec_path)
                dataset_info['fps'] = rec_metadata.get('fps')
                dataset_info['mpp'] = rec_metadata.get('mpp')
                dataset_info['exposure_ms'] = rec_metadata.get('exposure_ms')
                print(f"     [OK] {tif_path.name} (mpp={dataset_info['mpp']}, fps={dataset_info['fps']:.1f})"
                      if dataset_info['fps'] else f"     [OK] {tif_path.name}")
            else:
                print(f"     [!] {tif_path.name} (no .rec file)")

            datasets.append(dataset_info)

    print("\n" + "=" * 70)
    print(f"Total: {len(datasets)} TIFF files found")
    print("=" * 70)

    return datasets


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def fast_rolling_ball(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Fast rolling ball background subtraction using morphological operations.

    Args:
        image: 2D image array
        radius: Rolling ball radius in pixels

    Returns:
        Background-subtracted image
    """
    if radius < 1:
        return image
    k_size = int(radius * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(image, background)


def preprocess_stack(
    image_stack: np.ndarray,
    smooth: bool = True,
    sigma: float = 1.0,
    rolling_ball_radius: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply preprocessing: average subtraction + rolling ball background subtraction.

    Args:
        image_stack: 3D array (T, Y, X)
        smooth: Apply Gaussian smoothing before background subtraction
        sigma: Gaussian smoothing sigma
        rolling_ball_radius: Rolling ball radius for background subtraction

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
            frame = gaussian_filter(frame, sigma=sigma)
        preprocessed[i] = fast_rolling_ball(frame, rolling_ball_radius)

    return preprocessed, avg_image


# ============================================================================
# VISUALIZATION
# ============================================================================

class PreprocessingViewer:
    """Interactive viewer for comparing raw and preprocessed images."""

    def __init__(
        self,
        image_stack: np.ndarray,
        preprocessed: np.ndarray,
        avg_image: np.ndarray,
        title: str,
        mpp: Optional[float] = None,
        fps: Optional[float] = None
    ):
        self.image_stack = image_stack
        self.preprocessed = preprocessed
        self.avg_image = avg_image
        self.title = title
        self.mpp = mpp
        self.fps = fps

        self.root = tk.Tk()
        self.root.title(title)

        self.current_frame = tk.IntVar(value=0)
        self.setup_ui()

    def setup_ui(self):
        """Setup the tkinter UI."""
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Create matplotlib figure with 3 subplots
        self.fig = Figure(figsize=(16, 5))
        ax1 = self.fig.add_subplot(131)
        ax2 = self.fig.add_subplot(132)
        ax3 = self.fig.add_subplot(133)

        # Initial images
        self.im1 = ax1.imshow(self.image_stack[0], cmap='gray',
                              vmin=self.image_stack.min(), vmax=self.image_stack.max())
        ax1.set_title("Raw Image", fontsize=12, fontweight='bold')
        ax1.axis('off')
        self.fig.colorbar(self.im1, ax=ax1, fraction=0.046)

        self.im2 = ax2.imshow(self.avg_image, cmap='gray')
        ax2.set_title("Temporal Average (subtracted)", fontsize=12, fontweight='bold')
        ax2.axis('off')
        self.fig.colorbar(self.im2, ax=ax2, fraction=0.046)

        self.im3 = ax3.imshow(self.preprocessed[0], cmap='viridis',
                              vmin=self.preprocessed.min(), vmax=self.preprocessed.max())
        ax3.set_title("Preprocessed", fontsize=12, fontweight='bold')
        ax3.axis('off')
        self.fig.colorbar(self.im3, ax=ax3, fraction=0.046)

        self.fig.tight_layout()

        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Frame slider
        ttk.Label(control_frame, text="Frame:").pack(side=tk.LEFT)
        slider = ttk.Scale(
            control_frame, from_=0, to=len(self.image_stack)-1,
            variable=self.current_frame, orient=tk.HORIZONTAL,
            command=self.update_frame, length=400
        )
        slider.pack(side=tk.LEFT, padx=10)

        self.frame_label = ttk.Label(control_frame, text=f"Frame: 0/{len(self.image_stack)-1}")
        self.frame_label.pack(side=tk.LEFT)

        # Metadata display
        info_parts = []
        if self.mpp:
            info_parts.append(f"Scale: {self.mpp:.3f} µm/px")
        if self.fps:
            info_parts.append(f"FPS: {self.fps:.1f}")
        if info_parts:
            ttk.Label(control_frame, text=" | ".join(info_parts)).pack(side=tk.RIGHT, padx=10)

        # Close button
        ttk.Button(control_frame, text="Close", command=self.root.destroy).pack(side=tk.RIGHT)

    def update_frame(self, val):
        """Update displayed frame."""
        idx = int(float(val))
        self.im1.set_data(self.image_stack[idx])
        self.im3.set_data(self.preprocessed[idx])
        self.frame_label.config(text=f"Frame: {idx}/{len(self.image_stack)-1}")
        self.canvas.draw()

    def show(self):
        """Display the viewer."""
        self.root.mainloop()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: collect datasets and show preprocessing preview."""
    print("=" * 70)
    print("TIFF Preprocessing and Dataset Collection")
    print("=" * 70)

    # Ensure save path exists
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Collect all datasets
    datasets = collect_tiff_datasets(ROOT_PATH)

    if not datasets:
        print("\nNo datasets found!")
        return

    # Save dataset inventory
    datasets_df = pd.DataFrame(datasets)
    inventory_path = SAVE_PATH / "datasets_inventory.csv"
    datasets_df.to_csv(inventory_path, index=False)
    print(f"\nDataset inventory saved to {inventory_path}")

    # Group datasets by particle size
    datasets_df['particle_size_nm'] = datasets_df['particle_size_nm'].astype(float)
    datasets_by_size = datasets_df.groupby('particle_size_nm')

    # Interactive preprocessing visualization
    print("\n" + "=" * 70)
    print("PREPROCESSING VISUALIZATION")
    print("=" * 70)
    print("Showing one representative file per particle size...")

    for particle_size, group in datasets_by_size:
        if pd.isna(particle_size):
            print(f"\nSkipping datasets without particle size label")
            continue

        # Select first dataset for this particle size
        dataset = group.iloc[0]
        tif_path = Path(dataset['tif_path'])
        mpp = dataset.get('mpp')
        fps = dataset.get('fps')

        print(f"\nProcessing {particle_size:.0f} nm particles: {tif_path.name}")

        try:
            # Load TIFF stack
            image_stack = load_tiff_stack(tif_path)
            print(f"   Stack shape: {image_stack.shape}")
            print(f"   Metadata: mpp={mpp} µm/px, fps={fps}")

            # Apply preprocessing
            preprocessed, avg_image = preprocess_stack(
                image_stack,
                smooth=True,
                sigma=1.0,
                rolling_ball_radius=50
            )

            # Show comparison viewer
            title = f"Preprocessing: {particle_size:.0f} nm - {tif_path.name}"
            try:
                viewer = PreprocessingViewer(
                    image_stack,
                    preprocessed,
                    avg_image,
                    title,
                    mpp=mpp,
                    fps=fps
                )
                viewer.show()
                print(f"   Viewer closed. Continuing to next particle size.")
            except tk.TclError as e:
                print(f"   WARNING: Could not display viewer (Tkinter error): {e}")
            except Exception as e:
                print(f"   ERROR displaying viewer: {e}")

        except Exception as e:
            print(f"   ERROR loading {tif_path.name}: {e}")
            continue

    print("\n" + "=" * 70)
    print("Preprocessing visualization complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
