"""
Trajectory + SEM Overlay Visualization

Loads a TrackMate XML with its corresponding TIFF image and an SEM image.
Selects a random trajectory, displays it with a scalebar, and creates
a superposition with the SEM image (inverted, green, tiled).

Uses core modules for data loading and visualization where possible.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
import random

# Core imports
from core.io import single_file_data
from core.visualization import plot_trajectories

# SEM metadata function
from sem_particle_analysis_V2 import summarize_sem_metadata

# -----------------------------
# Configuration
# -----------------------------

# Trajectory data 
TRAJ_XML = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks\20nm_1d_A4_05_Tracks.xml")

# SEM image
SEM_IMAGE = Path(r"D:\SEM\2026_02-02_REM_20mg_42mg_hydrogel\20mg_150nm_80kx_004.tif")

# Output
SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations")

# Visualization settings
CROP_PADDING_UM = 1.0  # Padding around trajectory in µm
TRAJECTORY_COLOR = 'red'
TRAJECTORY_LINEWIDTH = 1.5
TRAJECTORY_ALPHA = 0.9
TRAJ_IMAGE_ALPHA = 0.6  # Opacity of trajectory image in overlay
SEM_ALPHA = 0.5  # Opacity of SEM tiles
SEM_BRIGHTNESS_BOOST = 1.5  # Boost SEM brightness after inversion (1.0 = no change)

# Random seed (set to None for truly random, or integer for reproducibility)
RANDOM_SEED = None


# -----------------------------
# Helper functions using core patterns
# -----------------------------

def load_tiff_from_result_dict(result_dict: dict) -> np.ndarray | None:
    """Load TIFF image from result_dict's tif_path."""
    tif_path = result_dict.get('tif_path')
    if tif_path is None:
        return None
    tif_path = Path(tif_path)
    if not tif_path.exists():
        return None
    img = Image.open(tif_path)
    return np.array(img.convert('L'))


def load_sem_image(sem_path: Path) -> tuple[np.ndarray, float]:
    """Load SEM image and extract pixel size in nm using core SEM metadata function."""
    meta = summarize_sem_metadata(sem_path)
    px_nm = 0.5 * (meta["summary"]["pixel_width_nm"] + meta["summary"]["pixel_height_nm"])

    img = Image.open(sem_path)
    # Keep original bit depth, don't convert to 'L' yet
    img_array = np.array(img)

    # Debug info
    print(f"  SEM raw dtype: {img_array.dtype}, min: {img_array.min()}, max: {img_array.max()}")

    # Normalize to 8-bit (0-255) with contrast stretch
    img_float = img_array.astype(np.float64)
    img_min, img_max = img_float.min(), img_float.max()

    if img_max > img_min:
        img_normalized = (img_float - img_min) / (img_max - img_min) * 255
    else:
        img_normalized = img_float

    img_8bit = img_normalized.astype(np.uint8)
    print(f"  SEM normalized: min: {img_8bit.min()}, max: {img_8bit.max()}")

    return img_8bit, px_nm


def select_random_trajectory(tracks_df, min_length: int = 10) -> int:
    """Select a random trajectory with at least min_length points."""
    track_lengths = tracks_df.groupby('particle').size()
    valid_tracks = track_lengths[track_lengths >= min_length].index.tolist()

    if not valid_tracks:
        return track_lengths.idxmax()

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    return random.choice(valid_tracks)


def get_trajectory_bounds(tracks_df, particle_id: int, mpp_um: float, padding_um: float) -> tuple:
    """Get bounding box for a trajectory in pixel coordinates."""
    traj = tracks_df[tracks_df['particle'] == particle_id]

    x_min_px = traj['x'].min()
    x_max_px = traj['x'].max()
    y_min_px = traj['y'].min()
    y_max_px = traj['y'].max()

    padding_px = padding_um / mpp_um

    x_min = int(max(0, x_min_px - padding_px))
    x_max = int(x_max_px + padding_px)
    y_min = int(max(0, y_min_px - padding_px))
    y_max = int(y_max_px + padding_px)

    return x_min, x_max, y_min, y_max


def scale_sem_to_spt(sem_image: np.ndarray, sem_px_nm: float, spt_mpp_um: float,
                     target_shape: tuple, max_tile_px: int = 2000) -> np.ndarray:
    """
    Scale SEM image so that its pixel size matches the SPT image pixel size,
    then tile to fill target shape. Memory-efficient: scales a smaller tile.

    Args:
        sem_image: SEM image array (grayscale)
        sem_px_nm: SEM pixel size in nanometers
        spt_mpp_um: SPT pixel size in micrometers per pixel
        target_shape: (height, width) of the target area to fill
        max_tile_px: Maximum tile size before scaling (to limit memory)

    Returns:
        Scaled and tiled SEM image matching target_shape
    """
    # Convert units to same scale (nm)
    sem_px_nm = float(sem_px_nm)
    spt_px_nm = spt_mpp_um * 1000.0  # µm to nm

    # Scale factor: how many SEM pixels fit in one SPT pixel
    scale_factor = spt_px_nm / sem_px_nm

    print(f"  SEM pixel size: {sem_px_nm:.2f} nm/px")
    print(f"  SPT pixel size: {spt_px_nm:.2f} nm/px")
    print(f"  Scale factor: {scale_factor:.2f}x (SEM will be {'enlarged' if scale_factor > 1 else 'shrunk'})")

    h_src, w_src = sem_image.shape[:2]
    h_target, w_target = target_shape[:2]

    # Calculate how large the scaled tile would be
    scaled_w = int(w_src * scale_factor)
    scaled_h = int(h_src * scale_factor)

    print(f"  SEM original size: {w_src} x {h_src} px")
    print(f"  Target size: {w_target} x {h_target} px")

    # If scaled size would be huge, use a crop of the SEM image
    if scaled_w > max_tile_px or scaled_h > max_tile_px:
        # Calculate how much of the original SEM we need for one tile
        crop_w = min(w_src, int(max_tile_px / scale_factor))
        crop_h = min(h_src, int(max_tile_px / scale_factor))

        # Take center crop
        start_x = (w_src - crop_w) // 2
        start_y = (h_src - crop_h) // 2
        sem_crop = sem_image[start_y:start_y+crop_h, start_x:start_x+crop_w]

        print(f"  Using center crop: {crop_w} x {crop_h} px (to limit memory)")

        # Scale the crop
        new_w = int(crop_w * scale_factor)
        new_h = int(crop_h * scale_factor)
    else:
        sem_crop = sem_image
        new_w = scaled_w
        new_h = scaled_h

    print(f"  Scaled tile size: {new_w} x {new_h} px")

    # Use PIL for high-quality resizing
    if len(sem_crop.shape) == 3:
        pil_img = Image.fromarray(sem_crop)
    else:
        pil_img = Image.fromarray(sem_crop, mode='L')

    pil_scaled = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    scaled_tile = np.array(pil_scaled)

    # Now tile the scaled image to fill target shape
    n_tiles_y = int(np.ceil(h_target / new_h))
    n_tiles_x = int(np.ceil(w_target / new_w))

    print(f"  Tiling: {n_tiles_x} x {n_tiles_y} tiles")

    if len(scaled_tile.shape) == 3:
        tiled = np.tile(scaled_tile, (n_tiles_y, n_tiles_x, 1))
    else:
        tiled = np.tile(scaled_tile, (n_tiles_y, n_tiles_x))

    # Crop to exact target size
    return tiled[:h_target, :w_target]


def invert_and_colorize_green(image: np.ndarray, brightness_boost: float = 1.5) -> np.ndarray:
    """
    Invert grayscale image and convert to green channel (RGB).

    Args:
        image: Grayscale image (should be 8-bit, 0-255)
        brightness_boost: Factor to boost brightness (1.0 = no change, >1.0 = brighter)
    """
    # Ensure 8-bit
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Invert: dark becomes bright, bright becomes dark
    inverted = 255 - image

    # Optional brightness boost
    if brightness_boost != 1.0:
        inverted = np.clip(inverted.astype(np.float64) * brightness_boost, 0, 255).astype(np.uint8)

    print(f"  SEM inverted: min: {inverted.min()}, max: {inverted.max()}")

    # Create RGB image with green channel only
    rgb = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
    rgb[:, :, 1] = inverted  # Green channel

    return rgb


def tile_image(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Tile an image to fill target shape."""
    h_target, w_target = target_shape[:2]
    h_src, w_src = image.shape[:2]

    n_tiles_y = int(np.ceil(h_target / h_src))
    n_tiles_x = int(np.ceil(w_target / w_src))

    if len(image.shape) == 3:
        tiled = np.tile(image, (n_tiles_y, n_tiles_x, 1))
    else:
        tiled = np.tile(image, (n_tiles_y, n_tiles_x))

    return tiled[:h_target, :w_target]


def plot_single_trajectory(
    ax,
    tracks_df,
    particle_id: int,
    x_offset: float = 0,
    y_offset: float = 0,
    color: str = 'red',
    linewidth: float = 1.5,
    alpha: float = 0.9,
    fading: bool = True,
    show_endpoints: bool = True
):
    """
    Plot a single trajectory on given axes.
    Based on core.visualization.plot_trajectories pattern but for single track.
    """
    traj = tracks_df[tracks_df['particle'] == particle_id].sort_values('frame')
    x_traj = traj['x'].values - x_offset
    y_traj = traj['y'].values - y_offset

    if fading:
        n_points = len(traj)
        for i in range(n_points - 1):
            fade_alpha = alpha * (i + 1) / n_points
            ax.plot(x_traj[i:i+2], y_traj[i:i+2],
                    color=color, alpha=fade_alpha, linewidth=linewidth)
    else:
        ax.plot(x_traj, y_traj, color=color, linewidth=linewidth, alpha=alpha)

    if show_endpoints:
        ax.scatter(x_traj[0], y_traj[0], color='lime', s=60, zorder=5,
                   marker='o', edgecolors='white', linewidths=1)
        ax.scatter(x_traj[-1], y_traj[-1], color='cyan', s=60, zorder=5,
                   marker='s', edgecolors='white', linewidths=1)

    return x_traj, y_traj


def add_scalebar(ax, mpp_um: float, location: str = 'lower right'):
    """Add scalebar to axes using matplotlib-scalebar."""
    scalebar = ScaleBar(mpp_um, 'um', length_fraction=0.25, location=location,
                       color='white', box_alpha=0.7)
    ax.add_artist(scalebar)


# -----------------------------
# Main visualization functions
# -----------------------------

def create_overlay_figure(
    result_dict: dict,
    traj_image: np.ndarray,
    sem_image: np.ndarray,
    sem_px_nm: float,
    particle_id: int,
    crop_bounds: tuple,
    save_path: Path | None = None
):
    """Create the complete overlay figure using result_dict pattern."""
    tracks_df = result_dict['tracks_df']
    mpp_um = result_dict['mpp']

    x_min, x_max, y_min, y_max = crop_bounds
    crop_shape = (y_max - y_min, x_max - x_min)

    # Scale SEM image to match SPT pixel size and tile to fill crop area
    print("\nScaling SEM image to match SPT pixel size...")
    sem_scaled = scale_sem_to_spt(sem_image, sem_px_nm, mpp_um, target_shape=crop_shape)

    # Prepare SEM image: invert, colorize green (already tiled by scale_sem_to_spt)
    sem_tiled = invert_and_colorize_green(sem_scaled, brightness_boost=SEM_BRIGHTNESS_BOOST)

    # Crop trajectory image
    traj_crop = traj_image[y_min:y_max, x_min:x_max]

    # Get trajectory info
    traj = tracks_df[tracks_df['particle'] == particle_id]
    n_frames = len(traj)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    # 1. Trajectory only
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(traj_crop, cmap='gray', origin='upper')
    plot_single_trajectory(ax1, tracks_df, particle_id, x_min, y_min,
                          color=TRAJECTORY_COLOR, linewidth=TRAJECTORY_LINEWIDTH,
                          alpha=TRAJECTORY_ALPHA, fading=True)
    add_scalebar(ax1, mpp_um)
    ax1.set_title(f'Trajectory {particle_id}\n({n_frames} frames)', fontsize=12)
    ax1.axis('off')

    # 2. SEM tiled (inverted green)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(sem_tiled, origin='upper')
    ax2.set_title('SEM (inverted, green, tiled)', fontsize=12)
    ax2.axis('off')

    # 3. Overlay / Superposition
    ax3 = fig.add_subplot(1, 3, 3)

    # Convert trajectory crop to RGB
    traj_rgb = np.stack([traj_crop, traj_crop, traj_crop], axis=-1)

    # Blend images
    blended = (TRAJ_IMAGE_ALPHA * traj_rgb.astype(float) +
               SEM_ALPHA * sem_tiled.astype(float))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    ax3.imshow(blended, origin='upper')

    # Draw trajectory on overlay (thicker line for visibility)
    plot_single_trajectory(ax3, tracks_df, particle_id, x_min, y_min,
                          color=TRAJECTORY_COLOR, linewidth=TRAJECTORY_LINEWIDTH + 0.5,
                          alpha=1.0, fading=False)

    add_scalebar(ax3, mpp_um)
    ax3.set_title('Superposition\n(Trajectory + SEM)', fontsize=12)
    ax3.axis('off')

    plt.tight_layout()

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig_path = save_path / f"trajectory_sem_overlay_{particle_id}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Figure saved: {fig_path}")

    plt.show()

    return fig


def create_simple_overlay(sem_image: np.ndarray, traj_image: np.ndarray,
                          tracks_df, mpp_um: float, save_path: Path | None = None):
    """
    Simple overlay: resize both images to same square size and overlay.
    No physical scaling - just visual comparison.
    """
    # Use a reasonable square size
    target_size = 600

    # Resize SEM to square
    pil_sem = Image.fromarray(sem_image, mode='L')
    sem_resized = np.array(pil_sem.resize((target_size, target_size), Image.Resampling.LANCZOS))

    # Resize trajectory image to square
    pil_traj = Image.fromarray(traj_image, mode='L')
    traj_resized = np.array(pil_traj.resize((target_size, target_size), Image.Resampling.LANCZOS))

    # Invert SEM and make green
    sem_inv = 255 - sem_resized
    sem_green = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    sem_green[:, :, 1] = sem_inv

    # Convert traj to RGB (grayscale)
    traj_rgb = np.stack([traj_resized, traj_resized, traj_resized], axis=-1)

    # Blend
    alpha_traj = 0.6
    alpha_sem = 0.5
    blended = (alpha_traj * traj_rgb.astype(float) + alpha_sem * sem_green.astype(float))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. SEM Original
    ax1 = axes[0, 0]
    ax1.imshow(sem_resized, cmap='gray')
    ax1.set_title('SEM (resized)')
    ax1.axis('off')

    # 2. SEM Inverted Green
    ax2 = axes[0, 1]
    ax2.imshow(sem_green)
    ax2.set_title('SEM Inverted (Green)')
    ax2.axis('off')

    # 3. Trajectory Image
    ax3 = axes[1, 0]
    ax3.imshow(traj_resized, cmap='gray')
    ax3.set_title('SPT Image (resized)')
    ax3.axis('off')

    # 4. Overlay
    ax4 = axes[1, 1]
    ax4.imshow(blended)
    ax4.set_title('Overlay (SPT + SEM)')
    ax4.axis('off')

    plt.suptitle('Simple Overlay (no physical scaling)', fontsize=14)
    plt.tight_layout()

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / "simple_overlay.png", dpi=150)
        print(f"Saved: {save_path / 'simple_overlay.png'}")

    plt.show()


def show_sem_debug(sem_image: np.ndarray, sem_px_nm: float, spt_mpp_um: float):
    """
    Debug function: Show the SEM image - original, inverted, and a scaled crop.
    """
    spt_px_nm = spt_mpp_um * 1000.0
    scale_factor = spt_px_nm / sem_px_nm

    print(f"\n=== SEM DEBUG ===")
    print(f"  SEM pixel: {sem_px_nm:.2f} nm/px")
    print(f"  SPT pixel: {spt_px_nm:.2f} nm/px")
    print(f"  Scale factor: {scale_factor:.1f}x")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Original SEM
    ax1 = axes[0, 0]
    ax1.imshow(sem_image, cmap='gray')
    ax1.set_title(f'SEM Original\n{sem_image.shape[1]}x{sem_image.shape[0]} px, {sem_px_nm:.2f} nm/px')
    ax1.axis('off')

    # 2. SEM Inverted
    ax2 = axes[0, 1]
    inverted = 255 - sem_image
    ax2.imshow(inverted, cmap='gray')
    ax2.set_title('SEM Inverted')
    ax2.axis('off')

    # 3. SEM Inverted Green
    ax3 = axes[1, 0]
    green = np.zeros((*sem_image.shape, 3), dtype=np.uint8)
    green[:, :, 1] = inverted
    ax3.imshow(green)
    ax3.set_title('SEM Inverted (Green)')
    ax3.axis('off')

    # 4. Small scaled crop (what 1 SPT pixel looks like)
    ax4 = axes[1, 1]
    # Take a small crop and scale it
    crop_size = min(50, sem_image.shape[0], sem_image.shape[1])  # 50 SEM pixels
    h, w = sem_image.shape[:2]
    cx, cy = w // 2, h // 2
    crop = sem_image[cy-crop_size//2:cy+crop_size//2, cx-crop_size//2:cx+crop_size//2]

    # Scale up to show what it would look like
    scaled_size = int(crop_size * scale_factor)
    if scaled_size > 0:
        pil_crop = Image.fromarray(crop, mode='L')
        pil_scaled = pil_crop.resize((min(scaled_size, 1000), min(scaled_size, 1000)), Image.Resampling.LANCZOS)
        scaled_crop = np.array(pil_scaled)

        # Invert and colorize
        scaled_inv = 255 - scaled_crop
        scaled_green = np.zeros((*scaled_crop.shape, 3), dtype=np.uint8)
        scaled_green[:, :, 1] = scaled_inv

        ax4.imshow(scaled_green)
        physical_size_nm = crop_size * sem_px_nm
        ax4.set_title(f'Scaled Crop ({crop_size}px = {physical_size_nm:.0f}nm)\n'
                     f'Scaled to {scaled_crop.shape[1]}x{scaled_crop.shape[0]} px')
    else:
        ax4.text(0.5, 0.5, 'Scale too small', ha='center', va='center')
        ax4.set_title('Scaled Crop')
    ax4.axis('off')

    plt.suptitle(f'SEM Debug View\nPhysical size: {sem_image.shape[1]*sem_px_nm/1000:.1f} x {sem_image.shape[0]*sem_px_nm/1000:.1f} µm',
                fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_all_trajectories_overview(result_dict: dict, save_path: Path | None = None):
    """
    Plot overview of all trajectories using core visualization function.
    This shows the full field of view with all tracked particles.
    """
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        overview_path = save_path / f"trajectories_overview_{result_dict['base_name']}.png"
        plot_trajectories(result_dict, overview_path, max_tracks=100, fading=True)


def main():
    print("=" * 60)
    print("TRAJECTORY + SEM OVERLAY VISUALIZATION")
    print("=" * 60)

    # Load trajectory data using core.io
    print(f"\nLoading trajectory: {TRAJ_XML.name}")
    result_dict = single_file_data(TRAJ_XML)

    if result_dict is None:
        raise ValueError(f"Could not load trajectory data from {TRAJ_XML}")

    tracks_df = result_dict['tracks_df']
    mpp_um = result_dict['mpp']

    print(f"  Tracks: {result_dict['num_tracks']}")
    print(f"  Frames: {result_dict['num_frames']}")
    print(f"  Pixel size: {mpp_um:.4f} µm/px ({mpp_um * 1000:.2f} nm/px)")

    # Load trajectory TIFF from result_dict
    traj_image = load_tiff_from_result_dict(result_dict)
    if traj_image is None:
        # Try parent directory
        tif_path = TRAJ_XML.parent.parent / (TRAJ_XML.stem.replace('_Tracks', '') + '.tif')
        if tif_path.exists():
            print(f"  TIFF (found): {tif_path.name}")
            traj_image = np.array(Image.open(tif_path).convert('L'))
        else:
            raise FileNotFoundError(f"Could not find TIFF for {TRAJ_XML}")
    else:
        print(f"  TIFF: {result_dict['tif_path']}")

    # Load SEM image
    print(f"\nLoading SEM: {SEM_IMAGE.name}")
    sem_image, sem_px_nm = load_sem_image(SEM_IMAGE)
    print(f"  SEM pixel size: {sem_px_nm:.2f} nm/px")
    print(f"  SEM image size: {sem_image.shape}")

    # Simple overlay: just show both images side by side and overlaid
    create_simple_overlay(sem_image, traj_image, tracks_df, mpp_um, SAVE_PATH)

    print("\nDone!")
    return  # Stop here for now

    # Select random trajectory
    particle_id = select_random_trajectory(tracks_df, min_length=10)
    traj = tracks_df[tracks_df['particle'] == particle_id]
    print(f"\nSelected trajectory: {particle_id} ({len(traj)} frames)")

    # Get crop bounds
    crop_bounds = get_trajectory_bounds(tracks_df, particle_id, mpp_um, CROP_PADDING_UM)
    x_min, x_max, y_min, y_max = crop_bounds
    print(f"  Crop region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print(f"  Crop size: {x_max - x_min} x {y_max - y_min} px")

    # Ensure crop is within image bounds
    img_h, img_w = traj_image.shape
    crop_bounds = (
        max(0, x_min),
        min(img_w, x_max),
        max(0, y_min),
        min(img_h, y_max)
    )

    # Plot overview of all trajectories first
    print("\nCreating trajectory overview...")
    plot_all_trajectories_overview(result_dict, SAVE_PATH)

    # Create overlay figure
    print("\nCreating overlay figure...")
    create_overlay_figure(
        result_dict, traj_image, sem_image, sem_px_nm, particle_id,
        crop_bounds, SAVE_PATH
    )

    print("\nDone!")


if __name__ == "__main__":
    print("Starte Main...")
    main()
