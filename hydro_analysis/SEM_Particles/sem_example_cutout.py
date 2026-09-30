"""
SEM example cutouts -- one cropped example image per nominal particle size,
saved once plain and once with the Cellpose-SAM particle outlines overlaid,
both with a scale bar. The images carry no heading; each matplotlib figure
is named after its particle size (figure window title, e.g. "35 nm").

For every size in EXAMPLES, a region of CROP_WIDTH_NM around the given centre
(fractions of the scanned area) is cut from the raw FEI SEM image; the crop
has the 1.42:1 aspect ratio of the output figure and lies inside the scanned
area (the FEI databar is never shown). Contrast: linear stretch between the
CONTRAST_PERCENTILES of the crop; RGB raw images are converted to grey by
channel mean.

Overlay: outlines of the particles that enter the size statistics, i.e.
used == True in hydro_analysis/SEM_Particles/cache/sem_feret_particles.pkl
(written by sem_feret_measure.py, which must be run first). Particles excluded
there (touching the scanned-area border, fragments < MIN_AREA_PX) are not
drawn. The annotation file of each image is found with the same rule as in
sem_feret_measure.py (find_annotations): label mask outlines are traced at
sub-pixel level (marching squares), ROI sets are drawn as their polygons.
Ellipses added by hand in sem_segmentation_review.py (review file, negative
particle ids) are drawn in the same colour when they are used.

Reads the raw SEM images and annotations in sem_feret_measure.DATA_DIR and the
cache pickle; writes into SAVE_PATH, per size:
  sem_example_<nominal>nm.png            cutout with scale bar
  sem_example_<nominal>nm_overlay.png    cutout with outlines and scale bar
No other script reads these outputs.

Styling follows Styleguide_Figures_Dissertation.md (v2): microscopy images as
PNG at 600 dpi in their final printed size (section 3, 9): width class
"half" (3.07 in) at 1.42:1, the image filling the whole figure, no
bbox_inches="tight". Figure names use the dissertation-wide particle labels
(core.io.get_dls_labels()). The scale bar has
the design of the other microscopy figures (add_scalebar() in
Trajectory/trajectory_plotter.py: matplotlib-scalebar, white bar with its
length above it, lower right, on a black box with alpha 0.6, Open Sans), with
the section 9 sizes: 8 pt text, bar height about 1.2 % of the image height.
add_scalebar() itself is not reused because it always labels in um (a 50 nm
bar would read "0.05 um"). Particle outlines in the detection colour #da00bd.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import roifile
import tifffile
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.measure import find_contours

from hydro_analysis.SEM_Particles.sem_feret_measure import (
    CACHE_PATH, DATA_DIR, SIZE_FOLDERS, added_ovals_for, ellipse_polygon, fei_scan_geometry, find_annotations,
    load_review,
)
from hydro_analysis.SEM_Particles.sem_feret_size_histogram import size_label

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "SEM_particle_size"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
SHOW_FIGURE = True

# nominal size -> (raw image, crop centre x, crop centre y as fractions of the scanned
# area, crop width in nm, scale bar length in nm)
EXAMPLES = {
    20: ("Latex particle_T2_20nm_007.tif", 0.5, 0.5, 300.0, 50.0),
    50: ("Latex particle_T2_50nm_002.tif", 0.5, 0.5, 500.0, 100.0),
    100: ("100nm_T1_04.tif", 0.5, 0.5, 1000.0, 200.0),
    200: ("Latex particle_Cpad_200nm_002.tif", 0.5, 0.5, 2000.0, 500.0),
    500: ("500nm_T2_04.tif", 0.5, 0.5, 5000.0, 1000.0),
    1000: ("1000nm_10kx_004.tif", 0.5, 0.5, 10000.0, 2000.0),
}

FIG_WIDTH_IN = 3.07                    # width class "half" (style guide §2)
RATIO = 1.42
DPI = 600
CONTRAST_PERCENTILES = (0.5, 99.9)
COLOR_OUTLINE = "#da00bd"              # detection colour (style guide §9)
OUTLINE_LW = 0.6
SCALEBAR_HEIGHT_FRAC = 0.012           # bar height / image height (style guide §9: 1-1.5 %)
SCALEBAR_FONT_PT = 8                   # style guide §4


def load_used_ids(cache_path: Path = CACHE_PATH) -> dict[str, set[int]]:
    """{raw image name: particle ids used in the size statistics} from the measurement cache."""
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} not found; run sem_feret_measure.py first.")
    with open(cache_path, "rb") as fh:
        particles = pickle.load(fh)["particles"]
    used = particles[particles["used"]]
    return {image: set(grp["particle_id"].astype(int)) for image, grp in used.groupby("image")}


def crop_window(width: int, height: int, px_nm: float, cx: float, cy: float,
                crop_width_nm: float) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) in px of a 1.42:1 crop centred at (cx, cy), shifted to lie inside the scanned area."""
    w = int(round(crop_width_nm / px_nm))
    h = int(round(w / RATIO))
    if w > width or h > height:
        raise ValueError(f"Crop of {crop_width_nm:g} nm ({w} x {h} px) exceeds the scanned area ({width} x {height} px).")
    x0 = int(np.clip(round(cx * width - w / 2), 0, width - w))
    y0 = int(np.clip(round(cy * height - h / 2), 0, height - h))
    return x0, y0, w, h


def load_grey(raw_path: Path) -> np.ndarray:
    image = tifffile.imread(raw_path).astype(float)
    return image.mean(axis=2) if image.ndim == 3 else image


def outlines_in_crop(ann_path: Path, used_ids: set[int], window: tuple[int, int, int, int],
                     ovals: list[list[float]] = ()) -> list[np.ndarray]:
    """Outlines (x, y in crop pixel coordinates) of the used particles and added ellipses overlapping the crop."""
    x0, y0, w, h = window
    outlines = []
    for k, oval in enumerate(ovals, start=1):
        xy = ellipse_polygon(*oval) - [x0, y0]
        if -k in used_ids and xy[:, 0].max() >= 0 and xy[:, 0].min() <= w and xy[:, 1].max() >= 0 and xy[:, 1].min() <= h:
            outlines.append(np.vstack([xy, xy[:1]]))
    if ann_path.suffix.lower() == ".zip":
        for i, roi in enumerate(roifile.roiread(ann_path), start=1):
            if i not in used_ids:
                continue
            xy = np.asarray(roi.coordinates(), dtype=float) - [x0, y0]
            if xy[:, 0].max() >= 0 and xy[:, 0].min() <= w and xy[:, 1].max() >= 0 and xy[:, 1].min() <= h:
                outlines.append(np.vstack([xy, xy[:1]]))
        return outlines

    # Pixel (r, c) covers [c, c + 1] x [r, r + 1] in ImageJ coordinates; contours of the
    # zero-padded binary mask at level 0.5 run through the pixel edges, hence the +0.5 shift.
    labels = tifffile.imread(ann_path)[y0:y0 + h, x0:x0 + w]
    for label_id in np.unique(labels):
        if label_id == 0 or int(label_id) not in used_ids:
            continue
        mask = np.pad(labels == label_id, 1).astype(float)
        for contour in find_contours(mask, 0.5):
            outlines.append(np.column_stack([contour[:, 1] - 0.5, contour[:, 0] - 0.5]))
    return outlines


def add_scalebar_nm(ax: plt.Axes, px_nm: float, bar_nm: float) -> None:
    """Scale bar in the design of trajectory_plotter.add_scalebar(), labelled in nm or um.

    The axes data unit is one image pixel of px_nm nanometres.
    """
    value, units = (bar_nm / 1000.0, "um") if bar_nm >= 1000 else (bar_nm, "nm")
    ax.add_artist(ScaleBar(
        px_nm, units="nm", dimension="si-length",
        fixed_value=value, fixed_units=units,
        location="lower right", width_fraction=SCALEBAR_HEIGHT_FRAC, sep=2,
        frameon=True, color="white", box_color="black", box_alpha=0.6,
        scale_loc="top", label_loc="bottom",
        scale_formatter=lambda v, unit: f"{v:g} {unit}",       # "50 nm" rather than "50.0 nm"
        font_properties={"family": "Open Sans", "size": SCALEBAR_FONT_PT},
    ))


def plot_cutout(crop: np.ndarray, px_nm: float, bar_nm: float, outlines: list[np.ndarray] | None,
                name: str) -> plt.Figure:
    """Figure `name` filled by the cutout, optional outlines and scale bar; call inside the style context."""
    h, w = crop.shape
    lo, hi = np.percentile(crop, CONTRAST_PERCENTILES)
    fig = plt.figure(num=name, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN / RATIO), layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi, interpolation="antialiased", extent=(0, w, h, 0))
    for xy in outlines or []:
        ax.plot(xy[:, 0], xy[:, 1], color=COLOR_OUTLINE, linewidth=OUTLINE_LW)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    add_scalebar_nm(ax, px_nm, bar_nm)
    return fig


def main(show: bool = SHOW_FIGURE):
    """Crop, plot and save the plain and the overlay cutout of every example image."""
    used_ids = load_used_ids()
    review = load_review()
    figures = {}
    with plt.style.context(STYLE_PATH):
        for nominal, (raw_name, cx, cy, crop_nm, bar_nm) in EXAMPLES.items():
            folder = DATA_DIR / SIZE_FOLDERS[nominal]
            annotations, _ = find_annotations(folder)
            px_nm, width, height = fei_scan_geometry(folder / raw_name)
            window = crop_window(width, height, px_nm, cx, cy, crop_nm)
            x0, y0, w, h = window
            crop = load_grey(folder / raw_name)[y0:y0 + h, x0:x0 + w]
            ovals = added_ovals_for(review, SIZE_FOLDERS[nominal], raw_name, annotations[raw_name])
            outlines = outlines_in_crop(folder / annotations[raw_name], used_ids.get(raw_name, set()), window, ovals)
            print(f"{nominal:>5} nm: {raw_name}  <-  {annotations[raw_name]}  crop x={x0} y={y0} {w} x {h} px "
                  f"({w * px_nm:.0f} x {h * px_nm:.0f} nm), {len(outlines)} outlines")
            label = size_label(nominal)
            figures[f"sem_example_{nominal}nm"] = plot_cutout(crop, px_nm, bar_nm, None, label)
            figures[f"sem_example_{nominal}nm_overlay"] = plot_cutout(crop, px_nm, bar_nm, outlines,
                                                                      f"{label} overlay")

        if show:
            plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(SAVE_PATH / f"{name}.png", format="png", dpi=DPI)
            print(f"Saved {len(figures)} images to {SAVE_PATH}")
    return figures


if __name__ == "__main__":
    main()
