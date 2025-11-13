"""Utilities for analysing SEM particle images using Hough circle detection.

This module exposes a command line interface that performs the following steps:

1. Load an image and read its DPI metadata to determine the pixel size.
2. Smooth the image using a Gaussian filter.
3. Detect edges with a Canny filter.
4. Locate circular particles with the Hough circle transform.
5. Summarise the detected particle population and optionally save an overlay
   image that visualises the detections.

The CLI can be invoked with ``python -m hydro_analysis.sem_particle_analysis``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image
from skimage import color, draw, feature, filters, io, util
from skimage.transform import hough_circle, hough_circle_peaks


MICRONS_PER_INCH = 25_400.0


@dataclass
class CircleDetection:
    """Container describing a detected circle in pixel coordinates."""

    y: float
    x: float
    radius: float
    strength: float

    def to_dict(self) -> dict:
        return {"y": self.y, "x": self.x, "radius": self.radius, "strength": self.strength}


@dataclass
class ParticleStatistics:
    """Summary statistics for a collection of detected particles."""

    count: int
    mean_diameter_um: float | None
    std_diameter_um: float | None
    mean_area_um2: float | None
    std_area_um2: float | None

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_diameter_um": self.mean_diameter_um,
            "std_diameter_um": self.std_diameter_um,
            "mean_area_um2": self.mean_area_um2,
            "std_area_um2": self.std_area_um2,
        }


def load_image_with_dpi(path: Path) -> tuple[np.ndarray, float | None]:
    """Load an image and return a normalised grayscale array and DPI metadata.

    Parameters
    ----------
    path:
        Path to the SEM image file.

    Returns
    -------
    image : numpy.ndarray
        2-D floating point array with values in ``[0, 1]``.
    dpi : float or None
        The average dots per inch stored in the image metadata. ``None`` if the
        metadata is missing.
    """

    with Image.open(path) as pil_img:
        # Convert to grayscale regardless of input mode.
        grayscale = pil_img.convert("L")
        dpi = None
        dpi_info = pil_img.info.get("dpi")
        if isinstance(dpi_info, tuple) and dpi_info:
            # Some formats store DPI as (x, y).
            dpi_values = [float(value) for value in dpi_info if value]
            if dpi_values:
                dpi = float(sum(dpi_values) / len(dpi_values))

        image = np.asarray(grayscale, dtype=np.float32)

    # Normalise to [0, 1] for skimage filters.
    image = util.img_as_float(image)
    return image, dpi


def smooth_image(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a Gaussian blur to reduce noise."""

    return filters.gaussian(image, sigma=sigma, preserve_range=False)


def detect_edges(image: np.ndarray, sigma: float, low_threshold: float, high_threshold: float) -> np.ndarray:
    """Run a Canny edge detector on the image."""

    return feature.canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)


def _build_radii(min_radius: int, max_radius: int, radius_step: int) -> List[int]:
    if min_radius <= 0 or max_radius <= 0:
        raise ValueError("Radii must be positive integers.")
    if max_radius < min_radius:
        raise ValueError("max_radius must be greater than or equal to min_radius.")
    if radius_step <= 0:
        raise ValueError("radius_step must be a positive integer.")

    return list(range(min_radius, max_radius + 1, radius_step))


def detect_circles(edges: np.ndarray, radii: Sequence[int], total_peaks: int) -> List[CircleDetection]:
    """Detect circles using the Hough transform."""

    if not radii:
        return []

    hough_res = hough_circle(edges, radii)
    accums, cx, cy, detected_radii = hough_circle_peaks(hough_res, radii, total_num_peaks=total_peaks)

    circles = [CircleDetection(y=float(y), x=float(x), radius=float(r), strength=float(a)) for a, x, y, r in zip(accums, cx, cy, detected_radii)]
    return circles


def pixel_to_microns(pixels: np.ndarray, dpi: float | None) -> np.ndarray | None:
    """Convert a length in pixels to microns using the image DPI metadata."""

    if dpi is None:
        return None
    microns_per_pixel = MICRONS_PER_INCH / dpi
    return pixels * microns_per_pixel


def summarise_particles(circles: Iterable[CircleDetection], dpi: float | None) -> ParticleStatistics:
    """Compute statistics for the detected particle diameters and areas."""

    circle_list = list(circles)
    if not circle_list:
        return ParticleStatistics(count=0, mean_diameter_um=None, std_diameter_um=None, mean_area_um2=None, std_area_um2=None)

    radii_pixels = np.array([circle.radius for circle in circle_list], dtype=float)
    diameters_pixels = 2.0 * radii_pixels
    diameters_um = pixel_to_microns(diameters_pixels, dpi)
    if dpi is not None:
        radii_um = pixel_to_microns(radii_pixels, dpi)
        assert radii_um is not None
        areas_um2 = np.pi * np.square(radii_um)
    else:
        areas_um2 = None

    def _summary(values: np.ndarray | None) -> tuple[float | None, float | None]:
        if values is None:
            return None, None
        return float(values.mean()), float(values.std(ddof=0))

    mean_diameter_um, std_diameter_um = _summary(diameters_um)
    mean_area_um2, std_area_um2 = _summary(areas_um2)

    return ParticleStatistics(
        count=len(circle_list),
        mean_diameter_um=mean_diameter_um,
        std_diameter_um=std_diameter_um,
        mean_area_um2=mean_area_um2,
        std_area_um2=std_area_um2,
    )


def create_overlay(image: np.ndarray, circles: Sequence[CircleDetection]) -> np.ndarray:
    """Generate an RGB overlay with detected circles highlighted."""

    if image.ndim != 2:
        raise ValueError("Input image must be a 2-D grayscale array.")

    overlay = color.gray2rgb(image)
    for circle in circles:
        rr, cc = draw.circle_perimeter(int(round(circle.y)), int(round(circle.x)), int(round(circle.radius)), shape=image.shape)
        overlay[rr, cc] = (1.0, 0.0, 0.0)
    return overlay


def save_overlay(image: np.ndarray, circles: Sequence[CircleDetection], path: Path) -> None:
    overlay = create_overlay(image, circles)
    io.imsave(path, util.img_as_ubyte(overlay))


def run_analysis(
    image_path: Path,
    gaussian_sigma: float,
    canny_sigma: float,
    canny_low: float,
    canny_high: float,
    min_radius: int,
    max_radius: int,
    radius_step: int,
    total_peaks: int,
    overlay_path: Path | None,
    output_json: Path | None,
) -> dict:
    """Execute the full analysis pipeline and return the resulting data."""

    image, dpi = load_image_with_dpi(image_path)
    smoothed = smooth_image(image, sigma=gaussian_sigma)
    edges = detect_edges(smoothed, sigma=canny_sigma, low_threshold=canny_low, high_threshold=canny_high)
    radii = _build_radii(min_radius, max_radius, radius_step)
    circles = detect_circles(edges, radii, total_peaks=total_peaks)
    stats = summarise_particles(circles, dpi)

    if overlay_path is not None:
        save_overlay(image, circles, overlay_path)

    result = {
        "image_path": str(image_path),
        "dpi": dpi,
        "gaussian_sigma": gaussian_sigma,
        "canny_sigma": canny_sigma,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "radius_step": radius_step,
        "total_peaks": total_peaks,
        "circles": [circle.to_dict() for circle in circles],
        "statistics": stats.to_dict(),
    }

    if output_json is not None:
        output_json.write_text(json.dumps(result, indent=2))

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse SEM particle images using Hough circle detection.")
    parser.add_argument("image", type=Path, help="Path to the SEM image file.")
    parser.add_argument("--gaussian-sigma", type=float, default=2.0, help="Sigma for the Gaussian smoothing filter.")
    parser.add_argument("--canny-sigma", type=float, default=1.0, help="Sigma for the Canny edge detector.")
    parser.add_argument("--canny-low", type=float, default=0.1, help="Low hysteresis threshold for the Canny detector.")
    parser.add_argument("--canny-high", type=float, default=0.3, help="High hysteresis threshold for the Canny detector.")
    parser.add_argument("--min-radius", type=int, default=5, help="Minimum circle radius to search for in pixels.")
    parser.add_argument("--max-radius", type=int, default=50, help="Maximum circle radius to search for in pixels.")
    parser.add_argument("--radius-step", type=int, default=2, help="Step size when enumerating radii in pixels.")
    parser.add_argument("--total-peaks", type=int, default=20, help="Maximum number of circle detections to return.")
    parser.add_argument("--overlay", type=Path, help="Optional path to save an overlay image with detections highlighted.")
    parser.add_argument("--output-json", type=Path, help="Optional path to save a JSON report of the analysis results.")
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    result = run_analysis(
        image_path=args.image,
        gaussian_sigma=args.gaussian_sigma,
        canny_sigma=args.canny_sigma,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        radius_step=args.radius_step,
        total_peaks=args.total_peaks,
        overlay_path=args.overlay,
        output_json=args.output_json,
    )

    print(json.dumps(result["statistics"], indent=2))
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
