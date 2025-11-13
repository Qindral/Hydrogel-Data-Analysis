"""Utilities for analysing SEM particle images using Hough circle detection.

This module exposes a command line interface that performs the following steps:

1. Load an image and read its DPI metadata to determine the pixel size.
2. Smooth the image using a Gaussian filter.
3. Detect edges with a Canny filter.
4. Locate circular particles with the Hough circle transform.
5. Summarise the detected particle population, optionally save an overlay
   image that visualises the detections, and present both graphical and
   textual summaries of the particle statistics.

The CLI can be invoked with ``python -m hydro_analysis.sem_particle_analysis``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, TYPE_CHECKING

import numpy as np
from PIL import Image
from skimage import color, draw, feature, filters, io, util
from skimage.transform import hough_circle, hough_circle_peaks


if TYPE_CHECKING:
    from matplotlib.figure import Figure


MICRONS_PER_INCH = 25_400.0


def _get_pyplot():
    from matplotlib import pyplot as plt

    return plt


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


def detect_edges(
    image: np.ndarray,
    *,
    method: str,
    sigma: float,
    low_threshold: float,
    high_threshold: float,
    edge_threshold: float,
) -> np.ndarray:
    """Detect edges using the requested algorithm."""

    method = method.lower()
    if method == "canny":
        return feature.canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

    if method == "sobel":
        gradient = filters.sobel(image)
    elif method == "scharr":
        gradient = filters.scharr(image)
    elif method == "prewitt":
        gradient = filters.prewitt(image)
    else:
        raise ValueError(f"Unbekannter Edge-Filter: {method}")

    if gradient.size == 0:
        return np.zeros_like(gradient, dtype=bool)

    gradient = util.img_as_float(gradient)
    max_value = float(np.max(gradient))
    if max_value == 0:
        return np.zeros_like(gradient, dtype=bool)

    if edge_threshold <= 0:
        threshold_value = filters.threshold_otsu(gradient)
    elif edge_threshold <= 1:
        threshold_value = edge_threshold * max_value
    else:
        threshold_value = edge_threshold

    return gradient >= threshold_value


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

    circles = [
        CircleDetection(y=float(y), x=float(x), radius=float(r), strength=float(a))
        for a, x, y, r in zip(accums, cx, cy, detected_radii)
    ]
    return circles


def perform_analysis(
    image: np.ndarray,
    dpi: float | None,
    *,
    gaussian_sigma: float,
    edge_method: str,
    canny_sigma: float,
    canny_low: float,
    canny_high: float,
    edge_threshold: float,
    min_radius: int,
    max_radius: int,
    radius_step: int,
    total_peaks: int,
    histogram_bins: int,
) -> tuple[
    List[CircleDetection],
    np.ndarray | None,
    tuple[np.ndarray, np.ndarray] | None,
    ParticleStatistics,
    np.ndarray,
    float | None,
]:
    """Analyse a preloaded SEM image."""

    if histogram_bins <= 0:
        raise ValueError("histogram_bins must be a positive integer.")

    smoothed = smooth_image(image, sigma=gaussian_sigma)
    edges = detect_edges(
        smoothed,
        method=edge_method,
        sigma=canny_sigma,
        low_threshold=canny_low,
        high_threshold=canny_high,
        edge_threshold=edge_threshold,
    )
    radii = _build_radii(min_radius, max_radius, radius_step)
    circles = detect_circles(edges, radii, total_peaks=total_peaks)
    stats = summarise_particles(circles, dpi)

    microns_per_pixel = microns_per_pixel_from_dpi(dpi)
    diameters_um: np.ndarray | None = None
    histogram: tuple[np.ndarray, np.ndarray] | None = None
    if circles and microns_per_pixel is not None:
        radii_pixels = np.array([circle.radius for circle in circles], dtype=float)
        diameters_pixels = 2.0 * radii_pixels
        diameters_um = pixel_to_microns(diameters_pixels, dpi)
        if diameters_um is not None and len(diameters_um) > 0:
            histogram = np.histogram(diameters_um, bins=histogram_bins)

    return circles, diameters_um, histogram, stats, edges, microns_per_pixel


def microns_per_pixel_from_dpi(dpi: float | None) -> float | None:
    """Return the physical size represented by a single pixel in microns."""

    if dpi is None:
        return None
    return MICRONS_PER_INCH / dpi


def pixel_to_microns(pixels: np.ndarray, dpi: float | None) -> np.ndarray | None:
    """Convert a length in pixels to microns using the image DPI metadata."""

    microns_per_pixel = microns_per_pixel_from_dpi(dpi)
    if microns_per_pixel is None:
        return None
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


def create_overlay_figure(image: np.ndarray, circles: Sequence[CircleDetection]) -> Figure:
    """Return a Matplotlib figure showing the detections on top of the image."""

    plt = _get_pyplot()
    overlay = create_overlay(image, circles)
    fig, ax = plt.subplots()
    ax.imshow(overlay, origin="upper")
    ax.set_title("Detected particles overlay")
    ax.axis("off")
    return fig


def create_statistics_figure(
    diameters_um: np.ndarray | None,
    stats: ParticleStatistics,
    histogram: tuple[np.ndarray, np.ndarray] | None,
) -> Figure | None:
    """Create a histogram figure summarising the diameter distribution."""

    if diameters_um is None or histogram is None or len(diameters_um) == 0:
        return None

    plt = _get_pyplot()
    counts, bin_edges = histogram
    fig, ax = plt.subplots()
    ax.hist(diameters_um, bins=bin_edges, color="tab:blue", edgecolor="black")
    ax.set_xlabel("Particle diameter (µm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Particle diameter distribution")

    text_lines = [f"Anzahl: {stats.count}"]
    if stats.mean_diameter_um is not None:
        text_lines.append(f"Mittelwert: {stats.mean_diameter_um:.3f} µm")
    if stats.std_diameter_um is not None:
        text_lines.append(f"Standardabweichung: {stats.std_diameter_um:.3f} µm")

    ax.text(
        0.98,
        0.95,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    fig.tight_layout()
    return fig


def write_diameter_report(path: Path, counts: np.ndarray, bin_edges: np.ndarray) -> None:
    """Persist a textual frequency table for the particle diameters."""

    lines = ["Durchmesserbereich (µm)\tAnzahl"]
    for start, end, count in zip(bin_edges[:-1], bin_edges[1:], counts):
        lines.append(f"{start:.3f} - {end:.3f}\t{int(count)}")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    image_path: Path,
    gaussian_sigma: float,
    edge_method: str,
    canny_sigma: float,
    canny_low: float,
    canny_high: float,
    edge_threshold: float,
    min_radius: int,
    max_radius: int,
    radius_step: int,
    total_peaks: int,
    overlay_path: Path | None,
    output_json: Path | None,
    histogram_bins: int,
) -> tuple[
    dict,
    np.ndarray,
    List[CircleDetection],
    np.ndarray | None,
    tuple[np.ndarray, np.ndarray] | None,
    ParticleStatistics,
    np.ndarray,
]:
    """Execute the full analysis pipeline and return the resulting data."""

    image, dpi = load_image_with_dpi(image_path)
    (
        circles,
        diameters_um,
        histogram,
        stats,
        edges,
        microns_per_pixel,
    ) = perform_analysis(
        image,
        dpi,
        gaussian_sigma=gaussian_sigma,
        edge_method=edge_method,
        canny_sigma=canny_sigma,
        canny_low=canny_low,
        canny_high=canny_high,
        edge_threshold=edge_threshold,
        min_radius=min_radius,
        max_radius=max_radius,
        radius_step=radius_step,
        total_peaks=total_peaks,
        histogram_bins=histogram_bins,
    )

    if overlay_path is not None:
        save_overlay(image, circles, overlay_path)

    result = {
        "image_path": str(image_path),
        "dpi": dpi,
        "gaussian_sigma": gaussian_sigma,
        "edge_method": edge_method,
        "canny_sigma": canny_sigma,
        "canny_low": canny_low,
        "canny_high": canny_high,
        "edge_threshold": edge_threshold,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "radius_step": radius_step,
        "total_peaks": total_peaks,
        "circles": [circle.to_dict() for circle in circles],
        "statistics": stats.to_dict(),
        "microns_per_pixel": microns_per_pixel,
        "diameters_um": diameters_um.tolist() if diameters_um is not None else None,
        "diameter_histogram": {
            "counts": histogram[0].tolist(),
            "bin_edges": histogram[1].tolist(),
        }
        if histogram is not None
        else None,
    }

    if output_json is not None:
        output_json.write_text(json.dumps(result, indent=2))

    return result, image, circles, diameters_um, histogram, stats, edges


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse SEM particle images using Hough circle detection.")
    parser.add_argument("image", type=Path, help="Path to the SEM image file.")
    parser.add_argument("--gaussian-sigma", type=float, default=2.0, help="Sigma for the Gaussian smoothing filter.")
    parser.add_argument(
        "--edge-method",
        choices=["canny", "sobel", "scharr", "prewitt"],
        default="canny",
        help="Edge detection algorithm used before the Hough-Transformation.",
    )
    parser.add_argument("--canny-sigma", type=float, default=1.0, help="Sigma for the Canny edge detector.")
    parser.add_argument("--canny-low", type=float, default=0.1, help="Low hysteresis threshold for the Canny detector.")
    parser.add_argument("--canny-high", type=float, default=0.3, help="High hysteresis threshold for the Canny detector.")
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.2,
        help=(
            "Schwellwert für Gradient-basierte Filter (0-1 relativ, >1 absolut, ≤0 automatische Bestimmung)."
        ),
    )
    parser.add_argument("--min-radius", type=int, default=5, help="Minimum circle radius to search for in pixels.")
    parser.add_argument("--max-radius", type=int, default=50, help="Maximum circle radius to search for in pixels.")
    parser.add_argument("--radius-step", type=int, default=2, help="Step size when enumerating radii in pixels.")
    parser.add_argument("--total-peaks", type=int, default=20, help="Maximum number of circle detections to return.")
    parser.add_argument("--overlay", type=Path, help="Optional path to save an overlay image with detections highlighted.")
    parser.add_argument("--output-json", type=Path, help="Optional path to save a JSON report of the analysis results.")
    parser.add_argument("--histogram-bins", type=int, default=10, help="Number of bins used for the diameter histogram.")
    parser.add_argument(
        "--stats-figure",
        type=Path,
        help="Path to save the statistics graphic (histogram of particle diameters).",
    )
    parser.add_argument(
        "--diameter-report",
        type=Path,
        help="Path to save the textual frequency table. Defaults to <image>_diameter_report.txt.",
    )
    parser.add_argument(
        "--no-show-figures",
        dest="show_figures",
        action="store_false",
        help="Disable interactive display of the overlay and statistics figures.",
    )
    parser.set_defaults(show_figures=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    (
        result,
        image,
        circles,
        diameters_um,
        histogram,
        stats,
        _edges,
    ) = run_analysis(
        image_path=args.image,
        gaussian_sigma=args.gaussian_sigma,
        edge_method=args.edge_method,
        canny_sigma=args.canny_sigma,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        edge_threshold=args.edge_threshold,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        radius_step=args.radius_step,
        total_peaks=args.total_peaks,
        overlay_path=args.overlay,
        output_json=args.output_json,
        histogram_bins=args.histogram_bins,
    )

    print(json.dumps(result["statistics"], indent=2))
    if result["microns_per_pixel"] is not None:
        print(f"Mikrometer pro Pixel: {result['microns_per_pixel']:.6f} µm")
    else:
        print("Mikrometer pro Pixel: nicht verfügbar (fehlende DPI-Angabe)")

    histogram_counts = histogram[0] if histogram is not None else None
    histogram_edges = histogram[1] if histogram is not None else None

    default_report_path = args.image.with_name(f"{args.image.stem}_diameter_report.txt")
    report_path = args.diameter_report or default_report_path
    if histogram_counts is not None and histogram_edges is not None:
        write_diameter_report(report_path, histogram_counts, histogram_edges)
        print(f"Häufigkeitstabelle gespeichert unter: {report_path}")
    else:
        if stats.count == 0 and result["microns_per_pixel"] is None:
            report_message = (
                "Keine Partikel erkannt und keine DPI-Angabe vorhanden – keine Häufigkeitsverteilung verfügbar."
            )
        elif stats.count == 0:
            report_message = "Keine Partikel erkannt – keine Häufigkeitsverteilung verfügbar."
        elif result["microns_per_pixel"] is None:
            report_message = "Durchmesser konnten mangels DPI nicht berechnet werden."
        else:
            report_message = "Keine Durchmesserwerte verfügbar."

        report_path.write_text(report_message, encoding="utf-8")
        print(f"Hinweis gespeichert unter: {report_path}")
        print(report_message)

    plt_module = None
    if args.show_figures or args.stats_figure is not None:
        plt_module = _get_pyplot()

    overlay_fig: Figure | None = None
    stats_fig: Figure | None = None

    if args.show_figures:
        overlay_fig = create_overlay_figure(image, circles)

    if args.show_figures or args.stats_figure is not None:
        stats_fig = create_statistics_figure(diameters_um, stats, histogram)

    if args.stats_figure is not None:
        if stats_fig is not None:
            stats_fig.savefig(args.stats_figure, dpi=300)
            print(f"Statistikdiagramm gespeichert unter: {args.stats_figure}")
            if not args.show_figures and plt_module is not None:
                plt_module.close(stats_fig)
                stats_fig = None
        else:
            print("Warnung: Keine Statistiken zum Speichern verfügbar.")

    if args.show_figures:
        figures = [fig for fig in (overlay_fig, stats_fig) if fig is not None]
        if figures:
            assert plt_module is not None
            plt_module.show()
            for fig in figures:
                plt_module.close(fig)
        else:
            print("Hinweis: Keine Figuren zum Anzeigen verfügbar.")

    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
