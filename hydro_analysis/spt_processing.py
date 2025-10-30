"""Single particle tracking preprocessing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import (
    gaussian_laplace,
    maximum_filter,
    shift as nd_shift,
    uniform_filter1d,
)


EPS = 1e-8


@dataclass
class SPTPreprocessingParams:
    """Configuration for the SPT preprocessing pipeline."""

    window: int = 21
    log_sigma: float = 1.5
    threshold: float = 1.5
    nms_radius: float = 3.0
    use_zscore: bool = True
    drift_correction: bool = False
    drift_update_interval: int = 20
    bleaching_mode: str = "none"  # "none", "poly", "exp"
    bleaching_order: int = 1
    min_presence_fraction: float = 0.0
    presence_threshold: float = 0.0


@dataclass
class SPTPreprocessingResult:
    """Outputs from the SPT preprocessing pipeline."""

    corrected_video: NDArray[np.float32]
    background_mean: NDArray[np.float32]
    background_std: NDArray[np.float32]
    variance_map: NDArray[np.float32]
    zscore_map: NDArray[np.float32]
    bandpass_map: NDArray[np.float32]
    peaks: NDArray[np.float32]
    peak_metrics: List[Dict[str, float]]
    drift_shifts: Optional[List[Tuple[float, float]]] = None


def _phase_correlation_shift(reference: NDArray[np.float32], frame: NDArray[np.float32]) -> Tuple[float, float]:
    """Estimate x/y shift between reference and frame via phase correlation."""

    ref_fft = np.fft.fftn(reference)
    frame_fft = np.fft.fftn(frame)
    cross_power = ref_fft * np.conj(frame_fft)
    denom = np.abs(cross_power) + EPS
    cross_power /= denom
    corr = np.fft.ifftn(cross_power)
    maxima = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    shifts: List[float] = []
    for dim, max_idx in enumerate(maxima):
        size = reference.shape[dim]
        if max_idx > size // 2:
            shift = max_idx - size
        else:
            shift = max_idx
        shifts.append(float(shift))
    # Return shifts as (y, x)
    return shifts[0], shifts[1]


def _apply_drift_correction(
    video: NDArray[np.float32],
    params: SPTPreprocessingParams,
    mask: Optional[NDArray[np.bool_]] = None,
) -> Tuple[NDArray[np.float32], List[Tuple[float, float]]]:
    """Apply phase correlation based drift correction."""

    corrected = np.empty_like(video)
    reference = video[0]
    if mask is not None:
        reference = reference * mask
    shifts: List[Tuple[float, float]] = [(0.0, 0.0)]
    for idx, frame in enumerate(video):
        if idx == 0:
            corrected[idx] = frame
            continue
        if params.drift_update_interval and idx % params.drift_update_interval == 0:
            reference = corrected[idx - 1]
            if mask is not None:
                reference = reference * mask
        working_frame = frame * mask if mask is not None else frame
        shift_y, shift_x = _phase_correlation_shift(reference, working_frame)
        shifts.append((shift_y, shift_x))
        corrected[idx] = nd_shift(frame, shift=(shift_y, shift_x), mode="nearest")
    return corrected, shifts


def _apply_bleaching_correction(
    video: NDArray[np.float32],
    params: SPTPreprocessingParams,
    mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float32]:
    """Normalize bleaching trends based on mean intensity in a mask."""

    if params.bleaching_mode == "none":
        return video
    pixels = video if mask is None else video[:, mask]
    frame_means = pixels.mean(axis=1)
    frames = np.arange(video.shape[0], dtype=np.float32)
    if params.bleaching_mode == "poly":
        order = max(1, int(params.bleaching_order))
        coeffs = np.polyfit(frames, frame_means, order)
        trend = np.polyval(coeffs, frames)
    elif params.bleaching_mode == "exp":
        order = max(1, int(params.bleaching_order))
        safe_means = np.clip(frame_means, EPS, None)
        coeffs = np.polyfit(frames, np.log(safe_means), order)
        trend = np.exp(np.polyval(coeffs, frames))
    else:
        return video
    norm = trend / (np.mean(trend) + EPS)
    corrected = video / norm[:, None, None]
    return corrected.astype(np.float32)


def _compute_background(
    video: NDArray[np.float32],
    window: int,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    window = max(1, int(window))
    mean = uniform_filter1d(video, size=window, axis=0, mode="nearest")
    mean_sq = uniform_filter1d(video ** 2, size=window, axis=0, mode="nearest")
    variance = np.clip(mean_sq - mean**2, 0.0, None)
    std = np.sqrt(variance + EPS)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_bandpass(map_image: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    if sigma <= 0:
        return map_image
    response = -gaussian_laplace(map_image, sigma=sigma)
    return response.astype(np.float32)


def _non_maximum_suppression(
    image: NDArray[np.float32],
    threshold: float,
    radius: float,
    mask: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.bool_]:
    size = max(1, int(round(radius)))
    if size % 2 == 0:
        size += 1
    footprint = np.ones((size, size), dtype=bool)
    local_max = image == maximum_filter(image, footprint=footprint, mode="nearest")
    candidates = local_max & (image >= threshold)
    if mask is not None:
        candidates &= mask
    return candidates


def _refine_subpixel(
    image: NDArray[np.float32],
    peak_coords: NDArray[np.int64],
) -> Tuple[NDArray[np.float32], List[Dict[str, float]]]:
    results: List[List[float]] = []
    metrics: List[Dict[str, float]] = []
    for y, x in peak_coords:
        y0 = max(0, y - 1)
        y1 = min(image.shape[0], y + 2)
        x0 = max(0, x - 1)
        x1 = min(image.shape[1], x + 2)
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        weights = patch - patch.min()
        weights_sum = weights.sum()
        if weights_sum <= EPS:
            yc = float(y)
            xc = float(x)
        else:
            yc = float((yy * weights).sum() / weights_sum)
            xc = float((xx * weights).sum() / weights_sum)
        value = float(image[y, x])
        results.append([yc, xc])
        metrics.append({"response": value})
    if not results:
        return np.empty((0, 2), dtype=np.float32), metrics
    return np.asarray(results, dtype=np.float32), metrics


def _validate_presence(
    detrended: NDArray[np.float32],
    coords: NDArray[np.float32],
    metrics: List[Dict[str, float]],
    params: SPTPreprocessingParams,
) -> Tuple[NDArray[np.float32], List[Dict[str, float]]]:
    if params.min_presence_fraction <= 0:
        return coords, metrics
    keep_indices: List[int] = []
    for idx, (y, x) in enumerate(coords):
        y_int = int(round(y))
        x_int = int(round(x))
        if not (0 <= y_int < detrended.shape[1] and 0 <= x_int < detrended.shape[2]):
            continue
        series = detrended[:, y_int, x_int]
        if params.presence_threshold > 0:
            thresh = params.presence_threshold
        else:
            thresh = series.mean() + series.std()
        fraction = float(np.count_nonzero(series > thresh)) / max(1, series.size)
        metrics[idx]["presence_fraction"] = fraction
        if fraction >= params.min_presence_fraction:
            keep_indices.append(idx)
    if not keep_indices:
        return np.empty((0, 2), dtype=np.float32), []
    filtered_coords = coords[keep_indices]
    filtered_metrics = [metrics[i] for i in keep_indices]
    return filtered_coords, filtered_metrics


def run_spt_preprocessing(
    video: ArrayLike,
    params: SPTPreprocessingParams,
    *,
    mask: Optional[ArrayLike] = None,
) -> SPTPreprocessingResult:
    """Run the SPT preprocessing pipeline on a video (t, y, x)."""

    data = np.asarray(video, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("Expected video array with shape (t, y, x)")
    roi_mask: Optional[NDArray[np.bool_]] = None
    if mask is not None:
        roi_mask = np.asarray(mask, dtype=bool)
        if roi_mask.shape != data.shape[1:]:
            raise ValueError("Mask must have shape (y, x)")
    if params.drift_correction:
        data, shifts = _apply_drift_correction(data, params, mask=roi_mask)
    else:
        shifts = [(0.0, 0.0) for _ in range(data.shape[0])]
    data = _apply_bleaching_correction(data, params, mask=roi_mask)
    background_mean_per_frame, background_std_per_frame = _compute_background(data, params.window)
    detrended = data - background_mean_per_frame
    variance_map = np.mean(detrended**2, axis=0)
    background_mean = background_mean_per_frame.mean(axis=0)
    background_std = np.sqrt(np.mean(background_std_per_frame**2, axis=0) + EPS)
    if params.use_zscore:
        mean_v = float(variance_map.mean())
        std_v = float(variance_map.std() + EPS)
        zscore_map = (variance_map - mean_v) / std_v
    else:
        zscore_map = variance_map
    bandpass_input = zscore_map if params.use_zscore else variance_map
    bandpass_map = _apply_bandpass(bandpass_input, params.log_sigma)
    detection_map = bandpass_map
    threshold_value = params.threshold
    if params.use_zscore and params.threshold <= 0:
        threshold_value = float(bandpass_map.mean() + bandpass_map.std())
    candidates_mask = _non_maximum_suppression(detection_map, threshold_value, params.nms_radius, mask=roi_mask)
    peak_coords_int = np.column_stack(np.nonzero(candidates_mask))
    refined_coords, metrics = _refine_subpixel(detection_map, peak_coords_int.astype(np.int64))
    refined_coords, metrics = _validate_presence(detrended, refined_coords, metrics, params)
    return SPTPreprocessingResult(
        corrected_video=data.astype(np.float32),
        background_mean=background_mean.astype(np.float32),
        background_std=background_std.astype(np.float32),
        variance_map=variance_map.astype(np.float32),
        zscore_map=zscore_map.astype(np.float32),
        bandpass_map=bandpass_map.astype(np.float32),
        peaks=refined_coords,
        peak_metrics=metrics,
        drift_shifts=shifts if params.drift_correction else None,
    )
