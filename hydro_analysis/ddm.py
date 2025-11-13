r"""Differential Dynamic Microscopy (DDM) utilities.

This module provides a lightweight implementation of the core DDM
calculation that can be applied to time resolved TIFF image stacks.  The
implementation follows the standard recipe:

1. Compute differences of frames separated by a lag time :math:`\Delta t`.
2. Transform the differences to Fourier space and average their power
   spectra.
3. Radially average the isotropic power spectrum to obtain the
   intermediate scattering function as a function of the scattering
   vector magnitude ``q`` and ``\Delta t``.

The resulting data can be used to extract diffusion coefficients or to
perform subsequent model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

from .data_loader import DatasetLoader


@dataclass
class DDMResult:
    """Container for the output of a DDM analysis."""

    lags: np.ndarray
    q_values: np.ndarray
    structure_function: np.ndarray
    pixel_size_um: Optional[float]
    time_step_s: Optional[float]

    def save(self, path: Path | str) -> None:
        """Persist the DDM result to a ``.npz`` file."""

        np.savez_compressed(
            path,
            lags=self.lags,
            q_values=self.q_values,
            structure_function=self.structure_function,
            pixel_size_um=self.pixel_size_um,
            time_step_s=self.time_step_s,
        )


def run_ddm_analysis(
    path: Path | str,
    *,
    max_lag: int = 50,
    q_bins: int = 30,
    frame_step: int = 1,
    subtract_mean: bool = True,
) -> DDMResult:
    r"""Run a Differential Dynamic Microscopy analysis on an image stack.

    Parameters
    ----------
    path:
        Location of the TIFF stack.  The first axis is expected to be the
        time dimension.  Additional dimensions (channels, z) are averaged
        prior to processing.
    max_lag:
        Largest frame separation :math:`\Delta t` to evaluate.
    q_bins:
        Number of bins used for the radial averaging in Fourier space.
    frame_step:
        Use every ``frame_step``-th frame to reduce the data volume.
    subtract_mean:
        Whether to subtract the temporal mean from the stack before the
        Fourier analysis.  This is typically recommended to suppress the
        static background.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if max_lag < 1:
        raise ValueError("max_lag muss ≥ 1 sein")
    if q_bins < 4:
        raise ValueError("q_bins muss ≥ 4 sein")
    if frame_step < 1:
        raise ValueError("frame_step muss ≥ 1 sein")

    loader = DatasetLoader(path)
    loaded = loader.load()
    data = _prepare_stack(loaded.data, axes=loaded.metadata.axes, frame_step=frame_step)

    if data.ndim != 3:
        raise ValueError(
            "DDM expects a 3-D array with axes (time, y, x) after preprocessing"
        )

    data = data.astype(np.float32, copy=False)
    if subtract_mean:
        data -= data.mean(axis=0, keepdims=True)

    max_lag = min(max_lag, data.shape[0] - 1)
    if max_lag < 1:
        raise ValueError("Need at least two frames to compute DDM")

    lags = np.arange(1, max_lag + 1, dtype=np.int32)
    q_values, structure_function = _compute_structure_function(
        data,
        lags,
        q_bins,
    )

    timestamps = loaded.metadata.timestamps
    time_step = None
    if timestamps and len(timestamps) >= 2:
        diffs = np.diff(timestamps)
        time_step = float(np.median(diffs))

    result = DDMResult(
        lags=lags.astype(float) * (time_step if time_step else 1.0),
        q_values=q_values,
        structure_function=structure_function,
        pixel_size_um=loaded.metadata.px_size_xy_um,
        time_step_s=time_step,
    )
    return result


def _prepare_stack(
    data: np.ndarray,
    *,
    axes: Optional[str] = None,
    frame_step: int = 1,
) -> np.ndarray:
    """Convert the raw stack into a 3-D ``(time, y, x)`` array."""

    arr = np.asarray(data)

    if axes:
        axes_list = list(axes)
        if "Y" not in axes_list or "X" not in axes_list:
            raise ValueError("DDM erfordert Y- und X-Achsen im Datensatz")
        if "T" not in axes_list:
            raise ValueError("DDM erfordert eine Zeitachse (T) im Datensatz")

        order: list[int] = []
        if "T" in axes_list:
            order.append(axes_list.index("T"))
        for idx, axis in enumerate(axes_list):
            if axis not in {"T", "Y", "X"}:
                order.append(idx)
        order.extend([axes_list.index("Y"), axes_list.index("X")])

        arr = np.transpose(arr, order)
        ordered_axes = [axes_list[i] for i in order]

        if ordered_axes[0] != "T":
            arr = arr[None, ...]
        if arr.ndim > 3:
            arr = arr.reshape(arr.shape[0], -1, arr.shape[-2], arr.shape[-1])
            arr = arr.mean(axis=1)
    else:
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim > 3:
            spatial_dims = arr.shape[-2:]
            arr = arr.reshape(-1, *spatial_dims)

    if arr.ndim != 3:
        raise ValueError("Konnte die Bilddaten nicht in (time, y, x) umformen")

    if frame_step > 1:
        arr = arr[::frame_step]

    return arr


def _compute_structure_function(
    data: np.ndarray,
    lags: Iterable[int],
    q_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the radially averaged DDM structure function."""

    n_frames, height, width = data.shape
    fy = np.fft.fftfreq(height)
    fx = np.fft.fftfreq(width)
    qy, qx = np.meshgrid(fy, fx, indexing="ij")
    q = np.sqrt(qx**2 + qy**2)

    positive_q = q[q > 0]
    q_min = float(positive_q.min()) if positive_q.size else 0.0
    q_max = float(q.max())
    q_edges = np.linspace(q_min, q_max, q_bins + 1)
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])

    lag_list = list(lags)
    structure = np.zeros((len(lag_list), q_bins), dtype=np.float32)
    filled = 0

    for idx, lag in enumerate(lag_list):
        if lag >= n_frames:
            break
        diffs = data[lag:] - data[:-lag]
        fft_vals = np.fft.fftn(diffs, axes=(1, 2))
        power_spectrum = np.mean(np.abs(fft_vals) ** 2, axis=0)
        structure[idx] = _radial_average(power_spectrum, q, q_edges)
        filled = idx + 1

    return q_centers, structure[:filled]


def _radial_average(
    image: np.ndarray,
    q: np.ndarray,
    q_edges: np.ndarray,
) -> np.ndarray:
    """Radially average ``image`` using bins defined by ``q_edges``."""

    flat_image = image.ravel()
    flat_q = q.ravel()
    bin_indices = np.digitize(flat_q, q_edges) - 1

    radial = np.zeros(len(q_edges) - 1, dtype=np.float32)
    counts = np.zeros_like(radial)

    for idx, value in zip(bin_indices, flat_image, strict=False):
        if 0 <= idx < len(radial):
            radial[idx] += value
            counts[idx] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        radial = np.divide(radial, counts, out=np.zeros_like(radial), where=counts > 0)
    return radial


__all__ = ["DDMResult", "run_ddm_analysis"]

