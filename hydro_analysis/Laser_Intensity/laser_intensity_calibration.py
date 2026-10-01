"""
Absolute illumination intensity in the sample plane for every wavelength and
laser setting: peak intensity, intensity at the edges of the field of view (FOV),
FOV-averaged intensity and the share of the laser power that reaches the FOV.

Measurement (per setting, both in the sample plane):
  (1) a mirror reflected the beam onto the SPT camera (pco.pixelfly USB,
      4 x 4 binning) -> TIFF stack with the relative beam profile;
  (2) a power sensor after the objective -> absolute power P (mW), noted in the
      Comment section of the matching .tif.rec file.
File names: Mirror_SizeofFWHM_<wavelength nm>[_<index>].tif. The index numbers
the measurements of one wavelength; it is not a repetition. There is no separate
laser setting (%, V), so the measured power P is the setting axis. Pixel size
(0.30 µm per binned pixel, Thorlabs 10 µm grid calibration), FOV (picture size)
and exposure time come from the .rec file via core.io.parse_rec_file; the FOV is
the recorded image (200 x 150 or 348 x 260 binned pixels).

Processing (parse -> load -> preprocess -> fit -> absolute -> metrics -> plot -> export):
  1. Parse wavelength and index from the file name, P from the .rec comment;
     print an overview of all files.
  2. Optional dark frame; frame-wise saturation check (frames with a pixel
     >= SATURATION_COUNTS are excluded with a warning); hot-pixel mask;
     consecutive valid frames averaged in blocks of FRAMES_PER_REP
     (one block = one repetition).
     A constant background is taken from the fitted offset c, not from an edge
     median: the beam reaches the FOV edge with a large fraction of its peak, so
     an edge median would subtract laser light, and a constant shift is absorbed
     by c anyway.
  3. 2D Gaussian in the 1/e^2 convention,
         I(x, y) = c + I0 exp(-2 (x'^2 / wx^2 + y'^2 / wy^2)),
     fitted with curve_fit (ROI ~4 w, start values from image moments, one refit
     with residual outliers masked); per-pixel weights from the temporal SD;
     uncertainties from the covariance, scaled by chi^2_red when > 1. The fit
     integral S = (pi/2) I0 wx wy is compared with the pixel sum.
  4. Absolute maps: the profile is normalised to integral 1 and multiplied by P,
         I_fit(x, y)  = P (G(x, y) - c) / (S A_px),
         I_meas(x, y) = P (image(x, y) - c) / (S A_px)   (A_px: pixel area).
     The total integral S must come from the Gaussian because the beam extends
     beyond the FOV; I_meas therefore integrates to the measured power in the FOV.
     If the residuals are structured (RMS > NON_GAUSSIAN_RESID_REL of I0 in the
     mean image of a file), the measured map is the selected source ("_sel").
  5. Per repetition and for both maps: I_peak = 2 P / (pi wx wy), intensity at
     the edge centres and corners of the FOV (absolute and % of peak), FOV mean,
     min/max and CV, power and power fraction inside the FOV, decentration of
     the beam centre from the FOV centre, wx, wy and FWHM = w sqrt(2 ln 2),
     energy E = P t and fluence F = I t (t = ENERGY_TIME_S, default exposure).
  6. Per wavelength: I_peak = a (P - P0) (proportional model with fewer than
     three valid settings, PCHIP if the residuals are systematic), inverse
     function, constancy of w over the settings.

Stage: compute stage. Reads the raw TIFF stacks and .rec files in DATA_DIR on
every run and unconditionally overwrites in OUTPUT_DIR:
  laser_intensity_per_image.csv       one row per repetition (frame block)
  laser_intensity_per_setting.csv     mean +- SD per (wavelength, P)
  laser_intensity.json                fit and calibration parameters
  laser_intensity_synthetic_test.csv  recovery test on synthetic beams
  intensity_map_<file>.png            A  absolute map + 1/e^2 ellipse + FOV (half, 600 dpi)
  line_profiles_<file>.pdf            B  x/y profiles, FOV limits, residuals (half)
  residual_map_<file>.png             residual map of the mean-image fit (half, 600 dpi)
  peak_intensity_vs_power.pdf         C  I_peak and FOV power vs. P (narrow)
  edge_intensity_relative.pdf         D  edge intensity in % of peak (half)
Consumers: other scripts import peak_intensity_for_power, power_for_peak_intensity,
fov_mean_intensity_for_power and fluence_for_power; these read only
laser_intensity.json and never touch raw data.

Figure A: linear colour scale; outside the FOV the Gaussian extrapolation is
drawn dimmed. Style: Styleguide_Figures_Dissertation.md v2 via
hydro_analysis/thesis.mplstyle, final printed size, no bbox_inches="tight".
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle
from scipy import stats
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.optimize import OptimizeWarning, curve_fit

from hydro_analysis.core.io import parse_rec_comment_metadata, parse_rec_file

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Laser_Intensity")
FILE_PATTERN = "Mirror_SizeofFWHM_*.tif"
FILENAME_RE = re.compile(r"_(?P<wavelength>\d{3})(?:_(?P<index>\d+))?\.tif$", re.IGNORECASE)
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
                  r"\Experiments and Results - Data\Laser_Intensity_Calibration")
RESULT_JSON = OUTPUT_DIR / "laser_intensity.json"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"

# Acquisition
PIXEL_SIZE_UM: float | None = None      # sample plane; None: .rec via core.io.parse_rec_file (0.30 µm, grid)
EXPOSURE_TIME_S: float | None = None    # None: exposure time from the .rec file
DARK_FRAME_PATH: Path | None = None     # dark TIFF (stack) at the same exposure; None: constant offset from the fit
SATURATION_COUNTS = 16383               # 14-bit ADC; frames with a pixel at or above this value are excluded
STUCK_PIXEL_FRACTION = 0.5              # saturated in more than this fraction of frames: stuck pixel, masked
POWER_OVERRIDES_MW: dict[str, float] = {}   # file name -> P (mW); default: first power in the .rec comment
ENERGY_TIME_S: float | None = None      # t for E = P t and F = I t; None: camera exposure time

# Fit
FRAMES_PER_REP = 10
MIN_VALID_REPS = 3                      # fewer valid repetitions: setting excluded from the calibration
ROI_HALF_WIDTH_W = 2.0                  # fit ROI = centre +- 2 w (about 4 w across), clipped to the image
MEDIAN_FILTER_PX = 5
HOT_PIXEL_SIGMA = 8.0
RESIDUAL_CLIP_SIGMA = 5.0
MAX_NFEV = 2000

# Metrics
EDGE_BOX_PX = 5                         # measured edge intensity: median of a box of this size at the edge pixel
NON_GAUSSIAN_RESID_REL = 0.03           # residual RMS / I0 above this: measured map is the selected source

# Quality flags
MIN_R2 = 0.95
MIN_FRACTION_IN_FOV = 0.4
W_OUTLIER_REL = 0.5
SYSTEMATIC_CHI2_RED = 3.0
SYSTEMATIC_RESID_SIGMA = 3.0
W_CV_WARN = 0.10
POWER_REL_UNC = 0.05                    # relative uncertainty of the sensor (assumed; use the datasheet value)

# Synthetic test
RUN_SYNTHETIC_TEST = True
SYNTHETIC_REALIZATIONS = 30
SYNTHETIC_SEED = 1

# Figures
SCALEBAR_UM = 20.0
PROFILE_BAND_PX = 1                     # line profiles averaged over centre row/column +- this many pixels
MAP_EXTEND_W = 1.15                     # map canvas covers the 1/e^2 ellipse times this factor ...
MAP_MAX_EXTEND_FOV = 0.5                # ... but at most this fraction of the FOV size beyond each edge
WIDTH_IN = {"full": 6.30, "narrow": 4.72, "half": 3.07, "third": 2.01}
RATIO = 1.42
COLOR_WAVELENGTH = {488: ("#3B6E8C", "#2A4F66"), 514: ("#237735", "#1a5e3c")}   # single series / special C
COLOR_FALLBACK = ("#555555", "#222222")
COLOR_X = ("#3B8C8C", "#2A6666")        # category A
COLOR_Y = ("#D98C3D", "#A6672D")        # category B
COLOR_ACCENT = "#da00bd"                # 1/e^2 ellipse
POINT_ALPHA = 0.7
ERR_KW = dict(elinewidth=0.8, capsize=2.0, capthick=0.8)

PARAM_NAMES = ("x0", "y0", "wx", "wy", "theta", "I0", "c")
EDGES = ("left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right")
EDGE_LABELS = ("L", "R", "T", "B", "TL", "TR", "BL", "BR")
FWHM_PER_W = np.sqrt(2.0 * np.log(2.0))
_WARNINGS: list[str] = []


def _warn(message: str) -> None:
    _WARNINGS.append(message)
    print(f"Warning: {message}")


# ── Parse and load ─────────────────────────────────────────────────────────────

def parse_filename(name: str) -> dict:
    """Wavelength (nm) and measurement index from the file name."""
    match = FILENAME_RE.search(name)
    if not match:
        raise ValueError(f"file name does not match FILENAME_RE: {name}")
    return {"wavelength_nm": float(match.group("wavelength")),
            "file_index": int(match.group("index")) if match.group("index") else 1}


def load_measurement(tif_path: Path) -> dict:
    """Frames, file-name fields and .rec metadata (P, pixel size, FOV, exposure) of one stack."""
    name = tif_path.name
    rec_path = tif_path.with_name(name + ".rec")
    if not rec_path.exists():
        raise FileNotFoundError(f"{rec_path.name} missing")
    rec = parse_rec_file(rec_path)
    meta = parse_rec_comment_metadata(rec_path)
    exposure_s = EXPOSURE_TIME_S or (rec["exposure_ms"] / 1000.0 if rec["exposure_ms"] else None)
    pixel_um = PIXEL_SIZE_UM or rec["mpp"]
    if not exposure_s or not pixel_um or not rec["size_x"]:
        raise ValueError("exposure time, pixel size or picture size not found in the .rec file")

    frames = tifffile.imread(tif_path)
    if frames.ndim == 2:
        frames = frames[None]
    if frames.shape[1:] != (rec["size_y"], rec["size_x"]):
        _warn(f"{name}: image {frames.shape[2]} x {frames.shape[1]} differs from the .rec picture size "
              f"{rec['size_x']} x {rec['size_y']}; the image size is used as FOV.")
    return {
        "file": name,
        **parse_filename(name),
        "P_mW": float(POWER_OVERRIDES_MW.get(name, meta["laser_power_mW"])),
        "frames_raw": frames,
        "exposure_s": float(exposure_s),
        "pixel_um": float(pixel_um),
        "fov_px": (int(frames.shape[2]), int(frames.shape[1])),
        "recorded_at": meta["recorded_at"],
    }


def load_all(data_dir: Path) -> list[dict]:
    measurements = []
    for tif_path in sorted(data_dir.glob(FILE_PATTERN)):
        try:
            m = load_measurement(tif_path)
        except Exception as exc:
            _warn(f"{tif_path.name}: could not be loaded ({exc}); skipped.")
            continue
        if not np.isfinite(m["P_mW"]):
            _warn(f"{m['file']}: no power in the .rec comment (set POWER_OVERRIDES_MW); relative values only.")
        measurements.append(m)
    measurements.sort(key=lambda m: (m["wavelength_nm"], m["P_mW"]))
    return measurements


def print_overview(measurements: list[dict]) -> None:
    print(f"\n{'file':<30} {'lambda':>6} {'idx':>3} {'P (mW)':>7} {'image (px)':>10} {'FOV (um)':>13} "
          f"{'px (um)':>7} {'exp (ms)':>8} {'frames':>6}  recorded")
    for m in measurements:
        nx, ny = m["fov_px"]
        print(f"{m['file']:<30} {m['wavelength_nm']:6.0f} {m['file_index']:3d} {m['P_mW']:7.3f} {nx:>4d} x {ny:<3d} "
              f"{nx * m['pixel_um']:5.1f} x {ny * m['pixel_um']:5.1f} {m['pixel_um']:7.3f} {m['exposure_s'] * 1e3:8.2f} "
              f"{len(m['frames_raw']):6d}  {m['recorded_at']}")
    print()


def load_dark_cps(path: Path | None, exposure_s: float) -> np.ndarray | None:
    if path is None:
        return None
    dark = tifffile.imread(path).astype(np.float64)
    return (dark.mean(axis=0) if dark.ndim == 3 else dark) / exposure_s


# ── Preprocess ─────────────────────────────────────────────────────────────────

def saturation_check(frames_raw: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (valid-frame mask, stuck-pixel mask); saturated frames are reported, never silently dropped."""
    saturated = frames_raw >= SATURATION_COUNTS
    stuck = saturated.mean(axis=0) > STUCK_PIXEL_FRACTION
    if stuck.any():
        _warn(f"{name}: {stuck.sum()} pixel(s) saturated in most frames, masked as stuck pixels.")
    bad_frames = (saturated & ~stuck).any(axis=(1, 2))
    if bad_frames.any():
        _warn(f"{name}: {bad_frames.sum()} of {len(bad_frames)} frames contain pixels >= {SATURATION_COUNTS} "
              f"counts (max {frames_raw.max()}); these frames are excluded.")
    return ~bad_frames, stuck


def find_hot_pixels(img: np.ndarray) -> np.ndarray:
    """Isolated outliers relative to a median-filtered image (robust MAD threshold)."""
    diff = img - median_filter(img, size=MEDIAN_FILTER_PX)
    mad = 1.4826 * np.median(np.abs(diff - np.median(diff)))
    return np.abs(diff) > HOT_PIXEL_SIGMA * max(mad, 1e-12)


def preprocess(meas: dict, dark_cps: np.ndarray | None) -> dict:
    """Counts/s, saturation and hot-pixel masks, repetition images and per-pixel noise."""
    name, exposure_s = meas["file"], meas["exposure_s"]
    valid, stuck = saturation_check(meas["frames_raw"], name)
    if not valid.any():
        raise ValueError("all frames saturated")
    cps = meas["frames_raw"][valid].astype(np.float64) / exposure_s
    if dark_cps is not None:
        if dark_cps.shape == cps.shape[1:]:
            cps -= dark_cps
        else:
            _warn(f"{name}: dark frame shape {dark_cps.shape} does not match {cps.shape[1:]}; not subtracted.")

    mean_img = cps.mean(axis=0)
    if len(cps) > 1:
        std_img = cps.std(axis=0, ddof=1)
    else:
        # Single frame: shot-noise estimate with the conversion factor of 1 e/count from the .rec header.
        std_img = np.sqrt(np.clip(mean_img * exposure_s, 1.0, None)) / exposure_s
    std_img = np.maximum(std_img, 0.1 * np.median(std_img))

    starts = list(range(0, len(cps), FRAMES_PER_REP))
    if len(starts) > 1 and len(cps) - starts[-1] < FRAMES_PER_REP / 2:
        starts.pop()                    # short remainder joins the previous block
    bounds = starts + [len(cps)]
    reps = [(cps[lo:hi].mean(axis=0), std_img / np.sqrt(hi - lo), hi - lo) for lo, hi in zip(bounds[:-1], bounds[1:])]
    return {"mean_img": mean_img, "sigma_mean": std_img / np.sqrt(len(cps)),
            "bad_px": find_hot_pixels(mean_img) | stuck, "reps": reps,
            "n_frames_valid": int(valid.sum()), "n_frames_total": int(len(valid))}


# ── Fit ────────────────────────────────────────────────────────────────────────

def gauss2d(xy, x0, y0, wx, wy, theta, I0, c):
    """Rotated elliptical Gaussian, 1/e^2 radii wx, wy (pixel units), plus constant offset c."""
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr = (x - x0) * ct + (y - y0) * st
    yr = -(x - x0) * st + (y - y0) * ct
    return c + I0 * np.exp(-2.0 * (xr ** 2 / wx ** 2 + yr ** 2 / wy ** 2))


def gauss2d_jac(xy, x0, y0, wx, wy, theta, I0, c):
    """Analytic Jacobian of gauss2d with respect to (x0, y0, wx, wy, theta, I0, c)."""
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr = (x - x0) * ct + (y - y0) * st
    yr = -(x - x0) * st + (y - y0) * ct
    e = np.exp(-2.0 * (xr ** 2 / wx ** 2 + yr ** 2 / wy ** 2))
    g = I0 * e
    ax_, ay_ = xr / wx ** 2, yr / wy ** 2
    jac = np.empty((np.size(x), 7))
    jac[:, 0] = 4.0 * g * (ax_ * ct - ay_ * st)
    jac[:, 1] = 4.0 * g * (ax_ * st + ay_ * ct)
    jac[:, 2] = 4.0 * g * xr ** 2 / wx ** 3
    jac[:, 3] = 4.0 * g * yr ** 2 / wy ** 3
    jac[:, 4] = -4.0 * g * xr * yr * (1.0 / wx ** 2 - 1.0 / wy ** 2)
    jac[:, 5] = e
    jac[:, 6] = 1.0
    return jac


def _normalize_orientation(p: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map theta into [-pi/4, pi/4]; a quarter turn swaps wx and wy (same ellipse)."""
    p, cov = p.copy(), cov.copy()
    k = int(np.round(p[4] / (np.pi / 2)))
    p[4] -= k * np.pi / 2
    if k % 2:
        order = [0, 1, 3, 2, 4, 5, 6]
        p, cov = p[order], cov[np.ix_(order, order)]
    return p, cov


def initial_guess(img: np.ndarray, bad_px: np.ndarray) -> np.ndarray:
    """Start values from first and second image moments of the median-filtered image."""
    med = median_filter(np.where(bad_px, np.median(img), img), size=MEDIAN_FILTER_PX)
    bg = np.percentile(med, 5)
    s = np.clip(med - bg, 0.0, None)
    total = s.sum()
    ny, nx = img.shape
    if total <= 0:
        return np.array([nx / 2, ny / 2, nx / 4, ny / 4, 0.0, med.max() - bg, bg])
    yy, xx = np.mgrid[:ny, :nx]
    x0, y0 = (s * xx).sum() / total, (s * yy).sum() / total
    sxx = (s * (xx - x0) ** 2).sum() / total
    syy = (s * (yy - y0) ** 2).sum() / total
    sxy = (s * (xx - x0) * (yy - y0)).sum() / total
    evals, evecs = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))
    theta = np.arctan2(evecs[1, 1], evecs[0, 1])
    w_major, w_minor = 2.0 * np.sqrt(np.clip(evals[::-1], 1.0, None))
    p, _ = _normalize_orientation(np.array([x0, y0, w_major, w_minor, theta, med.max() - bg, bg]), np.eye(7))
    return p


def roi_mask(shape: tuple[int, int], p: np.ndarray) -> np.ndarray:
    """Axis-aligned box of +- ROI_HALF_WIDTH_W * max(wx, wy) around the centre, clipped to the image."""
    ny, nx = shape
    half = ROI_HALF_WIDTH_W * max(p[2], p[3])
    yy, xx = np.mgrid[:ny, :nx]
    roi = (np.abs(xx - p[0]) <= half) & (np.abs(yy - p[1]) <= half)
    return roi if roi.sum() >= 50 else np.ones(shape, bool)


def fit_beam(img: np.ndarray, sigma: np.ndarray, bad_px: np.ndarray) -> dict:
    """
    2D Gaussian fit of one beam image (pixel units). Never raises: a failed fit
    returns status "failed" with the error message.
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx].astype(np.float64)
    scale = max(nx, ny)
    lower = np.array([-nx, -ny, 1.0, 1.0, -np.pi, 0.0, -np.inf])
    upper = np.array([2 * nx, 2 * ny, 20.0 * scale, 20.0 * scale, np.pi, np.inf, np.inf])

    try:
        p = np.clip(initial_guess(img, bad_px), lower + 1e-6, upper - 1e-6)
        use = ~bad_px
        n_clipped = 0
        for stage in range(2):
            roi = roi_mask(img.shape, p) & use
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                p, cov = curve_fit(gauss2d, (xx[roi], yy[roi]), img[roi], p0=p, sigma=sigma[roi],
                                   absolute_sigma=True, jac=gauss2d_jac, bounds=(lower, upper),
                                   method="trf", x_scale="jac", max_nfev=MAX_NFEV)
            if stage == 0:
                # Mask residual outliers (dust, cosmic rays) with a robust scale before the final fit.
                r = (img - gauss2d((xx, yy), *p)) / sigma
                r_roi = r[roi_mask(img.shape, p) & use]
                spread = 1.4826 * np.median(np.abs(r_roi - np.median(r_roi)))
                outlier = np.abs(r - np.median(r_roi)) > RESIDUAL_CLIP_SIGMA * spread
                n_clipped = int((outlier & use).sum())
                use &= ~outlier
    except Exception as exc:
        return {"status": "failed", "message": f"{type(exc).__name__}: {exc}"}

    roi_fit = roi_mask(img.shape, p) & use
    model = gauss2d((xx, yy), *p)
    resid = img[roi_fit] - model[roi_fit]
    dof = max(int(roi_fit.sum()) - 7, 1)
    chi2_red = float(np.sum((resid / sigma[roi_fit]) ** 2) / dof)
    p, cov = _normalize_orientation(p, cov * max(1.0, chi2_red))
    x0, y0, wx, wy, theta, I0, c = p

    S_fit = np.pi / 2.0 * I0 * wx * wy
    grad = np.pi / 2.0 * np.array([0.0, 0.0, I0 * wy, I0 * wx, 0.0, wx * wy, 0.0])
    filled = np.where(bad_px, median_filter(img, size=MEDIAN_FILTER_PX), img)
    S_sum = float(np.sum(filled - c))                 # pixel sum over the whole image (= FOV)
    S_model_fov = float(np.sum(model - c))

    flags = []
    if not (0 <= x0 < nx and 0 <= y0 < ny):
        flags.append("center_outside")
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((img[roi_fit] - img[roi_fit].mean()) ** 2)
    if r2 < MIN_R2:
        flags.append("low_R2")
    if S_model_fov / S_fit < MIN_FRACTION_IN_FOV:
        flags.append("beam_clipped")
    return {
        "status": "ok", "p": p, "perr": np.sqrt(np.clip(np.diag(cov), 0.0, None)), "cov": cov,
        "R2": float(r2), "chi2_red": chi2_red,
        "resid_rms_rel": float(np.sqrt(np.mean(resid ** 2)) / I0) if I0 > 0 else np.nan,
        "n_px_clipped": n_clipped,
        "S_fit": float(S_fit), "S_fit_err": float(np.sqrt(max(grad @ cov @ grad, 0.0))),
        "S_sum": S_sum, "S_model_fov": S_model_fov,
        "dev_pct": 100.0 * (S_sum - S_fit) / S_fit,
        "dev_fov_pct": 100.0 * (S_sum - S_model_fov) / S_model_fov,
        "flags": flags,
    }


# ── Absolute intensity and metrics ─────────────────────────────────────────────

def absolute_maps(img: np.ndarray, bad_px: np.ndarray, fit: dict, P_mW: float, pixel_um: float):
    """
    Intensity maps in W/cm^2: profile normalised to integral 1 (Gaussian total S),
    multiplied by P. Returns (I_fit, I_meas, scale) with scale = W/cm^2 per count/s.
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx].astype(np.float64)
    c = fit["p"][6]
    scale = P_mW * 1e-3 / (fit["S_fit"] * (pixel_um * 1e-4) ** 2)
    filled = np.where(bad_px, median_filter(img, size=MEDIAN_FILTER_PX), img)
    return scale * (gauss2d((xx, yy), *fit["p"]) - c), scale * (filled - c), scale


def peak_intensity(P_mW: float, wx_px: float, wy_px: float, pixel_um: float) -> float:
    """I_peak = 2 P / (pi wx wy) in W/cm^2."""
    return 2.0 * P_mW * 1e-3 / (np.pi * wx_px * wy_px * (pixel_um * 1e-4) ** 2)


def edge_points(shape: tuple[int, int]) -> dict:
    """Pixel indices (ix, iy) of the FOV edge centres and corners."""
    ny, nx = shape
    xc, yc = (nx - 1) // 2, (ny - 1) // 2
    return {"left": (0, yc), "right": (nx - 1, yc), "top": (xc, 0), "bottom": (xc, ny - 1),
            "top_left": (0, 0), "top_right": (nx - 1, 0), "bottom_left": (0, ny - 1), "bottom_right": (nx - 1, ny - 1)}


def map_metrics(I_map: np.ndarray, peak: float, pixel_um: float, P_mW: float, robust: bool) -> dict:
    """Edge intensities, FOV statistics and FOV power of one absolute map (suffix-free keys)."""
    smooth = median_filter(I_map, size=MEDIAN_FILTER_PX) if robust else I_map
    out = {}
    h = EDGE_BOX_PX // 2
    for name, (ix, iy) in edge_points(I_map.shape).items():
        if robust:
            val = float(np.median(I_map[max(iy - h, 0):iy + h + 1, max(ix - h, 0):ix + h + 1]))
        else:
            val = float(I_map[iy, ix])
        out[f"I_{name}_W_cm2"] = val
        out[f"I_{name}_rel_pct"] = 100.0 * val / peak
    edge_rel = [out[f"I_{n}_rel_pct"] for n in EDGES[:4]]
    corner_rel = [out[f"I_{n}_rel_pct"] for n in EDGES[4:]]
    P_fov_mW = float(I_map.sum() * (pixel_um * 1e-4) ** 2 * 1e3)
    out.update({
        "I_peak_map_W_cm2": peak,
        "edge_mean_rel_pct": float(np.mean(edge_rel)), "edge_min_rel_pct": float(np.min(edge_rel)),
        "corner_mean_rel_pct": float(np.mean(corner_rel)), "corner_min_rel_pct": float(np.min(corner_rel)),
        "I_fov_mean_W_cm2": float(I_map.mean()),
        "I_fov_min_W_cm2": float(smooth.min()), "I_fov_max_W_cm2": float(smooth.max()),
        "fov_min_max_ratio": float(smooth.min() / smooth.max()),
        "fov_cv": float(smooth.std() / smooth.mean()),
        "P_fov_mW": P_fov_mW, "fov_power_fraction": P_fov_mW / P_mW,
    })
    return out


def compute_row(meas: dict, rep, n_frames: int, img: np.ndarray, bad_px: np.ndarray, fit: dict) -> dict:
    """One row of the per-image table: fit parameters in physical units and all intensity metrics."""
    px, P = meas["pixel_um"], meas["P_mW"]
    t_s = ENERGY_TIME_S or meas["exposure_s"]
    nx, ny = meas["fov_px"]
    row = {"wavelength_nm": meas["wavelength_nm"], "P_mW": P, "file": meas["file"], "file_index": meas["file_index"],
           "rep": rep, "n_frames": n_frames, "fov_x_um": nx * px, "fov_y_um": ny * px, "pixel_um": px}
    if fit["status"] != "ok":
        row.update({"flag": "fit_failed", "fit_message": fit["message"]})
        return row
    (x0, y0, wx, wy, th, I0, c), e = fit["p"], fit["perr"]
    I_peak = peak_intensity(P, wx, wy, px)
    row.update({
        "x0_um": x0 * px, "y0_um": y0 * px,
        "decenter_x_um": (x0 - (nx - 1) / 2) * px, "decenter_y_um": (y0 - (ny - 1) / 2) * px,
        "wx_um": wx * px, "wx_err_um": e[2] * px, "wy_um": wy * px, "wy_err_um": e[3] * px,
        "fwhm_x_um": FWHM_PER_W * wx * px, "fwhm_y_um": FWHM_PER_W * wy * px,
        "theta_deg": np.degrees(th), "theta_err_deg": np.degrees(e[4]),
        "I0_cps": I0, "I0_err_cps": e[5], "c_cps": c, "c_err_cps": e[6],
        "S_fit_cps": fit["S_fit"], "S_fit_err_cps": fit["S_fit_err"], "S_sum_cps": fit["S_sum"],
        "dev_pct": fit["dev_pct"], "dev_fov_pct": fit["dev_fov_pct"],
        "R2": fit["R2"], "chi2_red": fit["chi2_red"], "resid_rms_rel": fit["resid_rms_rel"],
        "n_px_clipped": fit["n_px_clipped"],
        "I_peak_W_cm2": I_peak,
        "I_peak_err_W_cm2": I_peak * np.sqrt((e[2] / wx) ** 2 + (e[3] / wy) ** 2 + POWER_REL_UNC ** 2),
        "t_s": t_s, "E_mJ": P * t_s, "F_peak_J_cm2": I_peak * t_s,
    })
    row["decenter_um"] = float(np.hypot(row["decenter_x_um"], row["decenter_y_um"]))
    I_fit, I_meas, _ = absolute_maps(img, bad_px, fit, P, px)
    meas_peak = float(median_filter(I_meas, MEDIAN_FILTER_PX).max())
    for suffix, I_map, peak, robust in (("fit", I_fit, I_peak, False), ("meas", I_meas, meas_peak, True)):
        for key, val in map_metrics(I_map, peak, px, P, robust).items():
            row[f"{key}_{suffix}"] = val
        row[f"F_fov_mean_J_cm2_{suffix}"] = row[f"I_fov_mean_W_cm2_{suffix}"] * t_s
    row["flag"] = ";".join(fit["flags"])
    return row


def analyse_measurement(meas: dict, dark_cps: np.ndarray | None) -> tuple[list[dict], dict]:
    """Fit every repetition of one stack and its mean image (mean image: source selection and figures)."""
    pre = preprocess(meas, dark_cps)
    rows = []
    for i, (img, sigma, n) in enumerate(pre["reps"]):
        fit = fit_beam(img, sigma, pre["bad_px"])
        if fit["status"] != "ok":
            _warn(f"{meas['file']} rep {i}: fit failed ({fit['message']}).")
        rows.append(compute_row(meas, i, n, img, pre["bad_px"], fit))
    mean_fit = fit_beam(pre["mean_img"], pre["sigma_mean"], pre["bad_px"])
    if mean_fit["status"] != "ok":
        _warn(f"{meas['file']}: fit of the mean image failed ({mean_fit['message']}).")
        source = "measured"
    else:
        source = "measured" if mean_fit["resid_rms_rel"] > NON_GAUSSIAN_RESID_REL else "gaussian"
    detail = {"meas": {k: v for k, v in meas.items() if k != "frames_raw"}, "mean_img": pre["mean_img"],
              "bad_px": pre["bad_px"], "mean_fit": mean_fit, "profile_source": source,
              "n_frames_valid": pre["n_frames_valid"], "n_frames_total": pre["n_frames_total"]}
    return rows, detail


# ── Calibrate ──────────────────────────────────────────────────────────────────

def _add_flag(df: pd.DataFrame, mask: pd.Series, flag: str) -> None:
    df.loc[mask, "flag"] = df.loc[mask, "flag"].apply(lambda f: f"{f};{flag}" if f else flag)


def apply_series_flags(per_image: pd.DataFrame) -> None:
    """Flag repetitions whose beam radius deviates strongly from the median of their wavelength series."""
    w_eff = np.sqrt(per_image["wx_um"] * per_image["wy_um"])
    ok = per_image["flag"] == ""
    file_median = w_eff[ok].groupby([per_image["wavelength_nm"][ok], per_image["file"][ok]]).median()
    series_median = file_median.groupby(level=0).median()
    ref = per_image["wavelength_nm"].map(series_median)
    _add_flag(per_image, (w_eff / ref - 1.0).abs() > W_OUTLIER_REL, "w_outlier")


def summarize_settings(per_image: pd.DataFrame, sources: dict[str, str]) -> pd.DataFrame:
    """
    Mean +- SD per (wavelength, P) over valid repetitions; if none is valid,
    over all fitted repetitions (stats_basis = "all_fitted"). Metrics that
    exist for both maps also get "_sel" columns from the selected profile source.
    """
    numeric = [c for c in per_image.columns if per_image[c].dtype.kind in "fi"
               and c not in ("wavelength_nm", "P_mW", "rep", "file_index")]
    rows = []
    for (wl, P), g in per_image.groupby(["wavelength_nm", "P_mW"], dropna=False, sort=True):
        valid = g[g["flag"] == ""]
        fitted = g[g["flag"] != "fit_failed"]
        basis, use = ("valid", valid) if len(valid) else ("all_fitted", fitted)
        files = sorted(g["file"].unique())
        source = "measured" if any(sources.get(f) == "measured" for f in files) else "gaussian"
        flags = sorted({f for fl in g["flag"] for f in fl.split(";") if f})
        if len(valid) < MIN_VALID_REPS:
            flags.append("few_valid_reps")
        row = {"wavelength_nm": wl, "P_mW": P, "files": ";".join(files), "n_reps": len(g), "n_valid": len(valid),
               "included": bool(len(valid) >= MIN_VALID_REPS and np.isfinite(P)), "stats_basis": basis,
               "profile_source": source, "flags": ";".join(flags)}
        for col in numeric:
            row[f"{col}_mean"] = use[col].mean() if len(use) else np.nan
            row[f"{col}_sd"] = use[col].std(ddof=1) if len(use) > 1 else np.nan
        suffix = "meas" if source == "measured" else "fit"
        for col in numeric:
            if col.endswith(f"_{suffix}"):
                base = col[: -len(suffix) - 1]
                row[f"{base}_sel_mean"], row[f"{base}_sel_sd"] = row[f"{col}_mean"], row[f"{col}_sd"]
        rows.append(row)
    return pd.DataFrame(rows)


def _linear_threshold(P, a, P0):
    return a * (P - P0)


def _proportional(P, a):
    return a * P


def calibrate_wavelength(st: pd.DataFrame, wavelength_nm: float) -> dict:
    """I_peak(P) model with inverse, FOV factors and beam-radius constancy for one wavelength."""
    used = st[st["included"]].sort_values("P_mW")
    entry = {"wavelength_nm": wavelength_nm, "n_settings": int(len(used)),
             "settings_used_mW": used["P_mW"].tolist(),
             "settings_excluded": {f"{r.P_mW:g} mW": r.flags for r in st[~st["included"]].itertuples()}}
    if used.empty:
        _warn(f"{wavelength_nm:g} nm: no valid setting; no calibration.")
        return entry

    x = used["P_mW"].to_numpy(float)
    y = used["I_peak_W_cm2_mean"].to_numpy(float)
    sem = np.nan_to_num(used["I_peak_W_cm2_sd"].to_numpy(float) / np.sqrt(used["n_valid"].to_numpy(float)))
    sy = np.sqrt(sem ** 2 + (POWER_REL_UNC * y) ** 2)
    if len(x) >= 3:
        popt, pcov = curve_fit(_linear_threshold, x, y, p0=[y.mean() / x.mean(), 0.0], sigma=sy, absolute_sigma=True)
        (a, P0), (a_err, P0_err) = popt, np.sqrt(np.diag(pcov))
        y_fit, dof, name = _linear_threshold(x, a, P0), len(x) - 2, "linear_threshold"
    else:
        popt, pcov = curve_fit(_proportional, x, y, p0=[y[0] / x[0]], sigma=sy, absolute_sigma=True)
        a, a_err, P0, P0_err = popt[0], float(np.sqrt(pcov[0, 0])), 0.0, np.nan
        y_fit, dof, name = _proportional(x, a), len(x) - 1, "proportional"
        _warn(f"{wavelength_nm:g} nm: {len(x)} valid setting(s); proportional model I_peak = a P "
              "instead of the threshold model.")
    norm_resid = (y - y_fit) / sy
    chi2_red = float(np.sum(norm_resid ** 2) / dof) if dof > 0 else np.nan
    systematic = bool(dof > 0 and (chi2_red > SYSTEMATIC_CHI2_RED or np.max(np.abs(norm_resid)) > SYSTEMATIC_RESID_SIGMA))
    use_pchip = bool(systematic and np.all(np.diff(y) > 0))
    if systematic:
        _warn(f"{wavelength_nm:g} nm: systematic residuals of the I_peak(P) model (chi2_red = {chi2_red:.2f}); "
              + ("PCHIP interpolation used for conversions." if use_pchip else "I_peak not monotonic, model kept."))
    entry["peak_model"] = {"name": name, "equation": "I_peak_W_cm2 = a * (P_mW - P0_mW)", "a_W_cm2_per_mW": a,
                           "a_err_W_cm2_per_mW": a_err, "P0_mW": P0, "P0_err_mW": P0_err, "dof": dof,
                           "chi2_red": chi2_red, "residuals_norm": norm_resid.tolist(),
                           "systematic_residuals": systematic, "use_pchip": use_pchip,
                           "P_nodes_mW": x.tolist(), "I_peak_nodes_W_cm2": y.tolist()}

    beam = {}
    for col in ("wx", "wy", "fwhm_x", "fwhm_y"):
        means = used[f"{col}_um_mean"].to_numpy(float)
        mean = float(means.mean())
        sd = float(means.std(ddof=1)) if len(means) > 1 else float(np.nan_to_num(used[f"{col}_um_sd"].iloc[0]))
        beam[f"{col}_um"], beam[f"{col}_sd_um"], beam[f"{col}_cv"] = mean, sd, sd / mean
        if len(means) >= 3 and col in ("wx", "wy"):
            lr = stats.linregress(x, means)
            beam[f"{col}_slope_um_per_mW"], beam[f"{col}_slope_p"] = float(lr.slope), float(lr.pvalue)
    beam["theta_deg"] = float(used["theta_deg_mean"].mean())
    trend = [k for k in ("wx", "wy") if beam.get(f"{k}_slope_p", 1.0) < 0.05]
    spread = [k for k in ("wx", "wy") if beam[f"{k}_cv"] > W_CV_WARN]
    beam["constant"] = not (trend or spread)
    if not beam["constant"]:
        _warn(f"{wavelength_nm:g} nm: beam radius not constant over P (CV > {W_CV_WARN:.0%}: {spread or 'none'}; "
              f"significant trend: {trend or 'none'}).")
    entry["beam"] = beam

    entry.update({
        "fov_mean_W_cm2_per_mW": float(np.mean(used["I_fov_mean_W_cm2_sel_mean"] / used["P_mW"])),
        "fov_power_fraction": float(used["fov_power_fraction_sel_mean"].mean()),
        "edge_mean_rel_pct": float(used["edge_mean_rel_pct_sel_mean"].mean()),
        "corner_mean_rel_pct": float(used["corner_mean_rel_pct_sel_mean"].mean()),
        "fov_um": [float(used["fov_x_um_mean"].iloc[0]), float(used["fov_y_um_mean"].iloc[0])],
        "t_s": float(used["t_s_mean"].iloc[0]),
        "P_range_mW": [float(x.min()), float(x.max())],
    })
    if used["fov_x_um_mean"].nunique() > 1 or used["fov_y_um_mean"].nunique() > 1:
        _warn(f"{wavelength_nm:g} nm: FOV size differs between settings; FOV factors are averages over different FOVs.")
    return entry


def calibrate(per_setting: pd.DataFrame, details: dict) -> dict:
    cal = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "data_dir": str(DATA_DIR),
        "setting_axis": "P_mW: sample-plane power measured with the sensor after the objective (.rec comment)",
        "power_rel_unc": POWER_REL_UNC,
        "beam_model": "I = c + I0 exp(-2 (x'^2/wx^2 + y'^2/wy^2)), 1/e^2 radii, rotation theta",
        "definitions": {
            "I_map": "P (profile - c) / (S A_px), S = (pi/2) I0 wx wy (Gaussian total integral)",
            "I_peak_W_cm2": "2 P / (pi wx wy)", "fwhm_um": "w sqrt(2 ln 2)", "E_mJ": "P t", "F_J_cm2": "I t",
            "edge_points": "pixel at the centre of each FOV edge and at each corner; measured map: median of a "
                           f"{EDGE_BOX_PX} x {EDGE_BOX_PX} px box clipped at the edge",
            "_sel": f"measured map if residual RMS / I0 > {NON_GAUSSIAN_RESID_REL} in the file mean image, else Gaussian",
        },
        "frames_per_rep": FRAMES_PER_REP,
        "files": {},
        "wavelengths": {},
    }
    for name, d in details.items():
        f = d["mean_fit"]
        entry = {"wavelength_nm": d["meas"]["wavelength_nm"], "P_mW": d["meas"]["P_mW"],
                 "pixel_um": d["meas"]["pixel_um"], "fov_px": d["meas"]["fov_px"],
                 "exposure_s": d["meas"]["exposure_s"], "profile_source": d["profile_source"],
                 "n_frames_valid": d["n_frames_valid"], "n_frames_total": d["n_frames_total"]}
        if f["status"] == "ok":
            entry["mean_image_fit"] = {**dict(zip(PARAM_NAMES, f["p"])),
                                       **{f"{n}_err": v for n, v in zip(PARAM_NAMES, f["perr"])},
                                       "units": "pixel, rad, counts/s", "R2": f["R2"], "chi2_red": f["chi2_red"],
                                       "resid_rms_rel": f["resid_rms_rel"], "flags": f["flags"]}
        cal["files"][name] = entry
    for wl, st in per_setting.groupby("wavelength_nm"):
        cal["wavelengths"][f"{wl:g}"] = calibrate_wavelength(st, float(wl))
    return cal


# ── Functions for import ───────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _read_json(path_str: str) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_results(path: Path | str | None = None) -> dict:
    """Results written by this script (default RESULT_JSON)."""
    path = Path(path or RESULT_JSON)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run laser_intensity_calibration.py first.")
    return _read_json(str(path.resolve()))


def _entry(wavelength_nm: float, path) -> dict:
    cal = load_results(path)
    entry = cal["wavelengths"].get(f"{float(wavelength_nm):g}")
    if entry is None or "peak_model" not in entry:
        raise KeyError(f"no calibration for {wavelength_nm} nm; available: {list(cal['wavelengths'])}")
    return entry


def _peak_forward(m: dict, P: np.ndarray) -> np.ndarray:
    if m["use_pchip"]:
        return PchipInterpolator(m["P_nodes_mW"], m["I_peak_nodes_W_cm2"], extrapolate=False)(P)
    return m["a_W_cm2_per_mW"] * (P - m["P0_mW"])


def peak_intensity_for_power(P_mW, wavelength_nm: float, path=None) -> np.ndarray:
    """Peak intensity (W/cm^2) in the sample plane for a sample-plane power P (mW)."""
    return _peak_forward(_entry(wavelength_nm, path)["peak_model"], np.asarray(P_mW, float))


def power_for_peak_intensity(I_W_cm2, wavelength_nm: float, path=None) -> np.ndarray:
    """Sample-plane power P (mW) that gives the peak intensity I (W/cm^2)."""
    m = _entry(wavelength_nm, path)["peak_model"]
    I = np.asarray(I_W_cm2, float)
    if m["use_pchip"]:
        grid = np.linspace(min(m["P_nodes_mW"]), max(m["P_nodes_mW"]), 4001)
        return np.interp(I, _peak_forward(m, grid), grid, left=np.nan, right=np.nan)
    return I / m["a_W_cm2_per_mW"] + m["P0_mW"]


def fov_mean_intensity_for_power(P_mW, wavelength_nm: float, path=None) -> np.ndarray:
    """FOV-averaged intensity (W/cm^2) for a sample-plane power P (mW), FOV of the calibration images."""
    return _entry(wavelength_nm, path)["fov_mean_W_cm2_per_mW"] * np.asarray(P_mW, float)


def fluence_for_power(P_mW, t_s: float, wavelength_nm: float, path=None) -> np.ndarray:
    """Peak fluence F = I_peak t (J/cm^2) for a sample-plane power P (mW) and duration t (s)."""
    return peak_intensity_for_power(P_mW, wavelength_nm, path) * t_s


# ── Synthetic test ─────────────────────────────────────────────────────────────

SYNTHETIC_CASES = {
    # Like the 488 nm data: beam larger than half the FOV, nearly round, clipped by the image edge.
    "clipped_round": dict(shape=(150, 200), p=[88.0, 80.0, 122.0, 107.0, 0.08, 4200.0, 3000.0], n_frames=10),
    # Small, elliptical and rotated beam fully inside the image.
    "small_rotated": dict(shape=(150, 200), p=[95.3, 70.6, 30.0, 18.0, 0.45, 3000.0, 1000.0], n_frames=10),
}
SYNTHETIC_P_MW = 2.0
SYNTHETIC_PIXEL_UM = 0.30
SYNTHETIC_READ_NOISE = 10.0     # counts per frame
SYNTHETIC_HOT_PIXELS = 20


def make_synthetic_beam(shape, p, n_frames, rng) -> tuple[np.ndarray, np.ndarray]:
    """Mean of n_frames frames (counts): Gaussian beam + offset, shot and read noise, hot pixels."""
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx].astype(np.float64)
    model = gauss2d((xx, yy), *p)
    sigma = np.sqrt(SYNTHETIC_READ_NOISE ** 2 + model) / np.sqrt(n_frames)
    img = model + rng.normal(0.0, sigma)
    idx = rng.choice(img.size, SYNTHETIC_HOT_PIXELS, replace=False)
    img.flat[idx] += rng.uniform(2000.0, 10000.0, SYNTHETIC_HOT_PIXELS)
    return img, sigma


def run_synthetic_test() -> pd.DataFrame:
    """
    Fit SYNTHETIC_REALIZATIONS noisy realisations per case through fit_beam and
    the absolute-intensity step and compare with the true values. Pull =
    (fit - true) / covariance error; unbiased fit with correct errors: mean ~ 0, SD ~ 1.
    """
    rng = np.random.default_rng(SYNTHETIC_SEED)
    px, P = SYNTHETIC_PIXEL_UM, SYNTHETIC_P_MW
    area_mW = (px * 1e-4) ** 2 * 1e3
    rows = []
    print(f"Synthetic test: {SYNTHETIC_REALIZATIONS} realisations per case, P = {P} mW, pixel {px} um")
    for case, spec in SYNTHETIC_CASES.items():
        p_true = np.array(spec["p"])
        ny, nx = spec["shape"]
        yy, xx = np.mgrid[:ny, :nx].astype(np.float64)
        S_true = np.pi / 2 * p_true[5] * p_true[2] * p_true[3]
        frac_true = float(np.sum(gauss2d((xx, yy), *p_true) - p_true[6]) / S_true)
        I_peak_true = peak_intensity(P, p_true[2], p_true[3], px)
        fits, errs, I_peaks, frac_fit, frac_meas, n_failed = [], [], [], [], [], 0
        for _ in range(SYNTHETIC_REALIZATIONS):
            img, sigma = make_synthetic_beam(spec["shape"], p_true, spec["n_frames"], rng)
            bad = find_hot_pixels(img)
            fit = fit_beam(img, sigma, bad)
            if fit["status"] != "ok":
                n_failed += 1
                continue
            I_fit, I_meas, _ = absolute_maps(img, bad, fit, P, px)
            fits.append(fit["p"])
            errs.append(fit["perr"])
            I_peaks.append(peak_intensity(P, fit["p"][2], fit["p"][3], px))
            frac_fit.append(I_fit.sum() * area_mW / P)
            frac_meas.append(I_meas.sum() * area_mW / P)
        fits, errs = np.array(fits), np.array(errs)
        n = len(fits)
        print(f"  {case}: {n} fits, {n_failed} failed")
        print(f"    I_peak: true {I_peak_true:.3f} W/cm2, fit {np.mean(I_peaks):.3f} +- {np.std(I_peaks):.3f}")
        print(f"    power fraction in FOV: true {frac_true:.4f}, Gaussian map {np.mean(frac_fit):.4f} +- "
              f"{np.std(frac_fit):.4f}, measured map {np.mean(frac_meas):.4f} +- {np.std(frac_meas):.4f}")
        print(f"    {'param':>6} {'true':>10} {'mean fit':>10} {'SD fit':>9} {'mean err':>9} "
              f"{'pull mean':>9} {'pull SD':>8}  result")
        for j, name in enumerate(PARAM_NAMES):
            pulls = (fits[:, j] - p_true[j]) / errs[:, j]
            ok = abs(pulls.mean()) < 3.0 / np.sqrt(n) and 0.7 < pulls.std(ddof=1) < 1.4
            rows.append({"case": case, "quantity": name, "true": p_true[j], "mean_fit": fits[:, j].mean(),
                         "sd_fit": fits[:, j].std(ddof=1), "mean_err": errs[:, j].mean(), "pull_mean": pulls.mean(),
                         "pull_sd": pulls.std(ddof=1), "n_fits": n, "n_failed": n_failed, "pass": bool(ok)})
            print(f"    {name:>6} {p_true[j]:10.4g} {fits[:, j].mean():10.4g} {fits[:, j].std(ddof=1):9.3g} "
                  f"{errs[:, j].mean():9.3g} {pulls.mean():9.2f} {pulls.std(ddof=1):8.2f}  {'pass' if ok else 'CHECK'}")
        for name, true, vals in (("I_peak_W_cm2", I_peak_true, I_peaks), ("fov_power_fraction_fit", frac_true, frac_fit),
                                 ("fov_power_fraction_meas", frac_true, frac_meas)):
            vals = np.array(vals)
            ok = abs(vals.mean() - true) < max(3 * vals.std(ddof=1) / np.sqrt(n), 1e-3 * abs(true))
            rows.append({"case": case, "quantity": name, "true": true, "mean_fit": vals.mean(),
                         "sd_fit": vals.std(ddof=1), "n_fits": n, "n_failed": n_failed, "pass": bool(ok)})
    return pd.DataFrame(rows)


# ── Plot ───────────────────────────────────────────────────────────────────────

def new_figure(kind: str = "half", nrows: int = 1, ncols: int = 1, height: float | None = None, **kw):
    width = WIDTH_IN[kind]
    return plt.subplots(nrows, ncols, figsize=(width, height if height else width / RATIO), **kw)


def save_figure(fig: plt.Figure, stem: str, raster: bool = False) -> Path:
    """Plots as PDF, images (raster=True) as PNG at 600 dpi; never bbox_inches="tight"."""
    fmt = "png" if raster else "pdf"
    path = OUTPUT_DIR / f"{stem}.{fmt}"
    fig.savefig(path, format=fmt, dpi=600)
    plt.close(fig)
    return path


def _stem(file: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Path(file).stem.lower()).strip("_")


def _colors(wavelength_nm: float) -> tuple[str, str]:
    return COLOR_WAVELENGTH.get(int(round(wavelength_nm)) if np.isfinite(wavelength_nm) else -1, COLOR_FALLBACK)


def _draw_scalebar(ax: plt.Axes, x_right: float, y_bottom: float, height_px: float, pixel_um: float,
                   length_um: float, color: str = "white") -> None:
    length_px = length_um / pixel_um
    h = 0.012 * height_px
    ax.add_patch(Rectangle((x_right - length_px, y_bottom - h), length_px, h, color=color, lw=0, zorder=5))
    ax.text(x_right - length_px / 2, y_bottom - h - 0.015 * height_px, f"{length_um:g} µm", color=color,
            ha="center", va="bottom", fontsize=8, zorder=5)


def plot_intensity_map(detail: dict) -> Path:
    """A: absolute intensity map (measured inside the FOV, dimmed Gaussian outside), 1/e^2 ellipse, FOV frame."""
    meas, fit = detail["meas"], detail["mean_fit"]
    px, (nx, ny) = meas["pixel_um"], meas["fov_px"]
    x0, y0, wx, wy, theta, _, c = fit["p"]
    _, I_meas, scale = absolute_maps(detail["mean_img"], detail["bad_px"], fit, meas["P_mW"], px)

    hx = np.hypot(wx * np.cos(theta), wy * np.sin(theta)) * MAP_EXTEND_W
    hy = np.hypot(wx * np.sin(theta), wy * np.cos(theta)) * MAP_EXTEND_W
    xmin = max(min(-0.5, x0 - hx), -0.5 - MAP_MAX_EXTEND_FOV * nx)
    xmax = min(max(nx - 0.5, x0 + hx), nx - 0.5 + MAP_MAX_EXTEND_FOV * nx)
    ymin = max(min(-0.5, y0 - hy), -0.5 - MAP_MAX_EXTEND_FOV * ny)
    ymax = min(max(ny - 0.5, y0 + hy), ny - 0.5 + MAP_MAX_EXTEND_FOV * ny)
    xs = np.arange(np.floor(xmin + 0.5), np.ceil(xmax - 0.5) + 1)
    ys = np.arange(np.floor(ymin + 0.5), np.ceil(ymax - 0.5) + 1)
    gx, gy = np.meshgrid(xs, ys)
    canvas = scale * (gauss2d((gx, gy), *fit["p"]) - c)

    vmax = max(np.percentile(median_filter(I_meas, MEDIAN_FILTER_PX), 99.9), canvas.max())
    fig, ax = new_figure("half")
    kw = dict(cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest", origin="upper")
    ax.imshow(canvas, extent=(xs[0] - 0.5, xs[-1] + 0.5, ys[-1] + 0.5, ys[0] - 0.5), alpha=0.4, **kw)
    im = ax.imshow(I_meas, extent=(-0.5, nx - 0.5, ny - 0.5, -0.5), **kw)
    ax.add_patch(Rectangle((-0.5, -0.5), nx, ny, fill=False, ec="black", lw=1.0, zorder=4))
    ax.add_patch(Ellipse((x0, y0), 2 * wx, 2 * wy, angle=np.degrees(theta), fill=False, ec=COLOR_ACCENT,
                         lw=1.2, zorder=4))
    ax.plot(x0, y0, "+", color=COLOR_ACCENT, ms=6, mew=1.0, zorder=4)
    _draw_scalebar(ax, nx - 0.5 - 0.05 * nx, ny - 0.5 - 0.05 * ny, ny, px, SCALEBAR_UM)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_ylim(ys[-1] + 0.5, ys[0] - 0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label(r"Intensity $I$ (W cm$^{-2}$)")
    return save_figure(fig, f"intensity_map_{_stem(meas['file'])}", raster=True)


def plot_line_profiles(detail: dict) -> Path:
    """B: x/y line profiles through the beam centre (absolute), Gaussian fit extended beyond the FOV, residuals."""
    meas, fit = detail["meas"], detail["mean_fit"]
    px, (nx, ny) = meas["pixel_um"], meas["fov_px"]
    x0, y0, c = fit["p"][0], fit["p"][1], fit["p"][6]
    I_fit, I_meas, scale = absolute_maps(detail["mean_img"], detail["bad_px"], fit, meas["P_mW"], px)
    yc, xc = int(np.clip(round(y0), 0, ny - 1)), int(np.clip(round(x0), 0, nx - 1))
    rows = slice(max(yc - PROFILE_BAND_PX, 0), yc + PROFILE_BAND_PX + 1)
    cols = slice(max(xc - PROFILE_BAND_PX, 0), xc + PROFILE_BAND_PX + 1)
    reach = 1.3 * max(fit["p"][2], fit["p"][3])
    fine_x = np.linspace(min(-0.5, x0 - reach), max(nx - 0.5, x0 + reach), 600)
    fine_y = np.linspace(min(-0.5, y0 - reach), max(ny - 0.5, y0 + reach), 600)
    band = np.arange(-PROFILE_BAND_PX, PROFILE_BAND_PX + 1)
    model_x = np.mean([scale * (gauss2d((fine_x, np.full_like(fine_x, yc + b)), *fit["p"]) - c) for b in band], axis=0)
    model_y = np.mean([scale * (gauss2d((np.full_like(fine_y, xc + b), fine_y), *fit["p"]) - c) for b in band], axis=0)
    profiles = [
        ("x", (np.arange(nx) - x0) * px, I_meas[rows].mean(axis=0), I_fit[rows].mean(axis=0),
         (fine_x - x0) * px, model_x, ((-0.5 - x0) * px, (nx - 0.5 - x0) * px), COLOR_X),
        ("y", (np.arange(ny) - y0) * px, I_meas[:, cols].mean(axis=1), I_fit[:, cols].mean(axis=1),
         (fine_y - y0) * px, model_y, ((-0.5 - y0) * px, (ny - 0.5 - y0) * px), COLOR_Y),
    ]
    fig, (ax, axr) = new_figure("half", 2, 1, sharex=True, gridspec_kw={"height_ratios": [2.6, 1]})
    peak = 0.0
    for label, pos, data, mod_px, pos_fine, mod_fine, fov, (base, dark) in profiles:
        ax.plot(pos, data, color=base, lw=1.0, label=f"{label} profile")
        ax.plot(pos_fine, mod_fine, color=dark, lw=1.5)
        axr.plot(pos, data - mod_px, color=base, lw=1.0)
        for bound in fov:
            for a in (ax, axr):
                a.axvline(bound, color=dark, lw=0.8, ls=(0, (4, 3)), zorder=1)
        peak = max(peak, np.nanmax(data), mod_fine.max())
    axr.axhline(0.0, color="k", lw=0.8)
    ax.set_ylim(0.0, 1.3 * peak)                      # head room for the legend
    ax.set_ylabel(r"$I$ (W cm$^{-2}$)")
    axr.set_ylabel("Resid.")
    axr.set_xlabel("Position relative to beam center (µm)")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.25", lw=1.5, label="Gaussian fit"))
    ax.legend(handles=handles, loc="upper right", handlelength=1.2, borderaxespad=0.3)
    fig.align_ylabels((ax, axr))
    return save_figure(fig, f"line_profiles_{_stem(meas['file'])}")


def plot_residual_map(detail: dict) -> Path:
    """Residual map (image - fit) of the mean image in % of I0."""
    meas, fit = detail["meas"], detail["mean_fit"]
    img = np.where(detail["bad_px"], median_filter(detail["mean_img"], MEDIAN_FILTER_PX), detail["mean_img"])
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx].astype(np.float64)
    resid = 100.0 * (img - gauss2d((xx, yy), *fit["p"])) / fit["p"][5]
    lim = np.percentile(np.abs(resid), 99.5)
    fig, ax = new_figure("half")
    im = ax.imshow(resid, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    _draw_scalebar(ax, nx - 0.5 - 0.05 * nx, ny - 0.5 - 0.05 * ny, ny, meas["pixel_um"], SCALEBAR_UM, color="black")
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label(r"Residual (% of $I_0$)")
    return save_figure(fig, f"residual_map_{_stem(meas['file'])}", raster=True)


def _errorbar(ax, x, y, yerr, xerr, colors, included, marker="o", label=None):
    base, dark = colors
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=marker, ms=4, mfc=base if included else "none", mec=dark, mew=0.6,
                ecolor=dark, alpha=POINT_ALPHA if included else 0.5, zorder=3, label=label, **ERR_KW)


def plot_peak_vs_power(per_setting: pd.DataFrame, cal: dict) -> Path:
    """C: I_peak vs. P with the calibration model (A); power inside the FOV vs. P (B)."""
    fig, (ax_i, ax_p) = new_figure("narrow", 1, 2)
    handles = []
    p_max = per_setting["P_mW"].max()
    for wl, st in per_setting.groupby("wavelength_nm"):
        colors = _colors(wl)
        for r in st.itertuples():
            x_err = POWER_REL_UNC * r.P_mW
            i_err = np.hypot(np.nan_to_num(r.I_peak_W_cm2_sd), POWER_REL_UNC * r.I_peak_W_cm2_mean)
            _errorbar(ax_i, r.P_mW, r.I_peak_W_cm2_mean, i_err, x_err, colors, r.included)
            _errorbar(ax_p, r.P_mW, r.P_fov_mW_sel_mean, np.nan_to_num(r.P_fov_mW_sel_sd), x_err, colors, r.included)
        handles.append(Line2D([], [], ls="none", marker="o", ms=4, mfc=colors[0], mec=colors[1], mew=0.6,
                              label=f"{wl:g} nm"))
        m = cal["wavelengths"].get(f"{wl:g}", {}).get("peak_model")
        if m:
            grid = np.linspace(max(m["P0_mW"], 0.0), 1.1 * p_max, 200)
            ax_i.plot(grid, m["a_W_cm2_per_mW"] * (grid - m["P0_mW"]), color=colors[1], lw=1.5, zorder=2)
    grid = np.array([0.0, 1.1 * p_max])
    ax_p.plot(grid, grid, color="k", lw=1.2, ls=(0, (4, 3)), zorder=1)
    handles.append(Line2D([], [], color="k", lw=1.5, label="Fit"))
    if not per_setting["included"].all():
        handles.append(Line2D([], [], ls="none", marker="o", ms=4, mfc="none", mec="k", mew=0.6, label="Excluded"))
    ax_i.legend(handles=handles, loc="lower right", handlelength=1.2)
    ax_p.legend(handles=[Line2D([], [], color="k", lw=1.2, ls=(0, (4, 3)), label=r"$P_\mathrm{FOV} = P$")],
                loc="upper left", bbox_to_anchor=(0.0, 0.9), handlelength=1.8)
    for ax, label in ((ax_i, "A"), (ax_p, "B")):
        ax.set_xlim(0.0, 1.1 * p_max)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"Sample-plane power $P$ (mW)")
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_p.set_ylim(top=1.1 * p_max)
    ax_i.set_ylabel(r"Peak intensity $I_\mathrm{peak}$ (W cm$^{-2}$)")
    ax_p.set_ylabel(r"Power in field of view $P_\mathrm{FOV}$ (mW)")
    return save_figure(fig, "peak_intensity_vs_power")


def plot_edge_intensity(per_setting: pd.DataFrame) -> Path:
    """D: intensity at the FOV edge centres and corners in % of the peak, mean +- SD over valid settings."""
    fig, ax = new_figure("half")
    groups = list(per_setting.groupby("wavelength_nm"))
    offsets = np.linspace(-0.15, 0.15, len(groups)) if len(groups) > 1 else [0.0]
    pos = np.arange(len(EDGES))
    for (wl, st), dx in zip(groups, offsets):
        used = st[st["included"]]
        included = not used.empty
        used = used if included else st
        means = [used[f"I_{e}_rel_pct_sel_mean"].mean() for e in EDGES]
        sds = [used[f"I_{e}_rel_pct_sel_mean"].std(ddof=1) if len(used) > 1 else used[f"I_{e}_rel_pct_sel_sd"].iloc[0]
               for e in EDGES]
        _errorbar(ax, pos + dx, means, np.nan_to_num(sds), None, _colors(wl), included, label=f"{wl:g} nm")
    ax.set_xticks(pos, EDGE_LABELS)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlim(-0.5, len(EDGES) - 0.5)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("Position on field-of-view boundary")
    ax.set_ylabel(r"Intensity (% of $I_\mathrm{peak}$)")
    ax.legend(loc="upper right", handlelength=1.2)
    return save_figure(fig, "edge_intensity_relative")


# ── Export ─────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def export(per_image: pd.DataFrame, per_setting: pd.DataFrame, cal: dict) -> None:
    per_image.to_csv(OUTPUT_DIR / "laser_intensity_per_image.csv", index=False)
    per_setting.to_csv(OUTPUT_DIR / "laser_intensity_per_setting.csv", index=False)
    cal["warnings"] = list(_WARNINGS)
    RESULT_JSON.write_text(json.dumps(_json_safe(cal), indent=2, ensure_ascii=False), encoding="utf-8")
    _read_json.cache_clear()


def _fmt_intensity(value_W_cm2: float) -> str:
    return f"{value_W_cm2 / 1e3:.3g} kW/cm2" if value_W_cm2 >= 1e3 else f"{value_W_cm2:.3g} W/cm2"


def print_summary(per_setting: pd.DataFrame, cal: dict) -> None:
    print("\nSummary per wavelength and setting (mean over valid repetitions; * = excluded from calibration)")
    for wl, st in per_setting.groupby("wavelength_nm"):
        for r in st.itertuples():
            mark = " " if r.included else "*"
            edges = [getattr(r, f"I_{e}_rel_pct_sel_mean") for e in EDGES[:4]]
            print(f"{mark} {wl:.0f} nm, P = {r.P_mW:.3g} mW: I_peak = {_fmt_intensity(r.I_peak_W_cm2_mean)} "
                  f"(+- {r.I_peak_err_W_cm2_mean / r.I_peak_W_cm2_mean:.0%}), "
                  f"edge = {min(edges):.0f}-{max(edges):.0f} % of peak (corners {r.corner_mean_rel_pct_sel_mean:.0f} %), "
                  f"in FOV: {r.P_fov_mW_sel_mean:.3g} mW ({r.fov_power_fraction_sel_mean:.0%}), "
                  f"FOV mean {_fmt_intensity(r.I_fov_mean_W_cm2_sel_mean)}, CV {r.fov_cv_sel_mean:.0%}")
            print(f"      w = {r.wx_um_mean:.1f} x {r.wy_um_mean:.1f} um (FWHM {r.fwhm_x_um_mean:.1f} x "
                  f"{r.fwhm_y_um_mean:.1f} um), decentration {r.decenter_x_um_mean:+.1f} / {r.decenter_y_um_mean:+.1f} um, "
                  f"FOV {r.fov_x_um_mean:.0f} x {r.fov_y_um_mean:.0f} um, R2 {r.R2_mean:.3f}, map: {r.profile_source}, "
                  f"reps {r.n_valid}/{r.n_reps}" + (f", flags: {r.flags}" if r.flags else ""))
            print(f"      E = {r.E_mJ_mean * 1e3:.1f} uJ, F_peak = {r.F_peak_J_cm2_mean:.3g} J/cm2 "
                  f"(t = {r.t_s_mean * 1e3:.2f} ms)")
        e = cal["wavelengths"].get(f"{wl:g}", {})
        if "peak_model" in e:
            m, b = e["peak_model"], e["beam"]
            print(f"  {wl:.0f} nm calibration ({m['name']}): I_peak = {m['a_W_cm2_per_mW']:.4g} +- "
                  f"{m['a_err_W_cm2_per_mW']:.2g} W/cm2 per mW * (P - {m['P0_mW']:.3g} mW), chi2_red = {m['chi2_red']:.2f}; "
                  f"w constant: {b['constant']} (CV wx {b['wx_cv']:.1%}, wy {b['wy_cv']:.1%})\n")
    if _WARNINGS:
        print(f"{len(_WARNINGS)} warning(s), also stored in {RESULT_JSON.name}.")
    print(f"Output: {OUTPUT_DIR}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _WARNINGS.clear()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RUN_SYNTHETIC_TEST:
        synthetic = run_synthetic_test()
        synthetic.to_csv(OUTPUT_DIR / "laser_intensity_synthetic_test.csv", index=False)
        if not synthetic["pass"].all():
            _warn("synthetic test: at least one quantity outside the recovery criteria (see table above).")

    measurements = load_all(DATA_DIR)
    if not measurements:
        raise FileNotFoundError(f"no readable {FILE_PATTERN} stacks in {DATA_DIR}")
    print_overview(measurements)
    dark_cps = load_dark_cps(DARK_FRAME_PATH, measurements[0]["exposure_s"])

    rows, details = [], {}
    for meas in measurements:
        print(f"Fitting {meas['file']}")
        try:
            file_rows, details[meas["file"]] = analyse_measurement(meas, dark_cps)
            rows += file_rows
        except Exception as exc:
            _warn(f"{meas['file']}: analysis failed ({type(exc).__name__}: {exc}); skipped.")
        meas.pop("frames_raw")

    per_image = pd.DataFrame(rows)
    per_image["flag"] = per_image["flag"].fillna("")
    apply_series_flags(per_image)
    per_setting = summarize_settings(per_image, {f: d["profile_source"] for f, d in details.items()})
    cal = calibrate(per_setting, details)
    export(per_image, per_setting, cal)

    with plt.style.context(STYLE_PATH):
        for detail in details.values():
            if detail["mean_fit"]["status"] != "ok":
                continue
            plot_intensity_map(detail)
            plot_line_profiles(detail)
            plot_residual_map(detail)
        plot_peak_vs_power(per_setting, cal)
        plot_edge_intensity(per_setting)

    print_summary(per_setting, cal)


if __name__ == "__main__":
    main()
