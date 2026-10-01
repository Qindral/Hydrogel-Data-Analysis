"""
FRAP analysis v2, phase 1 (analysis and tables): FITC-dextran 70 kDa in water vs. in
eADF4(C16) hydrogel, Leica SP8 point-bleach FRAP (8-bit, FITC in the green channel).
Compute stage; figures are produced separately by frap_figures.py (phase 2) from the
cache written here, and only after the phase-1 tables have been reviewed.

Input: MEASUREMENTS_CSV (measurement_id, file, condition, batch, include, note,
r_nominal_um, gel_concentration_mg_ml); "file" is the measurement folder below DATA_ROOT
holding one "FRAP Pre ..." and one or more "FRAP Pb ..." Leica TIF series with their
MetaData XML. Pixel size and per-frame time stamps come from the Properties XML (read
with the existing loaders in frap_analysis.py); post-bleach series of different frame
intervals are joined on their absolute time stamps, t = 0 at the first post-bleach frame.
Conditions: water (free diffusion reference), gel (bulk hydrogel, batches from the CSV),
gel_interface (gel/glass interface, reported separately, never pooled with gel) and
gel_noFITC (autofluorescence / matrix-bleaching control, no diffusion analysis).

Per measurement (all geometry from the pixel data; the nominal radius r_n is metadata only):
  1. Bleach centre: intensity-weighted centroid of the smoothed deficit (pre mean - first
     post-bleach frame), then fixed for all frames.
  2. Profile broadening (primary D): azimuthally averaged I_post(r) / I_pre(r) per post-bleach
     frame, fit f(r) = A (1 - K exp(-2 r^2 / w^2)); the leading run of frames with a
     resolvable profile (K >= PROFILE_K_MIN, w <= PROFILE_W_MAX_FRACTION * r_max,
     R^2 >= PROFILE_R2_MIN) gives w^2(t) = w0^2 + 8 D_profile t by linear regression,
     weighted with 1/SE(w^2)^2 from the profile fits.
     Diagnostics: R^2 of that line and alpha of w^2 = w0^2 + Gamma t^alpha. r_e := w of the
     first post-bleach frame.
  3. Recovery curve: double normalisation (Phair & Misteli) in a disk of radius r_e,
       F_norm(t) = [(F_REF,pre - F_BG) / (F_REF(t) - F_BG)] * [(F_ROI(t) - F_BG) / (F_ROI,pre - F_BG)],
     reference = all pixels farther than r_ref from the centre. F_norm(0) is not forced to 0;
     K0 = 1 - F_norm(first post-bleach frame). Fit F(t) = F0 + (Finf - F0)(1 - exp(-t/tau)),
     i.e. M_f (1 - exp(-t/tau)) on the scale (F - F0)/(1 - F0), M_f = (Finf - F0)/(1 - F0),
     t_half = tau ln 2. Cross-validation: D_soumpasis = r_e^2/(4 tau), D_axelrod =
     0.88 r_e^2/(4 t_half), D_kang = (r_e^2 + r_n^2)/(8 t_half), D_soumpasis_nominal = r_n^2/(4 tau).
  F_BG = BACKGROUND_COUNTS (PMT offset 0 in all files). The noFITC controls cannot serve as a
  quantitative background (their intensity scatters by a factor of 5 at identical detector
  settings). A constant non-bleaching background cancels from tau, t_half, M_f and w(t), and
  only lowers the apparent K0, which is therefore a lower bound.

QC flags (boolean, with the failing reason written to qc_reasons; nothing is dropped silently):
  global   include, >= QC_MIN_PRE_FRAMES pre-bleach frames, K0 >= QC_MIN_BLEACH_DEPTH,
           saturated pixel fraction <= QC_MAX_SATURATED_FRACTION, centroid and r_e found;
  recovery global + recovery R^2 >= QC_MIN_RECOVERY_R2, duration >= QC_MIN_DURATION_THALF t_half,
           >= QC_MIN_FRAMES_TO_THALF frames up to t_half;
  profile  global + >= PROFILE_MIN_FRAMES frames in the w^2(t) window.
Every parameter is summarised only over measurements passing its gate (PARAMETER_GATES).
The experimental unit is the measurement; pixels and frames are never pooled for statistics.

Statistics: mean, SD, CV per condition, gel concentration and batch (concentrations are never
pooled; all present measurements are 20 mg/mL); hindrance ratio H = D_gel / D_water as the
ratio of geometric means, with a percentile bootstrap 95 % CI (BOOTSTRAP_N draws, fixed seed).
Main reference (MAIN_REFERENCE) is Stokes-Einstein (core.physics, R_h = DEXTRAN_RH_NM,
T = 20 degC); H against the measured water value is reported for comparison. Leave-one-out
over the 20 mg/mL bulk-gel measurements.

Before any real data are analysed, the profile method is validated on simulated 2D Gaussian
bleaches with known D, signal-dependent noise, static texture, acquisition bleaching and 8-bit
quantisation (SYNTHETIC_CASES, SYNTHETIC_REPLICATES noise realisations each); the mean D_profile
must match within SYNTHETIC_TOLERANCE, otherwise the run stops. The SD over the realisations
is the expected precision of a single measurement. Not simulated: line-scan timing within a
frame and the non-Gaussian profile left by a long bleach phase.

Writes (always recomputed, overwritten):
  cache/frap_v2_phase1.pkl                (this folder; read by frap_figures.py)
  OUTPUT_ROOT/results_per_measurement.csv, summary_per_condition.csv, ratio_gel_water.json,
    loo_gel.csv, nofitc_controls.csv, synthetic_validation.csv, results_per_measurement.tex,
    summary_per_condition.tex, report_phase1.md
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit, minimize_scalar
from scipy.special import j0

from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.FRAP.frap_analysis import (
    find_series_folders,
    load_tif_series,
    parse_frame_timestamps,
    parse_frap_roi,
    parse_properties_xml,
)

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"H:\Daten Promotion Sicherung\Confocal_Measure\2025_09_08_14_57_46--Project_TIF")
MEASUREMENTS_CSV = Path(__file__).resolve().parent / "measurements.csv"
OUTPUT_ROOT = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\FRAP_v2")
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "frap_v2_phase1.pkl"

# Image handling
SATURATION_COUNTS = 255                 # 8-bit images
BACKGROUND_COUNTS = 0.0                 # F_BG; PMT offset is 0 in all files (see module docstring)

# Bleach centre
CENTER_SMOOTH_SIGMA_PX = 5.0
CENTER_THRESHOLD_FRACTION = 0.1         # deficit below this fraction of the peak is ignored

# Radial profile and w^2(t) regression. The bleach spots are wide (w0 30-60 um, long zoom-in
# bleach phases), so the profile uses partial annuli out to the image corners.
PROFILE_R_MAX_UM = 95.0
PROFILE_BIN_UM = 0.6
PROFILE_K_MIN = 0.05                    # profile depth below this is no longer resolvable
PROFILE_W_MAX_FRACTION = 0.75           # w must stay below this fraction of r_max (baseline A needed)
PROFILE_MAX_REL_SE = 0.05               # SE(w)/w above this: width no longer determined
PROFILE_R2_MIN = 0.9                    # per-frame profile fit
PROFILE_MIN_FRAMES = 5                  # frames in the w^2(t) window

# Diffusion propagation (shape-free, drift-tracked): the depletion profile of the first post-bleach
# frame is propagated with the 2D diffusion equation (order-0 Hankel transform) and fitted to the
# later frames with D as the only shape parameter (per-frame baseline free). Radial profiles are taken
# around the per-frame depletion centroid, which removes a uniform drift / flow of the sample.
PROPAGATION_R_EXT_UM = 400.0            # zero-padded radial domain of the Hankel transform
PROPAGATION_DEPTH_MIN = 0.05            # frames with a shallower central depletion are not used
PROPAGATION_AREA_GROWTH = 2.0           # window ends at w0^2 + 8 D t = factor * w0^2 (spot area doubled)
PROPAGATION_TRACK_CENTRE = True
PROPAGATION_MIN_FRAMES = 5
PROPAGATION_SCAN_T_MAX_S = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 60.0)   # diagnostic only

# Recovery curve
REF_INNER_FACTOR = 3.0                 # reference region: r >= clip(factor * r_e, min, max)
REF_INNER_MIN_UM = 25.0
REF_INNER_MAX_UM = 65.0

# QC
QC_MIN_PRE_FRAMES = 5
QC_MIN_BLEACH_DEPTH = 0.3
QC_MIN_RECOVERY_R2 = 0.95
QC_MIN_DURATION_THALF = 5.0
QC_MIN_FRAMES_TO_THALF = 10
QC_MAX_SATURATED_FRACTION = 0.01        # pre-bleach pixels at SATURATION_COUNTS within r_max

PARAMETER_GATES = {
    "D_propagation_um2_s": "qc_propagation", "drift_um_per_s": "qc_propagation",
    "D_profile_um2_s": "qc_profile", "w2_r2": "qc_profile", "alpha": "qc_profile",
    "D_soumpasis_um2_s": "qc_recovery", "D_axelrod_um2_s": "qc_recovery", "D_kang_um2_s": "qc_recovery",
    "M_f": "qc_recovery", "t_half_s": "qc_recovery", "tau_s": "qc_recovery",
    "r_e_um": "qc_global", "K0": "qc_global",
}
D_METHODS = ["D_propagation_um2_s", "D_profile_um2_s", "D_soumpasis_um2_s", "D_axelrod_um2_s", "D_kang_um2_s"]

# Reference and statistics
TEMPERATURE_K = 293.15
WATER_VISCOSITY_PA_S = 1.002e-3
DEXTRAN_RH_NM = 6.5
MAIN_REFERENCE = "Stokes-Einstein"      # main water reference for H (measured water reported for comparison)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20251001
LOO_OUTLIER_Z = 3.0

# Synthetic validation of the profile method
SYNTHETIC_TOLERANCE = 0.05              # on the mean relative deviation over the replicates
SYNTHETIC_REPLICATES = 10               # independent noise realisations per case
SYNTHETIC_CASES = [                     # w0 and schedules as in the measured gel and water data
    {"D_um2_s": 2.0, "w0_um": 8.0, "schedule": ((0.374, 40), (2.0, 30), (10.0, 24))},
    {"D_um2_s": 10.0, "w0_um": 45.0, "schedule": ((0.374, 40), (2.0, 30))},
    {"D_um2_s": 25.0, "w0_um": 45.0, "schedule": ((0.374, 40), (2.0, 30))},
    {"D_um2_s": 33.0, "w0_um": 35.0, "schedule": ((0.374, 60),)},
    {"D_um2_s": 33.0, "w0_um": 35.0, "schedule": ((0.374, 60),), "label": "non-Gaussian (15 + 40 um components)",
     "components": [(0.35, 15.0), (0.25, 40.0)]},
    {"D_um2_s": 33.0, "w0_um": 35.0, "schedule": ((0.374, 60),), "label": "drift 2 um/s", "drift_um_s": 2.0},
]
SYNTHETIC_PARAMS = {"size_px": 512, "mpp_um": 0.283, "intensity": 80.0, "K0": 0.5, "noise_rel": 0.15,
                    "texture_rel": 0.1, "acq_bleach_tau_s": 2000.0, "n_pre": 7, "seed": 7}


# ── Loading ────────────────────────────────────────────────────────────────────

def _series_times(folder: Path, frame_idx: List[int]) -> List[datetime]:
    stamps = parse_frame_timestamps(folder / "MetaData" / f"{folder.name}_Properties.xml")
    out = []
    for i in frame_idx:
        if i < len(stamps) and stamps[i][1] is not None:
            out.append(stamps[i][1])
        else:
            raise ValueError(f"{folder.name}: no time stamp for frame {i}")
    return out


def load_measurement(folder: Path) -> dict:
    """Pre and post images (green channel, float) with times in s relative to the first post frame."""
    pre_folders, pb_folders = find_series_folders(folder)
    pre_imgs, pre_t, post_imgs, post_t = [], [], [], []
    for f in pre_folders:
        imgs, idx = load_tif_series(f, f.name)
        pre_imgs.append(imgs)
        pre_t += _series_times(f, idx)
    for f in pb_folders:
        imgs, idx = load_tif_series(f, f.name)
        post_imgs.append(imgs)
        post_t += _series_times(f, idx)
    post = np.concatenate(post_imgs)
    order = np.argsort(post_t)
    post_t = [post_t[i] for i in order]
    t0 = post_t[0]
    roi = parse_frap_roi(pb_folders[0] / "MetaData" / f"{pb_folders[0].name}_Properties.xml")
    props = parse_properties_xml(pb_folders[0] / "MetaData" / f"{pb_folders[0].name}_Properties.xml")
    phases = roi.get("phases", [])
    return {
        "pre": np.concatenate(pre_imgs),
        "pre_t_s": np.array([(t - t0).total_seconds() for t in pre_t]),
        "post": post[order],
        "post_t_s": np.array([(t - t0).total_seconds() for t in post_t]),
        "mpp_um": roi["mpp_um"],
        "bleach_laser_pct": roi.get("bleach_laser_pct", np.nan),
        "bleach_phase_s": phases[1]["time_s"] * phases[1]["frame_count"] if len(phases) > 1 else np.nan,
        "frame_interval_s": phases[0]["time_s"] if phases else np.nan,
        "objective": props.get("objective", ""),
        "pb_series": ", ".join(f.name for f in pb_folders),
    }


# ── Geometry from pixel data ───────────────────────────────────────────────────

def bleach_centroid(pre_mean: np.ndarray, post_first: np.ndarray) -> Optional[tuple]:
    """Weighted centroid (row, col) of the smoothed bleach deficit; None if there is no deficit."""
    deficit = gaussian_filter(pre_mean - post_first, CENTER_SMOOTH_SIGMA_PX)
    peak = deficit.max()
    if peak <= 0:
        return None
    weights = np.clip(deficit - CENTER_THRESHOLD_FRACTION * peak, 0, None)
    rows, cols = np.indices(deficit.shape)
    return float((weights * rows).sum() / weights.sum()), float((weights * cols).sum() / weights.sum())


class RadialBins:
    """Fixed azimuthal binning around a centre; profile(image) -> mean per bin."""

    def __init__(self, shape, centre, mpp_um, r_max_um=PROFILE_R_MAX_UM, bin_um=PROFILE_BIN_UM):
        rows, cols = np.indices(shape)
        r_um = np.hypot(rows - centre[0], cols - centre[1]) * mpp_um
        n_bins = int(np.ceil(r_max_um / bin_um))
        self.index = np.where(r_um < r_max_um, (r_um / bin_um).astype(int), -1).ravel()
        self.mask = self.index >= 0
        self.counts = np.bincount(self.index[self.mask], minlength=n_bins)
        self.r_um = (np.arange(n_bins) + 0.5) * bin_um
        self.r_max_um = r_max_um
        self.r_map_um = r_um

    def profile(self, image: np.ndarray) -> np.ndarray:
        sums = np.bincount(self.index[self.mask], weights=image.ravel()[self.mask], minlength=len(self.counts))
        with np.errstate(invalid="ignore", divide="ignore"):
            return sums / self.counts


def _gauss_profile(r, A, K, w):
    return A * (1.0 - K * np.exp(-2.0 * r ** 2 / w ** 2))


def fit_radial_profile(r_um, ratio, counts) -> dict:
    """Fit A (1 - K exp(-2 r^2/w^2)); returns parameters, standard errors and R^2."""
    ok = np.isfinite(ratio) & (counts > 0)
    r, y, n = r_um[ok], ratio[ok], counts[ok]
    try:
        K0 = float(np.clip(1 - y[:3].mean() / y[-5:].mean(), 0.01, 0.99))
        half = r[np.argmin(np.abs(y - y[-5:].mean() * (1 - K0 / 2)))]
        popt, pcov = curve_fit(_gauss_profile, r, y, p0=[y[-5:].mean(), K0, max(half * 1.7, 1.0)],
                               sigma=1 / np.sqrt(n), bounds=([0.2, 0.0, 0.3], [2.0, 1.0, 5 * r.max()]),
                               maxfev=20000)
    except (RuntimeError, ValueError):
        return {"success": False}
    resid = y - _gauss_profile(r, *popt)
    ss_tot = np.sum((y - y.mean()) ** 2)
    se = np.sqrt(np.diag(pcov))
    return {"success": True, "A": popt[0], "K": popt[1], "w_um": popt[2], "w_se_um": se[2],
            "r2": 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan}


def _power_law(t, w0sq, gamma, alpha):
    return w0sq + gamma * np.power(np.clip(t, 0, None), alpha)


def profile_analysis(pre: np.ndarray, post: np.ndarray, t_s: np.ndarray, mpp_um: float,
                     centre: Optional[tuple] = None) -> dict:
    """Centroid, per-frame Gaussian profile fits, w^2(t) window, D_profile and diagnostics."""
    pre_mean = pre.mean(axis=0)
    if centre is None:
        centre = bleach_centroid(pre_mean, post[0])
    if centre is None:
        return {"centre": None, "reason": "no bleach deficit"}
    bins = RadialBins(pre_mean.shape, centre, mpp_um)
    pre_prof = bins.profile(pre_mean - BACKGROUND_COUNTS)
    frames = []
    for i, img in enumerate(post):
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = bins.profile(img - BACKGROUND_COUNTS) / pre_prof
        fit = fit_radial_profile(bins.r_um, ratio, bins.counts)
        valid = bool(fit.get("success") and fit["K"] >= PROFILE_K_MIN
                     and fit["w_um"] <= PROFILE_W_MAX_FRACTION * bins.r_max_um and fit["r2"] >= PROFILE_R2_MIN
                     and fit["w_se_um"] <= PROFILE_MAX_REL_SE * fit["w_um"])
        frames.append({"frame": i, "t_s": t_s[i], "valid": valid, "ratio": ratio, **fit})
    out = {"centre": centre, "bins_r_um": bins.r_um, "bins_counts": bins.counts, "frames": frames,
           "r_e_um": frames[0]["w_um"] if frames[0].get("success") else np.nan,
           "K0_profile": frames[0]["K"] if frames[0].get("success") else np.nan,
           "D_profile_um2_s": np.nan, "D_profile_se_um2_s": np.nan, "w0_um": np.nan, "w2_r2": np.nan,
           "alpha": np.nan, "alpha_se": np.nan, "window": (np.nan, np.nan), "n_window": 0}
    valid_idx = [f["frame"] for f in frames if f["valid"]]
    if not valid_idx:
        return out
    start = valid_idx[0]
    stop = start
    while stop + 1 < len(frames) and frames[stop + 1]["valid"]:
        stop += 1
    window = range(start, stop + 1)
    t = np.array([frames[i]["t_s"] for i in window])
    w = np.array([frames[i]["w_um"] for i in window])
    w2 = w ** 2
    # SE of w^2 from the profile fit; late, shallow profiles are noisier and get less weight.
    w2_se = np.maximum(2 * w * np.array([frames[i]["w_se_um"] for i in window]), 1e-6)
    out.update({"window": (float(t[0]), float(t[-1])), "n_window": len(t), "w2_se": w2_se})
    if len(t) < 3:
        return out
    (slope, intercept), cov = np.polyfit(t, w2, 1, w=1 / w2_se, cov=True)
    pred = intercept + slope * t
    wts = 1 / w2_se ** 2
    w2_mean = np.sum(wts * w2) / wts.sum()
    out.update({"D_profile_um2_s": slope / 8.0, "D_profile_se_um2_s": np.sqrt(cov[0, 0]) / 8.0,
                "w0_um": np.sqrt(intercept) if intercept > 0 else np.nan,
                "w2_r2": 1 - np.sum(wts * (w2 - pred) ** 2) / np.sum(wts * (w2 - w2_mean) ** 2),
                "w2_slope": slope, "w2_intercept": intercept})
    if len(t) >= 4 and slope > 0:
        try:
            popt, pcov = curve_fit(_power_law, t, w2, p0=[max(intercept, 1e-3), slope, 1.0], sigma=w2_se,
                                   bounds=([0, 0, 0.1], [np.inf, np.inf, 3.0]), maxfev=20000)
            out.update({"alpha": popt[2], "alpha_se": float(np.sqrt(pcov[2, 2]))})
        except (RuntimeError, ValueError):
            pass
    return out


# ── Diffusion propagation (shape-free, drift-tracked) ─────────────────────────

def _hankel_operators(n_r: int, dr_um: float) -> tuple:
    """Order-0 Hankel forward/inverse matrices on a zero-padded radial grid (bin centres)."""
    r_ext = (np.arange(int(round(PROPAGATION_R_EXT_UM / dr_um))) + 0.5) * dr_um
    dk = 1.0 / (2 * PROPAGATION_R_EXT_UM)
    k = (np.arange(int(round(1 / (2 * dr_um) / dk))) + 0.5) * dk
    fwd = 2 * np.pi * j0(2 * np.pi * np.outer(k, r_ext)) * r_ext * dr_um
    inv = 2 * np.pi * j0(2 * np.pi * np.outer(r_ext, k)) * k * dk
    return k, fwd, inv


def propagation_analysis(pre: np.ndarray, post: np.ndarray, t_s: np.ndarray, mpp_um: float, w0_um: float) -> dict:
    """D from propagating the first post-bleach depletion profile with the 2D diffusion equation.

    delta0(r) = 1 - (I_post,0(r) / I_pre(r)) / a0 (a0: far-field baseline of frame 0) is evolved as
    delta(r, t) = H^-1[ H[delta0](k) exp(-4 pi^2 k^2 D t) ], and every later frame is modelled as
    A_i (1 - delta(r, t_i)) with a free baseline A_i (acquisition bleaching). This needs no assumption
    on the profile shape and stays valid when the spreading profile leaves the field of view, provided
    the depletion at t0 lies inside it. Profiles are taken around the per-frame depletion centroid
    (PROPAGATION_TRACK_CENTRE), which removes a uniform drift; non-uniform flow cannot be removed.
    The fit window ends when the spot area has grown by PROPAGATION_AREA_GROWTH
    (w0^2 + 8 D t = factor w0^2, iterated with the fitted D), or at the last frame with a central
    depletion >= PROPAGATION_DEPTH_MIN.
    """
    pre_mean = pre.mean(axis=0)
    centre0 = bleach_centroid(pre_mean, post[0])
    if centre0 is None:
        return {"D_propagation_um2_s": np.nan, "reason": "no bleach deficit"}
    bins0 = RadialBins(pre_mean.shape, centre0, mpp_um)
    n_r = len(bins0.r_um)
    k, fwd, inv = _hankel_operators(n_r, PROFILE_BIN_UM)
    prof0 = bins0.profile(post[0]) / bins0.profile(pre_mean)
    ok0 = np.isfinite(prof0) & (bins0.counts > 0)
    far0 = ok0 & (bins0.r_um > 0.8 * bins0.r_um[ok0].max())
    a0 = np.nanmedian(prof0[far0])
    delta0 = np.zeros(fwd.shape[1])
    delta0[:n_r][ok0] = 1.0 - prof0[ok0] / a0
    spec0 = fwd @ delta0

    frames = []
    for i in range(1, len(post)):
        centre = bleach_centroid(pre_mean, post[i]) if PROPAGATION_TRACK_CENTRE else centre0
        if centre is None:
            break
        bins = RadialBins(pre_mean.shape, centre, mpp_um) if PROPAGATION_TRACK_CENTRE else bins0
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = bins.profile(post[i]) / bins.profile(pre_mean)
        ok = np.isfinite(prof) & (bins.counts > 0)
        far = ok & (bins.r_um > 0.8 * bins.r_um[ok].max())
        depth = 1.0 - np.nanmean(prof[:8]) / np.nanmedian(prof[far])
        if depth < PROPAGATION_DEPTH_MIN:
            break
        frames.append({"t_s": float(t_s[i] - t_s[0]), "centre": centre, "profile": prof, "ok": ok,
                       "w": bins.counts.astype(float)})

    def predict(D, dt):
        return (inv @ (spec0 * np.exp(-4 * np.pi ** 2 * k ** 2 * D * dt)))[:n_r]

    def fit(t_max):
        use = [fr for fr in frames if fr["t_s"] <= t_max]
        if len(use) < 2:
            return np.nan, use

        def cost(log_d):
            total = 0.0
            for fr in use:
                model = 1.0 - predict(np.exp(log_d), fr["t_s"])[fr["ok"]]
                y, w = fr["profile"][fr["ok"]], fr["w"][fr["ok"]]
                A = np.sum(w * y * model) / np.sum(w * model ** 2)
                total += np.sum(w * (y - A * model) ** 2)
            return total
        res = minimize_scalar(cost, bounds=(np.log(0.05), np.log(500.0)), method="bounded", options={"xatol": 1e-5})
        return float(np.exp(res.x)), use

    out = {"centre0": centre0, "r_um": bins0.r_um, "delta0": delta0[:n_r], "a0": a0,
           "scan": {tm: fit(tm)[0] for tm in PROPAGATION_SCAN_T_MAX_S}, "frames": frames}
    D, t_end = np.nan, 3.0
    for _ in range(8):           # window from the area-growth rule, iterated with the fitted D
        D, use = fit(t_end)
        if not np.isfinite(D) or not np.isfinite(w0_um):
            break
        new_end = (PROPAGATION_AREA_GROWTH - 1.0) * w0_um ** 2 / (8.0 * D)
        if abs(new_end - t_end) < 0.02 * t_end:
            t_end = new_end
            break
        t_end = new_end
    D, use = fit(t_end)
    cents = np.array([centre0] + [fr["centre"] for fr in use]) * mpp_um
    times = np.array([0.0] + [fr["t_s"] for fr in use])
    disp = np.hypot(*(cents - cents[0]).T)
    out.update({
        "D_propagation_um2_s": D, "propagation_window_s": float(min(t_end, use[-1]["t_s"])) if use else np.nan,
        "n_propagation_frames": len(use), "max_drift_um": float(disp.max()) if len(disp) else np.nan,
        "drift_um_per_s": float(np.polyfit(times, disp, 1)[0]) if len(times) > 2 else np.nan,
        "used_times_s": [fr["t_s"] for fr in use],
        "fit_profiles": [{"t_s": fr["t_s"], "r_um": bins0.r_um[fr["ok"]], "data": fr["profile"][fr["ok"]],
                          "model": 1.0 - predict(D, fr["t_s"])[fr["ok"]]} for fr in use] if np.isfinite(D) else [],
        "centroid_track_um": [(float(ti), float(ci[0]), float(ci[1]))
                              for ti, ci in zip([0.0] + [fr["t_s"] for fr in frames],
                                                np.array([centre0] + [fr["centre"] for fr in frames]) * mpp_um)],
    })
    del out["frames"]
    return out


# ── Recovery curve ─────────────────────────────────────────────────────────────

def double_normalised_curve(pre: np.ndarray, post: np.ndarray, centre: tuple, roi_radius_um: float,
                            mpp_um: float) -> dict:
    """Phair/Misteli double normalisation in a disk of radius roi_radius_um."""
    rows, cols = np.indices(pre.shape[1:])
    r_um = np.hypot(rows - centre[0], cols - centre[1]) * mpp_um
    roi = r_um <= roi_radius_um
    ref_inner = float(np.clip(REF_INNER_FACTOR * roi_radius_um, REF_INNER_MIN_UM, REF_INNER_MAX_UM))
    ref = r_um >= ref_inner
    bg = BACKGROUND_COUNTS
    f_roi_pre = pre[:, roi].mean() - bg
    f_ref_pre = pre[:, ref].mean() - bg
    f_roi = post[:, roi].mean(axis=1) - bg
    f_ref = post[:, ref].mean(axis=1) - bg
    f_norm = (f_ref_pre / f_ref) * (f_roi / f_roi_pre)
    return {"F_norm": f_norm, "F_roi": f_roi, "F_ref": f_ref, "F_roi_pre": f_roi_pre, "F_ref_pre": f_ref_pre,
            "ref_inner_um": ref_inner, "n_roi_px": int(roi.sum()), "n_ref_px": int(ref.sum())}


def _recovery_model(t, F0, Finf, tau):
    return F0 + (Finf - F0) * (1.0 - np.exp(-t / tau))


def fit_recovery(t_s: np.ndarray, f_norm: np.ndarray) -> dict:
    """F(t) = F0 + (Finf - F0)(1 - exp(-t/tau)); M_f = (Finf - F0)/(1 - F0)."""
    try:
        popt, pcov = curve_fit(_recovery_model, t_s, f_norm,
                               p0=[f_norm[0], f_norm[-5:].mean(), max(t_s[-1] / 10, 0.1)],
                               bounds=([-0.5, 0.0, 1e-3], [1.5, 2.0, 100 * t_s[-1]]), maxfev=20000)
    except (RuntimeError, ValueError):
        return {"success": False}
    F0, Finf, tau = popt
    se = np.sqrt(np.diag(pcov))
    resid = f_norm - _recovery_model(t_s, *popt)
    ss_tot = np.sum((f_norm - f_norm.mean()) ** 2)
    return {"success": True, "F0": F0, "Finf": Finf, "tau_s": tau, "tau_se_s": se[2],
            "t_half_s": tau * np.log(2), "M_f": (Finf - F0) / (1 - F0) if F0 < 1 else np.nan,
            "recovery_r2": 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan,
            "residuals": resid}


# ── Per-measurement analysis ───────────────────────────────────────────────────

def analyse_fitc(row: pd.Series) -> tuple[dict, dict]:
    data = load_measurement(DATA_ROOT / row["file"])
    pre, post, t = data["pre"], data["post"], data["post_t_s"]
    mpp = data["mpp_um"]
    prof = profile_analysis(pre, post, t, mpp)
    res = {"measurement_id": row["measurement_id"], "file": row["file"], "condition": row["condition"],
           "batch": row["batch"], "include": bool(row["include"]), "note": row["note"],
           "r_n_um": row["r_nominal_um"], "gel_concentration_mg_ml": row["gel_concentration_mg_ml"],
           "mpp_um": mpp, "n_pre_frames": len(pre), "n_post_frames": len(post),
           "t_last_s": float(t[-1]), "bleach_laser_pct": data["bleach_laser_pct"],
           "bleach_phase_s": data["bleach_phase_s"], "frame_interval_s": data["frame_interval_s"]}
    centre = prof["centre"]
    res["centre_row_px"], res["centre_col_px"] = centre if centre else (np.nan, np.nan)
    rows, cols = np.indices(pre.shape[1:])
    if centre:
        near = np.hypot(rows - centre[0], cols - centre[1]) * mpp <= PROFILE_R_MAX_UM
    else:
        near = np.ones(pre.shape[1:], bool)
    res["saturated_fraction"] = float((pre[:, near] >= SATURATION_COUNTS).mean())
    res.update({k: prof.get(k, np.nan) for k in ("r_e_um", "D_profile_um2_s", "D_profile_se_um2_s", "w0_um",
                                                 "w2_r2", "alpha", "alpha_se", "n_window")})
    res["K0"] = prof.get("K0_profile", np.nan)
    prop = propagation_analysis(pre, post, t, mpp, res["r_e_um"])
    res.update({k: prop.get(k, np.nan) for k in ("D_propagation_um2_s", "propagation_window_s",
                                                 "n_propagation_frames", "max_drift_um", "drift_um_per_s")})
    res["profile_window_s"] = "{:.2f}-{:.2f}".format(*prof["window"]) if prof.get("n_window") else ""

    curve, rec = None, {"success": False}
    if centre and np.isfinite(res["r_e_um"]):
        curve = double_normalised_curve(pre, post, centre, res["r_e_um"], mpp)
        rec = fit_recovery(t, curve["F_norm"])
        res["K0_roi"] = 1.0 - curve["F_norm"][0]
        res["ref_inner_um"] = curve["ref_inner_um"]
        res["ref_n_px"] = curve["n_ref_px"]
        res["ref_rel_last"] = float(curve["F_ref"][-1] / curve["F_ref_pre"])
        res["ref_rel_min"] = float(curve["F_ref"].min() / curve["F_ref_pre"])
    if rec["success"]:
        r_e, r_n = res["r_e_um"], float(row["r_nominal_um"]) if pd.notna(row["r_nominal_um"]) else np.nan
        res.update({k: rec[k] for k in ("F0", "Finf", "tau_s", "tau_se_s", "t_half_s", "M_f", "recovery_r2")})
        res["D_soumpasis_um2_s"] = r_e ** 2 / (4 * rec["tau_s"])
        res["D_axelrod_um2_s"] = 0.88 * r_e ** 2 / (4 * rec["t_half_s"])
        res["D_kang_um2_s"] = (r_e ** 2 + r_n ** 2) / (8 * rec["t_half_s"]) if np.isfinite(r_n) else np.nan
        res["D_soumpasis_nominal_um2_s"] = r_n ** 2 / (4 * rec["tau_s"]) if np.isfinite(r_n) and r_n > 0 else np.nan
        res["frames_to_thalf"] = int(np.sum(t <= rec["t_half_s"]))
    apply_qc(res)
    detail = {"t_s": t, "pre_t_s": data["pre_t_s"], "profile": {k: v for k, v in prof.items()}, "propagation": prop,
              "curve": curve, "recovery": rec, "mpp_um": mpp, "folder": str(DATA_ROOT / row["file"])}
    return res, detail


def apply_qc(res: dict) -> None:
    reasons = {"global": [], "recovery": [], "profile": [], "propagation": []}
    checks_global = [
        (res["include"], "excluded in measurements.csv"),
        (res["n_pre_frames"] >= QC_MIN_PRE_FRAMES, f"pre frames {res['n_pre_frames']} < {QC_MIN_PRE_FRAMES}"),
        (res.get("K0", np.nan) >= QC_MIN_BLEACH_DEPTH, f"bleach depth K0 {res.get('K0', np.nan):.2f} < {QC_MIN_BLEACH_DEPTH}"),
        (res["saturated_fraction"] <= QC_MAX_SATURATED_FRACTION,
         f"saturated pixels {100 * res['saturated_fraction']:.1f} % > {100 * QC_MAX_SATURATED_FRACTION:.0f} %"),
        (np.isfinite(res.get("r_e_um", np.nan)), "no bleach centroid / r_e"),
    ]
    t_half = res.get("t_half_s", np.nan)
    checks_recovery = [
        (res.get("recovery_r2", np.nan) >= QC_MIN_RECOVERY_R2, f"recovery R2 {res.get('recovery_r2', np.nan):.3f} < {QC_MIN_RECOVERY_R2}"),
        (res["t_last_s"] >= QC_MIN_DURATION_THALF * t_half, f"duration {res['t_last_s']:.0f} s < {QC_MIN_DURATION_THALF:g} t_half ({QC_MIN_DURATION_THALF * t_half:.0f} s)"),
        (res.get("frames_to_thalf", 0) >= QC_MIN_FRAMES_TO_THALF, f"frames to t_half {res.get('frames_to_thalf', 0)} < {QC_MIN_FRAMES_TO_THALF}"),
    ]
    checks_profile = [(res.get("n_window", 0) >= PROFILE_MIN_FRAMES,
                       f"profile window {res.get('n_window', 0)} frames < {PROFILE_MIN_FRAMES}")]
    n_prop = res.get("n_propagation_frames", 0)
    checks_propagation = [(np.isfinite(res.get("D_propagation_um2_s", np.nan)) and n_prop >= PROPAGATION_MIN_FRAMES,
                           f"propagation window {n_prop} frames < {PROPAGATION_MIN_FRAMES}")]
    for key, checks in (("global", checks_global), ("recovery", checks_recovery), ("profile", checks_profile),
                        ("propagation", checks_propagation)):
        res[f"qc_{key}_checks_pass"] = all(bool(ok) for ok, _ in checks)
        reasons[key] = [msg for ok, msg in checks if not ok]
    res["qc_global"] = res["qc_global_checks_pass"]
    res["qc_recovery"] = res["qc_global"] and res["qc_recovery_checks_pass"]
    res["qc_profile"] = res["qc_global"] and res["qc_profile_checks_pass"]
    res["qc_propagation"] = res["qc_global"] and res["qc_propagation_checks_pass"]
    res["qc_reasons"] = "; ".join(f"{k}: {m}" for k in reasons for m in reasons[k])


def analyse_control(row: pd.Series, bleach_site: dict) -> tuple[dict, dict]:
    """Autofluorescence level, drift and matrix bleaching at the typical bleach site."""
    data = load_measurement(DATA_ROOT / row["file"])
    pre, post, t = data["pre"], data["post"], data["post_t_s"]
    mpp = data["mpp_um"]
    fov_post = post.mean(axis=(1, 2))
    slope = np.polyfit(t, fov_post, 1)[0] if len(t) > 1 else np.nan
    centre = (bleach_site["row_um"] / mpp, bleach_site["col_um"] / mpp)
    curve = double_normalised_curve(pre, post, centre, bleach_site["r_e_um"], mpp)
    res = {"measurement_id": row["measurement_id"], "file": row["file"], "include": bool(row["include"]),
           "mpp_um": mpp, "frame_interval_s": data["frame_interval_s"], "n_pre_frames": len(pre),
           "n_post_frames": len(post), "fov_mean_pre": float(pre.mean()), "fov_mean_post_first": float(fov_post[0]),
           "fov_mean_post_last": float(fov_post[-1]), "drift_pct_per_min": 100 * slope * 60 / pre.mean(),
           "matrix_bleach_depth_K0": float(1 - curve["F_norm"][0]),
           "matrix_F_norm_last": float(curve["F_norm"][-1]),
           "saturated_fraction": float((pre >= SATURATION_COUNTS).mean()),
           "bleach_site": f"centre ({bleach_site['row_um']:.1f}, {bleach_site['col_um']:.1f}) um, r = {bleach_site['r_e_um']:.1f} um"}
    return res, {"t_s": t, "curve": curve, "folder": str(DATA_ROOT / row["file"])}


# ── Statistics ─────────────────────────────────────────────────────────────────

def _stats(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    n = len(v)
    mean = v.mean() if n else np.nan
    sd = v.std(ddof=1) if n > 1 else np.nan
    return {"N": n, "mean": mean, "SD": sd, "CV": sd / mean if n > 1 and mean else np.nan,
            "median": np.median(v) if n else np.nan, "min": v.min() if n else np.nan, "max": v.max() if n else np.nan}


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Statistics per condition, gel concentration and batch; concentrations are never pooled."""
    rows = []
    conc = results["gel_concentration_mg_ml"].astype(float).fillna(-1)
    for (condition, c), cond_df in results.groupby([results["condition"], conc]):
        subsets = [("all", cond_df)] + [(b, g) for b, g in cond_df.groupby("batch") if isinstance(b, str) and b]
        for batch, sub in subsets:
            for param, gate in PARAMETER_GATES.items():
                for selection, chosen in (("qc_pass", sub[sub[gate]]), ("all_included", sub[sub["include"]])):
                    rows.append({"condition": condition, "gel_concentration_mg_ml": c if c >= 0 else np.nan,
                                 "batch": batch, "parameter": param, "selection": selection,
                                 "gate": gate, **_stats(chosen[param].to_numpy(float)),
                                 "measurements": ", ".join(chosen.loc[chosen[param].notna(), "measurement_id"])})
    return pd.DataFrame(rows)


def _bootstrap_log_ratio(gel: np.ndarray, ref: Optional[np.ndarray], ref_const: Optional[float], rng) -> tuple:
    lg = np.log(gel)
    point = lg.mean() - (np.log(ref).mean() if ref is not None else np.log(ref_const))
    draws = lg[rng.integers(0, len(lg), (BOOTSTRAP_N, len(lg)))].mean(axis=1)
    if ref is not None:
        lr = np.log(ref)
        draws = draws - lr[rng.integers(0, len(lr), (BOOTSTRAP_N, len(lr)))].mean(axis=1)
    else:
        draws = draws - np.log(ref_const)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.exp(point)), float(np.exp(lo)), float(np.exp(hi))


def hindrance_ratios(results: pd.DataFrame, d_se: float) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {"definition": "H = geometric mean D_gel / geometric mean D_reference; 95 % CI: percentile bootstrap "
                         f"({BOOTSTRAP_N} draws, seed {BOOTSTRAP_SEED}), measurements resampled within each group",
           "main_reference": MAIN_REFERENCE,
           "D_SE_water_um2_s": d_se, "R_h_nm": DEXTRAN_RH_NM, "T_K": TEMPERATURE_K,
           "eta_water_Pa_s": WATER_VISCOSITY_PA_S, "methods": {}}
    gate = {m: PARAMETER_GATES[m] for m in D_METHODS}
    for method in D_METHODS:
        water = results[(results["condition"] == "water") & results[gate[method]]][method].dropna().to_numpy()
        entry = {"water_N": len(water), "water_geomean_um2_s": float(np.exp(np.log(water).mean())) if len(water) else None,
                 "groups": {}}
        gel = results[(results["condition"] == "gel") & results[gate[method]]]
        groups = []
        for c, gel_c in gel.groupby(gel["gel_concentration_mg_ml"].astype(float)):
            groups += [(f"gel_{c:g}mg_all", gel_c)] + [(f"gel_{c:g}mg_{b}", gb) for b, gb in gel_c.groupby("batch")]
        for label, g in groups:
            vals = g[method].dropna().to_numpy()
            if len(vals) == 0:
                entry["groups"][label] = {"N": 0}
                continue
            item = {"N": len(vals), "geomean_um2_s": float(np.exp(np.log(vals).mean()))}
            h, lo, hi = _bootstrap_log_ratio(vals, None, d_se, rng)
            item["H_vs_StokesEinstein"] = {"H": h, "CI95": [lo, hi]}
            if len(water):
                h, lo, hi = _bootstrap_log_ratio(vals, water, None, rng)
                item["H_vs_measured_water"] = {"H": h, "CI95": [lo, hi],
                                               "note": "few water measurements: bootstrap CI is coarse" if len(water) < 5 else ""}
            else:
                item["H_vs_measured_water"] = {"H": None, "reason": "no water measurement passes QC for this method"}
            entry["groups"][label] = item
        out["methods"][method] = entry
    return out


def leave_one_out(results: pd.DataFrame, method: str = "D_profile_um2_s", concentration: float = 20.0) -> pd.DataFrame:
    gel = results[(results["condition"] == "gel") & results[PARAMETER_GATES[method]]
                  & np.isclose(results["gel_concentration_mg_ml"].astype(float), concentration)].dropna(subset=[method])
    rows = []
    for idx, row in gel.iterrows():
        rest = np.log(gel.drop(index=idx)[method].to_numpy())
        ln_out = np.log(row[method])
        sd = rest.std(ddof=1) if len(rest) > 1 else np.nan
        z = (ln_out - rest.mean()) / (sd * np.sqrt(1 + 1 / len(rest))) if np.isfinite(sd) and sd > 0 else np.nan
        all_geo = np.exp(np.log(gel[method]).mean())
        rows.append({"left_out": row["measurement_id"], "batch": row["batch"], "value_um2_s": row[method],
                     "geomean_rest_um2_s": float(np.exp(rest.mean())),
                     "change_of_geomean_pct": 100 * (np.exp(rest.mean()) / all_geo - 1),
                     "CV_rest": float(np.std(np.exp(rest), ddof=1) / np.mean(np.exp(rest))) if len(rest) > 1 else np.nan,
                     "z_studentized_log": z, "potential_outlier": bool(np.isfinite(z) and abs(z) > LOO_OUTLIER_Z)})
    return pd.DataFrame(rows)


# ── Synthetic validation ───────────────────────────────────────────────────────

def simulate_stack(D_um2_s: float, w0_um: float, schedule, p=SYNTHETIC_PARAMS, components=None,
                   drift_um_s: float = 0.0) -> tuple:
    """8-bit FRAP stack: depletion = sum of Gaussian components (amplitude, w0) diffusing with D,
    optionally translated along x with drift_um_s; static texture, shot-like noise, acquisition bleaching."""
    rng = np.random.default_rng(p["seed"])
    n, mpp = p["size_px"], p["mpp_um"]
    rows, cols = np.indices((n, n))
    texture = 1 + p["texture_rel"] * gaussian_filter(rng.standard_normal((n, n)), 2) / 0.14
    t = np.cumsum(np.r_[0, np.concatenate([np.full(k, dt) for dt, k in schedule])[:-1]])
    components = components or [(p["K0"], w0_um)]

    def frame(base):
        img = base * texture
        img = img + rng.standard_normal(img.shape) * p["noise_rel"] * np.sqrt(p["intensity"] * img.clip(0))
        return np.clip(np.round(img), 0, 255).astype(np.float32)

    pre = np.stack([frame(np.full((n, n), p["intensity"])) for _ in range(p["n_pre"])])
    post = []
    for ti in t:
        r2 = ((rows - n / 2 + 0.3) ** 2 + (cols - n / 2 - 0.4 - drift_um_s * ti / mpp) ** 2) * mpp ** 2
        depletion = 0.0
        for amp, w0 in components:
            w2 = w0 ** 2 + 8 * D_um2_s * ti
            depletion = depletion + amp * w0 ** 2 / w2 * np.exp(-2 * r2 / w2)
        acq = np.exp(-ti / p["acq_bleach_tau_s"])
        post.append(frame(p["intensity"] * acq * (1 - depletion)))
    return pre, np.stack(post), t


def synthetic_validation() -> pd.DataFrame:
    """Accuracy (mean over noise realisations) and precision (SD) of both profile-based methods.

    The noise of the averaged pre-bleach profile enters every post-bleach frame identically, so a
    single realisation scatters by a few percent; the acceptance criterion therefore applies to
    the mean over SYNTHETIC_REPLICATES realisations, the SD is the expected per-measurement spread.
    Each method is required to pass only the cases it is designed for: the Gaussian-width method
    (D_profile) assumes a Gaussian profile and no drift; the propagation method must pass all cases.
    """
    rows = []
    for case in SYNTHETIC_CASES:
        dev = {"D_profile_um2_s": [], "D_propagation_um2_s": []}
        alpha = []
        for seed in range(1, SYNTHETIC_REPLICATES + 1):
            pre, post, t = simulate_stack(case["D_um2_s"], case["w0_um"], case["schedule"],
                                          dict(SYNTHETIC_PARAMS, seed=seed), case.get("components"),
                                          case.get("drift_um_s", 0.0))
            prof = profile_analysis(pre, post, t, SYNTHETIC_PARAMS["mpp_um"])
            prop = propagation_analysis(pre, post, t, SYNTHETIC_PARAMS["mpp_um"], prof["r_e_um"])
            dev["D_profile_um2_s"].append(prof["D_profile_um2_s"] / case["D_um2_s"] - 1)
            dev["D_propagation_um2_s"].append(prop["D_propagation_um2_s"] / case["D_um2_s"] - 1)
            alpha.append(prof["alpha"])
        gaussian_no_drift = case.get("components") is None and case.get("drift_um_s", 0.0) == 0.0
        for method, values in dev.items():
            v = np.array(values)
            applicable = method == "D_propagation_um2_s" or gaussian_no_drift
            rows.append({"method": method, "D_true_um2_s": case["D_um2_s"], "w0_true_um": case["w0_um"],
                         "case": case.get("label", "Gaussian"),
                         "frame_schedule": " + ".join(f"{k} x {dt:g} s" for dt, k in case["schedule"]),
                         "replicates": len(v), "mean_rel_deviation": v.mean(), "sd_rel_deviation": v.std(ddof=1),
                         "min_rel_deviation": v.min(), "max_rel_deviation": v.max(),
                         "mean_alpha_profile": np.nanmean(alpha), "applicable": applicable,
                         "passed": bool(abs(v.mean()) <= SYNTHETIC_TOLERANCE) if applicable else None})
    return pd.DataFrame(rows)


# ── Output helpers ─────────────────────────────────────────────────────────────

def _fmt(v, digits=3):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return "--" if not np.isfinite(v) else f"{v:.{digits}g}"
    return str(v).replace("_", r"\_")


def write_latex(df: pd.DataFrame, columns: Dict[str, str], path: Path, caption: str) -> None:
    """booktabs table; numeric columns as siunitx S columns with table-format left to the document."""
    numeric = [pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in columns]
    spec = "".join("S" if n else "l" for n in numeric)
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\begin{{tabular}}{{{spec}}}",
             r"\toprule", " & ".join(h if h.startswith("{") else f"{{{h}}}" for h in columns.values()) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(row[c]) for c in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def _md(df: pd.DataFrame) -> str:
    """Markdown table without extra dependencies; floats with 3 significant digits."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.3g}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_report(path: Path, synth: pd.DataFrame, results: pd.DataFrame, summary: pd.DataFrame,
                 ratios: dict, loo: pd.DataFrame, controls: pd.DataFrame) -> None:
    L = ["# FRAP v2 - phase 1 report", f"Generated {datetime.now():%Y-%m-%d %H:%M}.", "",
         "## Synthetic validation of the profile-based methods", _md(synth), "",
         "## QC status per measurement"]
    qc_cols = ["measurement_id", "condition", "batch", "qc_global", "qc_propagation", "qc_profile", "qc_recovery",
               "qc_reasons"]
    L += [_md(results[qc_cols]), "", "## Key results per measurement"]
    key = ["measurement_id", "r_e_um", "K0", "D_propagation_um2_s", "propagation_window_s", "max_drift_um",
           "D_profile_um2_s", "w2_r2", "alpha", "n_window", "t_half_s", "M_f",
           "D_soumpasis_um2_s", "D_axelrod_um2_s", "D_kang_um2_s", "recovery_r2", "saturated_fraction"]
    L += [_md(results[key]), "", "## Summary (QC-passed, all batches)"]
    s = summary[(summary["selection"] == "qc_pass") & (summary["batch"] == "all")]
    L += [_md(s[["condition", "gel_concentration_mg_ml", "parameter", "N", "mean", "SD", "CV"]]), "",
          "## Hindrance ratio", "```json", json.dumps(ratios, indent=2), "```", "",
          "## Leave-one-out (bulk gel, D_profile)", _md(loo) if len(loo) else "no data",
          "", "## noFITC controls", _md(controls)]
    path.write_text("\n".join(L), encoding="utf-8")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    synth = synthetic_validation()
    print("Synthetic validation of the profile-based methods:")
    print(synth.drop(columns=["frame_schedule"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    failed = synth[synth["applicable"] & (synth["passed"] != True)]  # noqa: E712 (None for non-applicable)
    if len(failed):
        raise RuntimeError(f"Synthetic validation failed (tolerance {SYNTHETIC_TOLERANCE:.0%}):\n{failed}\n"
                           "Real data are not analysed.")

    table = pd.read_csv(MEASUREMENTS_CSV, dtype={"batch": str, "note": str}, keep_default_na=False,
                        na_values={"r_nominal_um": [""], "gel_concentration_mg_ml": [""]})
    table["include"] = table["include"].astype(str).str.lower().isin(["true", "1", "yes"])
    fitc = table[table["condition"] != "gel_noFITC"]
    results, details = [], {}
    for _, row in fitc.iterrows():
        print(f"Analysing {row['measurement_id']} ({row['file']})")
        res, det = analyse_fitc(row)
        results.append(res)
        details[row["measurement_id"]] = det
    results = pd.DataFrame(results)

    standard = results[np.isclose(results["mpp_um"], results["mpp_um"].mode()[0]) & results["qc_global"]]
    mpp0 = results["mpp_um"].mode()[0]
    bleach_site = {"row_um": float(standard["centre_row_px"].median() * mpp0),
                   "col_um": float(standard["centre_col_px"].median() * mpp0),
                   "r_e_um": float(standard.loc[standard["condition"] == "gel", "r_e_um"].median())}
    controls, control_details = [], {}
    for _, row in table[table["condition"] == "gel_noFITC"].iterrows():
        print(f"Analysing control {row['measurement_id']} ({row['file']})")
        res, det = analyse_control(row, bleach_site)
        controls.append(res)
        control_details[row["measurement_id"]] = det
    controls = pd.DataFrame(controls)

    d_se = calculate_theoretical_diffusion(2 * DEXTRAN_RH_NM, TEMPERATURE_K, WATER_VISCOSITY_PA_S)
    summary = summary_table(results)
    ratios = hindrance_ratios(results, d_se)
    loo = pd.concat([leave_one_out(results, m).assign(method=m) for m in ("D_propagation_um2_s", "D_profile_um2_s")],
                    ignore_index=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_ROOT / "results_per_measurement.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "summary_per_condition.csv", index=False)
    loo.to_csv(OUTPUT_ROOT / "loo_gel.csv", index=False)
    controls.to_csv(OUTPUT_ROOT / "nofitc_controls.csv", index=False)
    synth.to_csv(OUTPUT_ROOT / "synthetic_validation.csv", index=False)
    (OUTPUT_ROOT / "ratio_gel_water.json").write_text(json.dumps(ratios, indent=2), encoding="utf-8")
    write_latex(results, {"measurement_id": "Measurement", "condition": "Condition", "r_e_um": r"{$r_e$ (\si{\micro\metre})}",
                          "K0": "{$K_0$}", "D_profile_um2_s": r"{$D_\mathrm{profile}$ (\si{\micro\metre\squared\per\second})}",
                          "t_half_s": r"{$t_{1/2}$ (\si{\second})}", "M_f": "{$M_f$}",
                          "D_soumpasis_um2_s": r"{$D_\mathrm{Soumpasis}$}", "D_kang_um2_s": r"{$D_\mathrm{Kang}$}",
                          "qc_profile": "QC profile", "qc_recovery": "QC recovery"},
                OUTPUT_ROOT / "results_per_measurement.tex", "FRAP results per measurement.")
    s = summary[(summary["selection"] == "qc_pass") & summary["parameter"].isin(D_METHODS + ["M_f", "t_half_s", "r_e_um"])]
    write_latex(s, {"condition": "Condition", "gel_concentration_mg_ml": r"{$c$ (\si{\milli\gram\per\milli\litre})}",
                    "batch": "Batch", "parameter": "Parameter", "N": "{$N$}", "mean": "{Mean}",
                    "SD": "{SD}", "CV": "{CV}"}, OUTPUT_ROOT / "summary_per_condition.tex",
                "FRAP summary per condition (QC-passed measurements).")
    write_report(OUTPUT_ROOT / "report_phase1.md", synth, results, summary, ratios, loo, controls)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump({"created": datetime.now(), "results": results, "details": details, "controls": controls,
                     "control_details": control_details, "summary": summary, "ratios": ratios, "loo": loo,
                     "synthetic": synth, "bleach_site": bleach_site, "d_se_um2_s": d_se,
                     "config": {"profile_r_max_um": PROFILE_R_MAX_UM, "profile_bin_um": PROFILE_BIN_UM,
                                "background_counts": BACKGROUND_COUNTS, "data_root": str(DATA_ROOT)}}, fh)

    print("\nQC and key results:")
    print(results[["measurement_id", "r_e_um", "K0", "D_propagation_um2_s", "propagation_window_s", "max_drift_um",
                   "D_profile_um2_s", "alpha", "t_half_s", "M_f", "D_soumpasis_um2_s", "qc_propagation", "qc_profile",
                   "qc_recovery"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nQC reasons:")
    for _, r in results.iterrows():
        if r["qc_reasons"]:
            print(f"  {r['measurement_id']}: {r['qc_reasons']}")
    print(f"\nD_SE(water, R_h = {DEXTRAN_RH_NM} nm, {TEMPERATURE_K - 273.15:.0f} degC) = {d_se:.1f} um2/s")
    print(f"Outputs: {OUTPUT_ROOT}\nCache:   {CACHE_PATH}")


if __name__ == "__main__":
    main()
