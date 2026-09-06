"""
FRAP Analysis Script
====================
Reads Leica SP8 FRAP data (pre-bleach & post-bleach TIF series),
extracts metadata from XML, fits 2D Gaussian to bleach spot,
and performs full FRAP recovery analysis.
"""

import os
import re
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(r"H:\Daten Promotion Sicherung\Confocal_Measure\2025_09_08_14_57_46--Project_TIF\Water 3")

# --- Image loading ---
CHANNEL_INDEX = 0            # TIF channel (l-index) to load - only channel with FITC data
BLACK_FRAME_THRESHOLD = 1.0  # frames with mean intensity <= this are dropped (bad acquisition frames)

# --- Bleach-center detection ---
CENTER_SMOOTH_SIGMA_PX = 5   # Gaussian smoothing of the (pre - post) difference image before locating the center
ROI_TRACK_CENTER = False     # False: center is located once from t=0 and held fixed for the whole series.
                             # True:  center is re-estimated every frame (weighted centroid) to follow
                             #        sample/stage drift during long acquisitions. Costs one extra
                             #        centroid pass per frame; only worth it if drift is visible in panel A-C.

# --- Bleach / reference ROI (region averaged to build the recovery curve) ---
# The Soumpasis D = 0.224*w^2/t_half formula assumes the averaging ROI radius
# equals the actual bleach-spot radius w. "auto" enforces that by deriving the
# ROI from the measured spot size (sigma0) instead of an arbitrary constant -
# this is what makes the resulting D roi-radius-independent instead of a free
# parameter you have to guess correctly.
BLEACH_ROI_RADIUS_MODE = "auto"  # "auto" (recommended) or "fixed"
BLEACH_ROI_RADIUS_PX = 80        # only used when BLEACH_ROI_RADIUS_MODE == "fixed"
BLEACH_ROI_SIGMA_FACTOR = 2.5    # only used when mode == "auto": roi_r = factor * sigma0
REF_RING_GAP_PX = 30             # gap between bleach-ROI edge and reference-ring inner edge
REF_RING_WIDTH_PX = 40           # width of the reference ring (background/photobleaching correction)

# --- Per-frame Gaussian dip fit (bleach-spot width & auto-ROI sizing) ---
GAUSS_FIT_BOX_RADIUS_PX = 100  # half-width of the fit box - only needs to comfortably contain the
                                # dip (a few x sigma), NOT the whole frame (that was the main reason
                                # the old per-frame fit was slow: its box covered nearly the full image)
GAUSS_FIT_MIN_FRACTION = 0.1    # pixels with dip depth below this fraction of the peak are excluded
                                # from the fit, keeping the log-linearization stable against noise

# --- ROI-variation diffusion estimate (recommended - genuinely ROI-radius independent) ---
# For fast diffusers the bleach spot can be too small/noisy to resolve in the
# images at all (see analyze_diffusion_roi_variation docstring), which makes
# the single-ROI Soumpasis D above scale with whatever ROI radius you happen
# to pick. This method sidesteps that: it fits the recovery curve at several
# ROI radii and reads D off the SLOPE of t_half vs radius^2 (Kang et al. 2012,
# Traffic) - it never needs to know the bleach-spot size at all.
ROI_VARIATION_RADII_PX = [15, 25, 35, 50, 70, 90, 115, 140]

# --- Gaussian bleach-profile diffusion estimate (Axelrod 1976) ---
# A third, independent estimate: radially bin the first post-bleach frame
# around a centroid-detected center and fit the actual Gaussian bleach
# profile for its effective radius r_e, then use r_e (not any nominal ROI)
# both to build the recovery-curve ROI and in the closed-form D = r_e^2/(4*tau).
# Radial binning averages O(rho) pixels per bin, which is what lets r_e be
# resolved here even though a per-pixel 2D fit is noise-dominated on this
# kind of data (see fit_gaussian_dip_2d's notes on that failure mode).
GAUSS_PROFILE_MAX_RADIUS_PX = 150  # outer radius of the radial profile
GAUSS_PROFILE_N_BINS = 40          # number of radial bins
BACKGROUND_PATCH_PX = 20           # size of the 4 corner patches used as I_bg for
                                    # the Phair/Misteli normalization

# =============================================================================
# 1. Discover subfolders (Pre / Pb)
# =============================================================================
def find_series_folders(base_dir):
    """Find all Pre-bleach and Post-bleach (Pb) subfolders.

    Returns (pre_folders, pb_folders) both as sorted lists of Paths.
    """
    pre_folders = []
    pb_folders = []
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if "Pre" in name:
            pre_folders.append(entry)
        elif "Pb" in name:
            pb_folders.append(entry)
    if not pre_folders:
        raise FileNotFoundError(f"Could not find any Pre folders in {base_dir}")
    if not pb_folders:
        raise FileNotFoundError(f"Could not find any Pb folders in {base_dir}")
    return pre_folders, pb_folders


# =============================================================================
# 2. Read metadata from XML
# =============================================================================
def parse_series_xml(xml_path):
    """Parse timestamps and dimension info from the series XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Timestamps (Windows FILETIME: 100ns intervals since 1601-01-01)
    timestamps = []
    for tsl in root.iter("TimeStampList"):
        hex_stamps = tsl.text.strip().split()
        for h in hex_stamps:
            ft = int(h, 16)
            unix_us = (ft - 116444736000000000) / 10  # microseconds
            dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=unix_us)
            timestamps.append(dt)
        break  # use only the first TimeStampList

    # Dimension descriptions (pixel size, time interval)
    dimensions = {}
    for dd in root.iter("DimensionDescription"):
        dim_id = int(dd.get("DimID", 0))
        n_elem = int(dd.get("NumberOfElements", 1))
        length = float(dd.get("Length", 0))
        unit = dd.get("Unit", "")
        dimensions[dim_id] = {
            "n_elements": n_elem,
            "length": length,
            "unit": unit,
            "element_size": length / n_elem if n_elem > 0 else 0,
        }

    return timestamps, dimensions


def parse_properties_xml(xml_path):
    """Extract key acquisition parameters from the _Properties.xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    params = {}

    # Find the first ATLConfocalSettingDefinition (top-level acquisition settings)
    for setting in root.iter("ATLConfocalSettingDefinition"):
        params["image_width"] = int(setting.get("InDimension", 512))
        params["image_height"] = int(setting.get("OutDimension", 512))
        params["zoom"] = float(setting.get("Zoom", 1).replace(",", "."))
        params["frame_time_s"] = float(setting.get("FrameTime", 0).replace(",", "."))
        params["cycle_count"] = int(setting.get("CycleCount", 1))
        params["complete_time_s"] = float(setting.get("CompleteTime", 0).replace(",", "."))
        params["scan_speed"] = setting.get("ScanSpeed", "")
        params["objective"] = setting.get("ObjectiveName", "").strip()
        params["magnification"] = float(setting.get("Magnification", 1))
        params["na"] = float(setting.get("NumericalAperture", 0))
        params["immersion"] = setting.get("Immersion", "")
        params["pinhole_um"] = float(setting.get("Pinhole", 0).replace(",", ".").replace(" µm", ""))
        params["pinhole_airy"] = setting.get("PinholeAiry", "")
        params["bit_depth"] = int(setting.get("BitSize", 8))
        params["pixel_dwell_time_s"] = float(setting.get("PixelDwellTime", 0))
        params["scan_mode"] = setting.get("ScanMode", "")
        break

    # Bleach parameters
    for frap_block in root.iter("Block_FRAP"):
        params["bleach_repetitions"] = int(frap_block.get("BleachRepetitions", 1))
        break

    for bp in root.iter("Attachment"):
        if bp.get("Name") == "BLEACH_POINT":
            params["bleach_duration_ms"] = float(bp.get("Duration", 0))
            break

    # Bleach laser intensity
    for bps in root.iter("ATLConfocalBleachPointSettingDefinition"):
        for aotf in bps.iter("Aotf"):
            if aotf.get("LightSourceName") == "Visible Light":
                for laser in aotf.iter("LaserLineSetting"):
                    intensity = float(laser.get("IntensityDev", 0))
                    wl = laser.get("LaserLine", "")
                    if intensity > 0:
                        params[f"bleach_laser_{wl}nm_pct"] = intensity
        break

    # Imaging laser intensity
    for hw in root.iter("Attachment"):
        if hw.get("Name") != "HardwareSetting":
            continue
        for setting in hw.iter("ATLConfocalSettingDefinition"):
            if setting.get("WizardMode") == "0" or setting.get("UserSettingName", "").startswith("S"):
                if "CycleCount" in setting.attrib and int(setting.get("CycleCount", 0)) > 1:
                    for aotf in setting.iter("Aotf"):
                        if aotf.get("LightSourceName") == "Visible Light":
                            for laser in aotf.iter("LaserLineSetting"):
                                intensity = float(laser.get("IntensityDev", 0))
                                wl = laser.get("LaserLine", "")
                                if intensity > 0:
                                    params[f"imaging_laser_{wl}nm_pct"] = intensity
                    break
        break

    return params


def parse_frame_timestamps(xml_path):
    """Parse per-frame timestamps from _Properties.xml.

    Reads <TimeStamp RelativeTime="..." Date="..." Time="..." MiliSeconds="..."/>
    entries from Data > Image > TimeStampList.

    Returns list of (relative_time_s, absolute_datetime) for each frame.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    timestamps = []
    for ts in root.iter("TimeStamp"):
        rel_time = float(ts.get("RelativeTime", 0))
        date_str = ts.get("Date", "")
        time_str = ts.get("Time", "")
        ms_str = ts.get("MiliSeconds", "0")
        ms = int(ms_str)

        if date_str and time_str:
            dt = datetime.datetime.strptime(
                f"{date_str} {time_str}", "%m/%d/%Y %I:%M:%S %p"
            )
            dt = dt.replace(microsecond=ms * 1000)
        else:
            dt = None

        timestamps.append((rel_time, dt))

    return timestamps


def parse_frap_roi(xml_path):
    """Extract bleach ROI geometry, pixel size, phase timing and laser power
    from a Leica _Properties.xml file.

    Returns a dict (all keys optional, present only when found in the XML):
      mpp_m, mpp_um        – pixel size in m/px and µm/px
      roi_cx_px, roi_cy_px – bleach ROI centre in image pixels (col, row)
      roi_rx_um, roi_ry_um – bleach ROI half-axes in µm
      roi_rx_px, roi_ry_px – bleach ROI half-axes in pixels
      roi_area_um2         – ellipse area  π·rx·ry  in µm²
      phases               – list of {name, time_s, frame_count}
      bleach_laser_pct     – 488 nm bleach laser power (%)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}

    # --- Pixel size ---
    # Leica XML files are inconsistent about units: the series XML's Length/
    # NumberOfElements is in meters, but this Properties XML's Voxel attribute
    # is already in the dimension's own Unit (typically µm, e.g. Voxel="0.283"
    # with Unit="µm" - NOT 0.283 meters). Blindly treating every value as
    # meters silently produced mpp_um values off by 1e6 for such files.
    def _voxel_to_meters(value, unit):
        unit = (unit or "").strip().lower()
        if unit in ("m", "meter", "meters"):
            return value
        if unit in ("mm",):
            return value * 1e-3
        if unit in ("nm",):
            return value * 1e-9
        if unit in ("µm", "μm", "um", "micron", "microns", ""):
            return value * 1e-6
        # Unrecognized unit string (encoding issues etc.) - fall back to a
        # magnitude check: no real microscope pixel is >= 1 mm, so a value
        # that large must already be in µm, not meters.
        return value if value < 1e-3 else value * 1e-6

    mpp_m = None
    # Try DimID="X" format (Properties XML)
    dd_x = root.find(".//DimensionDescription[@DimID='X']")
    if dd_x is not None:
        voxel = dd_x.get("Voxel")
        unit = dd_x.get("Unit", "")
        if voxel:
            mpp_m = _voxel_to_meters(float(voxel), unit)
        else:
            n = int(dd_x.get("NumberOfElements", 1))
            L = float(dd_x.get("Length", 0))
            mpp_m = _voxel_to_meters(L / n, unit) if n > 0 else None
    # Fall back to integer DimID="1" (series XML format, Length always in meters)
    if not mpp_m:
        for dd in root.iter("DimensionDescription"):
            if dd.get("DimID") == "1":
                n = int(dd.get("NumberOfElements", 1))
                L = float(dd.get("Length", 0))
                mpp_m = L / n if n > 0 else None
                break
    if mpp_m and mpp_m > 0:
        result["mpp_m"]  = mpp_m
        result["mpp_um"] = mpp_m * 1e6

    # --- Bleach ROI geometry ---
    roi_elem = root.find(".//ROIDATA")
    if roi_elem is not None:
        scaling = roi_elem.find("REALWORLD_SCALING")
        if scaling is not None:
            size_x_m = float(scaling.get("SizeX", 0))
            size_y_m = float(scaling.get("SizeY", 0))
            pos_x_m  = float(scaling.get("PositionX", 0))
            pos_y_m  = float(scaling.get("PositionY", 0))
            mpp = result.get("mpp_m")
            if mpp and mpp > 0:
                # PositionX/Y = top-left corner of bounding box → convert to centre
                cx_px = (pos_x_m + size_x_m / 2) / mpp
                cy_px = (pos_y_m + size_y_m / 2) / mpp
                result["roi_cx_px"] = int(round(cx_px))
                result["roi_cy_px"] = int(round(cy_px))
                rx_m = size_x_m / 2
                ry_m = size_y_m / 2
                result["roi_rx_um"]   = rx_m * 1e6
                result["roi_ry_um"]   = ry_m * 1e6
                result["roi_rx_px"]   = rx_m / mpp
                result["roi_ry_px"]   = ry_m / mpp
                result["roi_area_um2"] = np.pi * result["roi_rx_um"] * result["roi_ry_um"]

    # --- Phase timing ---
    phases = []
    phase_names = ["Pre-Bleach", "Bleach", "Post-Bleach 1", "Post-Bleach 2"]
    for i, info in enumerate(root.iter("Block_FRAP_Time_Info")):
        name = phase_names[i] if i < len(phase_names) else f"Phase {i+1}"
        phases.append({
            "name":        name,
            "time_s":      float(info.get("Time", 0)),
            "frame_count": int(info.get("FrameCount", 0)),
        })
    if phases:
        result["phases"] = phases

    # --- Bleach laser power (488 nm) ---
    ll = root.find(".//Block_FRAP_Bleach_Info//LaserLineSetting[@LaserLine='488']")
    if ll is not None:
        result["bleach_laser_pct"] = float(ll.get("IntensityDev", 0))

    return result


# =============================================================================
# 3. Load TIF images
# =============================================================================
def load_tif_series(folder, folder_name, channel=CHANNEL_INDEX, black_threshold=BLACK_FRAME_THRESHOLD):
    """
    Load TIF time series for a given channel (l-index).
    Returns (images, frame_indices) where images is (n_frames, h, w) float64
    and frame_indices lists the original t-index for each kept frame.
    Skips black frames (mean intensity <= black_threshold).

    Handles two Leica naming conventions:
      - With channel:    {name}_l{ch}_t{idx}.tif  (e.g. Pb folders)
      - Without channel: {name}_t{idx}.tif         (e.g. Pre folders)
    """
    escaped = re.escape(folder_name)
    # Try pattern with channel index first
    pattern_ch = re.compile(escaped + r"_l" + str(channel) + r"_t(\d+)\.tif$", re.IGNORECASE)
    # Fallback: pattern without channel index
    pattern_no_ch = re.compile(escaped + r"_t(\d+)\.tif$", re.IGNORECASE)

    files = []
    for f in folder.iterdir():
        m = pattern_ch.match(f.name)
        if m:
            files.append((int(m.group(1)), f))
    if not files:
        for f in folder.iterdir():
            m = pattern_no_ch.match(f.name)
            if m:
                files.append((int(m.group(1)), f))
    files.sort(key=lambda x: x[0])

    if not files:
        raise FileNotFoundError(f"No TIF files found in {folder}")

    images = []
    frame_indices = []
    n_skipped = 0
    for t_idx, fpath in files:
        img = np.array(Image.open(fpath))
        if img.ndim == 3:
            img = img.sum(axis=2)  # sum RGB channels
        img_f = img.astype(np.float64)
        if img_f.mean() <= black_threshold:
            n_skipped += 1
            continue
        images.append(img_f)
        frame_indices.append(t_idx)

    if n_skipped > 0:
        print(f"    Skipped {n_skipped} black frames (mean <= {black_threshold})")

    return np.array(images), frame_indices


# =============================================================================
# 4. Find bleach spot
# =============================================================================
def find_bleach_center(pre_avg, post_first, sigma_smooth=CENTER_SMOOTH_SIGMA_PX):
    """
    Find the bleach spot center by computing the difference
    (pre - post) and locating the maximum.
    """
    diff = pre_avg - post_first
    diff_smooth = gaussian_filter(diff, sigma=sigma_smooth)
    center = np.unravel_index(diff_smooth.argmax(), diff_smooth.shape)
    return center  # (row, col)


# =============================================================================
# 5. Gaussian dip fitting (fast, closed-form)
# =============================================================================
def _box_bounds(shape, center, box_radius):
    """Clip a (2*box_radius) square around `center` to the image bounds."""
    cy, cx = center
    h, w = shape
    y_min = max(0, cy - box_radius)
    y_max = min(h, cy + box_radius)
    x_min = max(0, cx - box_radius)
    x_max = min(w, cx + box_radius)
    return y_min, y_max, x_min, x_max


def fit_gaussian_dip_2d(pre_avg, image, center, box_radius=GAUSS_FIT_BOX_RADIUS_PX,
                         min_fraction=GAUSS_FIT_MIN_FRACTION):
    """
    Fit a 2D Gaussian dip with a FREE center - not held fixed at `center` -
    using the true per-pixel pre-bleach baseline as the reference:

        z(x,y) = pre_avg(x,y) - image(x,y) = amplitude * exp(-((x-cx)^2+(y-cy)^2)/(2*sigma^2))

    `center` is only used to place the fit box; the returned (cy, cx) is the
    fitted position. Caruana's log-linearization extends to a free center
    because ln(z) is linear in (1, x, y, x^2+y^2):

        ln(z) = A + B*x + C*y + Dc*(x^2+y^2)
        sigma = sqrt(-1/(2*Dc)),  cx = -B/(2*Dc),  cy = -C/(2*Dc)

    so amplitude, sigma AND the center all come from one closed-form weighted
    least-squares solve (weights = z^2) - no iterative solver, and no need to
    trust a fixed center from a crude peak-detection step beforehand.

    Using pre_avg instead of a scalar background estimate also removes a
    second problem the previous version had: a box-percentile background
    drifts as the box grows, which was inflating sigma essentially without
    bound as box_radius increased.

    Returns (amplitude, sigma, (cy_fit, cx_fit)) or None when the box doesn't
    resolve a trustworthy, well-localized dip (sigma >= box_radius, or the
    fitted center falls outside the box - both mean the box is dominated by
    background/noise rather than an actual localized spot).
    """
    cy, cx = center
    y_min, y_max, x_min, x_max = _box_bounds(image.shape, center, box_radius)
    pre_roi = pre_avg[y_min:y_max, x_min:x_max]
    roi = image[y_min:y_max, x_min:x_max]
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    z = pre_roi - roi
    peak = z.max()
    if peak <= 0:
        return None

    mask = z > (min_fraction * peak)
    if mask.sum() < 8:  # need a clear margin over the 4 fitted unknowns
        return None

    x = xx[mask].astype(np.float64)
    y = yy[mask].astype(np.float64)
    zz = np.clip(z[mask], 1e-6, None)
    lnz = np.log(zz)
    w = zz ** 2

    design = np.column_stack([np.ones_like(x), x, y, x ** 2 + y ** 2])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(design * sw[:, None], lnz * sw, rcond=None)
    A, B, C, Dc = beta

    if Dc >= 0:
        return None  # not a valid dip (would give an imaginary sigma)

    sigma = np.sqrt(-1.0 / (2.0 * Dc))
    cx_fit = -B / (2.0 * Dc)
    cy_fit = -C / (2.0 * Dc)

    if sigma >= box_radius:
        return None  # width not resolved within the fit box - extrapolation, not a measurement
    if abs(cx_fit - cx) > box_radius or abs(cy_fit - cy) > box_radius:
        return None  # fitted center escaped the box - not trustworthy

    amplitude = np.exp(A - Dc * (cx_fit ** 2 + cy_fit ** 2))
    return amplitude, sigma, (cy_fit, cx_fit)


# =============================================================================
# 6. FRAP recovery curve fitting
# =============================================================================
def frap_recovery_single_exp(t, I_inf, I_0, tau):
    """Single-exponential FRAP recovery model."""
    return I_inf - (I_inf - I_0) * np.exp(-t / tau)


def analyze_frap_recovery(times_s, intensities, pre_intensity):
    """
    Fit FRAP recovery curve.
    Returns dict with fit parameters and derived quantities.
    """
    t = times_s - times_s[0]
    I_norm = intensities / pre_intensity

    # Initial guesses
    I_0_guess = float(np.clip(I_norm[0],  0.0, 2.0))
    I_inf_guess = float(np.clip(I_norm[-1], 0.0, 2.0))
    tau_guess = t[-1] / 4

    try:
        popt, pcov = curve_fit(
            frap_recovery_single_exp,
            t,
            I_norm,
            p0=[I_inf_guess, I_0_guess, tau_guess],
            bounds=([0, 0, 0.01], [2, 2, t[-1] * 10]),
            maxfev=10000,
        )
        I_inf_fit, I_0_fit, tau_fit = popt
        perr = np.sqrt(np.diag(pcov))

        t_half = tau_fit * np.log(2)
        mobile_fraction = (I_inf_fit - I_0_fit) / (1.0 - I_0_fit) if (1.0 - I_0_fit) > 0 else np.nan
        immobile_fraction = 1.0 - mobile_fraction

        return {
            "I_inf": I_inf_fit,
            "I_0": I_0_fit,
            "tau_s": tau_fit,
            "tau_err": perr[2],
            "t_half_s": t_half,
            "mobile_fraction": mobile_fraction,
            "immobile_fraction": immobile_fraction,
            "t_fit": t,
            "I_norm": I_norm,
            "I_fit": frap_recovery_single_exp(t, *popt),
        }
    except RuntimeError:
        return None


def analyze_diffusion_roi_variation(pb_images, pre_images, pb_times_s, bleach_center,
                                     mpp_um, radii_px, ref_gap_px, ref_width_px):
    """
    Determine D without ever needing to measure the bleach-spot size,
    by fitting the FRAP recovery curve at several different ROI radii and
    reading D off the SLOPE of t_half vs radius^2 (Kang et al. 2012, Traffic
    13:1589-1600): in the diffusion-dominated regime (ROI at least comparable
    to the true bleach spot) t_half grows linearly with radius^2, following
    the same Soumpasis prefactor already used elsewhere in this script:

        t_half(r) = (0.224 / D) * r^2 + t_half_0

    so D = 0.224 / slope. Because this reads D from how t_half *changes*
    between ROI choices rather than from its value at one specific radius, it
    is not sensitive to which radius you happen to test at - unlike the
    single-ROI Soumpasis estimate, which requires an accurate, independently
    measured bleach-spot radius w and is only as good as that measurement.
    This is the right tool when w can't be resolved in the images at all
    (e.g. a fast diffuser where the spot has already smoothed out by the
    first captured frame).

    All tested radii share ONE reference ring (placed just outside the
    largest radius) for the photobleaching-correction normalization, so only
    the bleach-disk radius itself varies between fits.

    Returns a dict with D_um2_s, the fitted slope/intercept, r2_fit (quality
    of the linear t_half-vs-r^2 fit), and the per-radius t_half values used -
    or None if there aren't enough usable radii to fit a slope.
    """
    h, w = pb_images.shape[1:]
    cy, cx = bleach_center
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    edge_margin = min(cy, h - cy, cx, w - cx)
    usable_radii = [r for r in radii_px if r + ref_gap_px + ref_width_px < edge_margin]
    dropped = sorted(set(radii_px) - set(usable_radii))
    if dropped:
        print(f"  ROI-variation: dropped radii too close to the image edge for a "
              f"reference ring: {dropped}")
    if len(usable_radii) < 3:
        print("  ROI-variation: fewer than 3 usable ROI radii - cannot fit a slope.")
        return None

    ref_inner = max(usable_radii) + ref_gap_px
    ref_outer = ref_inner + ref_width_px
    ref_mask = (dist2 >= ref_inner ** 2) & (dist2 <= ref_outer ** 2)
    pre_ref_I = np.array([img[ref_mask].mean() for img in pre_images]).mean()
    pb_ref = np.array([img[ref_mask].mean() for img in pb_images])

    r_list, t_half_list, curves = [], [], []
    for r in usable_radii:
        bleach_mask = dist2 <= r ** 2
        pre_I = np.array([img[bleach_mask].mean() for img in pre_images]).mean()
        pb_bleach = np.array([img[bleach_mask].mean() for img in pb_images])
        norm = (pb_bleach / pb_ref) / (pre_I / pre_ref_I)
        fit = analyze_frap_recovery(pb_times_s, norm, 1.0)
        if fit is not None:
            r_list.append(r)
            t_half_list.append(fit["t_half_s"])
            curves.append({
                "r_px": r,
                "r_um": r * mpp_um,
                "t_fit": fit["t_fit"],
                "I_norm": fit["I_norm"],
                "I_fit": fit["I_fit"],
                "t_half_s": fit["t_half_s"],
            })

    if len(r_list) < 3:
        print("  ROI-variation: fewer than 3 successful recovery fits - cannot fit a slope.")
        return None

    r2_um2 = (np.array(r_list) * mpp_um) ** 2
    t_half_s = np.array(t_half_list)

    design = np.vstack([r2_um2, np.ones_like(r2_um2)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, t_half_s, rcond=None)

    if slope <= 0:
        print("  ROI-variation: non-positive slope (t_half does not grow with ROI "
              "size) - D undefined.")
        return None

    D_um2_s = 0.224 / slope
    pred = slope * r2_um2 + intercept
    ss_res = np.sum((t_half_s - pred) ** 2)
    ss_tot = np.sum((t_half_s - t_half_s.mean()) ** 2)
    r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if r2_fit < 0.9:
        print(f"  ROI-variation: t_half-vs-r² fit quality is low (R²={r2_fit:.3f}) - "
              "some tested radii are probably too close to the (unresolved) bleach "
              "spot, where the linear t_half~r² relation breaks down. Try dropping "
              "the smallest radii from ROI_VARIATION_RADII_PX and re-running.")

    return {
        "D_um2_s": D_um2_s,
        "slope": slope,
        "intercept": intercept,
        "r2_fit": r2_fit,
        "radii_px": r_list,
        "radii_um2": r2_um2,
        "t_half_s": t_half_s,
        "curves": curves,
    }


# =============================================================================
# 7. Gaussian bleach-profile diffusion estimate (Axelrod 1976)
# =============================================================================
def find_bleach_centroid(pre_avg, post_first, smooth_sigma=CENTER_SMOOTH_SIGMA_PX,
                          threshold_frac=0.1):
    """
    Locate the bleach center as the intensity-weighted centroid of the bleach
    deficit (pre_avg - post_first) - not the single-pixel argmax used by
    find_bleach_center, and not derived from any assumed ROI. Smoothing
    suppresses pixel noise before thresholding; thresholding at
    `threshold_frac` of the peak deficit keeps far-field noise (zero-mean but
    nonzero variance) from pulling the centroid off the real spot.

    Returns a sub-pixel (cy, cx) float tuple.
    """
    diff = pre_avg - post_first
    diff_smooth = gaussian_filter(diff, sigma=smooth_sigma)
    peak = diff_smooth.max()
    if peak <= 0:
        raise ValueError("No bleach deficit detected in (pre_avg - post_first).")

    weights = np.clip(diff_smooth - threshold_frac * peak, 0, None)
    total = weights.sum()
    if total <= 0:
        cy, cx = np.unravel_index(diff_smooth.argmax(), diff_smooth.shape)
        return float(cy), float(cx)

    yy, xx = np.mgrid[0:diff.shape[0], 0:diff.shape[1]]
    cy = np.sum(weights * yy) / total
    cx = np.sum(weights * xx) / total
    return float(cy), float(cx)


def radial_profile(image, center, mpp_um, max_radius_px, n_bins):
    """
    Azimuthally-averaged radial intensity profile of `image` around a
    sub-pixel `center`. Averaging over all pixels at similar radius (O(rho)
    of them per bin) is what makes the bleach profile resolvable from noisy
    single-frame data even when a per-pixel fit isn't (see module notes).

    Returns (rho_um, mean_intensity, pixel_count) for bins that contain data.
    """
    cy, cx = center
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rho_px = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    mask = rho_px <= max_radius_px
    rho_px = rho_px[mask]
    vals = image[mask]

    bin_edges = np.linspace(0, max_radius_px, n_bins + 1)
    bin_idx = np.clip(np.digitize(rho_px, bin_edges) - 1, 0, n_bins - 1)

    rho_bin_um = 0.5 * (bin_edges[:-1] + bin_edges[1:]) * mpp_um
    mean_intensity = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        sel = bin_idx == i
        counts[i] = sel.sum()
        if counts[i] > 0:
            mean_intensity[i] = vals[sel].mean()

    valid = counts > 0
    return rho_bin_um[valid], mean_intensity[valid], counts[valid]


def gaussian_bleach_profile(rho, I0, K, r_e):
    """Axelrod (1976) Gaussian bleach profile: I(rho) = I0 - K*exp(-2*rho^2/r_e^2)."""
    return I0 - K * np.exp(-2 * rho ** 2 / r_e ** 2)


def fit_gaussian_bleach_profile(rho_um, intensity, counts):
    """
    Fit I(rho) = I0 - K*exp(-2*rho^2/r_e^2) to the radially-binned profile of
    the first post-bleach frame. r_e is the effective 1/e^2 bleach radius -
    the single geometric parameter the Axelrod D formula needs. Bins with
    more pixels (larger `counts`) get proportionally lower fit weight-error,
    since their mean is better determined.

    Returns a dict with I0, K, r_e_um (+ 1-sigma errors from the fit
    covariance) and the fitted curve, or None if the fit doesn't converge.
    """
    I0_guess = float(intensity[-1])
    K_guess = max(I0_guess - float(intensity.min()), 1e-3)
    re_guess = float(rho_um[len(rho_um) // 3]) if len(rho_um) > 3 else float(rho_um.max()) / 2
    p0 = [I0_guess, K_guess, re_guess]
    bounds = ([0, 0, 1e-3], [np.inf, np.inf, rho_um.max() * 5])
    sigma = 1.0 / np.sqrt(counts)

    try:
        popt, pcov = curve_fit(gaussian_bleach_profile, rho_um, intensity, p0=p0,
                                bounds=bounds, sigma=sigma, maxfev=10000)
    except RuntimeError:
        return None

    perr = np.sqrt(np.diag(pcov))
    I0_fit, K_fit, re_fit = popt
    fit_curve = gaussian_bleach_profile(rho_um, *popt)
    ss_res = np.sum((intensity - fit_curve) ** 2)
    ss_tot = np.sum((intensity - intensity.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "I0": I0_fit, "K": K_fit, "r_e_um": re_fit,
        "I0_err": perr[0], "K_err": perr[1], "r_e_err_um": perr[2],
        "r2": r2,
        "rho_um": rho_um, "intensity": intensity,
        "fit_curve": fit_curve,
    }


def phair_misteli_normalize(pb_images, pre_images, roi_mask, ref_mask, bg_mask=None):
    """
    Double-normalize the recovery curve (Phair & Misteli):

        I_norm(t) = [I_ref(pre)/I_ref(t)] * [I_roi(t) - I_bg(t)] / [I_roi(pre) - I_bg(t)]

    The I_ref(pre)/I_ref(t) factor cancels acquisition photobleaching; the
    background subtraction removes any constant offset (camera/dark counts)
    from both the numerator and the pre-bleach reference level.

    `bg_mask=None` means I_bg=0 for every frame - the right choice whenever
    there is no genuinely dark/no-sample region in the field of view (see
    analyze_gaussian_profile_diffusion, which auto-detects this and falls
    back to bg_mask=None rather than subtracting a "background" that is
    actually just more sample).
    """
    I_ref_pre = np.array([img[ref_mask].mean() for img in pre_images]).mean()
    I_roi_pre = np.array([img[roi_mask].mean() for img in pre_images]).mean()

    I_ref_t = np.array([img[ref_mask].mean() for img in pb_images])
    I_roi_t = np.array([img[roi_mask].mean() for img in pb_images])
    I_bg_t = np.array([img[bg_mask].mean() for img in pb_images]) if bg_mask is not None else 0.0

    return (I_ref_pre / I_ref_t) * (I_roi_t - I_bg_t) / (I_roi_pre - I_bg_t)


def analyze_gaussian_profile_diffusion(pre_images, pb_images, pb_times_s, pre_avg, mpp_um,
                                        max_radius_px=GAUSS_PROFILE_MAX_RADIUS_PX,
                                        n_bins=GAUSS_PROFILE_N_BINS,
                                        ref_gap_px=REF_RING_GAP_PX, ref_width_px=REF_RING_WIDTH_PX,
                                        bg_patch_px=BACKGROUND_PATCH_PX):
    """
    Third, independent diffusion estimate built entirely from the actual
    Gaussian bleach profile instead of any nominal or auto-derived ROI:

      1. locate the bleach center as the deficit centroid (find_bleach_centroid),
      2. radially bin the first post-bleach frame around that center and fit
         it for the effective bleach radius r_e (fit_gaussian_bleach_profile),
      3. build the recovery curve in a disk of radius r_e, double-normalized
         with the Phair/Misteli formula,
      4. fit the recovery curve (reusing analyze_frap_recovery) for tau, then:
           D_axelrod = r_e^2 / (4*tau)              - closed-form, matches the
                                                       Gaussian-profile derivation
           D_soumpasis_approx = 0.224*r_e^2/t_half  - disk-profile approximation,
                                                       reported only as a plausibility
                                                       cross-check against D_axelrod

    Returns a dict with the profile fit, recovery fit, r_e (+ error), tau
    (+ error), both D estimates (D_axelrod with propagated error), or None if
    any step fails.
    """
    h, w = pre_avg.shape

    center = find_bleach_centroid(pre_avg, pb_images[0])
    cy, cx = center

    rho_um, intensity, counts = radial_profile(pb_images[0], center, mpp_um, max_radius_px, n_bins)
    if len(rho_um) < 5:
        print("  Gaussian-profile method: too few radial bins with data.")
        return None

    profile_fit = fit_gaussian_bleach_profile(rho_um, intensity, counts)
    if profile_fit is None:
        print("  Gaussian-profile method: radial bleach-profile fit failed.")
        return None
    r_e_um = profile_fit["r_e_um"]
    r_e_err_um = profile_fit["r_e_err_um"]
    r_e_px = r_e_um / mpp_um

    edge_margin = min(cy, h - cy, cx, w - cx)
    if r_e_px + ref_gap_px + ref_width_px >= edge_margin:
        print(f"  Gaussian-profile method: fitted r_e={r_e_um:.2f} µm too large for "
              "a reference ring within the image - skipping.")
        return None

    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    roi_mask = dist2 <= r_e_px ** 2
    ref_inner = r_e_px + ref_gap_px
    ref_outer = ref_inner + ref_width_px
    ref_mask = (dist2 >= ref_inner ** 2) & (dist2 <= ref_outer ** 2)

    corner_mask = np.zeros((h, w), dtype=bool)
    corner_mask[:bg_patch_px, :bg_patch_px] = True
    corner_mask[:bg_patch_px, -bg_patch_px:] = True
    corner_mask[-bg_patch_px:, :bg_patch_px] = True
    corner_mask[-bg_patch_px:, -bg_patch_px:] = True
    corner_mask &= ~roi_mask  # guard against a very large r_e reaching a corner

    # The corner patches are only a valid I_bg if they are clearly darker than
    # the sample (e.g. no-cell/no-gel regions). If the FOV is filled with
    # signal everywhere (as for a bulk liquid/gel sample), the corners are
    # just more sample, and using them as I_bg makes I_roi_pre - I_bg near
    # zero (or negative) - blowing up I_norm. Check this before trusting it.
    I_roi_pre_check = np.array([img[roi_mask].mean() for img in pre_images]).mean()
    I_corner_pre_check = np.array([img[corner_mask].mean() for img in pre_images]).mean()
    if I_corner_pre_check < 0.5 * I_roi_pre_check:
        bg_mask = corner_mask
    else:
        print(f"  Gaussian-profile method: corner patches ({I_corner_pre_check:.1f}) are not "
              f"clearly darker than the ROI ({I_roi_pre_check:.1f}) - this FOV has no real "
              "background region, using I_bg=0 instead of subtracting sample signal.")
        bg_mask = None

    I_norm = phair_misteli_normalize(pb_images, pre_images, roi_mask, ref_mask, bg_mask)
    if not np.all(np.isfinite(I_norm)) or np.any(np.abs(I_norm) > 5):
        print("  Gaussian-profile method: normalized recovery curve is not physically "
              "plausible (non-finite or |I_norm|>5) - skipping.")
        return None

    recovery_fit = analyze_frap_recovery(pb_times_s, I_norm, 1.0)
    if recovery_fit is None:
        print("  Gaussian-profile method: recovery curve fit failed.")
        return None

    tau_s = recovery_fit["tau_s"]
    tau_err_s = recovery_fit["tau_err"]
    t_half_s = recovery_fit["t_half_s"]

    D_axelrod_um2_s = r_e_um ** 2 / (4 * tau_s)
    D_axelrod_err_um2_s = D_axelrod_um2_s * np.sqrt(
        (2 * r_e_err_um / r_e_um) ** 2 + (tau_err_s / tau_s) ** 2
    )
    D_soumpasis_approx_um2_s = 0.224 * r_e_um ** 2 / t_half_s

    return {
        "center": center,
        "r_e_um": r_e_um,
        "r_e_err_um": r_e_err_um,
        "profile_fit": profile_fit,
        "recovery_fit": recovery_fit,
        "I_norm": I_norm,
        "tau_s": tau_s,
        "tau_err_s": tau_err_s,
        "t_half_s": t_half_s,
        "D_axelrod_um2_s": D_axelrod_um2_s,
        "D_axelrod_err_um2_s": D_axelrod_err_um2_s,
        "D_soumpasis_approx_um2_s": D_soumpasis_approx_um2_s,
    }


# =============================================================================
# 8. Main analysis
# =============================================================================
def main():
    pre_folders, pb_folders = find_series_folders(BASE_DIR)

    print(f"Pre-bleach folders found: {len(pre_folders)}")
    for pf in pre_folders:
        print(f"  - {pf.name}")
    print(f"Post-bleach folders found: {len(pb_folders)}")
    for pf in pb_folders:
        print(f"  - {pf.name}")

    # ---- Metadata (from first Pb folder, for general acquisition info) ----
    pb0_name = pb_folders[0].name
    pb_props_xml = pb_folders[0] / "MetaData" / f"{pb0_name}_Properties.xml"
    pb_stamps_xml = pb_folders[0] / "MetaData" / f"{pb0_name}.xml"

    params = parse_properties_xml(pb_props_xml)
    _, pb_dims = parse_series_xml(pb_stamps_xml)

    # Pixel size from DimensionDescription (DimID 1 = X, 2 = Y)
    if 1 in pb_dims:
        params["mpp_m"] = pb_dims[1]["element_size"]          # meters/pixel
        params["mpp_um"] = pb_dims[1]["element_size"] * 1e6    # µm/pixel
        params["fov_um"] = pb_dims[1]["length"] * 1e6          # µm

    print("\n=== Acquisition Parameters (from first Pb folder) ===")
    print(f"  Objective:          {params.get('objective', '?')}")
    print(f"  Magnification:      {params.get('magnification', '?')}x")
    print(f"  NA:                 {params.get('na', '?')}")
    print(f"  Immersion:          {params.get('immersion', '?')}")
    print(f"  Pixel size:         {params.get('mpp_um', 0):.4f} µm/px")
    print(f"  Image size:         {params.get('image_width')}x{params.get('image_height')} px")
    print(f"  Zoom:               {params.get('zoom', '?')}")
    print(f"  Pinhole:            {params.get('pinhole_um', '?')} µm ({params.get('pinhole_airy', '?')})")
    print(f"  Bit depth:          {params.get('bit_depth', '?')} bit")
    print(f"  Scan mode:          {params.get('scan_mode', '?')}")
    print(f"  Bleach reps:        {params.get('bleach_repetitions', '?')}")
    print(f"  Bleach duration:    {params.get('bleach_duration_ms', '?')} ms")
    for k, v in params.items():
        if "laser" in k:
            print(f"  {k}: {v:.2f}%")

    # Field of view
    fov_um = params.get("fov_um", params.get("mpp_um", 0) * params.get("image_width", 512))
    print(f"  Field of view:      {fov_um:.1f} µm")

    # ---- Load pre-bleach images from ALL Pre folders ----
    print("\nLoading pre-bleach images...")
    all_pre_images = []
    all_pre_timestamps = []
    for pre_folder in pre_folders:
        pre_name = pre_folder.name
        print(f"  Reading {pre_name}...")
        imgs, frame_idx = load_tif_series(pre_folder, pre_name)
        all_pre_images.append(imgs)
        print(f"    {imgs.shape[0]} frames, shape {imgs.shape[1:]}")

        pre_xml = pre_folder / "MetaData" / f"{pre_name}.xml"
        if pre_xml.exists():
            stamps, _ = parse_series_xml(pre_xml)
            all_pre_timestamps.extend(stamps[:len(frame_idx)])
        else:
            print(f"    Warning: no XML found at {pre_xml}")

    pre_images = np.concatenate(all_pre_images, axis=0)
    print(f"  Total pre-bleach frames: {pre_images.shape[0]}")

    # ---- Load post-bleach images from ALL Pb folders ----
    # Per-frame real timestamps from each folder's _Properties.xml
    print("\nLoading post-bleach images...")
    all_pb_images = []
    all_pb_abs_times = []  # absolute datetime per frame (for stitching)
    print("\n  === Per-folder timing ===")
    for pb_folder in pb_folders:
        pb_name = pb_folder.name
        print(f"  Reading {pb_name}...")
        imgs, frame_idx = load_tif_series(pb_folder, pb_name)
        all_pb_images.append(imgs)

        # Parse per-frame timestamps from Properties XML
        props_xml = pb_folder / "MetaData" / f"{pb_name}_Properties.xml"
        frame_ts = parse_frame_timestamps(props_xml)

        # Compute actual frame interval from RelativeTime
        if len(frame_ts) >= 2:
            dt_interval = frame_ts[1][0] - frame_ts[0][0]
        elif len(frame_ts) == 1:
            dt_interval = frame_ts[0][0]
        else:
            dt_interval = 0
        total_dur = frame_ts[-1][0] if frame_ts else 0
        fps = 1.0 / dt_interval if dt_interval > 0 else 0
        print(f"    {imgs.shape[0]} frames, real interval = {dt_interval:.3f} s "
              f"({fps:.2f} fps), duration = {total_dur:.1f} s")

        # Map loaded frame indices to their absolute datetimes
        for idx in frame_idx:
            if idx < len(frame_ts) and frame_ts[idx][1] is not None:
                all_pb_abs_times.append(frame_ts[idx][1])
            elif frame_ts and frame_ts[-1][1] is not None:
                all_pb_abs_times.append(frame_ts[-1][1])
            else:
                # Last resort: use last known time
                if all_pb_abs_times:
                    all_pb_abs_times.append(all_pb_abs_times[-1])
                else:
                    all_pb_abs_times.append(datetime.datetime(2000, 1, 1))

    pb_images = np.concatenate(all_pb_images, axis=0)
    n_pb = pb_images.shape[0]

    # ---- Build continuous time axis from absolute timestamps ----
    t0 = all_pb_abs_times[0]
    pb_times_s = np.array([(t - t0).total_seconds() for t in all_pb_abs_times])

    # Pre-bleach timestamps (relative)
    if all_pre_timestamps:
        pre_times_s = np.array([(t - all_pre_timestamps[0]).total_seconds()
                                for t in all_pre_timestamps])
    else:
        pre_times_s = np.zeros(pre_images.shape[0])

    print(f"\n  Total pre-bleach frames:  {pre_images.shape[0]}")
    print(f"  Total post-bleach frames: {n_pb}")
    print(f"  Post-bleach time range:   0 - {pb_times_s[-1]:.1f} s")

    # ---- Pre-bleach average ----
    pre_avg = pre_images.mean(axis=0)
    pre_intensity_full = pre_avg.mean()
    print(f"\n  Pre-bleach mean intensity: {pre_intensity_full:.2f}")

    # ---- Find bleach center ----
    # find_bleach_center only gives a coarse, single-pixel starting guess
    # (argmax of a smoothed difference image). Refine it with the free-center
    # Gaussian fit below so the ROI/analysis center actually sits on the fitted
    # peak instead of that coarse guess.
    bleach_center = find_bleach_center(pre_avg, pb_images[0])
    print(f"  Bleach spot center (coarse): row={bleach_center[0]}, col={bleach_center[1]}")
    h, w = pre_avg.shape

    # ---- Determine the bleach-spot size (sigma0) and, from it, the ROI radius ----
    fit0 = fit_gaussian_dip_2d(pre_avg, pb_images[0], bleach_center)
    if fit0 is not None:
        _, sigma0_px, fitted_center0 = fit0
        bleach_center = (int(round(fitted_center0[0])), int(round(fitted_center0[1])))
        print(f"  Bleach spot center (Gaussian fit): row={bleach_center[0]}, col={bleach_center[1]}")
    else:
        sigma0_px = None

    if BLEACH_ROI_RADIUS_MODE == "auto" and sigma0_px is not None:
        roi_r = int(round(BLEACH_ROI_SIGMA_FACTOR * sigma0_px))
        print(f"  Bleach spot sigma0:  {sigma0_px:.1f} px -> ROI radius (auto, "
              f"{BLEACH_ROI_SIGMA_FACTOR}x sigma0) = {roi_r} px")
    else:
        if BLEACH_ROI_RADIUS_MODE == "auto":
            print("  Warning: initial Gaussian fit failed, falling back to fixed ROI radius.")
        roi_r = BLEACH_ROI_RADIUS_PX
        print(f"  Bleach ROI radius (fixed): {roi_r} px")

    ref_inner = roi_r + REF_RING_GAP_PX
    ref_outer = ref_inner + REF_RING_WIDTH_PX
    bleach_r_um = roi_r * params.get("mpp_um", 1)
    print(f"  Bleach ROI radius:   {roi_r} px = {bleach_r_um:.1f} µm")

    yy, xx = np.ogrid[:h, :w]

    def masks_for_center(center):
        cy, cx = center
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        return dist2 <= roi_r ** 2, (dist2 >= ref_inner ** 2) & (dist2 <= ref_outer ** 2)

    bleach_mask, ref_mask = masks_for_center(bleach_center)

    # ---- Pre-bleach reference intensities (fixed center; nothing to track pre-bleach) ----
    pre_I = np.array([img[bleach_mask].mean() for img in pre_images]).mean()
    pre_ref_I = np.array([img[ref_mask].mean() for img in pre_images]).mean()

    # ---- Post-bleach: per-frame ROI intensities + Gaussian dip fit in one pass ----
    print(f"\nFitting Gaussian dip to each post-bleach frame "
          f"({'tracked center' if ROI_TRACK_CENTER else 'fixed center'}="
          f"({bleach_center[0]},{bleach_center[1]}), box radius={GAUSS_FIT_BOX_RADIUS_PX} px)...")

    pb_bleach_roi = np.empty(n_pb)
    pb_ref_roi = np.empty(n_pb)
    gauss_amplitudes = np.full(n_pb, np.nan)
    gauss_sigmas = np.full(n_pb, np.nan)
    tracked_centers = np.empty((n_pb, 2), dtype=int)

    current_center = bleach_center
    for i in range(n_pb):
        fit = fit_gaussian_dip_2d(pre_avg, pb_images[i], current_center)
        if fit is not None:
            gauss_amplitudes[i], gauss_sigmas[i], fitted_center = fit
            if ROI_TRACK_CENTER:
                current_center = (int(round(fitted_center[0])), int(round(fitted_center[1])))
        # if the fit fails, current_center is left unchanged (hold last known position)

        if ROI_TRACK_CENTER:
            frame_bleach_mask, frame_ref_mask = masks_for_center(current_center)
        else:
            frame_bleach_mask, frame_ref_mask = bleach_mask, ref_mask
        tracked_centers[i] = current_center

        pb_bleach_roi[i] = pb_images[i][frame_bleach_mask].mean()
        pb_ref_roi[i] = pb_images[i][frame_ref_mask].mean()

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{n_pb}")
    print("  Done.")

    # Double normalization: correct for overall photobleaching
    pb_norm = (pb_bleach_roi / pb_ref_roi) / (pre_I / pre_ref_I)

    print(f"  Pre-bleach ROI mean: {pre_I:.2f}")
    print(f"  Post-bleach ROI (t=0): {pb_bleach_roi[0]:.2f}")
    print(f"  Post-bleach ROI (t=end): {pb_bleach_roi[-1]:.2f}")

    # ---- FRAP recovery fit (using double-normalized data) ----
    print("\nFitting FRAP recovery curve (double-normalized)...")
    frap_result = analyze_frap_recovery(pb_times_s, pb_norm, 1.0)

    if frap_result:
        # Diffusion coefficient (Soumpasis 1983): D = 0.224 * w^2 / t_half,
        # where w = bleach spot radius (from the Gaussian fit at t=0). This is
        # only meaningful because roi_r above was matched to sigma0 - an ROI
        # picked independently of the actual spot size biases D through t_half.
        sigma_px = gauss_sigmas[0] if not np.isnan(gauss_sigmas[0]) else roi_r
        sigma_um = sigma_px * params.get("mpp_um", 1)
        w_um = sigma_um
        D_um2_s = 0.224 * w_um**2 / frap_result["t_half_s"]

        print("\n=== FRAP Results ===")
        print(f"  I_0 (normalized):     {frap_result['I_0']:.4f}")
        print(f"  I_inf (normalized):   {frap_result['I_inf']:.4f}")
        print(f"  tau:                  {frap_result['tau_s']:.2f} ± {frap_result['tau_err']:.2f} s")
        print(f"  t_1/2:               {frap_result['t_half_s']:.2f} s")
        print(f"  Mobile fraction:     {frap_result['mobile_fraction']:.3f} ({frap_result['mobile_fraction']*100:.1f}%)")
        print(f"  Immobile fraction:   {frap_result['immobile_fraction']:.3f} ({frap_result['immobile_fraction']*100:.1f}%)")
        print(f"  Bleach spot sigma:   {sigma_um:.2f} µm ({sigma_px:.1f} px)")
        print(f"  D (Soumpasis, ROI r={roi_r}px): {D_um2_s:.4f} µm²/s  <- depends on roi_r, see below")
    else:
        print("  Recovery fit failed!")
        D_um2_s = np.nan

    # ---- ROI-independent diffusion estimate (recovery curve at several ROI radii) ----
    print("\nFitting FRAP recovery at multiple ROI radii (ROI-independent D)...")
    roi_var_result = analyze_diffusion_roi_variation(
        pb_images, pre_images, pb_times_s, bleach_center, params.get("mpp_um", 1),
        ROI_VARIATION_RADII_PX, REF_RING_GAP_PX, REF_RING_WIDTH_PX,
    )
    if roi_var_result:
        print(f"  D (ROI-variation, ROI-independent): {roi_var_result['D_um2_s']:.4f} µm²/s "
              f"(slope-fit R²={roi_var_result['r2_fit']:.3f}, radii tested: {roi_var_result['radii_px']} px)")
    else:
        print("  ROI-variation D estimate failed.")

    # ---- Gaussian bleach-profile diffusion estimate (Axelrod) ----
    print("\nFitting radial Gaussian bleach profile (Axelrod method)...")
    gauss_profile_result = analyze_gaussian_profile_diffusion(
        pre_images, pb_images, pb_times_s, pre_avg, params.get("mpp_um", 1)
    )
    if gauss_profile_result:
        print(f"  Bleach center (centroid): row={gauss_profile_result['center'][0]:.1f}, "
              f"col={gauss_profile_result['center'][1]:.1f}")
        print(f"  r_e (effective bleach radius): {gauss_profile_result['r_e_um']:.3f} ± "
              f"{gauss_profile_result['r_e_err_um']:.3f} µm "
              f"(radial-profile fit R²={gauss_profile_result['profile_fit']['r2']:.3f})")
        print(f"  tau: {gauss_profile_result['tau_s']:.3f} ± {gauss_profile_result['tau_err_s']:.3f} s")
        print(f"  D (Axelrod, closed-form):      {gauss_profile_result['D_axelrod_um2_s']:.4f} ± "
              f"{gauss_profile_result['D_axelrod_err_um2_s']:.4f} µm²/s")
        print(f"  D (Soumpasis approx., r_e):    {gauss_profile_result['D_soumpasis_approx_um2_s']:.4f} "
              "µm²/s  <- plausibility check only")
    else:
        print("  Gaussian bleach-profile method failed.")

    # ---- Plotting ----
    fig, axes = plt.subplots(3, 4, figsize=(24, 14))
    fig.suptitle("FRAP Analysis", fontsize=14, fontweight="bold")

    cy0, cx0 = bleach_center
    cy_end, cx_end = tracked_centers[-1]

    # (A) Pre-bleach vs. first post-bleach image
    ax = axes[0, 0]
    ax.imshow(pre_avg, cmap="gray", vmin=0, vmax=pre_avg.max())
    circle = plt.Circle((cx0, cy0), roi_r, color="red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    if gauss_profile_result:
        cy_g, cx_g = gauss_profile_result["center"]
        circle_g = plt.Circle((cx_g, cy_g), gauss_profile_result["r_e_um"] / params.get("mpp_um", 1),
                               color="cyan", fill=False, linewidth=1.5, linestyle="--")
        ax.add_patch(circle_g)
    ax.set_title("Pre-bleach (average)\nred=roi_r, cyan=Axelrod r_e")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    ax = axes[0, 1]
    ax.imshow(pb_images[0], cmap="gray", vmin=0, vmax=pre_avg.max())
    circle = plt.Circle((cx0, cy0), roi_r, color="red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_title("Post-bleach (t=0)")
    ax.set_xlabel("x [px]")

    ax = axes[0, 2]
    ax.imshow(pb_images[-1], cmap="gray", vmin=0, vmax=pre_avg.max())
    circle = plt.Circle((cx_end, cy_end), roi_r, color="red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_title(f"Post-bleach (t={pb_times_s[-1]:.0f}s)")
    ax.set_xlabel("x [px]")

    # (G) Radial Gaussian bleach profile + fit -> r_e
    ax = axes[0, 3]
    if gauss_profile_result:
        pf = gauss_profile_result["profile_fit"]
        ax.plot(pf["rho_um"], pf["intensity"], "o", markersize=4, color="tab:cyan", label="Radial mean")
        ax.plot(pf["rho_um"], pf["fit_curve"], "-", color="black", linewidth=2,
                label=f"Fit: r_e = {gauss_profile_result['r_e_um']:.2f} ± "
                      f"{gauss_profile_result['r_e_err_um']:.2f} µm, R²={pf['r2']:.3f}")
        ax.axvline(gauss_profile_result["r_e_um"], color="tab:red", linestyle=":", alpha=0.7)
        ax.legend(fontsize=8)
    ax.set_xlabel("ρ [µm]")
    ax.set_ylabel("Intensity [a.u.]")
    ax.set_title("Radial Bleach Profile (t=0)")

    # (B) Gaussian peak amplitude over time
    ax = axes[1, 0]
    valid = ~np.isnan(gauss_amplitudes)
    ax.plot(pb_times_s[valid], gauss_amplitudes[valid], "o-", markersize=2, color="tab:blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gaussian dip amplitude [a.u.]")
    ax.set_title("Gaussian Dip Amplitude vs. Time")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # (C) Gaussian sigma (bleach spot width) over time - diagnostic only.
    # Not used to derive D: for a fast diffuser the spot can be unresolvable
    # in the images at all (see analyze_diffusion_roi_variation docstring),
    # so this panel may be sparse/empty - that itself is informative.
    ax = axes[1, 1]
    sigma_um_arr = gauss_sigmas * params.get("mpp_um", 1)
    valid_s = ~np.isnan(sigma_um_arr)
    ax.plot(pb_times_s[valid_s], sigma_um_arr[valid_s], "o-", markersize=2, color="tab:orange")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gaussian sigma [µm]")
    ax.set_title("Bleach Spot Width vs. Time")

    # (H) Recovery curve in the r_e-based ROI (Phair/Misteli norm.) + Axelrod fit
    ax = axes[1, 3]
    if gauss_profile_result:
        rf = gauss_profile_result["recovery_fit"]
        ax.plot(rf["t_fit"], rf["I_norm"], "o", markersize=2, color="tab:cyan",
                label="Data (Phair/Misteli)", alpha=0.6)
        ax.plot(rf["t_fit"], rf["I_fit"], "-", color="black", linewidth=2, label="Fit")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Pre-bleach")
        info_text = (
            f"$r_e$ = {gauss_profile_result['r_e_um']:.2f} µm\n"
            f"$\\tau$ = {gauss_profile_result['tau_s']:.2f} s\n"
            f"D (Axelrod) = {gauss_profile_result['D_axelrod_um2_s']:.4f} µm²/s"
        )
        ax.text(0.95, 0.05, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))
        ax.legend(fontsize=8, loc="center right")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title("Recovery Curve, ROI=r_e (Axelrod method)")

    # Numeric summary
    ax = axes[1, 2]
    ax.axis("off")
    if frap_result:
        summary_lines = [
            "FRAP summary",
            f"$t_{{1/2}}$ = {frap_result['t_half_s']:.2f} s",
            f"Mobile fraction = {frap_result['mobile_fraction']*100:.1f}%",
            "",
            f"D (Soumpasis, roi_r={roi_r}px):",
            f"  {D_um2_s:.4f} µm²/s  (depends on roi_r)",
        ]
        if roi_var_result:
            summary_lines += [
                "",
                "D (ROI-variation, ROI-independent):",
                f"  {roi_var_result['D_um2_s']:.4f} µm²/s",
                f"  slope-fit R² = {roi_var_result['r2_fit']:.3f}",
            ]
        if gauss_profile_result:
            summary_lines += [
                "",
                f"r_e (Axelrod) = {gauss_profile_result['r_e_um']:.2f} ± "
                f"{gauss_profile_result['r_e_err_um']:.2f} µm",
                "D (Axelrod, closed-form):",
                f"  {gauss_profile_result['D_axelrod_um2_s']:.4f} ± "
                f"{gauss_profile_result['D_axelrod_err_um2_s']:.4f} µm²/s",
                "D (Soumpasis approx., r_e) [check]:",
                f"  {gauss_profile_result['D_soumpasis_approx_um2_s']:.4f} µm²/s",
            ]
        ax.text(0.02, 0.98, "\n".join(summary_lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", family="monospace")

    # (D) FRAP recovery curve at the single analysis ROI (roi_r)
    ax = axes[2, 0]
    if frap_result:
        ax.plot(frap_result["t_fit"], frap_result["I_norm"], "o", markersize=2,
                color="tab:green", label="Data (normalized)", alpha=0.6)
        t_fine = np.linspace(0, frap_result["t_fit"][-1], 500)
        I_fit_fine = frap_recovery_single_exp(
            t_fine, frap_result["I_inf"], frap_result["I_0"], frap_result["tau_s"]
        )
        ax.plot(t_fine, I_fit_fine, "-", color="black", linewidth=2, label="Fit")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Pre-bleach")
        ax.axhline(frap_result["I_inf"], color="tab:red", linestyle=":",
                    alpha=0.7, label=f"I_inf = {frap_result['I_inf']:.3f}")
        ax.legend(fontsize=8, loc="center right")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title(f"FRAP Recovery Curve (roi_r={roi_r}px)")

    # (E) FRAP recovery curves at every ROI-variation radius, overlaid
    ax = axes[2, 1]
    if roi_var_result:
        cmap = plt.get_cmap("viridis")
        n_curves = len(roi_var_result["curves"])
        for i, curve in enumerate(roi_var_result["curves"]):
            color = cmap(i / max(n_curves - 1, 1))
            ax.plot(curve["t_fit"], curve["I_norm"], "o", markersize=2, color=color, alpha=0.5)
            ax.plot(curve["t_fit"], curve["I_fit"], "-", color=color, linewidth=1.8,
                    label=f"r={curve['r_px']}px ({curve['r_um']:.1f}µm), "
                          f"$t_{{1/2}}$={curve['t_half_s']:.2f}s")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title("Recovery Curves at Different ROI Radii")

    # (F) ROI-variation: t_half vs ROI radius^2 - the ROI-independent D
    ax = axes[2, 2]
    if roi_var_result:
        ax.plot(roi_var_result["radii_um2"], roi_var_result["t_half_s"], "o",
                markersize=6, color="tab:purple", label="Data")
        r2_fine = np.linspace(0, roi_var_result["radii_um2"].max(), 100)
        t_half_fine = roi_var_result["slope"] * r2_fine + roi_var_result["intercept"]
        ax.plot(r2_fine, t_half_fine, "-", color="black", linewidth=2,
                label=f"Fit: D = {roi_var_result['D_um2_s']:.4f} µm²/s")
        ax.legend(fontsize=8)
    ax.set_xlabel("ROI radius² [µm²]")
    ax.set_ylabel("$t_{1/2}$ [s]")
    ax.set_title("t$_{1/2}$ vs. ROI² (ROI-independent D)")

    axes[2, 3].axis("off")

    plt.tight_layout()
    out_path = BASE_DIR / "frap_analysis_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
