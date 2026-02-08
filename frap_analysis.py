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
BASE_DIR = Path(r"D:\Diffusion in Hydrogel Data\2025_09_01_16_03_50--FRAP\FRAP 009")


# =============================================================================
# 1. Discover subfolders (Pre / Pb)
# =============================================================================
def find_series_folders(base_dir):
    """Find the Pre-bleach and Post-bleach subfolders."""
    pre_folder = None
    pb_folder = None
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if "Pre" in name:
            pre_folder = entry
        elif "Pb" in name:
            pb_folder = entry
    if pre_folder is None or pb_folder is None:
        raise FileNotFoundError(f"Could not find Pre/Pb folders in {base_dir}")
    return pre_folder, pb_folder


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


# =============================================================================
# 3. Load TIF images
# =============================================================================
def load_tif_series(folder, folder_name, channel=0, black_threshold=1.0):
    """
    Load TIF time series for a given channel (l-index).
    Returns (images, frame_indices) where images is (n_frames, h, w) float64
    and frame_indices lists the original t-index for each kept frame.
    Skips black frames (mean intensity <= black_threshold).
    """
    pattern = re.compile(
        re.escape(folder_name) + r"_l" + str(channel) + r"_t(\d+)\.tif$",
        re.IGNORECASE,
    )
    files = []
    for f in folder.iterdir():
        m = pattern.match(f.name)
        if m:
            files.append((int(m.group(1)), f))
    files.sort(key=lambda x: x[0])

    if not files:
        raise FileNotFoundError(f"No TIF files found for channel {channel} in {folder}")

    images = []
    frame_indices = []
    n_skipped = 0
    for t_idx, fpath in files:
        img = np.array(Image.open(fpath))
        if img.ndim == 3:
            img = img[:, :, 1]  # green channel
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
def find_bleach_center(pre_avg, post_first, sigma_smooth=5):
    """
    Find the bleach spot center by computing the difference
    (pre - post) and locating the maximum.
    """
    diff = pre_avg - post_first
    diff_smooth = gaussian_filter(diff, sigma=sigma_smooth)
    center = np.unravel_index(diff_smooth.argmax(), diff_smooth.shape)
    return center  # (row, col)


# =============================================================================
# 5. 2D Gaussian fitting
# =============================================================================
def make_gaussian_2d_fixed_center(cy, cx):
    """Return a 2D Gaussian function with fixed center (cy, cx)."""
    def gaussian_2d(coords, amplitude, sigma, offset):
        y, x = coords
        return offset - amplitude * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)
        )
    return gaussian_2d


def fit_gaussian_2d(image, center, roi_radius=60):
    """
    Fit a 2D Gaussian dip to a fixed region around `center`.
    Center position is kept constant; only amplitude, sigma, offset are fitted.
    Returns (amplitude, sigma, offset) or None if fit fails.
    """
    cy, cx = center
    h, w = image.shape
    y_min = max(0, cy - roi_radius)
    y_max = min(h, cy + roi_radius)
    x_min = max(0, cx - roi_radius)
    x_max = min(w, cx + roi_radius)

    roi = image[y_min:y_max, x_min:x_max]
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    coords = (yy.ravel(), xx.ravel())
    data = roi.ravel()

    bg = np.percentile(roi, 90)
    dip = bg - roi.min()
    gauss_func = make_gaussian_2d_fixed_center(cy, cx)
    p0 = [dip, 10.0, bg]
    bounds = ([0, 1, 0], [300, roi_radius, 300])

    try:
        popt, _ = curve_fit(gauss_func, coords, data, p0=p0, bounds=bounds, maxfev=5000)
        return popt  # amplitude, sigma, offset
    except RuntimeError:
        return None


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
    I_0_guess = I_norm[0]
    I_inf_guess = I_norm[-1]
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


# =============================================================================
# 7. Main analysis
# =============================================================================
def main():
    pre_folder, pb_folder = find_series_folders(BASE_DIR)
    pre_name = pre_folder.name
    pb_name = pb_folder.name

    print(f"Pre-bleach folder: {pre_name}")
    print(f"Post-bleach folder: {pb_name}")

    # ---- Metadata ----
    pb_props_xml = pb_folder / "MetaData" / f"{pb_name}_Properties.xml"
    pb_stamps_xml = pb_folder / "MetaData" / f"{pb_name}.xml"
    pre_stamps_xml = pre_folder / "MetaData" / f"{pre_name}.xml"

    params = parse_properties_xml(pb_props_xml)
    pb_timestamps, pb_dims = parse_series_xml(pb_stamps_xml)
    pre_timestamps, _ = parse_series_xml(pre_stamps_xml)

    # Pixel size from DimensionDescription (DimID 1 = X, 2 = Y)
    if 1 in pb_dims:
        params["mpp_m"] = pb_dims[1]["element_size"]          # meters/pixel
        params["mpp_um"] = pb_dims[1]["element_size"] * 1e6    # µm/pixel
        params["fov_um"] = pb_dims[1]["length"] * 1e6          # µm

    print("\n=== Acquisition Parameters ===")
    print(f"  Objective:          {params.get('objective', '?')}")
    print(f"  Magnification:      {params.get('magnification', '?')}x")
    print(f"  NA:                 {params.get('na', '?')}")
    print(f"  Immersion:          {params.get('immersion', '?')}")
    print(f"  Pixel size:         {params.get('mpp_um', 0):.4f} µm/px")
    print(f"  Image size:         {params.get('image_width')}x{params.get('image_height')} px")
    print(f"  Zoom:               {params.get('zoom', '?')}")
    print(f"  Pinhole:            {params.get('pinhole_um', '?')} µm ({params.get('pinhole_airy', '?')})")
    print(f"  Bit depth:          {params.get('bit_depth', '?')} bit")
    print(f"  Frame time:         {params.get('frame_time_s', 0):.3f} s")
    print(f"  Scan mode:          {params.get('scan_mode', '?')}")
    print(f"  Bleach reps:        {params.get('bleach_repetitions', '?')}")
    print(f"  Bleach duration:    {params.get('bleach_duration_ms', '?')} ms")
    for k, v in params.items():
        if "laser" in k:
            print(f"  {k}: {v:.2f}%")
    print(f"  Pre-bleach frames:  {len(pre_timestamps)}")
    print(f"  Post-bleach frames: {params.get('cycle_count', '?')}")

    # Field of view
    fov_um = params.get("fov_um", params.get("mpp_um", 0) * params.get("image_width", 512))
    print(f"  Field of view:      {fov_um:.1f} µm")

    # ---- Determine number of time points per channel from DimensionDescription ----
    # DimID 4 = time dimension -> n_elements = frames per channel
    n_timepoints_per_channel = None
    if 4 in pb_dims:
        n_timepoints_per_channel = pb_dims[4]["n_elements"]
        print(f"  Time points per channel (from XML): {n_timepoints_per_channel}")

    # ---- Load images (channel l0 = green/FITC) ----
    print("\nLoading pre-bleach images...")
    pre_images, pre_frame_idx = load_tif_series(pre_folder, pre_name, channel=0)
    print(f"  Loaded {pre_images.shape[0]} frames, shape {pre_images.shape[1:]}")

    print("Loading post-bleach images...")
    pb_images, pb_frame_idx = load_tif_series(pb_folder, pb_name, channel=0)
    print(f"  Loaded {pb_images.shape[0]} frames")

    # ---- Timestamps -> relative times in seconds ----
    # Timestamps in XML cover all channels sequentially.
    # For channel l0 (first channel), use timestamps[0 : n_timepoints_per_channel].
    if n_timepoints_per_channel is not None:
        all_ch0_stamps = pb_timestamps[:n_timepoints_per_channel]
    else:
        all_ch0_stamps = pb_timestamps

    # Map frame_indices to their timestamps (handles skipped black frames)
    pb_times = []
    for idx in pb_frame_idx:
        if idx < len(all_ch0_stamps):
            pb_times.append(all_ch0_stamps[idx])
        else:
            pb_times.append(all_ch0_stamps[-1])  # fallback
    t0 = pb_times[0]
    pb_times_s = np.array([(t - t0).total_seconds() for t in pb_times])
    n_pb = len(pb_times_s)

    # Pre-bleach timestamps
    pre_stamps_available = pre_timestamps[:len(pre_frame_idx)]
    pre_times_s = np.array([(t - pre_stamps_available[0]).total_seconds()
                            for t in pre_stamps_available]) if pre_stamps_available else np.zeros(len(pre_frame_idx))

    # ---- Pre-bleach average ----
    pre_avg = pre_images.mean(axis=0)
    pre_intensity_full = pre_avg.mean()
    print(f"\n  Pre-bleach mean intensity: {pre_intensity_full:.2f}")

    # ---- Find bleach center ----
    bleach_center = find_bleach_center(pre_avg, pb_images[0])
    print(f"  Bleach spot center:  row={bleach_center[0]}, col={bleach_center[1]}")

    # ---- Define ROIs ----
    cy, cx = bleach_center
    roi_r = 20  # radius in pixels for intensity measurement
    h, w = pre_avg.shape

    # Bleach ROI (circular mask)
    yy, xx = np.ogrid[:h, :w]
    bleach_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= roi_r**2

    # Reference ROI (ring around bleach, for normalization)
    ref_inner = roi_r + 10
    ref_outer = roi_r + 50
    ref_mask = (((yy - cy) ** 2 + (xx - cx) ** 2) >= ref_inner**2) & (
        ((yy - cy) ** 2 + (xx - cx) ** 2) <= ref_outer**2
    )

    bleach_r_um = roi_r * params.get("mpp_um", 1)
    print(f"  Bleach ROI radius:   {roi_r} px = {bleach_r_um:.1f} µm")

    # ---- Measure intensities ----
    pre_bleach_roi = np.array([img[bleach_mask].mean() for img in pre_images])
    pre_ref_roi = np.array([img[ref_mask].mean() for img in pre_images])
    pre_I = pre_bleach_roi.mean()
    pre_ref_I = pre_ref_roi.mean()

    pb_bleach_roi = np.array([img[bleach_mask].mean() for img in pb_images])
    pb_ref_roi = np.array([img[ref_mask].mean() for img in pb_images])

    # Double normalization: correct for overall photobleaching
    pb_norm = (pb_bleach_roi / pb_ref_roi) / (pre_I / pre_ref_I)

    print(f"  Pre-bleach ROI mean: {pre_I:.2f}")
    print(f"  Post-bleach ROI (t=0): {pb_bleach_roi[0]:.2f}")
    print(f"  Post-bleach ROI (t=end): {pb_bleach_roi[-1]:.2f}")

    # ---- 2D Gaussian fitting for each frame (fixed center & ROI) ----
    print(f"\nFitting 2D Gaussian to each post-bleach frame (fixed center=({cy},{cx}))...")
    gauss_amplitudes = []
    gauss_sigmas = []
    gauss_offsets = []

    for i in range(n_pb):
        result = fit_gaussian_2d(pb_images[i], bleach_center, roi_radius=60)
        if result is not None:
            amp, sigma, offset = result
            gauss_amplitudes.append(amp)
            gauss_sigmas.append(sigma)
            gauss_offsets.append(offset)
        else:
            gauss_amplitudes.append(np.nan)
            gauss_sigmas.append(np.nan)
            gauss_offsets.append(np.nan)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{n_pb}")

    gauss_amplitudes = np.array(gauss_amplitudes)
    gauss_sigmas = np.array(gauss_sigmas)
    gauss_offsets = np.array(gauss_offsets)
    gauss_center_intensity = gauss_offsets - gauss_amplitudes  # intensity at dip center
    print("  Done.")

    # ---- FRAP recovery fit (using double-normalized data) ----
    print("\nFitting FRAP recovery curve (double-normalized)...")
    frap_result = analyze_frap_recovery(pb_times_s, pb_norm, 1.0)

    if frap_result:
        # Diffusion coefficient (Soumpasis 1983)
        # D = 0.224 * w^2 / t_half
        # where w = bleach spot radius (from Gaussian sigma at t=0)
        sigma_px = gauss_sigmas[0] if not np.isnan(gauss_sigmas[0]) else roi_r
        sigma_um = sigma_px * params.get("mpp_um", 1)
        w_um = sigma_um  # Gaussian 1/e^2 radius ≈ sigma for our definition
        D_um2_s = 0.224 * w_um**2 / frap_result["t_half_s"]

        print("\n=== FRAP Results ===")
        print(f"  I_0 (normalized):     {frap_result['I_0']:.4f}")
        print(f"  I_inf (normalized):   {frap_result['I_inf']:.4f}")
        print(f"  tau:                  {frap_result['tau_s']:.2f} ± {frap_result['tau_err']:.2f} s")
        print(f"  t_1/2:               {frap_result['t_half_s']:.2f} s")
        print(f"  Mobile fraction:     {frap_result['mobile_fraction']:.3f} ({frap_result['mobile_fraction']*100:.1f}%)")
        print(f"  Immobile fraction:   {frap_result['immobile_fraction']:.3f} ({frap_result['immobile_fraction']*100:.1f}%)")
        print(f"  Bleach spot sigma:   {sigma_um:.2f} µm ({sigma_px:.1f} px)")
        print(f"  D (Soumpasis):       {D_um2_s:.4f} µm²/s")
    else:
        print("  Recovery fit failed!")
        D_um2_s = np.nan

    # ---- Plotting ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("FRAP Analysis", fontsize=14, fontweight="bold")

    # (A) Pre-bleach vs. first post-bleach image
    ax = axes[0, 0]
    ax.imshow(pre_avg, cmap="gray", vmin=0, vmax=pre_avg.max())
    circle = plt.Circle((cx, cy), roi_r, color="red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_title("Pre-bleach (average)")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    ax = axes[0, 1]
    ax.imshow(pb_images[0], cmap="gray", vmin=0, vmax=pre_avg.max())
    circle = plt.Circle((cx, cy), roi_r, color="red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_title("Post-bleach (t=0)")
    ax.set_xlabel("x [px]")

    ax = axes[0, 2]
    ax.imshow(pb_images[-1], cmap="gray", vmin=0, vmax=pre_avg.max())
    circle = plt.Circle((cx, cy), roi_r, color="red", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_title(f"Post-bleach (t={pb_times_s[-1]:.0f}s)")
    ax.set_xlabel("x [px]")

    # (B) Gaussian peak amplitude over time
    ax = axes[1, 0]
    valid = ~np.isnan(gauss_amplitudes)
    ax.plot(pb_times_s[valid], gauss_amplitudes[valid], "o-", markersize=2, color="tab:blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gaussian dip amplitude [a.u.]")
    ax.set_title("Gaussian Dip Amplitude vs. Time")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # (C) Gaussian sigma (bleach spot width) over time
    ax = axes[1, 1]
    sigma_um_arr = gauss_sigmas * params.get("mpp_um", 1)
    valid_s = ~np.isnan(sigma_um_arr)
    ax.plot(pb_times_s[valid_s], sigma_um_arr[valid_s], "o-", markersize=2, color="tab:orange")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gaussian sigma [µm]")
    ax.set_title("Bleach Spot Width vs. Time")

    # (D) FRAP recovery curve
    ax = axes[1, 2]
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

        info_text = (
            f"$\\tau$ = {frap_result['tau_s']:.1f} s\n"
            f"$t_{{1/2}}$ = {frap_result['t_half_s']:.1f} s\n"
            f"Mobile: {frap_result['mobile_fraction']*100:.1f}%\n"
            f"D = {D_um2_s:.4f} µm²/s"
        )
        ax.text(0.95, 0.05, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.legend(fontsize=8, loc="center right")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title("FRAP Recovery Curve")

    plt.tight_layout()
    out_path = BASE_DIR / "frap_analysis_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
