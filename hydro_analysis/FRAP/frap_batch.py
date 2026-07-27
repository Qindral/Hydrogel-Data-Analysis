"""
FRAP Batch Analysis
===================
Runs FRAP analysis on multiple experiment folders, collects diffusion
coefficients, and produces summary plots (individual + grouped with SEM).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Import analysis functions from frap_analysis
from frap_analysis import (
    find_series_folders,
    parse_properties_xml,
    parse_frame_timestamps,
    load_tif_series,
    fit_gaussian_2d,
    analyze_frap_recovery,
    parse_frap_roi,
)

# =============================================================================
# Configuration
# =============================================================================
DATA_ROOT  = Path(r"H:\Daten Promotion Sicherung\Confocal_Measure\2025_09_08_14_57_46--Project_TIF")
OUTPUT_DIR = DATA_ROOT / "analysis_output"

# Set True to display pre-bleach images during analysis (also always saved to OUTPUT_DIR)
SHOW_IMAGES = False

GROUPS = {
    "Water": [
        "Water",
        "Water 2",
        "Water 3",
    ],
    "no FITC": [
        "C16_noFITC",
        "C16_noFITC_2",
        "C16_noFITC_3",
    ],
    "C16 deep_B": [
        "C16 _deep B1",
        "C16 _deep B1 _2",
        "C16 _deep B1 _3",
        "C16 _deep B1 _4",
        "C16 _deep B1 _5",
        "C16 _deep B1 _6_306 um",
    ],
    "C16 deep_A": [
        "C16 _deep_A1_01",
        "C16 _deep_A1_02",
        "C16 _deep_A1_03",
    ],
    "C16 shallow": [
        "C16 shallow B2",
        "C16 shallow B2_02",
    ]

}

# Style guide rcParams
mpl.rcParams.update({
    "figure.figsize": (7.15, 5.00),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "legend.fontsize": 9,
    "legend.frameon": False,
})

# Color palette from style guide
COLORS_BASE = ["#0000da", "#004cff", "#00c4ff", "#49ffad",
               "#adff49", "#ffd700", "#ff6800", "#da0000"]
COLORS_DARK = ["#000099", "#0035b2", "#0089b2", "#33b279",
               "#79b233", "#b29600", "#b24900", "#990000"]
GROUP_COLORS = {
    "Water":      (COLORS_BASE[2], COLORS_DARK[2]),   # cyan
    "no FITC":    (COLORS_BASE[5], COLORS_DARK[5]),   # gold
    "C16 deep":   (COLORS_BASE[0], COLORS_DARK[0]),   # blue
    "C16 shallow":(COLORS_BASE[6], COLORS_DARK[6]),   # orange
}


# =============================================================================
# Gaussian fitting with floating center (tracks bleach spot movement)
# =============================================================================
def _fit_bleach_center(image, initial_center, search_radius=100):
    """Fit a 2D Gaussian dip with a free center near initial_center.

    Returns fitted (cy, cx).  Falls back to initial_center if fit fails.
    """
    from scipy.optimize import curve_fit as _curve_fit

    cy0, cx0 = initial_center
    h, w = image.shape
    y0 = max(0, cy0 - search_radius)
    y1 = min(h, cy0 + search_radius)
    x0 = max(0, cx0 - search_radius)
    x1 = min(w, cx0 + search_radius)

    roi = image[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    data = roi.ravel()

    bg  = np.percentile(roi, 90)
    dip = max(bg - roi.min(), 1.0)

    def _gauss_dip(coords, amp, cy, cx, sigma, offset):
        y, x = coords
        return offset - amp * np.exp(
            -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
        )

    p0     = [dip, float(cy0), float(cx0), 15.0, bg]
    bounds = ([0.0,  y0, x0, 2.0, 0.0],
              [1e6,  y1, x1, float(search_radius), 1e6])

    try:
        popt, _ = _curve_fit(
            _gauss_dip, (yy.ravel(), xx.ravel()), data,
            p0=p0, bounds=bounds, maxfev=5000,
        )
        return int(round(popt[1])), int(round(popt[2]))
    except RuntimeError:
        return initial_center


# =============================================================================
# Pre-bleach image display
# =============================================================================
def _plot_pre_images(label, pre_images, pre_avg, pb_first, cy, cx,
                     roi_r, ref_inner, ref_outer, save_dir=None):
    """Show individual pre-bleach frames, their average, and the first
    post-bleach frame (all with bleach + reference ROI overlays)."""
    from matplotlib.patches import Circle

    n_pre   = pre_images.shape[0]
    n_show  = min(n_pre, 6)          # cap at 6 thumbnails
    n_cols  = n_show + 2             # +2: average + first post-bleach

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(2.6 * n_cols, 3.2),
        constrained_layout=True,
    )

    def _add_rois(ax):
        ax.add_patch(Circle((cx, cy), roi_r,
                             fill=False, edgecolor="#0000da", linewidth=1.5))
        ax.add_patch(Circle((cx, cy), ref_inner,
                             fill=False, edgecolor="#da0000",
                             linewidth=1.0, linestyle="--"))
        ax.add_patch(Circle((cx, cy), ref_outer,
                             fill=False, edgecolor="#da0000",
                             linewidth=1.0, linestyle="--"))

    # Individual pre-bleach frames
    for i in range(n_show):
        ax = axes[i]
        ax.imshow(pre_images[i], cmap="gray", interpolation="nearest")
        title = f"Pre {i + 1}"
        if i == n_show - 1 and n_pre > n_show:
            title += f" (+{n_pre - n_show} more)"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Pre-bleach average with ROIs
    ax_avg = axes[n_show]
    ax_avg.imshow(pre_avg, cmap="gray", interpolation="nearest")
    _add_rois(ax_avg)
    ax_avg.set_title("Ø Pre-bleach", fontsize=8)
    ax_avg.axis("off")

    # First post-bleach frame with ROIs
    ax_pb = axes[n_show + 1]
    ax_pb.imshow(pb_first, cmap="gray", interpolation="nearest")
    _add_rois(ax_pb)
    ax_pb.set_title("Post-bleach t=0", fontsize=8)
    ax_pb.axis("off")

    fig.suptitle(label, fontsize=10, fontweight="semibold")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        safe = label.replace(" ", "_").replace("/", "_")
        fig.savefig(save_dir / f"prebleach_{safe}.png", dpi=300,
                    bbox_inches="tight")

    if SHOW_IMAGES:
        plt.show()
    else:
        plt.close(fig)


def _plot_control_images(label, pre_avg, pb_avg, save_dir=None):
    """Display average pre- and post-bleach images for a no-FITC control."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5), constrained_layout=True)
    vmax = max(pre_avg.max(), pb_avg.max())

    ax1.imshow(pre_avg, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax1.set_title("Pre-bleach (avg)", fontsize=9)
    ax1.axis("off")

    ax2.imshow(pb_avg, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax2.set_title("Post-bleach (avg)", fontsize=9)
    ax2.axis("off")

    fig.suptitle(f"{label} — Control (no FITC)", fontsize=10, fontweight="semibold")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        safe = label.replace(" ", "_").replace("/", "_")
        fig.savefig(save_dir / f"control_{safe}.png", dpi=300, bbox_inches="tight")

    if SHOW_IMAGES:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Single-experiment analysis (returns D or NaN)
# =============================================================================
def analyze_single(base_dir, save_dir=None, control=False):
    """Run FRAP analysis on a single experiment folder.

    control=True: no FITC experiment — just display pre/post averages, no D.
    Returns dict with results or None if analysis fails / control.
    """
    label = base_dir.name

    try:
        pre_folders, pb_folders = find_series_folders(base_dir)
    except FileNotFoundError as e:
        print(f"  SKIP {label}: {e}")
        return None

    # Load pre-bleach images from all Pre folders
    all_pre_images = []
    for pre_folder in pre_folders:
        pre_name = pre_folder.name
        try:
            imgs, _ = load_tif_series(pre_folder, pre_name, channel=0)
            all_pre_images.append(imgs)
        except FileNotFoundError:
            continue
    if not all_pre_images:
        print(f"  SKIP {label}: no pre-bleach images")
        return None
    pre_images = np.concatenate(all_pre_images, axis=0)

    # Load post-bleach images from all Pb folders,
    # using per-frame real timestamps from each folder's _Properties.xml
    all_pb_images = []
    all_pb_abs_times = []
    params = {}
    for pb_folder in pb_folders:
        pb_name = pb_folder.name
        try:
            imgs, frame_idx = load_tif_series(pb_folder, pb_name, channel=0)
        except FileNotFoundError:
            continue
        all_pb_images.append(imgs)

        # Read this folder's own metadata
        props_xml = pb_folder / "MetaData" / f"{pb_name}_Properties.xml"
        try:
            folder_params = parse_properties_xml(props_xml)
            frame_ts = parse_frame_timestamps(props_xml)
        except Exception as e:
            print(f"    Warning: metadata error in {pb_name}: {e}")
            continue

        if not params:
            params = folder_params

        # Actual frame interval from RelativeTime
        if len(frame_ts) >= 2:
            dt_interval = frame_ts[1][0] - frame_ts[0][0]
        else:
            dt_interval = 0
        fps = 1.0 / dt_interval if dt_interval > 0 else 0
        print(f"    {pb_name}: {imgs.shape[0]} frames, interval = {dt_interval:.3f} s ({fps:.2f} fps)")

        # Map frame indices to absolute datetimes
        for idx in frame_idx:
            if idx < len(frame_ts) and frame_ts[idx][1] is not None:
                all_pb_abs_times.append(frame_ts[idx][1])
            elif frame_ts and frame_ts[-1][1] is not None:
                all_pb_abs_times.append(frame_ts[-1][1])
            elif all_pb_abs_times:
                all_pb_abs_times.append(all_pb_abs_times[-1])

    if not all_pb_images or not all_pb_abs_times:
        print(f"  SKIP {label}: no post-bleach images")
        return None
    if not params:
        print(f"  SKIP {label}: no metadata could be read")
        return None
    pb_images = np.concatenate(all_pb_images, axis=0)

    # Build continuous time axis from absolute timestamps
    t0 = all_pb_abs_times[0]
    pb_times_s = np.array([(t - t0).total_seconds() for t in all_pb_abs_times])

    # Pre-bleach average
    pre_avg = pre_images.mean(axis=0)
    h, w   = pre_avg.shape

    # --- Parse exact ROI geometry + pixel size from Properties XML (no fallbacks) ---
    pb0_name = pb_folders[0].name
    props_xml_path = pb_folders[0] / "MetaData" / f"{pb0_name}_Properties.xml"
    try:
        frap_roi = parse_frap_roi(props_xml_path)
    except Exception as e:
        print(f"  SKIP {label}: parse_frap_roi failed: {e}")
        return None

    # Pixel size: must come from XML
    if "mpp_um" not in frap_roi:
        print(f"  SKIP {label}: pixel size not found in XML ({props_xml_path.name})")
        return None
    params["mpp_um"] = frap_roi["mpp_um"]

    # Bleach centre: must come from XML
    if "roi_cx_px" not in frap_roi or "roi_cy_px" not in frap_roi:
        print(f"  SKIP {label}: bleach ROI centre not found in XML")
        return None
    cx, cy = frap_roi["roi_cx_px"], frap_roi["roi_cy_px"]
    if not (0 <= cx < w and 0 <= cy < h):
        print(f"  SKIP {label}: XML bleach centre ({cx}, {cy}) outside image ({w}x{h})")
        return None
    bleach_center = (cy, cx)
    print(f"    Bleach centre (XML): ({cx}, {cy}) px")

    # ROI radius: must come from XML (area-equivalent radius of ellipse)
    if "roi_rx_px" not in frap_roi or "roi_ry_px" not in frap_roi:
        print(f"  SKIP {label}: bleach ROI radius not found in XML")
        return None
    roi_r = max(1, int(round(np.sqrt(frap_roi["roi_rx_px"] * frap_roi["roi_ry_px"]))))
    print(f"    Bleach ROI radius (XML): {roi_r} px"
          f"  |  area = {frap_roi.get('roi_area_um2', 0):.1f} µm²")
    if frap_roi.get("phases"):
        for ph in frap_roi["phases"]:
            print(f"      {ph['name']}: {ph['frame_count']} frames, {ph['time_s']:.3f} s")

    ref_inner = roi_r + 10
    ref_outer = roi_r + 50

    # Image display
    if control:
        _plot_control_images(label, pre_avg, pb_images.mean(axis=0), save_dir)
    else:
        _plot_pre_images(label, pre_images, pre_avg, pb_images[0],
                         cy, cx, roi_r, ref_inner, ref_outer, save_dir=save_dir)

    # --- Normalization (run for all experiments including control) ---
    # Imax: mean global brightness across ALL pre-bleach frames
    Imax = float(pre_images.mean())

    # Per post-bleach frame: track bleach center via free-center Gaussian fit,
    # measure mean intensity inside circular mask of radius roi_r at that center.
    yy, xx = np.ogrid[:h, :w]
    pb_bleach_roi  = np.empty(len(pb_images))
    current_center = bleach_center
    for i, img in enumerate(pb_images):
        cy_t, cx_t = _fit_bleach_center(img, current_center, search_radius=100)
        frame_mask = ((yy - cy_t) ** 2 + (xx - cx_t) ** 2) <= roi_r ** 2
        pb_bleach_roi[i] = img[frame_mask].mean()
        current_center = (cy_t, cx_t)

    Imin  = float(pb_bleach_roi[0])
    if control:
        Imin = 0.25 * Imax  # no real bleach: use fixed reference point
    denom = Imax - Imin
    if denom <= 0:
        print(f"  SKIP {label}: Imax ({Imax:.1f}) <= Imin ({Imin:.1f}), no bleach detected")
        return None

    # I_norm(t) = (I(t) - Imin) / (Imax - Imin)  →  0=bleach min, 1=full recovery
    pb_norm = (pb_bleach_roi - Imin) / denom
    print(f"    Imax (pre mean) = {Imax:.1f},  Imin (t=0) = {Imin:.1f}")

    # Control: return curve without D
    if control:
        return {
            "label":      label,
            "D_um2_s":    np.nan,
            "Imax":       Imax,
            "Imin":       Imin,
            "pb_times_s": pb_times_s,
            "pb_norm":    pb_norm,
        }

    # Gaussian fit at t=0 for bleach spot sigma (fixed initial center)
    gauss_result = fit_gaussian_2d(pb_images[0], bleach_center, roi_radius=200)
    sigma_px = gauss_result[1] if gauss_result is not None else float(roi_r)
    sigma_um = sigma_px * params.get("mpp_um", 1)

    # FRAP recovery fit
    frap_result = analyze_frap_recovery(pb_times_s, pb_norm, 1.0)
    if frap_result is None:
        print(f"  SKIP {label}: recovery fit failed")
        return None

    D_um2_s = 0.224 * sigma_um ** 2 / frap_result["t_half_s"]

    return {
        "label":           label,
        "D_um2_s":         D_um2_s,
        "tau_s":           frap_result["tau_s"],
        "tau_err":         frap_result["tau_err"],
        "t_half_s":        frap_result["t_half_s"],
        "mobile_fraction": frap_result["mobile_fraction"],
        "I_0":             frap_result["I_0"],
        "I_inf":           frap_result["I_inf"],
        "sigma_um":        sigma_um,
        "Imax":            Imax,
        "Imin":            Imin,
        # Recovery curve: 0 = bleach min, 1 = full pre-bleach level
        "pb_times_s":      pb_times_s,
        "pb_norm":         pb_norm,
    }


# =============================================================================
# Summary plotting
# =============================================================================
def plot_summary(all_results):
    """Create summary figure: individual D values + grouped mean/SEM."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.3, 5.0),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # --- Left panel: individual D values per measurement ---
    x_pos = 0
    x_ticks = []
    x_labels = []
    group_spans = {}  # group_name -> (x_start, x_end)

    for group_name, results in all_results.items():
        base_color, dark_color = GROUP_COLORS.get(
            group_name, (COLORS_BASE[0], COLORS_DARK[0]))
        x_start = x_pos

        for r in results:
            if r is None or np.isnan(r["D_um2_s"]):
                x_pos += 1
                continue
            ax1.scatter(x_pos, r["D_um2_s"], s=26, color=base_color,
                        edgecolors=dark_color, linewidths=0.8, zorder=3)
            x_ticks.append(x_pos)
            x_labels.append(r["label"])
            x_pos += 1

        group_spans[group_name] = (x_start, x_pos - 1)
        x_pos += 1  # gap between groups

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel(r"$D$ [$\mu$m$^2$/s]")
    ax1.set_title("Diffusion coefficient per measurement")

    # Group labels at bottom
    for group_name, (xs, xe) in group_spans.items():
        mid = (xs + xe) / 2
        ax1.annotate(group_name, xy=(mid, 0), xycoords=("data", "axes fraction"),
                     xytext=(0, -45), textcoords="offset points",
                     ha="center", fontsize=10, fontweight="semibold",
                     color=GROUP_COLORS.get(group_name, (COLORS_BASE[0],))[0])

    # --- Right panel: grouped bar with mean and SEM ---
    group_names = []
    group_means = []
    group_sems = []
    bar_colors = []

    for group_name, results in all_results.items():
        D_values = [r["D_um2_s"] for r in results
                    if r is not None and not np.isnan(r["D_um2_s"])]
        if not D_values:
            continue
        D_arr = np.array(D_values)
        group_names.append(group_name)
        group_means.append(np.mean(D_arr))
        group_sems.append(np.std(D_arr, ddof=1) / np.sqrt(len(D_arr)) if len(D_arr) > 1 else 0)
        bar_colors.append(GROUP_COLORS.get(group_name, (COLORS_BASE[0],))[0])

    x_bar = np.arange(len(group_names))
    bars = ax2.bar(x_bar, group_means, width=0.6, color=bar_colors, alpha=0.85,
                   edgecolor=[GROUP_COLORS.get(g, (COLORS_DARK[0],COLORS_DARK[0]))[1]
                              for g in group_names],
                   linewidth=1.0, zorder=2)
    ax2.errorbar(x_bar, group_means, yerr=group_sems, fmt="none",
                 ecolor="black", elinewidth=1.2, capsize=3.0, capthick=1.2, zorder=3)

    # Individual data points on bars
    for i, group_name in enumerate(group_names):
        D_values = [r["D_um2_s"] for r in all_results[group_name]
                    if r is not None and not np.isnan(r["D_um2_s"])]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(D_values))
        ax2.scatter(x_bar[i] + jitter, D_values, s=22, color="white",
                    edgecolors="black", linewidths=0.8, zorder=4, alpha=0.85)

    ax2.set_xticks(x_bar)
    ax2.set_xticklabels(group_names)
    ax2.set_ylabel(r"$D$ [$\mu$m$^2$/s]")
    ax2.set_title("Grouped (mean $\\pm$ SEM)")

    plt.tight_layout()
    return fig


def plot_recovery_comparison(all_results):
    """Overlay recovery curves grouped by condition, normalized to pre-bleach = 1."""

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

    for group_name, results in all_results.items():
        base_color, dark_color = GROUP_COLORS.get(
            group_name, (COLORS_BASE[0], COLORS_DARK[0])
        )
        valid = [r for r in results if r is not None and "pb_norm" in r]
        if not valid:
            continue

        # --- Individual curves (thin, semi-transparent) ---
        all_t, all_I = [], []
        for r in valid:
            t = r["pb_times_s"]
            I = r["pb_norm"]
            ax.plot(t, I, color=base_color, alpha=0.25, linewidth=0.9, zorder=2)
            all_t.append(t)
            all_I.append(I)

        # --- Mean ± SEM on a common time grid ---
        # Use the shortest curve to avoid extrapolation
        t_max_common = min(t[-1] for t in all_t)
        t_grid = np.linspace(0.0, t_max_common, 400)

        I_interp = np.array([
            np.interp(t_grid, t, I) for t, I in zip(all_t, all_I)
        ])
        I_mean = I_interp.mean(axis=0)

        if len(valid) > 1:
            I_sem = I_interp.std(axis=0, ddof=1) / np.sqrt(len(valid))
            ax.fill_between(
                t_grid, I_mean - I_sem, I_mean + I_sem,
                color=base_color, alpha=0.20, zorder=3,
            )

        ax.plot(
            t_grid, I_mean,
            color=dark_color, linewidth=2.2,
            label=f"{group_name} (n={len(valid)})",
            zorder=4,
        )

    # Pre-bleach reference line at I = 1
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)
    # Bleach event marker
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)

    ax.set_xlabel("Time after bleach (s)")
    ax.set_ylabel(r"$(I - I_\mathrm{min})\,/\,(I_\mathrm{max} - I_\mathrm{min})$")
    ax.set_title("FRAP Recovery Comparison")
    ax.set_ylim(bottom=0)
    ax.minorticks_on()
    ax.legend(loc="lower right")

    return fig


def print_summary_table(all_results):
    """Print a results table to the console."""
    print("\n" + "=" * 90)
    print(f"{'Group':<10} {'Folder':<30} {'D [um2/s]':>10} {'tau [s]':>10} "
          f"{'t1/2 [s]':>10} {'Mobile%':>10}")
    print("-" * 90)
    for group_name, results in all_results.items():
        for r in results:
            if r is None:
                continue
            if np.isnan(r["D_um2_s"]):
                print(f"{group_name:<10} {r['label']:<30} {'control':>10}")
                continue
            print(f"{group_name:<10} {r['label']:<30} {r['D_um2_s']:>10.4f} "
                  f"{r['tau_s']:>10.2f} {r['t_half_s']:>10.2f} "
                  f"{r['mobile_fraction']*100:>9.1f}%")
        # Group summary (only for groups with valid D values)
        D_vals = [r["D_um2_s"] for r in results
                  if r is not None and not np.isnan(r["D_um2_s"])]
        if len(D_vals) > 1:
            m = np.mean(D_vals)
            sem = np.std(D_vals, ddof=1) / np.sqrt(len(D_vals))
            print(f"{'':>10} {'>>> MEAN +/- SEM':<30} {m:>10.4f} +/- {sem:.4f}")
        print()
    print("=" * 90)


# =============================================================================
# Main
# =============================================================================
def main():
    all_results = {}

    for group_name, folder_names in GROUPS.items():
        print(f"\n{'='*60}")
        print(f"  GROUP: {group_name}")
        print(f"{'='*60}")
        results = []
        for folder_name in folder_names:
            base_dir = DATA_ROOT / folder_name
            if not base_dir.exists():
                print(f"  WARNING: {base_dir} does not exist, skipping.")
                results.append(None)
                continue
            is_control = (group_name == "no FITC")
            print(f"\n  Analyzing: {folder_name} {'[control]' if is_control else ''}...")
            result = analyze_single(base_dir, save_dir=OUTPUT_DIR, control=is_control)
            if result is not None and not np.isnan(result["D_um2_s"]):
                print(f"    D = {result['D_um2_s']:.4f} um2/s, "
                      f"tau = {result['tau_s']:.2f} s, "
                      f"mobile = {result['mobile_fraction']*100:.1f}%")
            results.append(result)
        all_results[group_name] = results

    # Summary
    print_summary_table(all_results)

    # --- Diffusion coefficient summary ---
    fig_summary = plot_summary(all_results)
    out_path = DATA_ROOT / "frap_batch_summary.png"
    fig_summary.savefig(out_path, dpi=600)
    print(f"\nSummary saved to:\n  {out_path}")

    # --- Recovery curve comparison ---
    fig_recovery = plot_recovery_comparison(all_results)
    out_recovery = DATA_ROOT / "frap_recovery_comparison.png"
    fig_recovery.savefig(out_recovery, dpi=600)
    print(f"Recovery comparison saved to:\n  {out_recovery}")

    plt.show()


if __name__ == "__main__":
    main()
