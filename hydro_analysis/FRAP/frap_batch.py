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
    find_bleach_center,
    fit_gaussian_2d,
    analyze_frap_recovery,
)


# =============================================================================
# Configuration
# =============================================================================
DATA_ROOT = Path(r"H:\Daten Promotion Sicherung\Confocal_Measure\2025_09_08_14_57_46--Project_TIF")

GROUPS = {
    "C16": [
        "C16 _ no FITC",
        "C16 _deep",
        "C16 _deep 2",
        "C16 _deep B1",
        "C16 _deep B1 _2",
        "C16 _deep B1 _3",
        "C16 _deep B1 _6_306 um",
        "C16 _deep3",
        "C16 _shallow B1",
        "C16 shallow B2",
    ],
    "FRAP": [
        "FRAP 001",
        "FRAP 002",
        "FRAP 003",
        "FRAP 004",
        "FRAP 005",
    ],
    "Water": [
        "Water",
        "Water 2",
        "Water 3",
    ],
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
    "C16":   (COLORS_BASE[0], COLORS_DARK[0]),
    "FRAP":  (COLORS_BASE[1], COLORS_DARK[1]),
    "Water": (COLORS_BASE[2], COLORS_DARK[2]),
}


# =============================================================================
# Single-experiment analysis (returns D or NaN)
# =============================================================================
def analyze_single(base_dir):
    """Run FRAP analysis on a single experiment folder.

    Returns dict with results or None if analysis fails.
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

    # Pixel size (from first Pb folder's series XML as fallback)
    if "mpp_um" not in params:
        from frap_analysis import parse_series_xml
        pb0_name = pb_folders[0].name
        try:
            _, dims = parse_series_xml(pb_folders[0] / "MetaData" / f"{pb0_name}.xml")
            if 1 in dims:
                params["mpp_um"] = dims[1]["element_size"] * 1e6
        except Exception:
            pass

    # Build continuous time axis from absolute timestamps
    t0 = all_pb_abs_times[0]
    pb_times_s = np.array([(t - t0).total_seconds() for t in all_pb_abs_times])

    # Pre-bleach average
    pre_avg = pre_images.mean(axis=0)

    # Find bleach center
    bleach_center = find_bleach_center(pre_avg, pb_images[0])
    cy, cx = bleach_center
    roi_r = 60
    h, w = pre_avg.shape

    # ROIs
    yy, xx = np.ogrid[:h, :w]
    bleach_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= roi_r**2
    ref_inner = roi_r + 10
    ref_outer = roi_r + 50
    ref_mask = (((yy - cy) ** 2 + (xx - cx) ** 2) >= ref_inner**2) & (
        ((yy - cy) ** 2 + (xx - cx) ** 2) <= ref_outer**2
    )

    # Intensities
    pre_I = np.array([img[bleach_mask].mean() for img in pre_images]).mean()
    pre_ref_I = np.array([img[ref_mask].mean() for img in pre_images]).mean()
    pb_bleach_roi = np.array([img[bleach_mask].mean() for img in pb_images])
    pb_ref_roi = np.array([img[ref_mask].mean() for img in pb_images])
    pb_norm = (pb_bleach_roi / pb_ref_roi) / (pre_I / pre_ref_I)

    # Gaussian fit at t=0 for bleach spot size
    gauss_result = fit_gaussian_2d(pb_images[0], bleach_center, roi_radius=200)
    if gauss_result is not None:
        sigma_px = gauss_result[1]
    else:
        sigma_px = float(roi_r)
    sigma_um = sigma_px * params.get("mpp_um", 1)

    # FRAP recovery fit
    frap_result = analyze_frap_recovery(pb_times_s, pb_norm, 1.0)
    if frap_result is None:
        print(f"  SKIP {label}: recovery fit failed")
        return None

    D_um2_s = 0.224 * sigma_um**2 / frap_result["t_half_s"]

    return {
        "label": label,
        "D_um2_s": D_um2_s,
        "tau_s": frap_result["tau_s"],
        "tau_err": frap_result["tau_err"],
        "t_half_s": frap_result["t_half_s"],
        "mobile_fraction": frap_result["mobile_fraction"],
        "I_0": frap_result["I_0"],
        "I_inf": frap_result["I_inf"],
        "sigma_um": sigma_um,
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
            if r is None:
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
        D_values = [r["D_um2_s"] for r in results if r is not None]
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
        D_values = [r["D_um2_s"] for r in all_results[group_name] if r is not None]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(D_values))
        ax2.scatter(x_bar[i] + jitter, D_values, s=22, color="white",
                    edgecolors="black", linewidths=0.8, zorder=4, alpha=0.85)

    ax2.set_xticks(x_bar)
    ax2.set_xticklabels(group_names)
    ax2.set_ylabel(r"$D$ [$\mu$m$^2$/s]")
    ax2.set_title("Grouped (mean $\\pm$ SEM)")

    plt.tight_layout()
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
            print(f"{group_name:<10} {r['label']:<30} {r['D_um2_s']:>10.4f} "
                  f"{r['tau_s']:>10.2f} {r['t_half_s']:>10.2f} "
                  f"{r['mobile_fraction']*100:>9.1f}%")
        # Group summary
        D_vals = [r["D_um2_s"] for r in results if r is not None]
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
            print(f"\n  Analyzing: {folder_name} ...")
            result = analyze_single(base_dir)
            if result is not None:
                print(f"    D = {result['D_um2_s']:.4f} um2/s, "
                      f"tau = {result['tau_s']:.2f} s, "
                      f"mobile = {result['mobile_fraction']*100:.1f}%")
            results.append(result)
        all_results[group_name] = results

    # Summary
    print_summary_table(all_results)

    fig = plot_summary(all_results)
    out_path = DATA_ROOT / "frap_batch_summary.png"
    fig.savefig(out_path, dpi=600)
    out_pdf = DATA_ROOT / "frap_batch_summary.pdf"
    fig.savefig(out_pdf)
    print(f"\nSummary saved to:\n  {out_path}\n  {out_pdf}")
    plt.show()


if __name__ == "__main__":
    main()
