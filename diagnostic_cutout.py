"""
Diagnostic script: count TrackMate particles within hydrogel cutout bounds
for +/-5 frames around the specified frame.

Conclusion from investigation:
- TrackMate XML x,y are already in PIXEL coordinates (not microns).
- Image dimensions are 200x150 px, track ranges confirm pixel coords.
- No mpp conversion is needed for cutout comparison.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydro_analysis.core import single_file_data, scan_xml_folder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HYDROGEL_ROOT = Path(r"E:\PhD Data Analysis\SPT 2025 II\Trajectory_Visualisation\hydrogel")

# Cutout definitions from Cutout.txt (pixel coordinates, origin top-left)
CUTOUTS = {
    20: {"frame": 1360, "x1": 116, "y1": 46, "x2": 184, "y2": 92},
    50: {"frame": 902,  "x1": 118, "y1": 30, "x2": 181, "y2": 35},
}

FRAME_RANGE = 5

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
xml_files = scan_xml_folder(HYDROGEL_ROOT)
print(f"Found {len(xml_files)} XML files:")
for xf in xml_files:
    print(f"  {xf}")
print()

datasets = {}
for xf in xml_files:
    result = single_file_data(xf)
    if result is None:
        print(f"  SKIP (no calibration): {xf.name}")
        continue
    size_nm = result["particle_size_nm"]
    datasets[size_nm] = result
    print(f"  Loaded {xf.name}: size={size_nm}nm, mpp={result['mpp']}, "
          f"fps={result['fps']:.2f}, tracks={result['num_tracks']}, frames={result['num_frames']}")
print()

# ---------------------------------------------------------------------------
# Diagnostic for each particle size
# ---------------------------------------------------------------------------
for size_nm, cutout in CUTOUTS.items():
    if size_nm not in datasets:
        print(f"=== {size_nm} nm: NO DATA FOUND ===\n")
        continue

    data = datasets[size_nm]
    df = data["tracks_df"]
    mpp = data["mpp"]
    center_frame = cutout["frame"]

    x_min = min(cutout["x1"], cutout["x2"])
    x_max = max(cutout["x1"], cutout["x2"])
    y_min = min(cutout["y1"], cutout["y2"])
    y_max = max(cutout["y1"], cutout["y2"])

    print("=" * 72)
    print(f"  {size_nm} nm  |  mpp = {mpp} um/px  |  center frame = {center_frame}")
    print(f"  Cutout bounds (pixels): x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print(f"  Cutout size: {x_max - x_min} x {y_max - y_min} pixels")
    print(f"  Track coord ranges: x=[{df['x'].min():.1f}, {df['x'].max():.1f}], "
          f"y=[{df['y'].min():.1f}, {df['y'].max():.1f}]")
    print(f"  NOTE: Track coords are in PIXELS (no mpp conversion needed)")
    print("=" * 72)

    # Show all particle positions in the frame range
    print(f"\n  All particle positions (frames {center_frame-FRAME_RANGE} to {center_frame+FRAME_RANGE}):")
    print(f"  {'frame':>7}  {'particle':>8}  {'x':>8}  {'y':>8}  {'x_in':>5}  {'y_in':>5}  status")
    print(f"  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*10}")

    for f in range(center_frame - FRAME_RANGE, center_frame + FRAME_RANGE + 1):
        fdf = df[df["frame"] == f]
        if len(fdf) == 0:
            print(f"  {f:>7}  {'(no particles)':>8}")
            continue
        for _, row in fdf.iterrows():
            in_x = x_min <= row["x"] <= x_max
            in_y = y_min <= row["y"] <= y_max
            status = "IN CUTOUT" if (in_x and in_y) else ""
            center_mark = " <-- center" if f == center_frame else ""
            print(f"  {f:>7}  {int(row['particle']):>8}  {row['x']:>8.2f}  {row['y']:>8.2f}"
                  f"  {str(in_x):>5}  {str(in_y):>5}  {status}{center_mark}")

    # Summary table
    print(f"\n  Summary table:")
    print(f"  {'frame':>7} | {'in_cutout':>10} | {'total_in_frame':>15}")
    print(f"  {'-'*7}-+-{'-'*10}-+-{'-'*15}")

    for f in range(center_frame - FRAME_RANGE, center_frame + FRAME_RANGE + 1):
        fdf = df[df["frame"] == f]
        total = len(fdf)
        if total == 0:
            print(f"  {f:>7} | {0:>10} | {0:>15}")
            continue
        in_cutout = ((fdf["x"] >= x_min) & (fdf["x"] <= x_max) &
                     (fdf["y"] >= y_min) & (fdf["y"] <= y_max)).sum()
        marker = " <-- center" if f == center_frame else ""
        print(f"  {f:>7} | {in_cutout:>10} | {total:>15}{marker}")

    print()
