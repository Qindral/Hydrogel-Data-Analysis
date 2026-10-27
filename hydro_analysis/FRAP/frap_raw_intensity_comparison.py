"""
FRAP v2: raw (not normalised) intensity of every bleaching, FITC-dextran measurements against the
noFITC controls -- appendix overview figure (user, 2026-10-22; ENTSCHEIDUNGEN.md E035).

Consumer stage: reads cache/frap_v2_phase1.pkl written by frap_analysis_v2.py (run that first) and
nothing is refitted. Like frap_depth_figures.py it reads, read-only, the pre-bleach TIF frames (the
cache holds only their mean) and the Leica MetaData XML of each measurement.

Quantity (E035 A): mean grey value in the bleach ROI, in raw counts of the 8-bit images,
BACKGROUND_COUNTS = 0 (PMT offset 0 in all files), no normalisation of any kind. The ROI is the
disk of radius r_e around the measurement's own bleach centre, as in frap_analysis_v2.py
(double_normalised_curve, F_roi); the noFITC controls use the median bleach site of the standard
measurements (cache "bleach_site", r = 46.4 um), as in frap_analysis_v2.analyse_control. Post-bleach
values come from the cache; pre-bleach values are recomputed per frame with the same function and
checked against the cached F_roi,pre.

Imaging laser power (E035 A): not uniform (488 nm at 1.0 % or 0.5 %, read from the first
ATLConfocalSettingDefinition of the Pre and Pb1 series XML; both must agree). All measurements are
shown as recorded, no conversion; 1.0 % solid, 0.5 % dashed. Detector gain 800, offset 0 and
pinhole 1 AU are the same in all files (checked and written to the CSV).

Figure (E035: one panel per bleaching in a grid, appendix only): frap_raw_intensity_by_measurement.pdf,
width class full, one panel per measurement (buffer, hydrogel shallow and deep, noFITC), shared
y axis in counts, each panel on its own time axis (measurement durations differ). Every FITC panel
also shows the noFITC controls of the cache in grey for comparison. water_2 stays left out
(frap_depth_figures.EXCLUDE_IDS, decided by user, 2026-10-06) and is logged in the exclusion CSV, as are
the noFITC controls excluded in frap_analysis_v2.py (cache "control_exclusions"; nofitc_2, nofitc_3,
FITC clearly present, user 2026-10-27).

Writes to FIGURE_DIR: frap_raw_intensity_by_measurement.pdf, frap_raw_intensity_by_measurement.csv
(per measurement: settings and raw ROI levels), frap_raw_intensity_by_measurement_exclusions.csv.
No other script reads these outputs.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hydro_analysis.FRAP.frap_analysis import find_series_folders, load_tif_series
from hydro_analysis.FRAP.frap_analysis_v2 import BACKGROUND_COUNTS, double_normalised_curve, load_measurement
from hydro_analysis.FRAP.frap_depth_figures import (
    CACHE_PATH, CONTROL_COLOR, DATA_ROOT, DISPLAY_LABEL, DISS_COLORS, EXCLUDE_IDS, FIGURE_DIR, STYLE_PATH,
    load_values,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FIG_STEM = "frap_raw_intensity_by_measurement"
FIG_WIDTH_IN = 6.30                  # width class "full" (appendix grid, E035)
PANEL_HEIGHT_IN = 1.45
N_COLS = 4
LASER_LINE_NM = 488
LASER_STYLE = {1.0: "-", 0.5: (0, (4, 2))}   # imaging power (%) -> line style; E035 A: marked, not converted
SATURATION_COUNTS = 255              # 8-bit images; upper limit of the y axis
GROUP_ORDER = ("Water", "Hydrogel", "noFITC")


# ── Metadata ───────────────────────────────────────────────────────────────────
def imaging_settings(series_dir: Path) -> dict:
    """Imaging laser power, active detector gain/offset and pinhole of one Leica series."""
    xml = series_dir / "MetaData" / f"{series_dir.name}.xml"
    text = xml.read_text("utf-8", "replace")
    start = text.find("<ATLConfocalSettingDefinition")
    end = text.find("</ATLConfocalSettingDefinition>", start)
    if start < 0 or end < 0:
        raise ValueError(f"{xml}: no ATLConfocalSettingDefinition")
    block = text[start:end]
    laser = re.findall(rf'LaserLine="{LASER_LINE_NM}" IntensityDev="([^"]+)"', block)
    detectors = re.findall(r'<Detector [^>]*IsActive="1"[^>]*>', block)
    pinhole = re.search(r'PinholeAiry="([^"]+)"', block)
    if len(laser) != 1 or len(detectors) != 1 or pinhole is None:
        raise ValueError(f"{xml}: expected one {LASER_LINE_NM} nm line, one active detector and a pinhole; "
                         f"found {len(laser)}, {len(detectors)}, {pinhole is not None}")
    return {"laser_pct": round(float(laser[0]), 1),
            "detector_gain": round(float(re.search(r'Gain="([^"]+)"', detectors[0]).group(1)), 1),
            "detector_offset": float(re.search(r'Offset="([^"]+)"', detectors[0]).group(1)),
            "pinhole_airy": round(float(pinhole.group(1)), 3)}


def measurement_settings(folder: Path) -> dict:
    """Settings of the Pre and the first post-bleach series; they must be identical."""
    pre_folders, pb_folders = find_series_folders(folder)
    pre, post = imaging_settings(pre_folders[0]), imaging_settings(pb_folders[0])
    if pre != post:
        raise ValueError(f"{folder}: imaging settings differ between {pre_folders[0].name} and {pb_folders[0].name}: "
                         f"{pre} vs {post}")
    return pre


# ── Raw ROI curves ─────────────────────────────────────────────────────────────
def raw_pre_fitc(mid: str, res: pd.Series, det: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame raw ROI mean of the pre-bleach frames on the cached centre and r_e."""
    pre_folders, _ = find_series_folders(DATA_ROOT / res["file"])
    imgs = np.concatenate([load_tif_series(f, f.name)[0] for f in pre_folders])
    t_pre = np.asarray(det["pre_t_s"], float)
    if len(imgs) != len(t_pre):
        raise ValueError(f"{mid}: {len(imgs)} pre-bleach frames on disk, {len(t_pre)} in the cache")
    curve = double_normalised_curve(imgs, imgs, (res["centre_row_px"], res["centre_col_px"]), res["r_e_um"],
                                    res["mpp_um"])
    if not np.isclose(curve["F_roi"].mean(), det["curve"]["F_roi_pre"], rtol=1e-9):
        raise ValueError(f"{mid}: recomputed pre-bleach ROI mean differs from the cache")
    return t_pre, curve["F_roi"]


def raw_control(mid: str, file: str, site: dict, cached: dict) -> dict:
    """Pre and post raw ROI curves of one noFITC control at the median bleach site."""
    data = load_measurement(DATA_ROOT / file)
    mpp = data["mpp_um"]
    centre = (site["row_um"] / mpp, site["col_um"] / mpp)
    pre = double_normalised_curve(data["pre"], data["pre"], centre, site["r_e_um"], mpp)
    post = double_normalised_curve(data["pre"], data["post"], centre, site["r_e_um"], mpp)
    if not np.allclose(post["F_roi"], cached["curve"]["F_roi"]):
        raise ValueError(f"{mid}: recomputed control ROI curve differs from the cache")
    return {"pre": (data["pre_t_s"], pre["F_roi"]), "post": (np.asarray(cached["t_s"], float), post["F_roi"]),
            "mpp_um": mpp, "roi_radius_um": site["r_e_um"]}


def test_roi_mean_synthetic() -> None:
    """Known answer: background 10 counts, disk of radius 5 px at 50 counts -> F_roi = 50 in every frame."""
    rows, cols = np.indices((64, 64))
    disk = np.hypot(rows - 32, cols - 30) <= 5
    stack = np.where(disk, 50.0, 10.0)[None].repeat(3, axis=0)
    curve = double_normalised_curve(stack, stack, (32, 30), 5.0, 1.0)
    print(f"Synthetic check F_roi: true 50 counts, recovered {curve['F_roi']}")
    assert np.allclose(curve["F_roi"], 50.0) and BACKGROUND_COUNTS == 0.0


# ── Figure ─────────────────────────────────────────────────────────────────────
def plot_grid(panels: list[dict], controls: dict) -> plt.Figure:
    n_rows = int(np.ceil(len(panels) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(FIG_WIDTH_IN, n_rows * PANEL_HEIGHT_IN + 0.5),
                             sharey=True, squeeze=False)
    y_max = 1.08 * max(max(np.max(p["pre"][1]), np.max(p["post"][1])) for p in panels)
    for ax, p in zip(axes.flat, panels):
        if p["group"] != "noFITC":
            for c in controls.values():
                for t, f in (c["pre"], c["post"]):
                    ax.plot(t, f, color=CONTROL_COLOR, lw=0.6, ls=LASER_STYLE[c["laser_pct"]], alpha=0.8, zorder=1)
        color = CONTROL_COLOR if p["group"] == "noFITC" else DISS_COLORS[p["group"]][1]
        for t, f in (p["pre"], p["post"]):
            ax.plot(t, f, color=color, lw=1.0, ls=LASER_STYLE[p["laser_pct"]], zorder=3)
        ax.axvline(0.0, color="0.75", lw=0.5, zorder=0)
        ax.set_title(p["title"], fontsize=7, fontweight="normal", pad=2, linespacing=1.1)
        ax.set_ylim(0, min(y_max, SATURATION_COUNTS))
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.supxlabel("Time after bleach $t$ (s)", fontsize=9)
    fig.supylabel("Mean intensity in the bleach ROI (counts)", fontsize=9)
    handles = [Line2D([], [], color=DISS_COLORS["Water"][1], lw=1.0), Line2D([], [], color=DISS_COLORS["Hydrogel"][1], lw=1.0),
               Line2D([], [], color=CONTROL_COLOR, lw=1.0), Line2D([], [], color="0.3", lw=1.0, ls=LASER_STYLE[1.0]),
               Line2D([], [], color="0.3", lw=1.0, ls=LASER_STYLE[0.5])]
    labels = ["Buffer", "Hydrogel", "noFITC control", f"{LASER_LINE_NM} nm imaging 1.0 %", "0.5 %"]
    fig.legend(handles, labels, loc="outside upper center", ncol=len(handles), frameon=False, fontsize=7,
               handlelength=2.2, columnspacing=1.0)
    return fig


def main():
    test_roi_mean_synthetic()
    df, _log, _curves = load_values()                     # FITC measurements, water_2 already left out
    cache = pickle.loads(CACHE_PATH.read_bytes())
    res = cache["results"].set_index("measurement_id")
    site = cache["bleach_site"]
    exclusions = [dict(file=res.loc[m, "file"], step="excluded by user", parameter="EXCLUDE_IDS", n_before=1, n_after=0,
                       reason=f"{m}: {why}") for m, why in EXCLUDE_IDS.items() if m in res.index]
    exclusions += cache.get("control_exclusions", [])

    panels, rows = [], []
    df = df.assign(group_order=df["sample_type"].map({g: i for i, g in enumerate(GROUP_ORDER)}))
    for r in df.sort_values(["group_order", "depth_um", "measurement_id"]).itertuples():
        full = res.loc[r.measurement_id]
        det = cache["details"][r.measurement_id]
        settings = measurement_settings(DATA_ROOT / r.file)
        pre = raw_pre_fitc(r.measurement_id, full, det)
        post = (np.asarray(det["t_s"], float), np.asarray(det["curve"]["F_roi"], float))
        depth = f"{r.depth_um:.0f} µm"
        panels.append({"group": r.sample_type, "pre": pre, "post": post, "laser_pct": settings["laser_pct"],
                       "title": f"{r.measurement_id}, {settings['laser_pct']:g} %\n{DISPLAY_LABEL[r.sample_type]}, {depth}"})
        rows.append({"measurement_id": r.measurement_id, "group": DISPLAY_LABEL[r.sample_type], "depth_um": r.depth_um,
                     **settings, "mpp_um": full.mpp_um, "roi_radius_um": full.r_e_um,
                     "roi_pre_mean_counts": float(np.mean(pre[1])), "roi_post_first_counts": float(post[1][0]),
                     "roi_post_last_counts": float(post[1][-1]), "t_last_s": float(post[0][-1])})

    controls = {}
    for c in cache["controls"].itertuples():
        settings = measurement_settings(DATA_ROOT / c.file)
        ctl = raw_control(c.measurement_id, c.file, site, cache["control_details"][c.measurement_id])
        controls[c.measurement_id] = {**ctl, "laser_pct": settings["laser_pct"]}
        panels.append({"group": "noFITC", "pre": ctl["pre"], "post": ctl["post"], "laser_pct": settings["laser_pct"],
                       "title": f"{c.measurement_id}, {settings['laser_pct']:g} %\nnoFITC control"})
        rows.append({"measurement_id": c.measurement_id, "group": "noFITC", "depth_um": np.nan, **settings,
                     "mpp_um": ctl["mpp_um"], "roi_radius_um": ctl["roi_radius_um"],
                     "roi_pre_mean_counts": float(np.mean(ctl["pre"][1])), "roi_post_first_counts": float(ctl["post"][1][0]),
                     "roi_post_last_counts": float(ctl["post"][1][-1]), "t_last_s": float(ctl["post"][0][-1])})

    table = pd.DataFrame(rows)
    print(table.round(2).to_string(index=False))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(FIGURE_DIR / f"{FIG_STEM}.csv", index=False)
    pd.DataFrame(exclusions, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        FIGURE_DIR / f"{FIG_STEM}_exclusions.csv", index=False)
    with plt.style.context(STYLE_PATH):
        fig = plot_grid(panels, controls)
        fig.savefig(FIGURE_DIR / f"{FIG_STEM}.pdf")   # final size, no bbox_inches="tight"
        plt.close(fig)
    print(f"Written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
