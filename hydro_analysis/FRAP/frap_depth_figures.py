"""
FRAP v2: diffusion of FITC-dextran 150 kDa (user, 2026-10-07) in water and in 20 mg/mL eADF4(C16)
hydrogel, near the surface (10 µm) and deep in the sample. Consumer stage: reads
cache/frap_v2_phase1.pkl written by frap_analysis_v2.py (run that first); nothing is refitted.
Reads, read-only, the z position of each measurement from its Leica MetaData XML below DATA_ROOT,
and the pre-bleach TIF frames, because the cache holds only their mean: the per-frame pre-bleach
curve is computed with frap_analysis_v2.double_normalised_curve on the cached centre and r_e
(controls: on the cached bleach site), checked against the cached F_ROI,pre.

Depth (user, 2026-10-06): the sample is assumed perfectly planar and the depth is anchored on the
measurement whose folder name states its depth (DEPTH_ANCHOR, 306 µm): depth = z - z_anchor +
anchor depth, with z the z-wide ZPosition (identical for all series of a measurement; checked).
The gel/glass interface measurements ("C16 shallow") are set to SHALLOW_DEPTH_UM by the user;
their planar depth is written to the CSV for comparison. water_2 (failed, 97 % saturated pixels)
is left out of every figure (EXCLUDE_IDS, user 2026-10-06) and logged in the exclusion CSV.
In the figures the depth is shown by the marker only: square = SHALLOW_DEPTH_UM, circle = deep.

D: D_propagation_um2_s (user, 2026-10-06), gate qc_propagation: filled symbols pass, open symbols
fail (shown, never hidden). No theoretical value is shown (user, 2026-10-06).
D0 for D_eff/D0 = arithmetic mean of the QC-passed deep water measurements (user, 2026-10-06).

Figures (OUTPUT_ROOT/figures/, narrow width class, PDF, each with a sidecar JSON):
  frap_d_vs_depth.pdf        D of every water and hydrogel measurement, x = water | hydrogel
  frap_deff_d0_vs_depth.pdf  D_eff / D0, mean +- SD of the QC-passed measurements per sample and depth only
      (user, 2026-10-07)
  frap_recovery_full_scale.pdf (full, linear time axis)  recovery curves, full-scale normalised: (F - F(0)) / (1 - F(0))
      of the double-normalised curve, F(0) = first post-bleach frame (user, 2026-10-06: start at 0); pre-bleach
      frames and the noFITC controls shown, x and y range fixed (user, 2026-10-07). The controls have no bleach
      (1 - F(0) ~ 0), so they are shown as the double-normalised curve F_norm, not full-scale normalised.
      Points outside the axis range are counted in the sidecar JSON.
Also writes frap_depth_values.csv and frap_depth_values_exclusions.csv. No other script reads these.
"""
from __future__ import annotations

import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hydro_analysis.FRAP.frap_analysis import find_series_folders, load_tif_series
from hydro_analysis.FRAP.frap_analysis_v2 import double_normalised_curve, load_measurement

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "frap_v2_phase1.pkl"
DATA_ROOT = Path(r"E:\Daten Promotion Sicherung\Confocal_Measure\2025_09_08_14_57_46--Project_TIF")
OUTPUT_ROOT = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\FRAP_v2")
FIGURE_DIR = OUTPUT_ROOT / "figures"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"

D_COLUMN, D_GATE = "D_propagation_um2_s", "qc_propagation"   # decided by user, 2026-10-06
DEPTH_ANCHOR = ("gel_B1_6", 306.0)                            # decided by user, 2026-10-06 (planar sample)
SHALLOW_DEPTH_UM = 10.0                                       # decided by user, 2026-10-06
SHALLOW_IDS_CONDITION = "gel_interface"                       # "C16 shallow" measurements
OUTER_LAYER_IDS: list[str] = []                               # water_2 was the outer layer; now excluded
EXCLUDE_IDS = {"water_2": "water measurement that failed (97 % saturated pixels); left out of every figure"}  # decided by user, 2026-10-06
SAMPLE_TYPE = {"water": "Water", "gel": "Hydrogel", "gel_interface": "Hydrogel"}   # noFITC not shown

FIG_SIZE = (4.72, 4.72 / 1.42)       # width class "narrow"
COLORS = {"Water": ("#3B8C8C", "#2A6666"), "Hydrogel": ("#D98C3D", "#A6672D")}   # style guide §11
MARKERS = {"shallow": "s", "deep": "o"}                       # decided by user, 2026-10-06
X_POS = {"Water": 0, "Hydrogel": 1}
SUB_OFFSET = {"shallow": -0.15, "deep": 0.15}                 # fixed sub-position per depth, not jitter
RATIO_SUB_OFFSET = {"shallow": -0.07, "deep": 0.07}           # D_eff/D0 figure: closer (user, 2026-10-07)
ALPHA = 0.7
CONTROL_COLOR = "#555555"                                      # noFITC control lines
RECOVERY_XLIM = (-15.0, 150.0)   # user, 2026-10-07: 0 to 150 s; left end -15 s so all pre-bleach frames (from -13.9 s) show
RECOVERY_YLIM = (0.0, 1.1)       # user, 2026-10-08: above 1 so the pre-bleach frames are visible


# ── Data ───────────────────────────────────────────────────────────────────────
def z_position_um(folder: Path) -> float:
    """z-wide focus position of a measurement; all its series must agree."""
    values = set()
    for xml in folder.glob("*/MetaData/*.xml"):
        if xml.name.endswith("_Properties.xml"):
            continue
        found = re.findall(r'ZUseModeName="z-wide" ZPosition="([^"]+)"', xml.read_text("utf-8", "replace"))
        if not found:
            raise ValueError(f"{xml}: no z-wide ZPosition.")
        values.update(round(float(v) * 1e6, 4) for v in found)
    if len(values) != 1:
        raise ValueError(f"{folder}: z-wide positions differ or missing: {sorted(values)}")
    return values.pop()


def load_values() -> tuple[pd.DataFrame, list[dict], dict]:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"{CACHE_PATH} missing: run frap_analysis_v2.py first.")
    cache = pickle.loads(CACHE_PATH.read_bytes())
    res = cache["results"]
    log = [dict(file=r.file, step="condition", parameter="SAMPLE_TYPE", n_before=1, n_after=0,
                reason=f"{r.measurement_id}: condition {r.condition} not shown")
           for r in res.itertuples() if r.condition not in SAMPLE_TYPE]
    for r in cache["controls"].itertuples():
        log.append(dict(file=r.file, step="condition", parameter="SAMPLE_TYPE", n_before=1, n_after=0,
                        reason=f"{r.measurement_id}: noFITC control, not in the D figures (shown in the recovery figure)"))
    df = res[res.condition.isin(SAMPLE_TYPE)].copy()
    for r in df[df.measurement_id.isin(EXCLUDE_IDS)].itertuples():
        log.append(dict(file=r.file, step="excluded by user", parameter="EXCLUDE_IDS", n_before=1, n_after=0,
                        reason=f"{r.measurement_id}: {EXCLUDE_IDS[r.measurement_id]}"))
    df = df[~df.measurement_id.isin(EXCLUDE_IDS)]
    for mid in OUTER_LAYER_IDS + [DEPTH_ANCHOR[0]]:
        if mid not in set(df.measurement_id):
            raise ValueError(f"{mid} not in the cache.")
    df["sample_type"] = df.condition.map(SAMPLE_TYPE)
    df["z_um"] = [z_position_um(DATA_ROOT / f) for f in df.file]
    z_anchor = float(df.loc[df.measurement_id == DEPTH_ANCHOR[0], "z_um"].iloc[0])
    df["depth_planar_um"] = df.z_um - z_anchor + DEPTH_ANCHOR[1]
    shallow = (df.condition == SHALLOW_IDS_CONDITION) | df.measurement_id.isin(OUTER_LAYER_IDS)
    df["position"] = np.where(shallow, "shallow", "deep")
    df["depth_um"] = np.where(shallow, SHALLOW_DEPTH_UM, df.depth_planar_um)
    df["depth_source"] = np.where(shallow, "set by user", "planar, anchored on " + DEPTH_ANCHOR[0])
    df["outer_layer"] = df.measurement_id.isin(OUTER_LAYER_IDS)
    df["D_um2_s"] = df[D_COLUMN]
    df["qc_pass"] = df[D_GATE].astype(bool)
    if df.D_um2_s.isna().any():
        raise ValueError("Measurements without D: " + ", ".join(df.loc[df.D_um2_s.isna(), "measurement_id"]))
    ref = df[(df.sample_type == "Water") & ~df.outer_layer & df.qc_pass]
    if ref.empty:
        raise ValueError("No QC-passed water measurement for D0.")
    d0 = float(ref.D_um2_s.mean())
    df["Deff_D0"] = df.D_um2_s / d0
    df.attrs["D0"] = d0
    df.attrs["D0_ids"] = list(ref.measurement_id)
    curves = {m: (np.asarray(cache["details"][m]["t_s"], float), np.asarray(cache["details"][m]["curve"]["F_norm"], float))
              for m in df.measurement_id}
    return df, log, curves


def _check_pre(mid, curve, cached):
    for key in ("F_roi_pre", "F_ref_pre"):
        if not np.isclose(curve[key], cached[key], rtol=1e-9):
            raise ValueError(f"{mid}: recomputed {key} {curve[key]} differs from the cache {cached[key]}.")


def load_pre_bleach(df) -> tuple[dict, dict]:
    """Per-frame double-normalised pre-bleach curves (FITC) and the noFITC control curves (pre and post)."""
    cache = pickle.loads(CACHE_PATH.read_bytes())
    res = cache["results"].set_index("measurement_id")
    pre = {}
    for m in df.measurement_id:
        r, det = res.loc[m], cache["details"][m]
        pre_folders, _ = find_series_folders(DATA_ROOT / r.file)
        imgs = np.concatenate([load_tif_series(f, f.name)[0] for f in pre_folders])
        t_pre = np.asarray(det["pre_t_s"], float)
        if len(imgs) != len(t_pre):
            raise ValueError(f"{m}: {len(imgs)} pre-bleach frames on disk, {len(t_pre)} in the cache.")
        curve = double_normalised_curve(imgs, imgs, (r.centre_row_px, r.centre_col_px), r.r_e_um, r.mpp_um)
        _check_pre(m, curve, det["curve"])
        pre[m] = (t_pre, curve["F_norm"])
    site = cache["bleach_site"]
    controls = {}
    for r in cache["controls"].itertuples():
        data = load_measurement(DATA_ROOT / r.file)
        mpp = data["mpp_um"]
        centre = (site["row_um"] / mpp, site["col_um"] / mpp)
        curve_pre = double_normalised_curve(data["pre"], data["pre"], centre, site["r_e_um"], mpp)
        curve_post = double_normalised_curve(data["pre"], data["post"], centre, site["r_e_um"], mpp)
        cached = cache["control_details"][r.measurement_id]
        _check_pre(r.measurement_id, curve_pre, cached["curve"])
        if not np.allclose(curve_post["F_norm"], cached["curve"]["F_norm"]):
            raise ValueError(f"{r.measurement_id}: recomputed control curve differs from the cache.")
        controls[r.measurement_id] = {"pre": (data["pre_t_s"], curve_pre["F_norm"]),
                                      "post": (np.asarray(cached["t_s"], float), curve_post["F_norm"])}
    return pre, controls


# ── Plotting ───────────────────────────────────────────────────────────────────
def _points(ax, df, column):
    for (stype, pos), sub in df.groupby(["sample_type", "position"]):
        base, dark = COLORS[stype]
        x = X_POS[stype] + SUB_OFFSET[pos]
        for ok in (True, False):
            s = sub[sub.qc_pass == ok]
            ax.plot(np.full(len(s), x), s[column], MARKERS[pos], ms=4, mfc=base if ok else "none", mec=dark,
                    mew=0.6, ls="none", alpha=ALPHA, zorder=3)
    ax.set_xticks(list(X_POS.values()), list(X_POS.keys()))
    ax.set_xlim(-0.6, len(X_POS) - 0.4)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    handles = [Line2D([], [], marker=MARKERS["shallow"], ms=4, mfc="#bbbbbb", mec="#555555", mew=0.6, ls="none",
                      label=f"{SHALLOW_DEPTH_UM:g} µm" + (" (water: outer layer)" if df.outer_layer.any() else "")),
               Line2D([], [], marker=MARKERS["deep"], ms=4, mfc="#bbbbbb", mec="#555555", mew=0.6, ls="none",
                      label=f"Deep (hydrogel {DEPTH_ANCHOR[1]:g} µm)")]
    if (~df.qc_pass).any():
        handles.append(Line2D([], [], marker="o", ms=4, mfc="none", mec="#555555", mew=0.6, ls="none",
                              label="Fails QC"))
    return handles


def save(fig, stem, meta):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(path, format="pdf")   # final size, no bbox_inches="tight"
    plt.close(fig)
    (FIGURE_DIR / f"{stem}.json").write_text(json.dumps(
        {"figure": path.name, "created": f"{datetime.now():%Y-%m-%d %H:%M}", "cache": str(CACHE_PATH), **meta},
        indent=2, default=str), encoding="utf-8")


def plot_d(df):
    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        handles = _points(ax, df, "D_um2_s")
        ax.set_ylim(0, 1.25 * df.D_um2_s.max())
        ax.set_ylabel(r"Diffusion coefficient $D$ (µm$^2$ s$^{-1}$)")
        ax.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, handletextpad=0.3,
                  columnspacing=0.8)
        save(fig, "frap_d_vs_depth", {"value": D_COLUMN, "gate": D_GATE, "depth_anchor": DEPTH_ANCHOR,
                                      "shallow_depth_um": SHALLOW_DEPTH_UM, "left_out": list(EXCLUDE_IDS)})


def plot_ratio(df):
    """Mean +- SD (ddof = 1) of D_eff/D0 over the QC-passed measurements per sample type and depth."""
    d0 = df.attrs["D0"]
    df = df[~df.outer_layer]
    groups = {}
    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.axhline(1.0, color="0.65", lw=0.8, zorder=1)
        for (stype, pos), sub in df.groupby(["sample_type", "position"]):
            base, dark = COLORS[stype]
            v = sub.loc[sub.qc_pass, "Deff_D0"]
            mean, sd = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan
            ax.errorbar(X_POS[stype] + RATIO_SUB_OFFSET[pos], mean, yerr=sd, fmt=MARKERS[pos], ms=4, mfc=base,
                        mec=dark, mew=0.6, ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=3)
            groups[f"{stype} {pos}"] = {"n": len(v), "mean": mean, "SD": sd, "measurements": list(sub.loc[sub.qc_pass, "measurement_id"]),
                                        "qc_failed_not_in_mean": list(sub.loc[~sub.qc_pass, "measurement_id"])}
        ax.set_xticks(list(X_POS.values()), list(X_POS.keys()))
        ax.set_xlim(-0.6, len(X_POS) - 0.4)
        ax.tick_params(axis="x", which="both", top=False, bottom=False)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel(r"$D_\mathrm{eff}\,/\,D_0$")
        handles = [Line2D([], [], marker=MARKERS[p], ms=4, mfc="#bbbbbb", mec="#555555", mew=0.6, ls="none",
                          label=p.capitalize()) for p in ("shallow", "deep")]
        ax.legend(handles=handles, loc="center right", handletextpad=0.2, borderaxespad=0.3, handlelength=1.0)
        save(fig, "frap_deff_d0_vs_depth", {"value": f"{D_COLUMN} / D0", "gate": D_GATE,
                                            "shown": "mean +- SD (ddof = 1) of the QC-passed measurements",
                                            "D0_um2_s": d0, "D0_from": df.attrs.get("D0_ids"), "groups": groups,
                                            "left_out": list(EXCLUDE_IDS)})
    return groups


def full_scale(f: np.ndarray) -> np.ndarray:
    """Full-scale normalisation of a double-normalised curve: 0 at the first post-bleach frame, 1 = pre-bleach."""
    return (f - f[0]) / (1.0 - f[0])


def _mean_curve(curves):
    """Mean of the curves on a common grid up to the shortest series (as in frap_figures.py)."""
    t_end = min(t[-1] for t, _ in curves)
    grid = np.linspace(0.0, t_end, 200)
    return grid, np.mean([np.interp(grid, t, f) for t, f in curves], axis=0)


def _outside(t, f):
    """Points of one curve beyond the x range, and inside it but beyond the y range."""
    in_x = (t >= RECOVERY_XLIM[0]) & (t <= RECOVERY_XLIM[1])
    return int((~in_x).sum()), int((in_x & ((f < RECOVERY_YLIM[0]) | (f > RECOVERY_YLIM[1]))).sum())


def plot_recovery(df, curves, pre, controls):
    """Full-scale normalised recovery: single measurements thin (pre-bleach frames included), group means bold
    with depth markers, noFITC controls as double-normalised F_norm in grey."""
    outside = {}
    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=(6.30, 6.30 / 1.42))   # width class "full"
        handles, meta = [], {}
        for (stype, pos), sub in df.groupby(["sample_type", "position"], sort=False):
            base, dark = COLORS[stype]
            shown = []
            for row in sub.itertuples():
                t, f = curves[row.measurement_id]
                ff = full_scale(f)
                t_pre, f_pre = pre[row.measurement_id]
                ff_pre = (f_pre - f[0]) / (1.0 - f[0])
                ls = "-" if row.qc_global else "--"
                ax.plot(t, ff, color=base, lw=0.6, alpha=0.6, ls=ls, zorder=2)
                ax.plot(t_pre, ff_pre, color=base, lw=0.6, alpha=0.6, ls=ls, zorder=2)
                outside[row.measurement_id] = [a + b for a, b in zip(_outside(t, ff), _outside(t_pre, ff_pre))]
                if row.qc_global:
                    shown.append((t, ff))
            outer = bool(sub.outer_layer.any())
            lo, hi = sub.depth_um.min(), sub.depth_um.max()
            depth = f"{lo:.0f} µm" if np.isclose(lo, hi) else f"{lo:.0f}–{hi:.0f} µm"
            label = f"{stype}, {'outer layer' if outer else depth}"
            if shown:
                grid, mean = _mean_curve(shown)
                ax.plot(grid, mean, color=dark, lw=1.5, marker=MARKERS[pos], ms=4, mfc=base, mec=dark, mew=0.6,
                        markevery=20, zorder=3)
                handles.append(Line2D([], [], color=dark, lw=1.5, marker=MARKERS[pos], ms=4, mfc=base, mec=dark,
                                      mew=0.6, label=f"{label} (n = {len(shown)})"))
            else:
                handles.append(Line2D([], [], color=base, lw=0.8, ls="--", label=f"{label} (fails QC, n = {len(sub)})"))
            meta[label] = {"measurements": list(sub.measurement_id), "in_mean": len(shown)}
        for mid, c in controls.items():
            for t, f in (c["pre"], c["post"]):
                ax.plot(t, f, color=CONTROL_COLOR, lw=0.6, alpha=0.6, zorder=2)
            outside[mid] = [a + b for a, b in zip(_outside(*c["pre"]), _outside(*c["post"]))]
        handles.append(Line2D([], [], color=CONTROL_COLOR, lw=0.8, label=f"noFITC control, $F_\\mathrm{{norm}}$ (n = {len(controls)})"))
        meta["noFITC control"] = {"measurements": list(controls), "normalisation": "double-normalised F_norm only"}
        ax.set_xlim(*RECOVERY_XLIM)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.set_ylim(*RECOVERY_YLIM)
        ax.set_xlabel("Time after bleach $t$ (s)")
        ax.set_ylabel("Normalized fluorescence recovery")
        ax.legend(handles=handles, loc="lower right", fontsize=7)
        save(fig, "frap_recovery_full_scale", {
            "normalisation": "full scale: (F - F(0)) / (1 - F(0)) of the double-normalised curve (Phair & Misteli), "
                             "F(0) = first post-bleach frame; 1 = pre-bleach level, plateau = mobile fraction; "
                             "pre-bleach frames with the same F(0); noFITC controls: double-normalised F_norm",
            "time_axis": "linear (user, 2026-10-06)", "xlim_s": RECOVERY_XLIM, "ylim": RECOVERY_YLIM,
            "points_outside_axes": {m: {"beyond_x": a, "beyond_y_within_x": b} for m, (a, b) in outside.items()},
            "group_means": "mean of the qc_global-passed curves on a common grid up to the shortest series",
            "dashed": "qc_global failed (shown, not in the mean)", "groups": meta})
    return outside


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df, log, curves = load_values()
    d0 = df.attrs["D0"]
    print(f"D0 = mean of {df.attrs['D0_ids']} = {d0:.3f} um2/s")
    cols = ["measurement_id", "sample_type", "position", "outer_layer", "file", "z_um", "depth_planar_um",
            "depth_um", "depth_source", "D_um2_s", "Deff_D0", "qc_pass"]
    print(df[cols].round(3).to_string(index=False))
    for (t, p), g in df[df.qc_pass].groupby(["sample_type", "position"]):
        print(f"{t} {p}: n = {len(g)}, D = {g.D_um2_s.mean():.2f} +- {g.D_um2_s.std(ddof=1):.2f}, "
              f"Deff/D0 = {g.Deff_D0.mean():.3f} +- {g.Deff_D0.std(ddof=1):.3f}")
    pre, controls = load_pre_bleach(df)
    plot_d(df)
    for name, g in plot_ratio(df).items():
        print(f"D_eff/D0 {name}: n = {g['n']}, {g['mean']:.3f} +- {g['SD']:.3f} (SD)")
    for m, (a, b) in plot_recovery(df, curves, pre, controls).items():
        print(f"recovery {m}: {a} points beyond x range, {b} within x range beyond y range")
    df[cols].to_csv(FIGURE_DIR / "frap_depth_values.csv", index=False)
    pd.DataFrame(log, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        FIGURE_DIR / "frap_depth_values_exclusions.csv", index=False)
    print(f"Written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
