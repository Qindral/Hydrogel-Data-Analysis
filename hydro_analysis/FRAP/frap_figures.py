"""
FRAP v2, phase 2: publication figures for FITC-dextran 70 kDa in water vs. in 20 mg/mL
eADF4(C16) hydrogel. Consumer stage: reads cache/frap_v2_phase1.pkl written by
frap_analysis_v2.py (run that first and review its tables); nothing is refitted here. Only
Fig. 1 reloads raw images (with frap_analysis_v2.load_measurement) because it shows them.

Figures (OUTPUT_ROOT/figures/, each with a sidecar JSON of parameters, fit windows and paths):
  fig1_frap_image_sequence.png  (full, 600 dpi)  water and gel at pre-bleach, first post-bleach
      frame, frame closest to t_half and last frame; one contrast per row (pre-bleach
      percentiles), crop 1.42:1 around the bleach centre, 10 um scale bar, r_e circle (solid)
      and bleach point (+; point bleach, r_n = 0, so no nominal circle exists)
  fig2_profile_broadening.pdf   (full)  A: radial profiles I_post/I_pre with Gaussian fits at five
      times of the representative gel measurement; B: w^2(t) of every measurement passing the
      profile QC, with the weighted linear fits
  fig3_recovery_curves.pdf      (full)  double-normalised recovery on a symlog time axis
      (linear below one frame interval so t = 0 stays visible), single measurements thin,
      group means bold, noFITC controls grey; residual panel below
  fig4_diffusion_mobile_fraction.pdf (narrow)  A: D_profile per measurement (log y) with
      geometric means and the Stokes-Einstein water value; B: mobile fraction M_f
  figS1_all_measurements.pdf    (full)  recovery + fit and w^2(t) for every measurement; panels of
      measurements failing the respective QC gate are shaded and labelled
  figS2_method_comparison.pdf   (narrow)  D_soumpasis, D_axelrod, D_kang vs. D_profile, identity line

Selection: gel measurements of FOCUS_CONCENTRATION_MG_ML only; filled symbols pass the QC gate
of the plotted quantity, open symbols fail it and are excluded from means (never hidden).
Representative measurement of a group: the QC-passed (profile) measurement with the largest
bleach depth K0. Style: Styleguide_Figures_Dissertation.md v2 (thesis.mplstyle, width classes,
no bbox_inches="tight", plots as PDF, images as PNG 600 dpi).
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from hydro_analysis.FRAP.frap_analysis_v2 import DATA_ROOT, OUTPUT_ROOT, load_measurement

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "frap_v2_phase1.pkl"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
FIGURE_DIR = OUTPUT_ROOT / "figures"
FOCUS_CONCENTRATION_MG_ML = 20.0

WIDTH_IN = {"full": 6.30, "narrow": 4.72, "half": 3.07, "third": 2.01}
RATIO = 1.42
DASH_THEORY = (0, (4, 3))

GROUP_STYLE = {   # (base, dark), marker, label; colours per style guide §11, batches by marker
    "water": (("#3B8C8C", "#2A6666"), "D", "Water"),
    "gel_A1": (("#D98C3D", "#A6672D"), "o", "Gel A1"),
    "gel_B1": (("#D98C3D", "#A6672D"), "s", "Gel B1"),
    "interface": (("#54ad49", "#1a5e3c"), "^", "Gel/glass interface"),
    "nofitc": (("#bdbdbd", "#6e6e6e"), "v", "Gel, no FITC"),
}
GEL_STYLE = (("#D98C3D", "#A6672D"), "o", "Gel")
QC_FAIL_SHADE = "#f6e3e3"
WINDOW_SHADE_LIGHT = "#e6e6e6"

IMAGE_CONTRAST_PERCENTILES = (0.5, 99.5)
SCALEBAR_UM = 10.0
PROFILE_PANEL_TIMES = 5
PROFILE_PANEL_MARKER_STEP = 4          # plot every n-th radial bin as a marker


# ── Helpers ────────────────────────────────────────────────────────────────────

def _group(row) -> str:
    if row["condition"] == "water":
        return "water"
    if row["condition"] == "gel_interface":
        return "interface"
    return f"gel_{row['batch']}"


def _focus(results: pd.DataFrame) -> pd.DataFrame:
    conc = results["gel_concentration_mg_ml"].astype(float)
    keep = (results["condition"] == "water") | np.isclose(conc, FOCUS_CONCENTRATION_MG_ML)
    out = results[keep].copy()
    out["group"] = out.apply(_group, axis=1)
    return out


def _representative(results: pd.DataFrame, condition: str) -> pd.Series:
    sub = results[(results["condition"] == condition) & results["qc_profile"]]
    return sub.loc[sub["K0"].idxmax()]


def new_figure(kind="full", nrows=1, ncols=1, height=None, **kw):
    w = WIDTH_IN[kind]
    return plt.subplots(nrows, ncols, figsize=(w, height or w / RATIO), **kw)


def save(fig, stem: str, fmt: str, meta: dict) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{stem}.{fmt}"
    fig.savefig(path, format=fmt, dpi=600)
    plt.close(fig)
    sidecar = {"figure": path.name, "created": f"{datetime.now():%Y-%m-%d %H:%M}", "cache": str(CACHE_PATH),
               "focus_concentration_mg_ml": FOCUS_CONCENTRATION_MG_ML, **meta}
    (FIGURE_DIR / f"{stem}.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")


def _panel_label(ax, letter):
    ax.text(-0.14, 1.03, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def _recovery_model(t, F0, Finf, tau):
    return F0 + (Finf - F0) * (1.0 - np.exp(-t / tau))


def _symlog_time(ax, linthresh):
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.4)
    ax.xaxis.set_major_locator(mticker.SymmetricalLogLocator(base=10, linthresh=linthresh))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(mticker.SymmetricalLogLocator(base=10, linthresh=linthresh, subs=np.arange(2, 10)))


# ── Figure 1: image sequence ───────────────────────────────────────────────────

def fig1_image_sequence(cache: dict, results: pd.DataFrame) -> None:
    rows = [("Water", _representative(results, "water")),
            (f"Gel {FOCUS_CONCENTRATION_MG_ML:g} mg mL$^{{-1}}$", _representative(results, "gel"))]
    width = WIDTH_IN["full"]
    img_w = (width - 0.35) / 4
    fig, axes = plt.subplots(2, 4, figsize=(width, 2 * img_w / RATIO + 0.6), layout="none",
                             gridspec_kw={"wspace": 0.03, "hspace": 0.22})
    meta_rows = []
    for r, (label, rec) in enumerate(rows):
        data = load_measurement(DATA_ROOT / rec["file"])
        t = data["post_t_s"]
        i_half = int(np.argmin(np.abs(t - rec["t_half_s"])))
        frames = [("Pre-bleach", data["pre"][-1]), ("t = 0 s", data["post"][0]),
                  (f"t = {t[i_half]:.1f} s (≈ t$_{{1/2}}$)", data["post"][i_half]),
                  (f"t = {t[-1]:.0f} s (plateau)", data["post"][-1])]
        vmin, vmax = np.percentile(data["pre"][-1], IMAGE_CONTRAST_PERCENTILES)
        mpp = data["mpp_um"]
        n_rows_px, n_cols_px = data["pre"].shape[1:]
        crop_h = int(round(n_cols_px / RATIO))
        top = int(np.clip(round(rec["centre_row_px"] - crop_h / 2), 0, n_rows_px - crop_h))
        for c, (title, img) in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(img[top:top + crop_h], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            cy, cx = rec["centre_row_px"] - top, rec["centre_col_px"]
            ax.add_patch(Circle((cx, cy), rec["r_e_um"] / mpp, fill=False, ec="white", lw=1.0))
            ax.plot(cx, cy, "+", color="white", ms=6, mew=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8, fontweight="normal", pad=3)
            if c == 3:
                bar_px = SCALEBAR_UM / mpp
                x0, y0 = n_cols_px - bar_px - 20, crop_h - 20
                ax.add_patch(Rectangle((x0, y0), bar_px, 4, color="white", lw=0))
                ax.text(x0 + bar_px, y0 - 6, f"{SCALEBAR_UM:g} µm", color="white", fontsize=8, ha="right", va="bottom")
        axes[r, 0].set_ylabel(label, fontsize=9)
        meta_rows.append({"row": label, "measurement_id": rec["measurement_id"], "folder": str(DATA_ROOT / rec["file"]),
                          "frames_shown_t_s": [None, 0.0, float(t[i_half]), float(t[-1])],
                          "contrast_vmin_vmax": [float(vmin), float(vmax)], "crop_rows_px": [top, top + crop_h],
                          "r_e_um": float(rec["r_e_um"]), "centre_px": [float(rec["centre_row_px"]), float(rec["centre_col_px"])]})
    fig.subplots_adjust(left=0.05, right=0.995, top=0.93, bottom=0.01)
    save(fig, "fig1_frap_image_sequence", "png", {
        "selection_rule": "QC-passed (profile) measurement with the largest bleach depth K0 per condition",
        "contrast": f"linear, percentiles {IMAGE_CONTRAST_PERCENTILES} of the last pre-bleach frame, identical within a row",
        "overlays": "solid circle: r_e (1/e^2 radius of the first post-bleach profile); +: bleach point (point bleach, r_n = 0)",
        "rows": meta_rows})


# ── Figure 2: profile broadening ───────────────────────────────────────────────

def fig2_profile_broadening(cache: dict, results: pd.DataFrame) -> None:
    rep = _representative(results, "gel")
    prof = cache["details"][rep["measurement_id"]]["profile"]
    fig, (ax_a, ax_b) = new_figure("full", 1, 2, height=2.9)

    window = [f for f in prof["frames"] if f["valid"] and prof["window"][0] <= f["t_s"] <= prof["window"][1]]
    picks = [window[int(round(i))] for i in np.linspace(0, len(window) - 1, PROFILE_PANEL_TIMES)]
    colours = plt.get_cmap("cividis")(np.linspace(0.0, 0.85, len(picks)))
    r = prof["bins_r_um"]
    rr = np.linspace(0, r.max(), 300)
    for f, col in zip(picks, colours):
        ok = np.isfinite(f["ratio"])
        idx = np.flatnonzero(ok)[::PROFILE_PANEL_MARKER_STEP]
        ax_a.plot(r[idx], f["ratio"][idx], "o", ms=2.5, mfc=col, mec="none", alpha=0.6)
        ax_a.plot(rr, f["A"] * (1 - f["K"] * np.exp(-2 * rr ** 2 / f["w_um"] ** 2)), color=col, lw=1.5,
                  label=f"t = {f['t_s']:.1f} s, w = {f['w_um']:.0f} µm")
    ax_a.set_xlabel("Radius r (µm)")
    ax_a.set_ylabel("I$_{post}$(r) / I$_{pre}$(r)")
    ax_a.set_xlim(0, r.max())
    ax_a.legend(loc="lower right", handlelength=1.2)
    _panel_label(ax_a, "A")

    meta = []
    focus = results[results["qc_profile"]]
    for _, row in focus.sort_values("group").iterrows():
        (base, dark), marker, _ = GROUP_STYLE[row["group"]]
        p = cache["details"][row["measurement_id"]]["profile"]
        sel = [f for f in p["frames"] if f["valid"] and p["window"][0] <= f["t_s"] <= p["window"][1]]
        t = np.array([f["t_s"] for f in sel])
        w2 = np.array([f["w_um"] ** 2 for f in sel])
        ax_b.plot(t, w2, marker, ms=3, mfc=base, mec=dark, mew=0.5, alpha=0.6, ls="none")
        tt = np.array([t.min(), t.max()])
        ax_b.plot(tt, p["w2_intercept"] + p["w2_slope"] * tt, color=dark, lw=1.2)
        meta.append({"measurement_id": row["measurement_id"], "window_s": list(p["window"]), "n_frames": len(t),
                     "D_profile_um2_s": float(row["D_profile_um2_s"]), "w2_r2": float(row["w2_r2"])})
    ax_b.set_xlabel("Time t (s)")
    ax_b.set_ylabel("w$^2$ (µm$^2$)")
    ax_b.set_xlim(left=0)
    groups = [g for g in GROUP_STYLE if g in set(focus["group"])]
    ax_b.legend(handles=[Line2D([], [], marker=GROUP_STYLE[g][1], ls="-", color=GROUP_STYLE[g][0][1],
                                mfc=GROUP_STYLE[g][0][0], ms=4, label=f"{GROUP_STYLE[g][2]} (n = {(focus['group'] == g).sum()})")
                         for g in groups], loc="lower right")
    _panel_label(ax_b, "B")
    save(fig, "fig2_profile_broadening", "pdf", {
        "panel_A_measurement": rep["measurement_id"], "panel_A_times_s": [float(f["t_s"]) for f in picks],
        "panel_A_model": "A (1 - K exp(-2 r^2 / w^2)), azimuthal average around the fixed bleach centroid",
        "panel_B_fit": "w^2 = w0^2 + 8 D t, weighted by 1/SE(w^2)^2, frames of the profile window only",
        "panel_B_measurements": meta})


# ── Figure 3: recovery curves ──────────────────────────────────────────────────

def _mean_curve(curves):
    t_end = min(t[-1] for t, _ in curves)
    t_first = min(t[1] for t, _ in curves)
    grid = np.r_[0.0, np.geomspace(t_first, t_end, 80)]
    return grid, np.mean([np.interp(grid, t, f) for t, f in curves], axis=0)


def fig3_recovery(cache: dict, results: pd.DataFrame) -> None:
    fig, (ax, ax_r) = plt.subplots(2, 1, figsize=(WIDTH_IN["full"], 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    linthresh = float(results["frame_interval_s"].median())
    groups = {"water": results[results["condition"] == "water"],
              "gel": results[results["condition"] == "gel"],
              "interface": results[results["condition"] == "gel_interface"]}
    handles, meta = [], {}
    for name, sub in groups.items():
        (base, dark), marker, label = GEL_STYLE if name == "gel" else GROUP_STYLE[name]
        used = sub[sub["qc_global"]]
        curves = []
        for _, row in used.iterrows():
            det = cache["details"][row["measurement_id"]]
            t, f = det["t_s"], det["curve"]["F_norm"]
            ax.plot(t, f, color=base, lw=0.6, alpha=0.5)
            if det["recovery"]["success"]:
                ax_r.plot(t, det["recovery"]["residuals"], color=base, lw=0.6, alpha=0.6)
            curves.append((t, f))
        if curves:
            grid, mean = _mean_curve(curves)
            ax.plot(grid, mean, color=dark, lw=2.0)
            handles.append(Line2D([], [], color=dark, lw=2.0, label=f"{label} (n = {len(curves)})"))
            meta[name] = {"measurements": list(used["measurement_id"]), "mean_grid_end_s": float(grid[-1]),
                          "not_shown_qc_global_failed": list(sub.loc[~sub["qc_global"], "measurement_id"])}
    ctrl_curves = []
    for det in cache["control_details"].values():
        ax.plot(det["t_s"], det["curve"]["F_norm"], color=GROUP_STYLE["nofitc"][0][0], lw=0.6)
        ctrl_curves.append((det["t_s"], det["curve"]["F_norm"]))
    grid, mean = _mean_curve(ctrl_curves)
    ax.plot(grid, mean, color=GROUP_STYLE["nofitc"][0][1], lw=2.0, ls=DASH_THEORY)
    handles.append(Line2D([], [], color=GROUP_STYLE["nofitc"][0][1], lw=2.0, ls=DASH_THEORY,
                          label=f"Gel, no FITC (n = {len(ctrl_curves)})"))
    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.set_ylabel("F$_{norm}$ (double-normalised)")
    ax.legend(handles=handles, loc="lower right")
    ax_r.axhline(0.0, color="black", lw=0.6)
    ax_r.set_ylabel("Residual")
    ax_r.set_xlabel("Time after bleach t (s)")
    _symlog_time(ax_r, linthresh)
    ax_r.set_xlim(0, max(det["t_s"][-1] for det in cache["details"].values()) * 1.1)
    _panel_label(ax, "A")
    _panel_label(ax_r, "B")
    save(fig, "fig3_recovery_curves", "pdf", {
        "time_axis": f"symlog, linear below {linthresh:.3f} s (one frame interval), t = 0 at the first post-bleach frame",
        "normalisation": "Phair/Misteli double normalisation, ROI = disk of radius r_e, reference r >= ref_inner_um",
        "fit_model": "F0 + (Finf - F0)(1 - exp(-t/tau)); residuals = data - fit",
        "group_means": "mean of the single curves interpolated on a common grid up to the shortest series",
        "groups": meta, "nofitc_bleach_site": cache["bleach_site"]})


# ── Figure 4: D and M_f ────────────────────────────────────────────────────────

def _strip(ax, results, value, gate, log=False):
    order = [g for g in ["water", "gel_A1", "gel_B1", "interface"] if g in set(results["group"])]
    meta = {}
    for i, g in enumerate(order):
        (base, dark), marker, _ = GROUP_STYLE[g]
        sub = results[results["group"] == g].dropna(subset=[value])
        ok = sub[gate].astype(bool)
        ax.plot(np.full(ok.sum(), i), sub.loc[ok, value], marker, ms=5, mfc=base, mec=dark, mew=0.6, alpha=0.7, ls="none")
        ax.plot(np.full((~ok).sum(), i), sub.loc[~ok, value], marker, ms=5, mfc="none", mec=dark, mew=0.6, ls="none")
        vals = sub.loc[ok, value].to_numpy(float)
        if len(vals):
            centre = np.exp(np.log(vals).mean()) if log else vals.mean()
            ax.plot([i - 0.25, i + 0.25], [centre, centre], color="black", lw=1.5)
        meta[g] = {"passed": list(sub.loc[ok, "measurement_id"]), "failed": list(sub.loc[~ok, "measurement_id"]),
                   "centre": float(centre) if len(vals) else None}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([GROUP_STYLE[g][2].replace("Gel/glass interface", "Interface") for g in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    return meta


def fig4_d_and_mobile_fraction(cache: dict, results: pd.DataFrame) -> None:
    fig, (ax_a, ax_b) = new_figure("narrow", 1, 2, width_ratios=[1, 1])
    meta_a = _strip(ax_a, results, "D_profile_um2_s", "qc_profile", log=True)
    ax_a.set_yscale("log")
    d_vals = results["D_profile_um2_s"].dropna()
    ax_a.set_ylim(0.8 * d_vals.min(), 3.0 * d_vals.max())     # headroom for the legend above the data
    ax_a.axhline(cache["d_se_um2_s"], color="black", lw=1.2, ls=DASH_THEORY)
    ax_a.legend(handles=[Line2D([], [], color="black", lw=1.2, ls=DASH_THEORY, label="Stokes–Einstein, water"),
                         Line2D([], [], color="black", lw=1.5, label="Geometric mean")], loc="upper right")
    ax_a.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_a.yaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v)), 6) in (2, 5) else ""))
    ax_a.set_ylabel("D$_{profile}$ (µm$^2$ s$^{-1}$)")
    _panel_label(ax_a, "A")
    meta_b = _strip(ax_b, results, "M_f", "qc_recovery")
    ax_b.set_ylabel("Mobile fraction M$_f$")
    _panel_label(ax_b, "B")
    for ax in (ax_a, ax_b):
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "fig4_diffusion_mobile_fraction", "pdf", {
        "panel_A": {"value": "D_profile_um2_s", "gate": "qc_profile", "centre_line": "geometric mean of QC-passed",
                    "reference": f"Stokes-Einstein water value {cache['d_se_um2_s']:.2f} um2/s", "groups": meta_a},
        "panel_B": {"value": "M_f", "gate": "qc_recovery", "centre_line": "arithmetic mean of QC-passed", "groups": meta_b},
        "symbols": "filled: passes the QC gate; open: fails it (not in the mean)"})


# ── Supplementary figures ──────────────────────────────────────────────────────

def figS1_all_measurements(cache: dict, results: pd.DataFrame) -> None:
    rows = results.sort_values(["group", "measurement_id"]).reset_index(drop=True)
    n_rows = int(np.ceil(len(rows) / 2))
    fig, axes = plt.subplots(n_rows, 4, figsize=(WIDTH_IN["full"], 1.15 * n_rows + 0.3), layout="none")
    meta = []
    for k, row in rows.iterrows():
        ax_rec, ax_w = axes[k // 2, 2 * (k % 2)], axes[k // 2, 2 * (k % 2) + 1]
        (base, dark), marker, _ = GROUP_STYLE[row["group"]]
        det = cache["details"][row["measurement_id"]]
        t, f = det["t_s"], det["curve"]["F_norm"] if det["curve"] else None
        if f is not None:
            ax_rec.plot(t, f, ".", ms=1.5, color=base)
            rec = det["recovery"]
            if rec["success"]:
                ax_rec.plot(t, _recovery_model(t, rec["F0"], rec["Finf"], rec["tau_s"]), color=dark, lw=1.0)
        p = det["profile"]
        frames = [fr for fr in p.get("frames", []) if fr.get("success")]
        tw = np.array([fr["t_s"] for fr in frames])
        w2 = np.array([fr["w_um"] ** 2 for fr in frames])
        inwin = np.array([fr["valid"] and p["window"][0] <= fr["t_s"] <= p["window"][1] for fr in frames]) if frames else np.array([])
        if len(tw):
            ax_w.plot(tw[~inwin], w2[~inwin], ".", ms=1.5, color="#bbbbbb")
            ax_w.plot(tw[inwin], w2[inwin], ".", ms=2, color=base)
            if np.isfinite(row["D_profile_um2_s"]):
                tt = np.array([p["window"][0], p["window"][1]])
                ax_w.plot(tt, p["w2_intercept"] + p["w2_slope"] * tt, color=dark, lw=1.0)
            ax_w.set_xlim(0, 2.5 * max(p["window"][1], 5) if np.isfinite(p["window"][1]) else None)
            ax_w.set_ylim(0, 1.3 * w2[inwin].max() if inwin.any() else None)
        for ax, gate in ((ax_rec, "qc_recovery"), (ax_w, "qc_profile")):
            if not row[gate]:
                ax.set_facecolor(QC_FAIL_SHADE)
            ax.tick_params(labelsize=6, length=2)
        ax_rec.set_title(f"{row['measurement_id']}" + ("" if row["qc_recovery"] else " (QC fail)"),
                         fontsize=7, fontweight="normal", pad=2, loc="left")
        d_text = f"D = {row['D_profile_um2_s']:.1f}" if np.isfinite(row["D_profile_um2_s"]) else "no D"
        ax_w.set_title(d_text + ("" if row["qc_profile"] else " (QC fail)"), fontsize=7, fontweight="normal", pad=2, loc="left")
        meta.append({"measurement_id": row["measurement_id"], "qc_reasons": row["qc_reasons"]})
    for k in range(len(rows), 2 * n_rows):
        axes[k // 2, 2 * (k % 2)].axis("off")
        axes[k // 2, 2 * (k % 2) + 1].axis("off")
    fig.supxlabel("Time t (s)", fontsize=9)
    fig.text(0.01, 0.5, "F$_{norm}$ (columns 1, 3)  /  w$^2$ in µm$^2$ (columns 2, 4)", rotation=90, va="center", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.05, wspace=0.35, hspace=0.75)
    save(fig, "figS1_all_measurements", "pdf", {
        "layout": "per measurement: recovery (data + fit) and w^2(t) (window frames coloured, other fitted frames grey, "
                  "weighted linear fit); shaded panel = QC gate of that quantity failed", "measurements": meta})


def figS2_method_comparison(cache: dict, results: pd.DataFrame) -> None:
    fig, ax = new_figure("narrow")
    methods = [("D_soumpasis_um2_s", "Soumpasis", "o", ("#0000da", "#000099")),
               ("D_axelrod_um2_s", "Axelrod", "s", ("#00c4ff", "#0089b2")),
               ("D_kang_um2_s", "Kang", "^", ("#ff6800", "#b24900"))]
    sub = results.dropna(subset=["D_profile_um2_s"])
    for col, label, marker, (base, dark) in methods:
        ok = (sub["qc_profile"] & sub["qc_recovery"]).to_numpy()
        ax.plot(sub.loc[ok, "D_profile_um2_s"], sub.loc[ok, col], marker, ms=4, mfc=base, mec=dark, mew=0.6,
                alpha=0.7, ls="none", label=label)
        ax.plot(sub.loc[~ok, "D_profile_um2_s"], sub.loc[~ok, col], marker, ms=4, mfc="none", mec=dark, mew=0.6, ls="none")
    lim = [5, 100]
    ax.plot(lim, lim, color="black", lw=1.2, ls=DASH_THEORY, label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v)), 6) in (2, 5) else ""))
    ax.set_xlabel("D$_{profile}$ (µm$^2$ s$^{-1}$)")
    ax.set_ylabel("D from recovery curve (µm$^2$ s$^{-1}$)")
    ax.legend(loc="upper left")
    save(fig, "figS2_method_comparison", "pdf", {
        "methods": {"Soumpasis": "r_e^2 / (4 tau)", "Axelrod": "0.88 r_e^2 / (4 t_half)", "Kang": "(r_e^2 + r_n^2) / (8 t_half), r_n = 0"},
        "symbols": "filled: passes profile and recovery QC; open: fails one of them",
        "measurements": list(sub["measurement_id"])})


def figS3_water_diagnostics(cache: dict, results: pd.DataFrame) -> None:
    """Per water measurement: recovery + fit, propagation fit of the radial profiles, D vs. fit window,
    drift of the depletion centroid."""
    water = results[results["condition"] == "water"].sort_values("measurement_id")
    (base, dark), _, _ = GROUP_STYLE["water"]
    d_se = cache["d_se_um2_s"]
    fig, axes = plt.subplots(len(water), 4, figsize=(WIDTH_IN["full"], 1.75 * len(water) + 0.35), squeeze=False)
    meta = []
    for r, (_, row) in enumerate(water.iterrows()):
        det = cache["details"][row["measurement_id"]]
        prop = det["propagation"]
        ax_rec, ax_prof, ax_scan, ax_drift = axes[r]
        note = "" if row["qc_global"] else " (excluded: saturated)" if row["saturated_fraction"] > 0.01 else " (excluded)"
        title = f"{row['measurement_id']}, bleach {row['bleach_laser_pct']:.0f} %{note}"

        t, f = det["t_s"], det["curve"]["F_norm"]
        ax_rec.plot(t, f, ".", ms=2, color=base)
        rec = det["recovery"]
        if rec["success"]:
            ax_rec.plot(t, _recovery_model(t, rec["F0"], rec["Finf"], rec["tau_s"]), color=dark, lw=1.2)
            ax_rec.text(0.97, 0.05, f"t$_{{1/2}}$ = {rec['t_half_s']:.1f} s", transform=ax_rec.transAxes,
                        ha="right", va="bottom", fontsize=7)
        ax_rec.set_ylabel("F$_{norm}$", fontsize=8)
        ax_rec.set_title(title, fontsize=8, fontweight="normal", loc="left", pad=3)

        fits = prop.get("fit_profiles", [])
        if fits:
            picks = [fits[int(round(i))] for i in np.linspace(0, len(fits) - 1, min(4, len(fits)))]
            colours = plt.get_cmap("cividis")(np.linspace(0, 0.85, len(picks)))
            for fp, col in zip(picks, colours):
                step = max(1, len(fp["r_um"]) // 40)
                ax_prof.plot(fp["r_um"][::step], fp["data"][::step], "o", ms=1.8, mfc=col, mec="none", alpha=0.7)
                ax_prof.plot(fp["r_um"], fp["model"] * np.median(fp["data"][-20:]) / np.median(fp["model"][-20:]),
                             color=col, lw=1.0, label=f"{fp['t_s']:.1f} s")
            ax_prof.legend(fontsize=6, loc="lower right", handlelength=1.0, labelspacing=0.2)
        ax_prof.set_ylabel("I$_{post}$/I$_{pre}$", fontsize=8)

        scan = {float(k): v for k, v in prop.get("scan", {}).items() if np.isfinite(v)}
        if scan:
            ax_scan.plot(list(scan), list(scan.values()), "o-", ms=3, lw=1.0, color=dark, mfc=base)
        ax_scan.axhline(d_se, color="black", lw=1.0, ls=DASH_THEORY)
        if np.isfinite(row["D_propagation_um2_s"]):
            ax_scan.axvline(row["propagation_window_s"], color="#777777", lw=0.8, ls=":")
            ax_scan.plot(row["propagation_window_s"], row["D_propagation_um2_s"], "D", ms=5, mfc=dark, mec="black")
            ax_scan.annotate(f"D = {row['D_propagation_um2_s']:.1f}", (row["propagation_window_s"], row["D_propagation_um2_s"]),
                             xytext=(5, -6), textcoords="offset points", ha="left", va="top", fontsize=7)
        ax_scan.set_xscale("log")
        ax_scan.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_scan.xaxis.set_minor_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v)), 6) in (2, 5) else ""))
        ax_scan.set_ylabel("D (µm$^2$ s$^{-1}$)", fontsize=8)

        track = np.array(prop.get("centroid_track_um", []))
        if len(track):
            ax_drift.plot(track[:, 0], track[:, 2] - track[0, 2], "-", color=dark, lw=1.0, label="x")
            ax_drift.plot(track[:, 0], track[:, 1] - track[0, 1], "--", color=base, lw=1.0, label="y")
            if np.isfinite(row["propagation_window_s"]):
                ax_drift.axvspan(0, row["propagation_window_s"], color=WINDOW_SHADE_LIGHT, lw=0, zorder=0)
            ax_drift.legend(fontsize=6, loc="best", handlelength=1.2)
        ax_drift.set_ylabel("Centroid shift (µm)", fontsize=8)
        for ax in axes[r]:
            ax.tick_params(labelsize=7)
        meta.append({"measurement_id": row["measurement_id"], "D_propagation_um2_s": row["D_propagation_um2_s"],
                     "propagation_window_s": row["propagation_window_s"], "max_drift_um": row["max_drift_um"],
                     "D_vs_window": scan, "D_profile_gaussian_width_um2_s": row["D_profile_um2_s"],
                     "qc_reasons": row["qc_reasons"]})
    for ax, label in zip(axes[-1], ["Time t (s)", "Radius r (µm)", "Fit window t$_{max}$ (s)", "Time t (s)"]):
        ax.set_xlabel(label, fontsize=8)
    for ax, letter in zip(axes[0], "ABCD"):
        _panel_label(ax, letter)
    save(fig, "figS3_water_diagnostics", "pdf", {
        "columns": ["A: double-normalised recovery with single-exponential fit",
                    "B: radial profiles around the per-frame centroid (points) and the diffusion propagation of the "
                    "first post-bleach profile with the fitted D (lines), four times of the fit window",
                    "C: propagation D as a function of the fit-window end; diamond = window from the area-growth rule; "
                    f"dashed = Stokes-Einstein {d_se:.1f} um2/s",
                    "D: displacement of the depletion centroid (x solid, y dashed); shaded = fit window"],
        "measurements": meta})


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"{CACHE_PATH} not found; run frap_analysis_v2.py first.")
    with open(CACHE_PATH, "rb") as fh:
        cache = pickle.load(fh)
    results = _focus(cache["results"])
    with plt.style.context(str(STYLE_PATH)):
        fig1_image_sequence(cache, results)
        fig2_profile_broadening(cache, results)
        fig3_recovery(cache, results)
        fig4_d_and_mobile_fraction(cache, results)
        figS1_all_measurements(cache, results)
        figS2_method_comparison(cache, results)
        figS3_water_diagnostics(cache, results)
    print(f"Figures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
