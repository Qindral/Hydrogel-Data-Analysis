"""
Figures for the oscillatory rheology of the 20 mg/mL hydrogel (mechanical
stability over gel age; all runs from one hydrogel batch, cast on two
dates). Consumer stage: reads only
cache/rheo_hyd20_results.pkl written by Rheo_Hyd20_Compute.py (run that
first); raw .tri files are never touched and nothing is refitted here.

Figures (OUTPUT_ROOT/figures/...):
  frequency_sweeps/   fs_<run>        G'(f) filled, G''(f) open, selected
                                      weakly frequency-dependent window shaded
                                      with its log-log fit, excluded points grey,
                                      0.5 Hz marked;
                      fs_<run>_tandelta  tan(delta)(f)
  amplitude_sweeps/   as_<run>        G'(gamma), G''(gamma) with the low-strain
                                      reference, 5 %/10 % deviation strains and
                                      the G'/G'' crossover
  stability/          stability_a_frequency_sweeps   G' (left) and G'' (right) of
                                      every analysable sweep; colour and marker
                                      per age, line style per cast
                      stability_b_normalised_modulus G'_rep / mean G'_rep at the
                                      youngest age
                      stability_c_absolute_modulus   G'_rep vs. age
                      stability_d_viscoelastic_mean  mean behaviour per age:
                                      mean G'(f), G''(f) and tan(delta)(f) +- SD
                                      (table mean_frequency_sweeps), and G', G'',
                                      tan(delta) at 0.5 Hz vs. age (mean +- SD, N)
  mesh_size/          mesh_size_vs_age   network mesh size xi = (g k_B T / G')^(1/3)
                                      per specimen and per age, pooled mean over
                                      all ages (affine, and phantom f = 4 as reference)
                      pore_size_phantom_network  diamond cell (f = 4) to scale with
                                      cavity sphere; cavity and throat diameter vs.
                                      strand diameter for f = 4 and f = 6
  validation/         tri_vs_xls_parity   reconstructed vs. TRIOS-exported moduli

In B and C both casts share one age axis, because they come from the same
hydrogel batch. Every run is shown at its real age (no jitter); colour
encodes the cast, marker shape the measuring gap, and runs excluded from the
qc_filtered statistics are open symbols. Thin lines connect only the same
specimen (cast + position) measured at different ages; the mean +- SD over
specimens of the same age (FIG_SELECTION, casts pooled) is drawn as a
black-edged diamond and is not connected across ages.

Style: Styleguide_Figures_Dissertation.md v2 via hydro_analysis/thesis.mplstyle,
final printed size (width classes half / full), saved without
bbox_inches="tight". File formats as requested for this analysis: PDF and SVG
(vector) plus PNG at 600 dpi (FIGURE_FORMATS).
"""
from __future__ import annotations

import pickle
import re
import time
from itertools import product
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY
from hydro_analysis.Rheology.network_geometry import lattice_segments

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "rheo_hyd20_results.pkl"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
OUTPUT_ROOT: Path | None = None           # None: OUTPUT_ROOT stored in the cache
FIGURE_FORMATS = ("pdf", "svg", "png")
FIG_SELECTION = "qc_filtered"             # statistics shown as mean +- SD ("all_analysable" / "qc_filtered")
SHOW_FIGURES = False
SAVE_RETRIES = 5

WIDTH_IN = {"full": 6.30, "narrow": 4.72, "half": 3.07, "third": 2.01}
RATIO = 1.42

COLOR_SINGLE = ("#3B6E8C", "#2A4F66")                  # style guide §11, single series (base, dark)
COLOR_CAST = [("#3B8C8C", "#2A6666"), ("#D98C3D", "#A6672D")]    # style guide category A / B
CAST_LINESTYLES = ["-", (0, (4, 2))]
AGE_PALETTE = [("#0000da", "#000099"), ("#00c4ff", "#0089b2"), ("#ff6800", "#b24900"),
               ("#da0000", "#990000"), ("#004cff", "#0035b2"), ("#ffd700", "#b29600")]
AGE_MARKERS = ["o", "s", "^", "D", "v", "P"]
GAP_MARKERS = {116: "o", 250: "s", 500: "^"}           # rounded gap (um) -> marker
COLOR_EXCLUDED = "#9a9a9a"
COLOR_REFERENCE = "#555555"
WINDOW_SHADE = "#d9d9d9"
POINT_ALPHA = 0.7
EXCLUDED_ALPHA = 0.35              # sweeps excluded from qc_filtered statistics (figure A)
PLAIN_LABEL_MAX_DECADES = 1.6     # log axes spanning fewer decades get plain-number labels


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_name(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def new_figure(kind: str = "half", nrows: int = 1, ncols: int = 1, height: float | None = None, **kw):
    width = WIDTH_IN[kind]
    return plt.subplots(nrows, ncols, figsize=(width, height if height else width / RATIO), **kw)


def save(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for fmt in FIGURE_FORMATS:
        # Retries: another process (Explorer thumbnails, virus scanner) can briefly lock an existing file.
        for attempt in range(SAVE_RETRIES):
            try:
                fig.savefig(folder / f"{stem}.{fmt}", format=fmt, dpi=600)
                break
            except OSError:
                if attempt == SAVE_RETRIES - 1:
                    raise
                time.sleep(1.0)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)


def _gap_marker(gap_um: float) -> str:
    if not np.isfinite(gap_um):
        return "X"
    key = min(GAP_MARKERS, key=lambda g: abs(g - gap_um))
    return GAP_MARKERS[key] if abs(key - gap_um) < 20 else "X"


def _log_axes(ax: plt.Axes, logx: bool = True, logy: bool = True) -> None:
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    for axis, is_log, lim in ((ax.xaxis, logx, ax.get_xlim()), (ax.yaxis, logy, ax.get_ylim())):
        if not is_log:
            continue
        axis.set_minor_locator(mticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=100))
        if np.log10(max(lim) / min(lim)) < PLAIN_LABEL_MAX_DECADES:
            # Narrow log range: label 1, 2 and 5 x 10^n as plain numbers so the values stay readable.
            plain = mticker.FuncFormatter(lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v)), 6) in (1, 2, 5) else "")
            axis.set_major_formatter(plain)
            axis.set_minor_formatter(plain)
        else:
            axis.set_major_formatter(mticker.LogFormatterSciNotation())
            axis.set_minor_formatter(mticker.NullFormatter())


# ── Per-run figures ────────────────────────────────────────────────────────────

def plot_frequency_sweep(run: str, data: pd.DataFrame, res: dict, target_Hz: float, folder: Path) -> None:
    f, gp, gpp = data["frequency_Hz"], data["Gp_Pa"], data["Gpp_Pa"]
    valid = data["point_flag"] == ""
    pos = (gp > 0) & (gpp > 0)
    base, dark = COLOR_SINGLE

    fig, ax = new_figure("half")
    w = res["window"]
    if w.get("found"):
        ax.axvspan(w["f_min_Hz"], w["f_max_Hz"], color=WINDOW_SHADE, lw=0, zorder=0)
    ax.axvline(target_Hz, color=COLOR_REFERENCE, lw=0.8, ls=":", zorder=1)
    ax.plot(f[valid & pos], gp[valid & pos], "o", ms=4, mfc=base, mec=dark, zorder=3, label="G'")
    ax.plot(f[valid & pos], gpp[valid & pos], "s", ms=4, mfc="white", mec=dark, zorder=3, label="G''")
    if (~valid & pos).any():
        ax.plot(f[~valid & pos], gp[~valid & pos], "o", ms=4, mfc=COLOR_EXCLUDED, mec=COLOR_EXCLUDED, zorder=2,
                label="excluded")
        ax.plot(f[~valid & pos], gpp[~valid & pos], "s", ms=4, mfc="white", mec=COLOR_EXCLUDED, zorder=2)
    if w.get("found"):
        xf = np.logspace(np.log10(w["f_min_Hz"]), np.log10(w["f_max_Hz"]), 50)
        ax.plot(xf, 10 ** (w["intercept"] + w["slope"] * np.log10(xf)), color="black", lw=1.5, zorder=4,
                label=f"fit, m = {w['slope']:.3f}")
    _log_axes(ax)
    ax.set_xlabel("Frequency f (Hz)")
    ax.set_ylabel("Modulus G', G'' (Pa)")
    ax.legend(loc="center right", handlelength=1.5)
    save(fig, folder, f"fs_{_safe_name(run)}")

    fig, ax = new_figure("half")
    tan = data["tan_delta"]
    ax.plot(f[valid & pos], tan[valid & pos], "o-", ms=4, lw=1.0, color=dark, mfc=base, mec=dark, zorder=3)
    if (~valid & pos).any():
        ax.plot(f[~valid & pos], tan[~valid & pos], "o", ms=4, mfc=COLOR_EXCLUDED, mec=COLOR_EXCLUDED, zorder=2)
    if w.get("found"):
        ax.axvspan(w["f_min_Hz"], w["f_max_Hz"], color=WINDOW_SHADE, lw=0, zorder=0)
    _log_axes(ax, logy=False)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Frequency f (Hz)")
    ax.set_ylabel("Loss factor tan δ")
    save(fig, folder, f"fs_{_safe_name(run)}_tandelta")


def plot_amplitude_sweep(run: str, data: pd.DataFrame, res: dict, folder: Path) -> None:
    x, gp, gpp = data["strain_pct"], data["Gp_Pa"], data["Gpp_Pa"]
    valid = (data["point_flag"] == "") & (gp > 0) & (gpp > 0)
    base, dark = COLOR_SINGLE
    fig, ax = new_figure("half")
    ax.plot(x[valid], gp[valid], "o", ms=4, mfc=base, mec=dark, zorder=3, label="G'")
    ax.plot(x[valid], gpp[valid], "s", ms=4, mfc="white", mec=dark, zorder=3, label="G''")
    if "Gp_ref_Pa" in res:
        ax.axhline(res["Gp_ref_Pa"], color=COLOR_REFERENCE, lw=0.8, zorder=1)
        for tag, ls in (("5pct", ":"), ("10pct", _DASH_THEORY)):
            gamma = res.get(f"strain_{tag}_pct", np.nan)
            if np.isfinite(gamma):
                ax.axvline(gamma, color="black", lw=1.0, ls=ls, zorder=2,
                           label=f"γ$_{{{tag[:-3]}\\%}}$ = {gamma:.3g} %")
        if res.get("crossover_status") == "found":
            ax.plot(res["crossover_strain_pct"], res["crossover_modulus_Pa"], "*", ms=8, mfc="black", mec="black",
                    zorder=4, label=f"G' = G'' at {res['crossover_strain_pct']:.3g} %")
    _log_axes(ax)
    ax.set_xlabel("Strain amplitude γ (%)")
    ax.set_ylabel("Modulus G', G'' (Pa)")
    ax.legend(loc="lower left", handlelength=1.5)
    save(fig, folder, f"as_{_safe_name(run)}")


# ── Stability figures ──────────────────────────────────────────────────────────

def _age_styles(ages) -> Dict[float, tuple]:
    return {age: (AGE_PALETTE[i % len(AGE_PALETTE)], AGE_MARKERS[i % len(AGE_MARKERS)])
            for i, age in enumerate(sorted(ages))}


def _cast_styles(casts) -> Dict[str, tuple]:
    """Cast -> ((base, dark), line style); colour and line style both encode the cast."""
    return {cast: (COLOR_CAST[i % len(COLOR_CAST)], CAST_LINESTYLES[i % len(CAST_LINESTYLES)])
            for i, cast in enumerate(sorted(casts))}


def _cast_label(cast: str) -> str:
    return cast.replace("cast ", "Cast ")


def plot_stability_sweeps(cache: dict, folder: Path) -> None:
    """Panel A: G'(f), panel B: G''(f) of every analysable sweep; colour + marker = age, line style = cast."""
    fs = cache["tables"]["frequency_sweep_summary"]
    fs = fs[(fs["qc_status"] != "FAIL") & fs["Gp_rep_Pa"].notna()]
    ages = _age_styles(fs["age_d"].unique())
    casts = _cast_styles(fs["cast"].unique())
    fig, axes = new_figure("full", 1, 2, height=3.0, sharex=True)
    for _, row in fs.sort_values(["age_d", "run"]).iterrows():
        data = cache["results"][row["run"]]["data"]
        valid = (data["point_flag"] == "") & (data["Gp_Pa"] > 0) & (data["Gpp_Pa"] > 0)
        (base, dark), marker = ages[row["age_d"]]
        ls = casts[row["cast"]][1]
        alpha = EXCLUDED_ALPHA if row["excluded_by_qc_codes"] else 1.0
        for ax, col in zip(axes, ("Gp_Pa", "Gpp_Pa")):
            ax.plot(data["frequency_Hz"][valid], data[col][valid], marker=marker, ms=3.5, lw=1.0, ls=ls,
                    color=dark, mfc=base, mec=dark, alpha=alpha)
    for ax, label in zip(axes, ("Storage modulus G' (Pa)", "Loss modulus G'' (Pa)")):
        _log_axes(ax)
        ax.set_xlabel("Frequency f (Hz)")
        ax.set_ylabel(label)
    handles = [Line2D([], [], marker=ages[a][1], color=ages[a][0][1], mfc=ages[a][0][0], ms=4, lw=1.0,
                      label=f"{a:g} d") for a in sorted(ages)]
    handles += [Line2D([], [], color="#444444", lw=1.0, ls=casts[c][1], label=_cast_label(c)) for c in sorted(casts)]
    handles.append(Line2D([], [], marker="o", color="#444444", mfc="#888888", ms=4, lw=1.0, alpha=EXCLUDED_ALPHA,
                          label="excluded (QC)"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    save(fig, folder, "stability_a_frequency_sweeps")


def plot_stability_summary(cache: dict, folder: Path, normalised: bool) -> None:
    """G'_rep vs. age, casts on one axis; group mean +- SD pooled over casts (FIG_SELECTION)."""
    fs = cache["tables"]["frequency_sweep_summary"]
    fs = fs[(fs["qc_status"] != "FAIL") & fs["Gp_rep_Pa"].notna()]
    stab = cache["tables"]["stability_summary"]
    stab = stab[(stab["selection"] == FIG_SELECTION) & stab["Gp_rep_Pa_mean"].notna()].sort_values("age_d")
    norm = stab["Gp_rep_Pa_mean"].iloc[0] if normalised and len(stab) else 1.0
    casts = _cast_styles(fs["cast"].unique())

    fig, ax = new_figure("full", height=3.0)
    spec = cache["tables"]["stability_specimen_level"]
    spec = spec[(spec["selection"] == FIG_SELECTION) & (spec["sweep_type"] == "frequency_sweep")]
    for specimen, group in spec.groupby("specimen_id"):
        if group["age_d"].nunique() > 1:
            g = group.sort_values("age_d")
            (_, dark), ls = casts[g["cast"].iloc[0]]
            ax.plot(g["age_d"], g["Gp_rep_Pa"] / norm, color=dark, lw=0.8, ls=ls, alpha=0.6, zorder=1)
    for _, row in fs.iterrows():
        (base, dark), _ = casts[row["cast"]]
        excluded = bool(row["excluded_by_qc_codes"])
        ax.plot(row["age_d"], row["Gp_rep_Pa"] / norm, _gap_marker(row["gap_um"]), ms=4,
                mfc="white" if excluded else base, mec=dark, alpha=POINT_ALPHA, zorder=3)
    ax.errorbar(stab["age_d"], stab["Gp_rep_Pa_mean"] / norm, yerr=stab["Gp_rep_Pa_sd"] / norm, fmt="D", ms=5,
                mfc="#444444", mec="black", mew=0.8, ecolor="black", elinewidth=0.8, capsize=2.0, capthick=0.8,
                zorder=4)
    if normalised:
        ax.axhline(1.0, color=COLOR_REFERENCE, lw=0.8, ls=_DASH_THEORY, zorder=0)
    ax.set_xlim(0, fs["age_d"].max() + 2)
    _log_axes(ax, logx=False, logy=True)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.set_xlabel("Gel age t (d)")
    ax.set_ylabel(f"G'(t) / G'({stab['age_d'].iloc[0]:g} d)" if normalised else "Storage modulus G' (Pa)")

    handles = [Line2D([], [], marker="o", ls="", mfc=casts[c][0][0], mec=casts[c][0][1], ms=4, label=_cast_label(c))
               for c in sorted(casts)]
    gaps = sorted({min(GAP_MARKERS, key=lambda g: abs(g - x)) for x in fs["gap_um"].dropna()})
    handles += [Line2D([], [], marker=GAP_MARKERS[g], ls="", mfc="#bbbbbb", mec="#444444", ms=4, label=f"gap {g} µm")
                for g in gaps]
    handles += [Line2D([], [], marker="o", ls="", mfc="white", mec="#444444", ms=4, label="excluded (QC)"),
                Line2D([], [], color="#444444", lw=0.8, label="same specimen"),
                Line2D([], [], marker="D", ls="", mfc="#444444", mec="black", ms=5, label="mean ± SD")]
    fig.legend(handles=handles, loc="outside right upper")
    save(fig, folder, "stability_b_normalised_modulus" if normalised else "stability_c_absolute_modulus")


def plot_viscoelastic_mean(cache: dict, folder: Path) -> None:
    """Mean viscoelastic behaviour per gel age (FIG_SELECTION, casts pooled, N = specimens).

    A  mean G'(f) (filled) and G''(f) (open) per age, SD band where N >= 2
    B  mean tan(delta)(f) per age, SD band where N >= 2
    C  G' (representative value at 0.5 Hz) and G'' (interpolated at 0.5 Hz) vs. age, mean +- SD
    D  tan(delta) at 0.5 Hz vs. age, mean +- SD
    """
    curves = cache["tables"]["mean_frequency_sweeps"]
    curves = curves[curves["selection"] == FIG_SELECTION]
    stab = cache["tables"]["stability_summary"]
    stab = stab[(stab["selection"] == FIG_SELECTION) & (stab["fs_N_specimens"] > 0)].sort_values("age_d")
    ages = _age_styles(curves["age_d"].unique())
    base, dark = COLOR_SINGLE

    fig, axes = new_figure("full", 2, 2, height=4.9)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    for age, group in curves.groupby("age_d"):
        g = group.sort_values("frequency_Hz")
        (a_base, a_dark), marker = ages[age]
        band = g["N_specimens"] >= 2
        for col, mfc, ls in (("Gp_Pa", a_base, "-"), ("Gpp_Pa", "white", ":")):
            ax_a.plot(g["frequency_Hz"], g[f"{col}_mean"], marker=marker, ms=3.5, lw=1.0, ls=ls,
                      color=a_dark, mfc=mfc, mec=a_dark, zorder=3)
            ax_a.fill_between(g["frequency_Hz"], (g[f"{col}_mean"] - g[f"{col}_sd"]).where(band),
                              (g[f"{col}_mean"] + g[f"{col}_sd"]).where(band), color=a_base, alpha=0.2, lw=0, zorder=1)
        ax_b.plot(g["frequency_Hz"], g["tan_delta_mean"], marker=marker, ms=3.5, lw=1.0, color=a_dark,
                  mfc=a_base, mec=a_dark, zorder=3)
        ax_b.fill_between(g["frequency_Hz"], (g["tan_delta_mean"] - g["tan_delta_sd"]).where(band),
                          (g["tan_delta_mean"] + g["tan_delta_sd"]).where(band), color=a_base, alpha=0.2, lw=0, zorder=1)
    _log_axes(ax_a)
    ax_a.set_xlabel("Frequency f (Hz)")
    ax_a.set_ylabel("Modulus G', G'' (Pa)")
    _log_axes(ax_b, logy=False)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel("Frequency f (Hz)")
    ax_b.set_ylabel("Loss factor tan δ")

    errkw = dict(ms=5, elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=3)
    ax_c.errorbar(stab["age_d"], stab["Gp_rep_Pa_mean"], yerr=stab["Gp_rep_Pa_sd"], fmt="o", mfc=base, mec=dark,
                  ecolor=dark, label="G'", **errkw)
    ax_c.errorbar(stab["age_d"], stab["Gpp_0p5Hz_interp_Pa_mean"], yerr=stab["Gpp_0p5Hz_interp_Pa_sd"], fmt="s",
                  mfc="white", mec=dark, ecolor=dark, label="G''", **errkw)
    ax_d.errorbar(stab["age_d"], stab["tan_delta_0p5Hz_interp_mean"], yerr=stab["tan_delta_0p5Hz_interp_sd"],
                  fmt="o", mfc=base, mec=dark, ecolor=dark, **errkw)
    # Number of specimens per age (same for C and D), written once along the bottom of D.
    for _, row in stab.iterrows():
        ax_d.text(row["age_d"], 0.04, f"N = {row['fs_N_specimens']:.0f}", transform=ax_d.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=8)
    for ax in (ax_c, ax_d):
        ax.set_xlim(0, curves["age_d"].max() + 2)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.set_xlabel("Gel age t (d)")
    _log_axes(ax_c, logx=False, logy=True)
    ax_c.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax_c.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax_c.set_ylabel(f"Modulus at {cache['config']['target_frequency_Hz']:g} Hz (Pa)")
    ax_c.legend(loc="center right")
    ax_d.set_ylim(0, 1.3 * np.nanmax(stab["tan_delta_0p5Hz_interp_mean"] + stab["tan_delta_0p5Hz_interp_sd"].fillna(0)))
    ax_d.set_ylabel(f"tan δ at {cache['config']['target_frequency_Hz']:g} Hz")

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        ax.text(-0.16, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    handles = [Line2D([], [], marker=ages[a][1], color=ages[a][0][1], mfc=ages[a][0][0], ms=4, lw=1.0,
                      label=f"{a:g} d") for a in sorted(ages)]
    handles += [Line2D([], [], marker="o", color="#444444", mfc="#888888", ms=4, lw=1.0, label="G' (filled)"),
                Line2D([], [], marker="o", color="#444444", mfc="white", ms=4, lw=1.0, ls=":", label="G'' (open)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    save(fig, folder, "stability_d_viscoelastic_mean")


def plot_mesh_size(cache: dict, folder: Path) -> None:
    """Network mesh size xi (affine network) vs. gel age; pooled mean over all ages as band.

    Specimen values coloured by cast, mean +- SD per age as black-edged diamonds, pooled
    mean +- SD over all specimens as solid line and grey band, the phantom-network (f = 4)
    pooled mean as dashed reference line.
    """
    mesh = cache["tables"]["mesh_size"]
    mesh = mesh[mesh["selection"] == FIG_SELECTION]
    spec = mesh[mesh["level"] == "specimen"]
    ages = mesh[mesh["level"] == "age"].sort_values("age_d")
    pooled = mesh[mesh["level"] == "pooled_all_ages"].iloc[0]
    casts = _cast_styles(spec["cast"].unique())
    dark = COLOR_SINGLE[1]

    fig, ax = new_figure("narrow")
    x_max = spec["age_d"].max() + 2
    mean, sd = pooled["xi_affine_nm_mean"], pooled["xi_affine_nm_sd"]
    ax.axhspan(mean - sd, mean + sd, color=WINDOW_SHADE, lw=0, zorder=0)
    ax.axhline(mean, color=dark, lw=1.5, zorder=1,
               label=f"all ages, affine: {mean:.1f} ± {sd:.1f} nm (N = {pooled['N_specimens']:.0f})")
    ax.axhline(pooled["xi_phantom_f4_nm_mean"], color="black", lw=1.2, ls=_DASH_THEORY, zorder=1,
               label=f"all ages, phantom (f = 4): {pooled['xi_phantom_f4_nm_mean']:.1f} nm")
    for _, row in spec.iterrows():
        (c_base, c_dark), _ = casts[row["cast"]]
        ax.plot(row["age_d"], row["xi_affine_nm_mean"], "o", ms=4, mfc=c_base, mec=c_dark, alpha=POINT_ALPHA, zorder=3)
    ax.errorbar(ages["age_d"], ages["xi_affine_nm_mean"], yerr=ages["xi_affine_nm_sd"], fmt="D", ms=5,
                mfc="#444444", mec="black", mew=0.8, ecolor="black", elinewidth=0.8, capsize=2.0, capthick=0.8,
                zorder=4, label="mean ± SD per age")
    for _, row in ages.iterrows():
        ax.text(row["age_d"], 0.04, f"N = {row['N_specimens']:.0f}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.5 * np.nanmax(spec["xi_affine_nm_mean"]))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.set_xlabel("Gel age t (d)")
    ax.set_ylabel("Mesh size ξ (nm)")
    handles = ax.get_legend_handles_labels()[0]
    handles += [Line2D([], [], marker="o", ls="", mfc=casts[c][0][0], mec=casts[c][0][1], ms=4, label=_cast_label(c))
                for c in sorted(casts)]
    ax.legend(handles=handles, loc="upper left", ncol=1)
    save(fig, folder, "mesh_size_vs_age")


def _clip_segment_to_unit_cube(p0: np.ndarray, p1: np.ndarray):
    """Part of the segment p0-p1 inside [0, 1]^3 (Liang-Barsky), or None."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for i in range(3):
        for p, q in ((-d[i], p0[i]), (d[i], 1.0 - p0[i])):
            if abs(p) < 1e-12:
                if q < -1e-12:
                    return None
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t1 - t0 < 1e-9:
        return None
    return np.array([p0 + t0 * d, p0 + t1 * d])


def plot_pore_size(cache: dict, folder: Path) -> None:
    """Pore geometry of the phantom network (pooled over all ages, FIG_SELECTION).

    A  diamond-lattice cell (f = 4) to scale for the pooled mean G': strands (d_f = 0),
       junctions, and the cavity sphere
    B  cavity and throat diameter vs. strand diameter d_f, diamond (f = 4, solid) and simple
       cubic (f = 6, dashed) lattice, mean +- SD over specimens as band
    """
    pores = cache["tables"]["pore_size_phantom_network"]
    pooled = pores[(pores["level"] == "pooled_all_ages") & (pores["selection"] == FIG_SELECTION)]
    geo = cache["lattice_geometry"][4]
    ref = pooled[(pooled["functionality"] == 4) & (pooled["strand_diameter_nm"] == 0)].iloc[0]
    a = ref["cell_edge_nm_mean"]

    fig = plt.figure(figsize=(WIDTH_IN["full"], 3.1))
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2)

    # A: one cubic cell of the diamond lattice (periodic images clipped to the cell), lengths in nm
    nodes = []
    for shift in product((-1, 0, 1), repeat=3):
        for p0, p1 in lattice_segments(4):
            clipped = _clip_segment_to_unit_cube(p0 + np.array(shift), p1 + np.array(shift))
            if clipped is not None:
                ax_a.plot(*(clipped * a).T, color=COLOR_SINGLE[1], lw=1.2)
                nodes += [p for p in (p0 + np.array(shift), p1 + np.array(shift)) if np.all((p >= -1e-9) & (p <= 1 + 1e-9))]
    nodes = np.unique(np.round(np.array(nodes), 6), axis=0) * a
    ax_a.scatter(*nodes.T, s=8, color=COLOR_SINGLE[1], depthshade=False)
    for corner in product((0, 1), repeat=3):
        for axis in range(3):
            if corner[axis] == 0:
                end = list(corner)
                end[axis] = 1
                ax_a.plot(*(np.array([corner, end]) * a).T, color="#9a9a9a", lw=0.6)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 15))
    r = ref["cavity_diameter_nm_mean"] / 2
    c = geo["cavity_centre"] * a
    ax_a.plot_surface(c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v), c[2] + r * np.cos(v),
                      color=COLOR_CAST[0][0], alpha=0.35, lw=0)
    ax_a.set_box_aspect((1, 1, 1))
    ax_a.grid(False)
    for axis in (ax_a.xaxis, ax_a.yaxis, ax_a.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.set_ticks([0, round(a / 10) * 10])
    ax_a.set_xlabel("x (nm)", labelpad=-8)
    ax_a.set_ylabel("y (nm)", labelpad=-8)
    ax_a.set_zlabel("z (nm)", labelpad=-8)
    ax_a.tick_params(pad=-3)
    ax_a.text2D(0.5, -0.04, f"cell edge a = {a:.1f} nm, strand length b = {ref['strand_length_nm_mean']:.1f} nm,\n"
                f"cavity sphere d = {ref['cavity_diameter_nm_mean']:.1f} nm (d$_f$ = 0)",
                transform=ax_a.transAxes, ha="center", va="top", fontsize=8)

    # B: diameters vs. strand diameter
    for f, ls, marker in ((4, "-", "o"), (6, _DASH_THEORY, "s")):
        sub = pooled[pooled["functionality"] == f].sort_values("strand_diameter_nm")
        lattice = sub["lattice"].iloc[0]
        for kind, (base, dark) in (("cavity", COLOR_CAST[0]), ("throat", COLOR_CAST[1])):
            m, s = sub[f"{kind}_diameter_nm_mean"], sub[f"{kind}_diameter_nm_sd"]
            ax_b.fill_between(sub["strand_diameter_nm"], m - s, m + s, color=base, alpha=0.15, lw=0)
            ax_b.plot(sub["strand_diameter_nm"], m, ls=ls, marker=marker, ms=4, lw=1.2, color=dark, mfc=base, mec=dark,
                      label=f"{kind}, {lattice} (f = {f})")
    ax_b.set_xlabel("Strand diameter d$_f$ (nm)")
    ax_b.set_ylabel("Pore diameter (nm)")
    ax_b.set_ylim(0, None)
    ax_b.legend(loc="lower left")

    ax_a.text2D(0.0, 1.02, "A", transform=ax_a.transAxes, fontsize=10, fontweight="bold")
    ax_b.text(-0.16, 1.02, "B", transform=ax_b.transAxes, fontsize=10, fontweight="bold", va="bottom")
    save(fig, folder, "pore_size_phantom_network")


# ── Validation ─────────────────────────────────────────────────────────────────

def plot_validation(cache: dict, folder: Path) -> None:
    pts = cache["validation_points"]
    if pts is None or len(pts) == 0:
        return
    base, dark = COLOR_SINGLE
    fig, ax = new_figure("half")
    ax.plot(pts["Gp_xls_Pa"], pts["Gp_tri_Pa"], "o", ms=4, mfc=base, mec=dark, alpha=POINT_ALPHA, label="G'")
    ax.plot(pts["Gpp_xls_Pa"], pts["Gpp_tri_Pa"], "s", ms=4, mfc="white", mec=dark, alpha=POINT_ALPHA, label="G''")
    lim = [0.5 * min(pts[["Gp_xls_Pa", "Gpp_xls_Pa"]].min()), 2 * max(pts[["Gp_xls_Pa", "Gpp_xls_Pa"]].max())]
    ax.plot(lim, lim, color="black", lw=1.2, ls=_DASH_THEORY, label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _log_axes(ax)
    ax.set_xlabel("TRIOS export (.xls) (Pa)")
    ax.set_ylabel("Reconstructed from .tri (Pa)")
    ax.legend(loc="upper left")
    save(fig, folder, "tri_vs_xls_parity")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"{CACHE_PATH} not found; run Rheo_Hyd20_Compute.py first.")
    with open(CACHE_PATH, "rb") as fh:
        cache = pickle.load(fh)
    root = OUTPUT_ROOT or Path(cache["output_root"])
    figs = root / "figures"
    target = cache["config"]["target_frequency_Hz"]

    with plt.style.context(str(STYLE_PATH)):
        for run, res in cache["results"].items():
            if res["fs"] is not None and len(res["data"]):
                plot_frequency_sweep(run, res["data"], res["fs"], target, figs / "frequency_sweeps")
            elif res["as"] is not None and "error" not in res["as"]:
                plot_amplitude_sweep(run, res["data"], res["as"], figs / "amplitude_sweeps")
        plot_stability_sweeps(cache, figs / "stability")
        plot_stability_summary(cache, figs / "stability", normalised=True)
        plot_stability_summary(cache, figs / "stability", normalised=False)
        plot_viscoelastic_mean(cache, figs / "stability")
        plot_mesh_size(cache, figs / "mesh_size")
        plot_pore_size(cache, figs / "mesh_size")
        plot_validation(cache, figs / "validation")
    print(f"Figures written to {figs}")


if __name__ == "__main__":
    main()
