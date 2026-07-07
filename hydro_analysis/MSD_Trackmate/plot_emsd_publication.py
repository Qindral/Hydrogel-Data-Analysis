"""
Publication-quality log-log MSD figure for single-particle tracking.

Single-column layout (5.5 × 4 in), matplotlib mathtext rendering (no LaTeX
required).  Math uses the DejaVu Sans font set (``mathtext.fontset=dejavusans``)
so text and math characters are visually consistent.

Typical usage
-------------
from hydro_analysis.MSD_Trackmate.plot_emsd_publication import plot_emsd_publication
from hydro_analysis.core.physics import calculate_theoretical_diffusion

fit  = result["fit_results_MSD"]
D_th = calculate_theoretical_diffusion(particle_size_nm=100.0)

fig, ax = plot_emsd_publication(
    imsd       = fit["imsd"],
    emsd       = fit["emsd"],
    D_theo     = D_th,
    n_fit      = MSD_FIT_POINTS,
    fit_result = fit,
    save_path  = Path("output/"),
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ── colour / style constants ──────────────────────────────────────────────────
_COL_IMSD    = "black"
_COL_EMSD    = "#1a1aee"
_COL_THEORY  = "black"
_COL_FIT     = "#cc2200"
_COL_SPAN    = "#bbbbbb"

_ALPHA_IMSD  = 0.01
_LW_IMSD     = 0.65
_LW_EMSD     = 0.8
_MS_EMSD     = 3
_LW_THEORY   = 1.2
_LW_FIT      = 1.0
_DASH_THEORY = (0, (4, 3))

_RC = {
    "text.usetex":        False,
    "mathtext.fontset":   "dejavusans",
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Open Sans", "Arial", "DejaVu Sans"],
    "axes.linewidth":     0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   5.0,
    "ytick.major.size":   5.0,
    "xtick.minor.size":   3.0,
    "ytick.minor.size":   3.0,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.width":  0.8,
    "ytick.minor.width":  0.8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.labelsize":     10,
    "legend.fontsize":    8,
}


def _darken(hex_col: str, factor: float = 0.60) -> str:
    r, g, b = mcolors.to_rgb(hex_col)
    return mcolors.to_hex((r * factor, g * factor, b * factor))


def _add_log_minor_ticks(ax: plt.Axes) -> None:
    loc = mticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=100)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(mticker.NullFormatter())


def plot_emsd_publication(
    imsd: pd.DataFrame,
    emsd: pd.Series,
    D_theo: float,
    n_fit: int,
    fit_result: Optional[dict] = None,
    save_path: Optional[Path | str] = None,
    filename: str = "emsd_publication",
    particle_size_nm: Optional[float] = None,
    sample_label: Optional[str] = None,
    xlim: tuple = (1e-2, 2.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a publication-quality log-log MSD plot with a normalized inset.

    Parameters
    ----------
    imsd : pd.DataFrame
        tp.imsd() output.  Index = lag time (s), columns = particle IDs.
    emsd : pd.Series
        tp.emsd() output.  Index = lag time (s), values = MSD (µm²).
    D_theo : float
        Stokes–Einstein diffusion coefficient (µm²/s).
    n_fit : int
        Number of early lag-time points used for the power-law eMSD fit.
    fit_result : dict, optional
        Keys ``'A'`` (µm²/s^n) and ``'exponent'``.
    save_path : Path or str, optional
        Output directory.  Saves <filename>.pdf and <filename>.png (300 dpi).
    filename : str
        Base filename without extension.
    particle_size_nm : float, optional
        DLS-corrected particle diameter (nm) — shown in the legend labels.
    sample_label : str, optional
        Short sample description shown inside the figure (e.g. "20 mg/mL C16").
    xlim : tuple
        (x_min, x_max) lag-time axis limits in seconds.

    Returns
    -------
    fig, ax
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))

        lag   = emsd.index.values.astype(float)
        ev    = emsd.values.astype(float)
        valid = ev > 0

        # ── Stokes–Einstein theory line ──────────────────────────────────
        t_th = np.logspace(np.log10(lag[valid].min()),
                           np.log10(lag[valid].max()), 300)
        ax.plot(t_th, 4.0 * D_theo * t_th,
                color=_COL_THEORY, linewidth=_LW_THEORY,
                linestyle=_DASH_THEORY, zorder=3)

        # ── power-law fit line: computed from n_fit points, drawn full range ─
        A_fit, n_exp = None, None
        if fit_result is not None:
            A_fit = fit_result.get("A")
            n_exp = fit_result.get("exponent", fit_result.get("n"))
        if A_fit is not None and n_exp is not None:
            ax.plot(t_th, float(A_fit) * t_th ** float(n_exp),
                    color=_COL_FIT, linewidth=_LW_FIT, zorder=4)

        # ── individual MSDs ──────────────────────────────────────────────
        for col in imsd.columns:
            s = imsd[col].dropna()
            s = s[s > 0]
            if not s.empty:
                ax.plot(s.index.values, s.values,
                        color=_COL_IMSD, linewidth=_LW_IMSD,
                        alpha=_ALPHA_IMSD, zorder=1, rasterized=True)

        # ── ensemble MSD: thin line + one marker per lag time ────────────
        ax.plot(lag[valid], ev[valid],
                color=_COL_EMSD, linewidth=_LW_EMSD, zorder=5,
                marker="o", markersize=_MS_EMSD, markeredgewidth=0)

        # ── axis scales + fixed ranges for cross-sample comparability ────
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(1e-2, 1e2)
        _add_log_minor_ticks(ax)

        ax.set_xlabel(r"Lag time $\tau$ (s)")
        ax.set_ylabel(r"MSD ($\mu\mathrm{m}^2$)")

        # ── legend upper left ────────────────────────────────────────────
        if particle_size_nm is not None:
            s = f"{particle_size_nm:.0f}"
            imsd_label = f"iMSD ({s} nm)"
            emsd_label = f"eMSD ({s} nm)"
            se_label   = rf"Stokes–Einstein $4D_{{0}}^{{{s}\,\mathrm{{nm}}}}\tau$"
        else:
            imsd_label = "individual MSD"
            emsd_label = "ensemble MSD"
            se_label   = r"Stokes–Einstein ($4D_0\tau$)"
        handles = [
            Line2D([0], [0], color=_COL_IMSD, linewidth=0.9, alpha=0.5,
                   label=imsd_label),
            Line2D([0], [0], color=_COL_EMSD, linewidth=_LW_EMSD,
                   marker="o", markersize=_MS_EMSD, markeredgewidth=0,
                   label=emsd_label),
            Line2D([0], [0], color=_COL_THEORY, linewidth=_LW_THEORY,
                   linestyle=_DASH_THEORY, label=se_label),
        ]
        if A_fit is not None and n_exp is not None:
            handles.append(
                Line2D([0], [0], color=_COL_FIT, linewidth=_LW_FIT,
                       label=r"power-law fit ($A\tau^n$)"),
            )
        ax.legend(handles=handles, loc="upper left", frameon=False)

        if sample_label:
            ax.text(0.03, 0.03, sample_label, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, fontstyle="italic")

        # ── inset lower right: normalized MSD = MSD(τ) / (4 D_0 τ) ──────
        # Square panel, no zoom connector.
        # Theory → flat line at 1.  Deviations show sub-/superdiffusion.
        axin = ax.inset_axes([0.58, 0.08, 0.38, 0.38])

        fit_mask = valid & (np.arange(len(lag)) < n_fit)
        lag_fit  = lag[fit_mask]
        ev_fit   = ev[fit_mask]

        norm_y_all = [1.0]

        # black: theory normalized → 1
        t_dense = np.logspace(np.log10(lag_fit[0]), np.log10(lag_fit[-1]), 150)
        axin.axhline(1.0, color=_COL_THEORY, linewidth=1.0,
                     linestyle=_DASH_THEORY, zorder=3)

        # red: power-law fit normalized
        if A_fit is not None and n_exp is not None:
            y_fit_norm = (float(A_fit) * t_dense ** float(n_exp)) / (4.0 * D_theo * t_dense)
            axin.plot(t_dense, y_fit_norm,
                      color=_COL_FIT, linewidth=_LW_FIT, zorder=4)
            norm_y_all.extend(y_fit_norm.tolist())

        # blue: eMSD normalized, marker per point
        if lag_fit.size >= 2:
            y_emsd_norm = ev_fit / (4.0 * D_theo * lag_fit)
            axin.plot(lag_fit, y_emsd_norm,
                      color=_COL_EMSD, linewidth=_LW_EMSD, zorder=5,
                      marker="o", markersize=_MS_EMSD, markeredgewidth=0)
            norm_y_all.extend(y_emsd_norm.tolist())

        axin.set_xscale("log")
        axin.set_yscale("log")
        _add_log_minor_ticks(axin)
        axin.tick_params(which="both", direction="in",
                         top=True, right=True, labelsize=7, width=0.6)
        axin.tick_params(which="major", length=3.5)
        axin.tick_params(which="minor", length=2.0)
        for sp in axin.spines.values():
            sp.set_linewidth(0.6)

        axin.set_xlim(lag_fit[0] * 0.85, lag_fit[-1] * 1.15)
        finite_y = [v for v in norm_y_all if np.isfinite(v) and v > 0]
        if finite_y:
            axin.set_ylim(min(finite_y) * 0.5, max(finite_y) * 2.0)

        if particle_size_nm is not None:
            _s = f"{particle_size_nm:.0f}"
            _inset_ylabel = rf"MSD $/ 4D_0^{{{_s}\,\mathrm{{nm}}}}\tau$"
        else:
            _inset_ylabel = r"MSD $/ 4D_0\tau$"
        axin.text(0.97, 0.04, r"$\tau$ (s)", transform=axin.transAxes,
                  ha="right", va="bottom", fontsize=7)
        axin.text(0.04, 0.96, _inset_ylabel, transform=axin.transAxes,
                  ha="left", va="top", fontsize=7)

        # ── layout + output ───────────────────────────────────────────────
        fig.tight_layout(pad=0.4)

        if save_path is not None:
            out = Path(save_path)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{filename}.pdf", bbox_inches="tight")
            fig.savefig(out / f"{filename}.png", dpi=600, bbox_inches="tight")

        plt.show()
        return fig, ax


def plot_fit_lines_overview(
    fits_by_size: dict[float, list[dict]],
    theory_by_size: dict[float, float],
    dls_labels: dict[float, int],
    fits_hydrogel_by_size: Optional[dict[float, list[dict]]] = None,
    hydrogel_label: str = "20 mg/mL C16",
    save_path: Optional[Path | str] = None,
    filename: str = "emsd_fits_overview",
    sample_label: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Overview chart: per-file D₀ fit lines plus optional hydrogel fit lines,
    all colour-coded by particle size.

    Parameters
    ----------
    fits_by_size          {nominal_nm: [{"A", "n", "tau_min", "tau_max"}, ...]}
                          D₀ water — thin, semi-transparent lines.
    theory_by_size        {nominal_nm: D_theo (µm²/s)}
    dls_labels            {nominal_nm: label_nm}
    fits_hydrogel_by_size Same structure as fits_by_size — drawn thicker and
                          darker to show the hydrogel slowing.
    hydrogel_label        Legend text for the hydrogel layer.
    """
    _LW_D0  = 0.7    # D₀ fit lines
    _LW_HG  = 1.6    # hydrogel fit lines — thicker
    _A_D0   = 0.45   # D₀ alpha
    _A_HG   = 0.75   # hydrogel alpha

    all_sizes = sorted(set(fits_by_size) | set(fits_hydrogel_by_size or {}))

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        handles = []

        for size_nm in all_sizes:
            d0_fits = fits_by_size.get(size_nm, [])
            D_theo  = theory_by_size.get(size_nm, 0.0)
            label_nm = dls_labels.get(size_nm, int(size_nm))
            col      = _SIZE_COLORS.get(size_nm, "#333333")
            col_dark = _darken(col, 0.60)

            # Stokes-Einstein theory line
            t_th = np.logspace(-2, np.log10(6.0), 400)
            ax.plot(t_th, 4.0 * D_theo * t_th,
                    color=col, linewidth=_LW_THEORY,
                    linestyle=_DASH_THEORY, zorder=3, alpha=0.9)

            # D₀ individual fit lines (thin, semi-transparent, lighter colour)
            for fit in d0_fits:
                A, n = fit["A"], fit["n"]
                t = np.logspace(np.log10(fit["tau_min"]),
                                np.log10(fit["tau_max"]), 200)
                ax.plot(t, A * t ** n,
                        color=col, linewidth=_LW_D0, alpha=_A_D0, zorder=2)

            # hydrogel fit lines (thicker, darker)
            if fits_hydrogel_by_size:
                for fit in fits_hydrogel_by_size.get(size_nm, []):
                    A, n = fit["A"], fit["n"]
                    t = np.logspace(np.log10(fit["tau_min"]),
                                    np.log10(fit["tau_max"]), 200)
                    ax.plot(t, A * t ** n,
                            color=col_dark, linewidth=_LW_HG,
                            alpha=_A_HG, zorder=4)

            handles.append(
                Line2D([0], [0], color=col, linewidth=1.5,
                       label=f"{label_nm} nm")
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-2, 6.0)
        ax.set_ylim(1e-2, 1e3)
        _add_log_minor_ticks(ax)
        ax.set_xlabel(r"Lag time $\tau$ (s)")
        ax.set_ylabel(r"MSD ($\mu\mathrm{m}^2$)")

        style_handles = [
            Line2D([0], [0], color="gray", linewidth=_LW_D0, alpha=_A_D0,
                   label=r"$D_0$ fit (per file)"),
        ]
        if fits_hydrogel_by_size:
            style_handles.append(
                Line2D([0], [0], color="#444444", linewidth=_LW_HG,
                       alpha=_A_HG, label=f"{hydrogel_label} fit"),
            )
        style_handles.append(
            Line2D([0], [0], color="gray", linewidth=_LW_THEORY,
                   linestyle=_DASH_THEORY, label=r"Stokes–Einstein $4D_0\tau$"),
        )
        ax.legend(handles=handles + style_handles, loc="upper left",
                  frameon=False, fontsize=7)

        if sample_label:
            ax.text(0.03, 0.03, sample_label, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, fontstyle="italic")

        fig.tight_layout(pad=0.4)

        if save_path is not None:
            out = Path(save_path)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{filename}.pdf", bbox_inches="tight")
            fig.savefig(out / f"{filename}.png", dpi=600, bbox_inches="tight")

        plt.show()
        return fig, ax


# ── 6-colour palette (one per nominal size, light → dark) ────────────────────
_SIZE_COLORS: dict[float, str] = {
    20.0:   "#0072B2",   # blue
    50.0:   "#009E73",   # green
    100.0:  "#E69F00",   # amber
    200.0:  "#D55E00",   # vermillion
    500.0:  "#CC79A7",   # pink
    1000.0: "#56B4E9",   # sky blue
}


def plot_emsd_overview(
    pub_data: dict,
    dls_labels: dict[float, int],
    save_path: Optional[Path | str] = None,
    filename: str = "emsd_overview",
    sample_label: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Single log-log chart with all particle sizes overlaid.

    For each size one colour is used for:
      • eMSD data points (markers + thin line)
      • power-law fit line (solid)
      • Stokes-Einstein theory line (dashed, same colour, lighter)

    Parameters
    ----------
    pub_data   dict keyed by nominal_nm → {emsd_agg, fit_agg, D_theo}
    dls_labels {nominal_nm: label_nm}
    save_path  output directory
    filename   base name without extension
    sample_label  optional annotation (e.g. "D₀ water")
    """
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        handles = []

        for size_nm in sorted(pub_data.keys()):
            entry    = pub_data[size_nm]
            emsd     = entry["emsd_agg"]
            fit      = entry["fit_agg"]
            D_theo   = entry["D_theo"]
            label_nm = dls_labels.get(size_nm, int(size_nm))
            col      = _SIZE_COLORS.get(size_nm, "#333333")
            col_th   = col   # theory same colour, distinguished by linestyle

            lag   = emsd.index.values.astype(float)
            ev    = emsd.values.astype(float)
            valid = ev > 0

            t_th = np.logspace(np.log10(lag[valid].min()),
                               np.log10(lag[valid].max()), 300)

            # Stokes-Einstein theory (dashed)
            ax.plot(t_th, 4.0 * D_theo * t_th,
                    color=col_th, linewidth=_LW_THEORY,
                    linestyle=_DASH_THEORY, alpha=0.7, zorder=2)

            # power-law fit (solid)
            A_fit = fit.get("A")
            n_exp = fit.get("exponent", fit.get("n"))
            if A_fit is not None and n_exp is not None:
                ax.plot(t_th, float(A_fit) * t_th ** float(n_exp),
                        color=col, linewidth=_LW_FIT, zorder=4)

            # eMSD data points
            ax.plot(lag[valid], ev[valid],
                    color=col, linewidth=_LW_EMSD * 0.7, zorder=5,
                    marker="o", markersize=_MS_EMSD, markeredgewidth=0, alpha=0.85)

            handles.append(
                Line2D([0], [0], color=col, linewidth=_LW_FIT,
                       marker="o", markersize=_MS_EMSD, markeredgewidth=0,
                       label=f"{label_nm} nm")
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-2, 6.0)
        ax.set_ylim(1e-2, 1e3)
        _add_log_minor_ticks(ax)
        ax.set_xlabel(r"Lag time $\tau$ (s)")
        ax.set_ylabel(r"MSD ($\mu\mathrm{m}^2$)")
        ax.legend(handles=handles, loc="upper left", frameon=False, title="particle size")

        if sample_label:
            ax.text(0.03, 0.03, sample_label, transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, fontstyle="italic")

        # legend entry for line styles
        style_handles = [
            Line2D([0], [0], color="gray", linewidth=_LW_FIT, label="power-law fit"),
            Line2D([0], [0], color="gray", linewidth=_LW_THEORY,
                   linestyle=_DASH_THEORY, alpha=0.7, label=r"Stokes–Einstein $4D_0\tau$"),
        ]
        ax.legend(
            handles=handles + style_handles,
            loc="upper left", frameon=False,
            ncol=1, fontsize=7,
        )

        fig.tight_layout(pad=0.4)

        if save_path is not None:
            out = Path(save_path)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{filename}.pdf", bbox_inches="tight")
            fig.savefig(out / f"{filename}.png", dpi=600, bbox_inches="tight")

        plt.show()
        return fig, ax
