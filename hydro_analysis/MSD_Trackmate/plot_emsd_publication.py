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
