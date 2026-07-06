"""
Interactive Method Comparison — Step Size (Gaussian) vs MSD (power-law).

tp.imsd() is cached per track-length filter so adjusting exponent or
fit-point sliders does not trigger a full recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
import numpy as np
from scipy import stats
import trackpy as tp

from hydro_analysis.core.io import (
    extract_particle_size_from_path,
    single_file_data,
    get_dls_sizes,
    get_dls_labels,
)
from hydro_analysis.core.analysis import (
    calculate_step_sizes,
    fit_gaussian_diffusion_1d,
    diffusion_2d_from_1d_fits,
    fit_powerlaw_with_errors,
    MIN_TRACK_LENGTH,
)
from hydro_analysis.core.physics import calculate_theoretical_diffusion

tp.quiet()

# ── Style (mirrors plot_emsd_publication) ─────────────────────────────────────
_COL_IMSD    = "black"
_COL_EMSD    = "#1a1aee"
_COL_THEORY  = "black"
_COL_FIT     = "#cc2200"
_COL_STEP    = "#228822"

_LW_IMSD     = 0.5
_LW_EMSD     = 0.8
_LW_THEORY   = 1.2
_LW_FIT      = 1.0
_MS_EMSD     = 3
_DASH_THEORY = (0, (4, 3))
_ALPHA_IMSD  = 0.15

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
    "xtick.major.size":   4.0,
    "ytick.major.size":   4.0,
    "xtick.minor.size":   2.5,
    "ytick.minor.size":   2.5,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.labelsize":     10,
    "legend.fontsize":    8,
    "axes.grid":          False,
}

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_STEP_INTERVAL   = 1
DEFAULT_FIT_POINTS      = 6
DEFAULT_MIN_EXPONENT    = 0.85
DEFAULT_MAX_EXPONENT    = 1.15
DEFAULT_MAX_TRACK       = 500


def _add_log_minor_ticks(ax: plt.Axes) -> None:
    loc = mticker.LogLocator(subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=100)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(mticker.NullFormatter())


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _filter_tracks(tracks_df, min_len: int, max_len: int):
    lengths = tracks_df.groupby("particle").size()
    valid   = lengths[(lengths >= min_len) & (lengths <= max_len)].index
    return tracks_df[tracks_df["particle"].isin(valid)].copy(), len(valid), len(lengths)


def _step_analysis(rd: dict, step_interval: int, min_len: int, max_len: int) -> dict | None:
    tracks = rd.get("tracks_df")
    if tracks is None or tracks.empty:
        return None
    filtered, n_valid, n_total = _filter_tracks(tracks, min_len, max_len)
    if filtered.empty:
        return None
    step_df = calculate_step_sizes(filtered, step_interval=step_interval)
    if step_df.empty:
        return None
    mpp, fps = rd["mpp"], rd["fps"]
    fx = fit_gaussian_diffusion_1d(step_df["dx"], mpp, fps, frame_interval=step_interval, axis="x")
    fy = fit_gaussian_diffusion_1d(step_df["dy"], mpp, fps, frame_interval=step_interval, axis="y")
    f2 = diffusion_2d_from_1d_fits(fx, fy)
    return {
        **f2,
        "dx_nm":       step_df["dx"].values * mpp * 1000.0,
        "dy_nm":       step_df["dy"].values * mpp * 1000.0,
        "n_valid":     n_valid,
        "n_total":     n_total,
        "step_interval": step_interval,
    }


def _msd_from_imsd(imsd, fit_points: int, min_n: float, max_n: float) -> dict | None:
    """Filter particles by exponent and return ensemble fit — does NOT call tp.imsd()."""
    valid, exponents, rej_lo, rej_hi = [], [], 0, 0
    for pid in imsd.columns:
        s = imsd[pid].dropna()
        if len(s) < fit_points:
            continue
        try:
            pr = fit_powerlaw_with_errors(s, points=fit_points)
            n  = float(pr["n"][0])
            if n < min_n:
                rej_lo += 1
            elif n > max_n:
                rej_hi += 1
            else:
                valid.append(pid)
                exponents.append(n)
        except Exception:
            continue
    if not valid:
        return None
    f_imsd  = imsd[valid]
    emsd    = f_imsd.mean(axis=1)
    pr      = fit_powerlaw_with_errors(emsd, points=fit_points)
    A       = float(pr["A"][0]);   A_err = float(pr["A_err"][0])
    n       = float(pr["n"][0]);   n_err = float(pr["n_err"][0])
    return {
        "D":         A / 4.0,
        "D_err":     A_err / 4.0,
        "A":         A,   "A_err": A_err,
        "n":         n,   "n_err": n_err,
        "emsd":      emsd,
        "imsd":      f_imsd,
        "n_valid":   len(valid),
        "n_total":   len(imsd.columns),
        "rej_lo":    rej_lo,
        "rej_hi":    rej_hi,
        "mean_n":    float(np.mean(exponents)),
        "std_n":     float(np.std(exponents)),
    }


# ── Interactive figure ────────────────────────────────────────────────────────

class InteractiveComparison:
    def __init__(self, xml_path: Path, particle_size: float | None = None):
        self.xml_path = xml_path
        self.rd = single_file_data(xml_path)
        if self.rd is None:
            raise ValueError(f"Could not load: {xml_path}")

        self.mpp = self.rd["mpp"]
        self.fps = self.rd["fps"]

        # track length limits
        lengths = self.rd["tracks_df"].groupby("particle").size()
        self.max_track_possible = int(lengths.max())
        self.n_tracks_total     = len(lengths)

        # DLS mapping from PKL
        dls_sizes  = get_dls_sizes()   # {nominal: z_mean_nm}
        dls_labels = get_dls_labels()  # {nominal: label_nm}

        # determine nominal size
        nom = particle_size or self.rd.get("particle_size_nm") or self._guess_size(xml_path)
        self.nominal_nm = float(nom)
        self.z_mean_nm  = dls_sizes.get(self.nominal_nm, self.nominal_nm)
        self.label_nm   = dls_labels.get(self.nominal_nm, int(self.nominal_nm))
        self.D_theo     = calculate_theoretical_diffusion(particle_size_nm=self.z_mean_nm)

        print(f"  nominal: {self.nominal_nm:.0f} nm  |  DLS z_mean: {self.z_mean_nm:.1f} nm"
              f"  |  label: {self.label_nm} nm  |  D_theo: {self.D_theo:.3e} µm²/s")
        print(f"  mpp={self.mpp} µm/px  fps={self.fps}  tracks={self.n_tracks_total}")

        # slider state
        self.step_interval  = DEFAULT_STEP_INTERVAL
        self.fit_points     = DEFAULT_FIT_POINTS
        self.min_n          = DEFAULT_MIN_EXPONENT
        self.max_n          = DEFAULT_MAX_EXPONENT
        self.min_len        = MIN_TRACK_LENGTH
        self.max_len        = min(DEFAULT_MAX_TRACK, self.max_track_possible)

        # imsd cache — only invalidated when track-length filter changes
        self._imsd_cache     = None
        self._imsd_cache_key = None

        with plt.rc_context(_RC):
            self.fig = plt.figure(figsize=(20, 13))
            self._build_layout()
            self._build_controls()
            self.update()

    # ── size extraction ───────────────────────────────────────────────────────
    def _guess_size(self, xml_path: Path) -> float:
        import re
        for src in [xml_path.name] + [p.name for p in xml_path.parents[:3]]:
            m = re.search(r"(\d+)\s*nm", src, re.IGNORECASE)
            if m:
                return float(m.group(1))
        s = extract_particle_size_from_path(xml_path.parent)
        return float(s) if s else 100.0

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        gs = GridSpec(2, 3, figure=self.fig,
                      height_ratios=[1, 1],
                      width_ratios=[1, 1, 1.3],
                      hspace=0.38, wspace=0.30,
                      left=0.06, right=0.97, top=0.92, bottom=0.25)
        self.ax_dx  = self.fig.add_subplot(gs[0, 0])
        self.ax_dy  = self.fig.add_subplot(gs[0, 1])
        self.ax_msd = self.fig.add_subplot(gs[0:2, 2])
        self.ax_cmp = self.fig.add_subplot(gs[1, 0:2])

    # ── controls ──────────────────────────────────────────────────────────────
    def _build_controls(self):
        sh, sw = 0.015, 0.17
        max_l  = min(self.max_track_possible, 1000)

        def _sl(x, y, label, vmin, vmax, vinit, vstep=None, valfmt=None):
            ax = plt.axes([x, y, sw, sh])
            kw = {}
            if vstep  is not None: kw["valstep"] = vstep
            if valfmt is not None: kw["valfmt"]  = valfmt
            s = Slider(ax, label, vmin, vmax, valinit=vinit, **kw)
            s.on_changed(self._on_change)
            return s

        c1, c2, c3 = 0.07, 0.32, 0.58

        self.sl_step    = _sl(c1, 0.18, "Step interval", 1, 10, self.step_interval, vstep=1)
        self.sl_fit     = _sl(c2, 0.18, "MSD fit pts",   3, 20, self.fit_points,    vstep=1)
        self.sl_min_n   = _sl(c2, 0.13, "n min",         0.0, 2.0, self.min_n,      valfmt="%.2f")
        self.sl_max_n   = _sl(c2, 0.08, "n max",         0.0, 2.0, self.max_n,      valfmt="%.2f")
        self.sl_min_len = _sl(c3, 0.18, "Track min",     1, max_l, self.min_len,    vstep=1)
        self.sl_max_len = _sl(c3, 0.13, "Track max",     1, max_l, self.max_len,    vstep=1)

        ax_btn = plt.axes([c3 + 0.03, 0.05, 0.10, 0.04])
        self.btn = Button(ax_btn, "Update")
        self.btn.on_clicked(lambda _: self.update())

        for cx, label in [(c1, "STEP SIZE"), (c2, "MSD FILTER"), (c3, "TRACK LENGTH")]:
            self.fig.text(cx + sw/2, 0.22, label, ha="center", fontsize=9, fontweight="bold")

    def _on_change(self, _):
        self.step_interval = int(self.sl_step.val)
        self.fit_points    = int(self.sl_fit.val)
        self.min_n         = float(self.sl_min_n.val)
        self.max_n         = max(float(self.sl_max_n.val), self.min_n)
        self.min_len       = int(self.sl_min_len.val)
        self.max_len       = max(int(self.sl_max_len.val), self.min_len)

    # ── cached imsd ───────────────────────────────────────────────────────────
    def _get_imsd(self):
        key = (self.min_len, self.max_len)
        if self._imsd_cache is None or self._imsd_cache_key != key:
            tracks = self.rd["tracks_df"]
            filtered, _, _ = _filter_tracks(tracks, self.min_len, self.max_len)
            if filtered.empty:
                self._imsd_cache = None
            else:
                self._imsd_cache = tp.imsd(filtered, mpp=self.mpp, fps=self.fps)
            self._imsd_cache_key = key
        return self._imsd_cache

    # ── main update ───────────────────────────────────────────────────────────
    def update(self):
        self._on_change(None)
        for ax in [self.ax_dx, self.ax_dy, self.ax_msd, self.ax_cmp]:
            ax.clear()

        with plt.rc_context(_RC):
            step_r = _step_analysis(self.rd, self.step_interval, self.min_len, self.max_len)
            imsd   = self._get_imsd()
            msd_r  = _msd_from_imsd(imsd, self.fit_points, self.min_n, self.max_n) if imsd is not None else None

            self._plot_hist(self.ax_dx, step_r, "dx" if step_r else None)
            self._plot_hist(self.ax_dy, step_r, "dy" if step_r else None)
            self._plot_msd(msd_r)
            self._plot_comparison(step_r, msd_r)

            self.fig.suptitle(
                f"{self.label_nm} nm  |  {self.xml_path.name}  |  "
                f"D_theo = {self.D_theo:.3e} µm²/s  (DLS z_mean = {self.z_mean_nm:.1f} nm)",
                fontsize=11, y=0.97,
            )
        self.fig.canvas.draw_idle()

    # ── histogram panel ───────────────────────────────────────────────────────
    def _plot_hist(self, ax: plt.Axes, step_r: dict | None, axis: str | None):
        if step_r is None or axis is None:
            ax.set_visible(False)
            return
        ax.set_visible(True)
        data  = step_r[f"{axis}_nm"]
        mu    = step_r[f"mu_{axis[-1]}_nm"]
        sigma = step_r[f"sigma_{axis[-1]}_nm"]
        color = "#4488cc" if axis == "dx" else "#cc6644"

        ax.hist(data, bins=60, density=True, color=color, alpha=0.55, linewidth=0)
        xr = np.linspace(data.min(), data.max(), 300)
        ax.plot(xr, stats.norm.pdf(xr, mu, sigma),
                color=_COL_FIT, linewidth=_LW_FIT + 0.5,
                label=f"μ={mu:.1f} nm\nσ={sigma:.1f} nm")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel(f"d{axis[-1]} (nm)")
        ax.set_ylabel("density")
        ax.set_title(f"Δ{axis[-1]}  (Δt={step_r['step_interval']},  "
                     f"{step_r['n_valid']}/{step_r['n_total']} tracks)", fontsize=9)
        ax.legend(fontsize=8, frameon=False)

    # ── MSD panel ─────────────────────────────────────────────────────────────
    def _plot_msd(self, msd_r: dict | None):
        ax = self.ax_msd
        if msd_r is None:
            ax.text(0.5, 0.5, "no MSD data", transform=ax.transAxes, ha="center")
            return

        emsd = msd_r["emsd"]
        imsd = msd_r["imsd"]
        lag  = emsd.index.values.astype(float)
        ev   = emsd.values.astype(float)
        t_th = np.logspace(np.log10(lag[lag > 0].min()), np.log10(lag.max()), 300)

        # iMSD
        for col in imsd.columns[:min(200, len(imsd.columns))]:
            s = imsd[col].dropna()
            s = s[s > 0]
            if not s.empty:
                ax.plot(s.index.values, s.values,
                        color=_COL_IMSD, linewidth=_LW_IMSD, alpha=_ALPHA_IMSD, zorder=1)

        # eMSD
        ax.plot(lag[ev > 0], ev[ev > 0],
                color=_COL_EMSD, linewidth=_LW_EMSD,
                marker="o", markersize=_MS_EMSD, markeredgewidth=0, zorder=5)

        # theory
        ax.plot(t_th, 4.0 * self.D_theo * t_th,
                color=_COL_THEORY, linewidth=_LW_THEORY, linestyle=_DASH_THEORY, zorder=3)

        # power-law fit
        ax.plot(t_th, msd_r["A"] * t_th ** msd_r["n"],
                color=_COL_FIT, linewidth=_LW_FIT, zorder=4)

        ax.set_xscale("log");  ax.set_yscale("log")
        _add_log_minor_ticks(ax)
        ax.set_xlabel(r"Lag time $\tau$ (s)")
        ax.set_ylabel(r"MSD ($\mu\mathrm{m}^2$)")

        s = str(self.label_nm)
        handles = [
            Line2D([0],[0], color=_COL_IMSD,   lw=0.9, alpha=0.4,  label=f"iMSD ({s} nm)"),
            Line2D([0],[0], color=_COL_EMSD,   lw=_LW_EMSD,
                   marker="o", markersize=_MS_EMSD, markeredgewidth=0, label=f"eMSD ({s} nm)"),
            Line2D([0],[0], color=_COL_THEORY, lw=_LW_THEORY, linestyle=_DASH_THEORY,
                   label=rf"Stokes–Einstein $4D_0^{{{s}\,\mathrm{{nm}}}}\tau$"),
            Line2D([0],[0], color=_COL_FIT,    lw=_LW_FIT,
                   label=rf"fit  $n$={msd_r['n']:.3f}±{msd_r['n_err']:.3f}"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False)
        ax.text(0.03, 0.03,
                f"{msd_r['n_valid']}/{msd_r['n_total']} particles  "
                f"({self.fit_points} fit pts)",
                transform=ax.transAxes, fontsize=7, va="bottom")

    # ── comparison panel ──────────────────────────────────────────────────────
    def _plot_comparison(self, step_r: dict | None, msd_r: dict | None):
        ax = self.ax_cmp
        methods, D_vals, D_errs, cols = [], [], [], []

        if step_r:
            D = step_r["D_um2_per_s"]
            methods.append(f"Step Size\n(Gaussian)\nΔt={step_r['step_interval']}")
            D_vals.append(D); D_errs.append(step_r["D_dir_disagreement_um2_per_s"])
            cols.append(_COL_STEP)

        if msd_r:
            methods.append(f"MSD\n(power-law)\nn={msd_r['n']:.3f}")
            D_vals.append(msd_r["D"]); D_errs.append(msd_r["D_err"])
            cols.append(_COL_EMSD)

        ax.axhline(self.D_theo, color=_COL_THEORY, linewidth=_LW_THEORY,
                   linestyle=_DASH_THEORY,
                   label=rf"$D_0^{{{self.label_nm}\,\mathrm{{nm}}}}$ = {self.D_theo:.3e} µm²/s")
        ax.axhspan(self.D_theo * 0.9, self.D_theo * 1.1, color="lightgray", alpha=0.25)

        for i, (_, D, De, c) in enumerate(zip(methods, D_vals, D_errs, cols)):
            ax.errorbar(i, D, yerr=De, fmt="o", markersize=10,
                        color=c, ecolor=c, elinewidth=1.2, capsize=4, capthick=1.2)
            dev = (D - self.D_theo) / self.D_theo * 100
            ax.text(i, D + De * 1.3, f"{dev:+.1f}%",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_xlim(-0.6, max(len(methods) - 0.4, 0.4))
        ax.set_ylabel(r"D (µm²/s)")
        ax.legend(frameon=False, fontsize=8)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    xml_path = Path(r"E:\PhD Data Analysis\SPT 2025 II\2025.10.01\Tracks\A4_ontop_1d_50_20mg_processed_Tracks.xml")
    particle_size = None

    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        particle_size = float(sys.argv[2])

    if not xml_path.exists():
        print(f"ERROR: not found: {xml_path}")
        print(f"Usage: python {Path(__file__).name} <xml_file> [particle_size_nm]")
        return

    tool = InteractiveComparison(xml_path, particle_size)
    plt.show()


if __name__ == "__main__":
    main()
