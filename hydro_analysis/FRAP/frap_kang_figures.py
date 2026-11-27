"""
FRAP, Kang et al. 2012 result figure (user, 2026-11-26: all recovery curves pooled and the final D0 and
D_eff in one figure, no comparison with other methods). Consumer stage: reads cache/frap_kang_result.pkl
written by frap_kang_analysis.py (run that first); nothing is refitted.

frap_kang_recovery_d.pdf (full width, two panels, PDF + sidecar JSON in FIGURE_DIR):
  A  recovery curves of all measurements in one panel, F(t) in the drift-tracked nominal bleach ROI
     (radius r_n from the XML), background- and fading-corrected and normalised to the pre-bleach value
     (F_i = 1) as in Kang et al.; buffer and hydrogel (shallow and deep pooled, user 2026-10-14) in the
     category colours of the style guide (section 11). Dashed: qc_kang failed (shown, not in the mean).
     x range RECOVERY_XLIM as in the existing recovery figure (user, 2026-10-07); points beyond it are
     counted in the JSON.
  B  D after Kang eq. 5 per measurement (left sub-position; open = qc_kang failed) and mean +- 95 % t
     confidence interval over the QC-passed measurements (right sub-position): buffer = D0, hydrogel = D_eff.
The numbers are in FRAP_v2/kang/frap_kang_summary.csv (frap_kang_analysis.py). No other script reads the
figure or its JSON.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

# ── Configuration ──────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "cache" / "frap_kang_result.pkl"
FIGURE_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\FRAP")
STYLE_PATH = HERE.parent / "thesis.mplstyle"
FIG_SIZE = (6.30, 2.60)                     # class "full", two panels; height PROPOSED, not approved
WIDTH_RATIOS = (2.0, 1.0)                   # PROPOSED, not approved
COLORS = {"Buffer": ("#3B8C8C", "#2A6666"), "Hydrogel": ("#D98C3D", "#A6672D")}   # style guide §11, categories A/B
X_POS = {"Buffer": 0, "Hydrogel": 1}
SUB_OFFSET = {"points": -0.13, "mean": 0.13}  # fixed sub-positions, not jitter
TICK_LABEL = {"Buffer": "Buffer\n($D_0$)", "Hydrogel": "Hydrogel\n($D_\\mathrm{eff}$)"}
RECOVERY_XLIM = (0.0, 150.0)                # user, 2026-10-07 (recovery figure)
RECOVERY_YLIM = (0.0, 1.1)                  # user, 2026-10-08


def _outside(t, f):
    in_x = (t >= RECOVERY_XLIM[0]) & (t <= RECOVERY_XLIM[1])
    return int((~in_x).sum()), int((in_x & ((f < RECOVERY_YLIM[0]) | (f > RECOVERY_YLIM[1]))).sum())


def _panel_label(ax, text):
    ax.text(-0.13, 1.0, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="right")


def plot(cache: dict) -> dict:
    df, summary, curves = cache["results"], cache["summary"].set_index("sample_type"), cache["curves"]
    outside = {}
    with plt.style.context(STYLE_PATH):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIG_SIZE, layout="constrained",
                                         gridspec_kw={"width_ratios": WIDTH_RATIOS})
        handles = []
        for stype in X_POS:
            base, dark = COLORS[stype]
            sub = df[df.sample_type == stype]
            for row in sub.itertuples():
                c = curves[row.measurement_id]
                ax_a.plot(c["t_s"], c["F"], color=dark if row.qc_kang else base, lw=1.0, alpha=0.7,
                          ls="-" if row.qc_kang else "--", zorder=3 if stype == "Buffer" else 2)
                outside[row.measurement_id] = _outside(c["t_s"], c["F"])
            handles.append(Line2D([], [], color=dark, lw=1.0, label=f"{stype} (n = {len(sub)})"))
        if (~df.qc_kang).any():
            handles.append(Line2D([], [], color="0.4", lw=1.0, ls="--", label="Fails QC"))
        ax_a.axhline(1.0, color="0.65", lw=0.8, zorder=1)
        ax_a.set_xlim(*RECOVERY_XLIM)
        ax_a.set_ylim(*RECOVERY_YLIM)
        ax_a.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_a.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_a.set_xlabel("Time after bleach $t$ (s)")
        ax_a.set_ylabel(r"Normalized fluorescence $F\,/\,F_\mathrm{i}$")
        ax_a.legend(handles=handles, loc="lower right")
        _panel_label(ax_a, "A")

        for stype, x in X_POS.items():
            base, dark = COLORS[stype]
            sub = df[df.sample_type == stype]
            for ok in (True, False):
                v = sub[sub.qc_kang == ok]
                ax_b.plot(np.full(len(v), x + SUB_OFFSET["points"]), v.D_kang_um2_s, "o", ms=4,
                          mfc=base if ok else "none", mec=dark, mew=0.6, alpha=0.6, ls="none", zorder=3)
            s = summary.loc[stype]
            ax_b.errorbar(x + SUB_OFFSET["mean"], s.D_mean_um2_s, yerr=s.D_CI95_half_width_um2_s, fmt="o", ms=4,
                          mfc=base, mec=dark, mew=0.6, ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8,
                          zorder=3)
        ax_b.set_xticks(list(X_POS.values()), [TICK_LABEL[k] for k in X_POS])
        ax_b.set_xlim(-0.6, len(X_POS) - 0.4)
        ax_b.tick_params(axis="x", which="both", top=False, bottom=False)
        ax_b.set_ylim(0, 1.15 * df.D_kang_um2_s.max())
        ax_b.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax_b.set_ylabel(r"Diffusion coefficient $D$ (µm$^2$ s$^{-1}$)")
        _panel_label(ax_b, "B")

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_DIR / "frap_kang_recovery_d.pdf"
        fig.savefig(path, format="pdf")          # fixed size, no bbox_inches="tight"
        plt.close(fig)
    meta = {"figure": path.name, "created": f"{datetime.now():%Y-%m-%d %H:%M}", "cache": str(CACHE_PATH),
            "cache_created": cache["created"], "method": "Kang et al. 2012 eq. 5, D = (r_e^2 + r_n^2) / (8 t_1/2)",
            "panel_A": "F(t) in the drift-tracked r_n ROI, fading-corrected with the reference region, F_i = 1",
            "panel_B": "points: D per measurement (open = qc_kang failed); right: mean +- 95 % t CI of the QC-passed",
            "xlim_s": RECOVERY_XLIM, "ylim": RECOVERY_YLIM,
            "points_outside_axes": {m: {"beyond_x": a, "beyond_y_within_x": b} for m, (a, b) in outside.items()},
            "groups": {k: {c: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                           for c, v in cache["summary"].set_index("sample_type").loc[k].items()} for k in X_POS}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return outside


def main():
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"{CACHE_PATH} missing: run frap_kang_analysis.py first.")
    cache = pickle.loads(CACHE_PATH.read_bytes())
    for m, (a, b) in plot(cache).items():
        print(f"{m}: {a} points beyond x range, {b} within x range beyond y range")
    print(f"Written to {FIGURE_DIR / 'frap_kang_recovery_d.pdf'}")


if __name__ == "__main__":
    main()
