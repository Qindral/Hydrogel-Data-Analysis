"""
AFM force curves of latex colloidal probes, glass control versus hydrogel film,
one figure per particle category (PEG-modified, BSA-modified, bare). Approach
curves only, with an inset zoomed on the contact region around F = 0.

Data files, categories, loading, baseline correction, binning and the
optional contact shift are imported from afm_force_distance_curve.py, so all
AFM figures are processed identically (see its header for details):
- Baseline: per curve, the mean force at separation < BASELINE_MAX_NM is
  subtracted, so F = 0 before contact.
- x axis: indentation delta as exported, delta = 0 at the contact point set in
  the AFM software. On glass the force rises there as a vertical wall (the
  probe cannot indent); the glass curve is never shifted, the hydrogel curve
  only if SHIFT_TO_CONTACT is True.
The glass wall (about 125-245 nN) leaves the main axes at the top; its maximum
force is annotated. The main y range is set by the hydrogel curve.

Styling follows Styleguide_Figures_Dissertation.md (v2): width class "narrow"
(4.72 in, \\includegraphics[width=0.75\\linewidth]), thesis.mplstyle, vector
PDF with rasterized data points, no bbox_inches="tight". Glass is blue and
hydrogel red (discrete series 2 and 8 of style guide §11); the particle
category is given as legend title.

Reads the raw text files directly; no cache is read or written and nothing
needs to be run first. Shows each figure (if SHOW_FIGURE), then writes into
SAVE_PATH, per category <cat> in peg / bsa / bare:
  afm_force_indentation_<cat>_glass_vs_hydrogel.pdf / .json
The .json sidecars hold files, baseline offsets, contact shifts and limits.
No other script reads these outputs.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from hydro_analysis.AFM_Agnes.afm_force_distance_curve import (
    BASELINE_MAX_NM,
    BIN_NM,
    CATEGORIES,
    DATA_DIR,
    FIG_W,
    RATIO,
    SHIFT_TO_CONTACT,
    STYLE_PATH,
    add_inset,
    add_legend,
    draw,
    inset_ylim,
    prepare_measurement,
    save_figure,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SEGMENT = "Ex"               # approach curves only
SURFACES = {                 # key: (legend label, base colour, dark colour)
    "glass":    ("Glass (control)", "#004cff", "#0035b2"),
    "hydrogel": ("Hydrogel",        "#da0000", "#990000"),
}
FIG_STEM   = "afm_force_indentation_{cat}_glass_vs_hydrogel"
SHOW_FIGURE = True

MAIN_XMIN  = -900.0          # nm; the right limit follows the hydrogel data
MAIN_YPAD  = 1.07            # main y top = MAIN_YPAD * hydrogel maximum force
INSET_XLIM = (-250.0, 100.0) # nm
ARROW_KW = dict(arrowstyle="-|>", mutation_scale=10, lw=0.8)


def build_figure(meas: dict[str, dict], title: str):
    """Glass vs hydrogel approach curves with zoom inset; call inside plt.style.context(STYLE_PATH)."""
    raw = {s: m["raw"][SEGMENT] for s, m in meas.items()}
    binned = {s: m["binned"][SEGMENT] for s, m in meas.items()}
    colors = {s: (base, dark) for s, (_, base, dark) in SURFACES.items()}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W / RATIO))
    draw(ax, raw, binned, colors)
    gel = raw["hydrogel"]
    y_top = MAIN_YPAD * gel["force_nN"].max()
    ax.set_xlim(MAIN_XMIN, gel["delta_nm"].max() + 30.0)
    ax.set_ylim(-0.05 * y_top, y_top)
    ax.set_xlabel(r"Indentation $\delta$ (nm)")
    ax.set_ylabel("Force F (nN)")
    ax.minorticks_on()
    glass_color = SURFACES["glass"][2]
    ax.annotate(f"{raw['glass']['force_nN'].max():.0f} nN", xy=(0.0, y_top),
                xytext=(40.0, 0.83 * y_top), fontsize=8, color=glass_color,
                arrowprops={**ARROW_KW, "color": glass_color})

    ylim = inset_ylim(binned, INSET_XLIM)
    add_inset(ax, raw, binned, colors, INSET_XLIM, ylim)
    add_legend(ax, {s: label for s, (label, _, _) in SURFACES.items()}, colors, title)
    return fig, ylim


def main(show: bool = SHOW_FIGURE) -> None:
    for cat, info in CATEGORIES.items():
        meas = {s: prepare_measurement(DATA_DIR / info[s], SHIFT_TO_CONTACT and s == "hydrogel")
                for s in SURFACES}
        with plt.style.context(STYLE_PATH):
            fig, ylim = build_figure(meas, info["label"])
            sidecar = {
                "segment": "approach",
                "baseline_region_nm": f"separation < {BASELINE_MAX_NM}",
                "bin_nm": BIN_NM,
                "samples": {
                    SURFACES[s][0]: {
                        "file": str(DATA_DIR / info[s]),
                        "baseline_offset_pN": round(m["offsets"][SEGMENT] * 1e3, 3),
                        "contact_shift_nm": round(m["x_contact_nm"], 2),
                        "max_force_nN": round(float(m["raw"][SEGMENT]["force_nN"].max()), 3),
                    }
                    for s, m in meas.items()
                },
                "inset_xlim_nm": INSET_XLIM,
                "inset_ylim_nN": [round(v, 3) for v in ylim],
            }
            pdf_path = save_figure(fig, FIG_STEM.format(cat=cat), sidecar, show)
        print(f"Saved {pdf_path.name}")


if __name__ == "__main__":
    main()
