"""
AFM force curves (approach and retract) of latex colloidal probes, one figure
per measurement, each with an inset zoomed on the contact region around F = 0.

Data (Agnes, Igor text exports in DATA_DIR; CR line endings, header row, four
columns Sep/Force for retract "Re" and approach "Ex", in m and N). There are
three particle categories, each measured once on the hydrogel film and once
on glass as control (CATEGORIES). The files "... 2.txt" and "peg-mod_2.txt"
are byte-identical copies of their "1" counterparts and are not used.

On the hydrogel the x axis is the indentation depth delta exported by the AFM
software (SepCPH columns): delta = 0 is the contact point set there, delta < 0
the non-contact region, delta > 0 the indentation into the hydrogel. On glass
the exported separation (Sep columns) puts contact at 0, where the force rises
as a vertical wall; the probe cannot indent, so the x axis is labelled as
distance and ends at about 0.

Baseline correction: for each segment the mean force of the non-contact
region (separation < BASELINE_MAX_NM) is subtracted, so that F = 0 wherever
the probe is not yet in contact with the sample. A constant offset is used
because a linear fit of that region shows no relevant tilt in any file.

Optional contact shift (SHIFT_TO_CONTACT, default False, hydrogel only): at
the exported contact point the force is already 0.1-0.2 nN; the rise out of
the baseline starts roughly 120 nm earlier. With SHIFT_TO_CONTACT = True, x = 0
is moved to that onset: the binned approach force is taken from the point
where it last rises above CONTACT_SIGMA times the binned baseline noise up to
CONTACT_FMAX_NN, a straight line fitted to the raw points in that window is
extrapolated back to F = 0, and the same shift is applied to the retract
curve. A power-law contact model (F ~ (x - x0)^n) was not used for this
because its x0 moved by more than 70 nm with the fitted force range.

Each raw data point is drawn as a small rasterized dot; on top, the mean force
in BIN_NM separation bins is drawn as a line, because the single-point noise
(30-70 pN) would otherwise hide the onset of contact in the inset.

Styling follows Styleguide_Figures_Dissertation.md (v2): width class "narrow"
(4.72 in, \\includegraphics[width=0.75\\linewidth]), thesis.mplstyle, vector
PDF with rasterized data points, no bbox_inches="tight". Approach is blue and
retract red (discrete series 2 and 8 of style guide §11), as in the original
AFM software plots. The particle category is given as legend title.

Reads the raw text files directly; no cache is read or written and nothing
needs to be run first. Shows each figure (if SHOW_FIGURE), then writes into
SAVE_PATH, per category <cat> in peg / bsa / bare:
  afm_force_indentation_<cat>_hydrogel.pdf / .json
  afm_force_distance_<cat>_glass.pdf / .json
The .json sidecars hold file, baseline offsets, contact shift and plot limits.
afm_force_distance_glass_vs_hydrogel.py imports the loading and correction
functions and CATEGORIES from this script; no script reads its outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = Path(r"E:\Daten Promotion Sicherung\Daten Agnes")
CATEGORIES = {   # key: legend title, hydrogel file, glass control file
    "peg":  {"label": "PEG-modified particle", "hydrogel": "peg-mod_1.txt",
             "glass": "Kontrolle PEG 1.txt"},
    "bsa":  {"label": "BSA-modified particle", "hydrogel": "BSA.txt",
             "glass": "Kontrolle BSA 1.txt"},
    "bare": {"label": "Bare particle",         "hydrogel": "bare latex particles.txt",
             "glass": "Kontrolle bare 1.txt"},
}
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "AFM_force_distance"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
SHOW_FIGURE = True

BASELINE_MAX_NM = -400.0     # separation below this is taken as non-contact baseline
BIN_NM          = 5.0        # bin width of the averaged force line
SHIFT_TO_CONTACT = False     # False: keep the exported contact point at x = 0
CONTACT_SIGMA   = 3.0        # lower end of the contact-fit window, in binned baseline SD
CONTACT_FMAX_NN = 0.10       # upper end of the contact-fit window (nN)

SURFACES = {   # x label, main x limits (None: data range), inset x limits (nm), file stem
    "hydrogel": {"xlabel": r"Indentation $\delta$ (nm)", "xlim": None,
                 "inset_xlim": (-250.0, 100.0), "stem": "afm_force_indentation_{cat}_hydrogel"},
    "glass":    {"xlabel": "Distance (nm)", "xlim": (-900.0, 100.0),
                 "inset_xlim": (-250.0, 50.0), "stem": "afm_force_distance_{cat}_glass"},
}
INSET_YTOP   = 0.60          # nN; the inset bottom follows the lowest binned force
INSET_BOUNDS = (0.09, 0.42, 0.44, 0.50)   # axes fraction: x0, y0, width, height
LEGEND_ANCHOR = (0.07, 0.10)             # axes fraction, free area below the inset

FIG_W, RATIO = 4.72, 1.42    # width class "narrow"

SEGMENTS = {                 # key: (legend label, base colour, dark colour)
    "Ex": ("Approach", "#004cff", "#0035b2"),
    "Re": ("Retract",  "#da0000", "#990000"),
}
RAW_ALPHA   = 0.15
RAW_SIZE    = 0.8
ZERO_COLOR  = "0.55"


def load_segments(path: Path) -> dict[str, pd.DataFrame]:
    """Return {segment: DataFrame(sep_nm, force_nN)} with NaN padding removed.

    Accepts both export variants, <id>_SepCPH_<seg> and <id>_Sep_<seg>.
    """
    raw = pd.read_csv(path, sep=r"\s+", lineterminator="\r")
    segments = {}
    for seg in SEGMENTS:
        sep_col = next(c for c in raw.columns if "_Sep" in c and c.endswith(f"_{seg}"))
        force_col = next(c for c in raw.columns if c.endswith(f"_Force_{seg}"))
        df = raw[[sep_col, force_col]].dropna()
        df.columns = ["sep_nm", "force_nN"]
        segments[seg] = df * 1e9
    return segments


def subtract_baseline(df: pd.DataFrame, max_sep_nm: float) -> tuple[pd.DataFrame, float]:
    """Shift the force so that the non-contact region averages to zero."""
    offset = float(df.loc[df["sep_nm"] < max_sep_nm, "force_nN"].mean())
    out = df.copy()
    out["force_nN"] -= offset
    return out, offset


def binned_mean(df: pd.DataFrame, bin_nm: float) -> pd.DataFrame:
    """Mean separation and force per separation bin, sorted by separation."""
    edges = np.arange(df["sep_nm"].min(), df["sep_nm"].max() + bin_nm, bin_nm)
    bins = pd.cut(df["sep_nm"], edges, include_lowest=True).rename("bin")
    means = df.groupby(bins, observed=True).mean().dropna()
    return means.reset_index(drop=True).sort_values("sep_nm")


def contact_point(df: pd.DataFrame, binned: pd.DataFrame) -> tuple[float, tuple[float, float]]:
    """Separation where the initial rise of the approach force extrapolates to F = 0.

    Returns the contact point and the separation window used for the linear fit.
    """
    x, F = binned["sep_nm"].to_numpy(), binned["force_nN"].to_numpy()
    f_lo = CONTACT_SIGMA * F[x < BASELINE_MAX_NM].std()
    i_lo = len(F) - np.argmax((F <= f_lo)[::-1])          # start of the sustained rise
    i_hi = i_lo + np.argmax(F[i_lo:] >= CONTACT_FMAX_NN)
    x_lo, x_hi = float(x[i_lo]), float(x[i_hi])
    window = df[(df["sep_nm"] >= x_lo) & (df["sep_nm"] <= x_hi)]
    slope, intercept = np.polyfit(window["sep_nm"], window["force_nN"], 1)
    return float(-intercept / slope), (x_lo, x_hi)


def prepare_measurement(path: Path, shift: bool) -> dict:
    """Load both segments, subtract baselines, bin, and add the plotted x column delta_nm."""
    corrected, binned, offsets = {}, {}, {}
    for seg, df in load_segments(path).items():
        corrected[seg], offsets[seg] = subtract_baseline(df, BASELINE_MAX_NM)
        binned[seg] = binned_mean(corrected[seg], BIN_NM)
    x_contact, fit_window = 0.0, None
    if shift:
        x_contact, fit_window = contact_point(corrected["Ex"], binned["Ex"])
    for data in (corrected, binned):
        for df in data.values():
            df["delta_nm"] = df["sep_nm"] - x_contact
    return {"raw": corrected, "binned": binned, "offsets": offsets,
            "x_contact_nm": x_contact, "fit_window_nm": fit_window}


def inset_ylim(binned: dict[str, pd.DataFrame], xlim: tuple[float, float]) -> tuple[float, float]:
    """Fixed top; bottom low enough for the lowest binned force (e.g. adhesion) in the inset."""
    f_min = min(df.loc[df["delta_nm"].between(*xlim), "force_nN"].min() for df in binned.values())
    return min(-0.1, 1.15 * f_min), INSET_YTOP


def draw(ax, raw: dict[str, pd.DataFrame], binned: dict[str, pd.DataFrame],
         colors: dict[str, tuple[str, str]]) -> None:
    """Zero lines, raw points (rasterized) and binned mean lines; colors: {key: (base, dark)}."""
    ax.axhline(0.0, color=ZERO_COLOR, lw=0.6, zorder=1)
    ax.axvline(0.0, color=ZERO_COLOR, lw=0.6, zorder=1)
    for key, (base, _) in colors.items():
        ax.scatter(raw[key]["delta_nm"], raw[key]["force_nN"], s=RAW_SIZE, color=base,
                   alpha=RAW_ALPHA, linewidths=0, rasterized=True, zorder=2)
    for key, (_, dark) in colors.items():
        ax.plot(binned[key]["delta_nm"], binned[key]["force_nN"], color=dark, lw=1.2, zorder=3)


def add_inset(ax, raw, binned, colors, xlim, ylim):
    """Zoom inset in the upper left, connected to its region in the main axes."""
    axins = ax.inset_axes(INSET_BOUNDS)
    draw(axins, raw, binned, colors)
    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_xticks([t for t in (-200, -100, 0, 100) if xlim[0] <= t <= xlim[1]])
    axins.minorticks_on()
    axins.tick_params(which="both", top=True, right=True, direction="in")
    ax.indicate_inset_zoom(axins, edgecolor="0.3", linewidth=0.6, alpha=1.0)
    return axins


def add_legend(ax, labels: dict[str, str], colors: dict[str, tuple[str, str]], title: str) -> None:
    handles = [Line2D([], [], color=colors[key][1], lw=1.2, label=label)
               for key, label in labels.items()]
    ax.legend(handles=handles, title=title, title_fontsize=8, loc="lower left",
              bbox_to_anchor=LEGEND_ANCHOR, alignment="left")


def build_figure(meas: dict, surface: str, title: str):
    """Approach/retract figure with zoom inset; call inside plt.style.context(STYLE_PATH)."""
    cfg = SURFACES[surface]
    colors = {seg: (base, dark) for seg, (_, base, dark) in SEGMENTS.items()}
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W / RATIO))
    draw(ax, meas["raw"], meas["binned"], colors)
    if cfg["xlim"] is not None:
        ax.set_xlim(*cfg["xlim"])
    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("Force F (nN)")
    ax.minorticks_on()
    ylim = inset_ylim(meas["binned"], cfg["inset_xlim"])
    add_inset(ax, meas["raw"], meas["binned"], colors, cfg["inset_xlim"], ylim)
    add_legend(ax, {seg: label for seg, (label, _, _) in SEGMENTS.items()}, colors, title)
    return fig, ylim


def save_figure(fig, stem: str, sidecar: dict, show: bool) -> Path:
    """Show (optional), then save as PDF at final size plus a JSON sidecar of the same name."""
    if show:
        plt.show()
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    pdf_path = SAVE_PATH / f"{stem}.pdf"
    fig.savefig(pdf_path, format="pdf")   # final size, no bbox_inches="tight" (style guide §3)
    plt.close(fig)
    (SAVE_PATH / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))
    return pdf_path


def main(show: bool = SHOW_FIGURE) -> None:
    for cat, info in CATEGORIES.items():
        for surface in SURFACES:
            path = DATA_DIR / info[surface]
            meas = prepare_measurement(path, SHIFT_TO_CONTACT and surface == "hydrogel")
            with plt.style.context(STYLE_PATH):
                fig, ylim = build_figure(meas, surface, info["label"])
                stem = SURFACES[surface]["stem"].format(cat=cat)
                sidecar = {
                    "data_path": str(path),
                    "baseline_region_nm": f"separation < {BASELINE_MAX_NM}",
                    "baseline_offset_pN": {SEGMENTS[s][0]: round(v * 1e3, 3)
                                           for s, v in meas["offsets"].items()},
                    "contact_shift_nm": round(meas["x_contact_nm"], 2),
                    "contact_fit_window_separation_nm": meas["fit_window_nm"],
                    "max_force_nN": round(max(df["force_nN"].max() for df in meas["raw"].values()), 3),
                    "min_binned_force_nN": round(min(df["force_nN"].min() for df in meas["binned"].values()), 3),
                    "bin_nm": BIN_NM,
                    "inset_xlim_nm": SURFACES[surface]["inset_xlim"],
                    "inset_ylim_nN": [round(v, 3) for v in ylim],
                }
                pdf_path = save_figure(fig, stem, sidecar, show)
            print(f"Saved {pdf_path.name}")


if __name__ == "__main__":
    main()
