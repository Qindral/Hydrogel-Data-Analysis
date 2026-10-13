"""
DLS particle-size figure for the appendix: the reference figure of litesizer_visualization.py
extended by every additional LiteSizer size measurement, plus a polydispersity figure.
The main-text figure (litesizer_visualization.py) is not changed by this script.

Data
  Reference measurements: REFERENCE_XLSX (Jan 2026, 20 °C), grouped and PDI-filtered with
  litesizer_visualization.group_accepted_measurements (PDI > 42 % excluded, E007).
  Additional measurements: rows with origin == "added" in dls_size_table_extended.csv, written by
  litesizer_measurement_inventory.py (run that first). That table only contains additional rows
  with PDI <= 42 %. The nominal size is taken from the measurement name; rows whose name holds
  none of SIZES are not plotted.
  Temperature of additional rows: the instrument setting where it is recorded; where it is not
  (CSV-only exports), 20 °C if the name says so, otherwise 25 °C, the instrument default
  (stated by user, 2026-10-06). Temperatures and solvent models are pooled (user, 2026-10-06).
  1000 nm: additional measurements whose d_H deviates by more than OUTLIER_WINDOW_NM from the
  mean d_H of the accepted reference measurements are excluded (user, 2026-10-06).

Figures (all at final printed size, PDF, no bbox_inches="tight"; style guide v2):
  A  max-normalised intensity distributions, reference measurements only (unchanged from the
     main figure; CSV-only exports contain no distribution)
  B  d_H vs nominal diameter: reference circles, additional 20 °C diamonds, additional 25 °C
     triangles, mean +-SD over all plotted measurements
  PDI figure: PDI of every plotted measurement, same markers, exclusion limit dashed.

Reads REFERENCE_XLSX and EXTENDED_CSV; writes no pickle. Writes to SAVE_PATH:
  dls_particle_size_reference_all_measurements.pdf
  dls_polydispersity_all_measurements.pdf
  dls_particle_size_reference_all_measurements_statistics.csv
  dls_particle_size_reference_all_measurements_exclusions.csv
No other script reads these outputs.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import hydro_analysis.Litesizer.litesizer_visualization as lv
from hydro_analysis.Litesizer.litesizer_parser import load_litesizer_xlsx
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY, _add_log_minor_ticks

# ── Configuration ──────────────────────────────────────────────────────────────
REFERENCE_XLSX = Path(r"E:\Daten Promotion Sicherung\Lite Sizer Particle Measurements\Size_repitition_All Sizes.xlsx")
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
EXTENDED_CSV = RESULTS / "DLS_measurement_inventory" / "dls_size_table_extended.csv"
SAVE_PATH = RESULTS / "DLS_particle_size_reference_appendix"
FIG_STEM = "dls_particle_size_reference_all_measurements"
PDI_FIG_STEM = "dls_polydispersity_all_measurements"
PREVIEW_PATH: Path | None = RESULTS / "_scratch" / "DLS_particle_size_reference_appendix"  # PNG check copies

SIZES = lv.SIZES
MAX_PDI = 42.0                  # decided by user, 2026-10-06 (E007)
DEFAULT_TEMPERATURE_C = 25.0    # stated by user, 2026-10-06: instrument default when not recorded
OUTLIER_SIZE_NM = 1000          # decided by user, 2026-10-06
OUTLIER_WINDOW_NM = 400.0       # decided by user, 2026-10-06: |d_H - reference mean| <= 400 nm kept

MARKERS = {"reference": ("o", 9), 20.0: ("D", 8), 25.0: ("^", 12)}   # marker, scatter size s
LEGEND_TEXT = {"reference": "Reference, 20 °C", 20.0: "Additional, 20 °C", 25.0: "Additional, 25 °C"}
PDI_FIG_SIZE = (4.72, 3.32)     # width class "narrow"


# ── Data ───────────────────────────────────────────────────────────────────────
def load_additional(ref_mean: dict[int, float]) -> tuple[pd.DataFrame, list[dict]]:
    """Additional measurements to plot, and one exclusion-log row per measurement left out."""
    if not EXTENDED_CSV.is_file():
        raise FileNotFoundError(f"{EXTENDED_CSV} missing: run litesizer_measurement_inventory.py first.")
    table = pd.read_csv(EXTENDED_CSV)
    added = table[table.origin == "added"].copy()
    log = []

    def drop(mask, step, parameter, reason):
        for _, r in added[mask].iterrows():
            log.append(dict(file=r.sources, step=step, parameter=parameter, n_before=1, n_after=0,
                            reason=f"{r.measurement_id} '{r['name']}': {reason(r)}"))
        return added[~mask]

    added = drop(added.pdi_pct > MAX_PDI, "PDI criterion", f"MAX_PDI={MAX_PDI:g}",
                 lambda r: f"PDI {r.pdi_pct:.1f} %")
    added = drop(~added.name_size_nm.isin(SIZES), "nominal size", "SIZES",
                 lambda r: "no nominal size of SIZES in the name")
    added["size"] = added.name_size_nm.astype(int)
    window = (added["size"] == OUTLIER_SIZE_NM) & (
        (added.dh_nm - ref_mean[OUTLIER_SIZE_NM]).abs() > OUTLIER_WINDOW_NM)
    added = drop(window, "1000 nm outlier", f"|d_H - {ref_mean[OUTLIER_SIZE_NM]:.1f}| > {OUTLIER_WINDOW_NM:g} nm",
                 lambda r: f"d_H {r.dh_nm:.1f} nm")

    added["temperature_source"] = np.where(added.T_C.notna(), "instrument", "")
    named_20 = added.T_C.isna() & added["name"].str.contains(r"20\s*°C", flags=re.IGNORECASE)
    added.loc[named_20, ["T_C", "temperature_source"]] = [20.0, "measurement name"]
    unknown = added.T_C.isna()
    added.loc[unknown, ["T_C", "temperature_source"]] = [DEFAULT_TEMPERATURE_C, "instrument default (user)"]
    if not added.T_C.isin([20.0, 25.0]).all():
        raise ValueError(f"Unexpected temperatures: {sorted(added.T_C.unique())}")
    return added, log


def summarize(groups, added) -> pd.DataFrame:
    rows = []
    for s in SIZES:
        ref = groups.get(s, [])
        new = added[added["size"] == s]
        dh = [m.hydrodynamic_diameter_nm for m in ref] + list(new.dh_nm)
        pdi = [m.polydispersity_pct for m in ref] + list(new.pdi_pct)
        diff = [m.diffusion_coefficient_um2s for m in ref if m.diffusion_coefficient_um2s is not None] \
            + list(new.D_um2_s.dropna())
        if not dh:
            continue
        n = len(dh)
        sd = np.std(dh, ddof=1) if n > 1 else np.nan
        rows.append({
            "nominal_size_nm": s, "n_measurements": n, "n_reference": len(ref),
            "n_additional_20C": int((new.T_C == 20.0).sum()), "n_additional_25C": int((new.T_C == 25.0).sum()),
            "dls_diameter_mean_nm": np.mean(dh), "dls_diameter_sd_nm": sd, "dls_diameter_sem_nm": sd / np.sqrt(n),
            "n_diffusion": len(diff), "diffusion_mean_um2s": np.mean(diff) if diff else np.nan,
            "diffusion_sd_um2s": np.std(diff, ddof=1) if len(diff) > 1 else np.nan,
            "pdi_mean_pct": np.mean(pdi), "pdi_sd_pct": np.std(pdi, ddof=1) if n > 1 else np.nan,
        })
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────
def _scatter_groups(ax, s, groups, added, value):
    """Reference and additional points of one size at the nominal x value (no jitter)."""
    base, dark = lv._size_colors(s)
    sets = [("reference", [value(m) for m in groups.get(s, [])])]
    for t in (20.0, 25.0):
        sets.append((t, list(added.loc[(added["size"] == s) & (added.T_C == t), value.__name__])))
    for key, vals in sets:
        marker, size = MARKERS[key]
        ax.scatter(np.full(len(vals), s), vals, s=size, marker=marker, alpha=lv.POINT_ALPHA,
                   facecolor=base, edgecolor=dark, linewidth=0.6, zorder=2)


def _mean_sd(ax, s, mean, sd):
    base, dark = lv._size_colors(s)
    ax.errorbar(s, mean, yerr=None if np.isnan(sd) else sd, fmt="s", markersize=4, markerfacecolor=base,
                markeredgecolor=dark, markeredgewidth=0.6, ecolor=dark, elinewidth=0.8, capsize=2.0,
                capthick=0.8, linestyle="None", zorder=3)


def _legend_handles(ax):
    grey_face, grey_edge = "#bbbbbb", "#555555"
    handles = [Line2D([], [], marker=MARKERS[k][0], markersize=3.5, linestyle="None", markerfacecolor=grey_face,
                      markeredgecolor=grey_edge, markeredgewidth=0.6, alpha=lv.POINT_ALPHA)
               for k in ("reference", 20.0, 25.0)]
    handles.append(ax.errorbar([], [], yerr=[], fmt="s", markersize=4, markerfacecolor=grey_face,
                               markeredgecolor=grey_edge, markeredgewidth=0.6, ecolor=grey_edge,
                               elinewidth=0.8, capsize=2.0, capthick=0.8, linestyle="None"))
    texts = [LEGEND_TEXT[k] for k in ("reference", 20.0, 25.0)] + ["Mean ± SD (all)"]
    return handles, texts


def dh_nm(m):
    return m.hydrodynamic_diameter_nm


def pdi_pct(m):
    return m.polydispersity_pct


def plot_panel_b(ax, summary, groups, added):
    lo, hi = lv.B_AXIS_LIM
    ax.plot([lo, hi], [lo, hi], color=lv.COLOR_REFERENCE, linestyle=_DASH_THEORY, linewidth=1.2, zorder=1)
    for row in summary.itertuples(index=False):
        s = int(row.nominal_size_nm)
        _scatter_groups(ax, s, groups, added, dh_nm)
        _mean_sd(ax, s, row.dls_diameter_mean_nm, row.dls_diameter_sd_nm)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    _add_log_minor_ticks(ax)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator(lv.B_TICKS))
        axis.set_major_formatter(mticker.FixedFormatter([str(t) for t in lv.B_TICKS]))
    ax.set_xlabel("Nominal diameter (nm)")
    ax.set_ylabel(r"DLS diameter $d_\mathrm{H}$ (nm)")
    handles, texts = _legend_handles(ax)
    handles.append(Line2D([], [], color=lv.COLOR_REFERENCE, linestyle=_DASH_THEORY, linewidth=1.2))
    texts.append(r"$d_\mathrm{H}$ = nominal")
    ax.legend(handles, texts, loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.3,
              fontsize=7, labelspacing=0.3)


def plot_reference_figure(replicates, summary, groups, added) -> plt.Figure:
    """Same layout and legend as litesizer_visualization.plot_dls_reference_figure."""
    sizes = list(summary.nominal_size_nm)
    labels = lv._particle_labels(sizes)
    fig, ax_a, ax_b = lv._create_panel_axes()
    lv.plot_normalized_dls_distributions(ax_a, replicates)
    plot_panel_b(ax_b, summary, groups, added)
    lv._panel_label(ax_a, "A")
    lv._panel_label(ax_b, "B")
    handles = [Line2D([], [], color=lv._size_colors(s)[1], linewidth=1.2) for s in sizes]
    texts = [labels[s] for s in sizes]
    handles.append(Line2D([], [], color="#555555", linestyle=(0, (1, 2)), linewidth=0.8))
    texts.append("Nominal diameter")
    x0, x1 = ax_a.get_position().x0, ax_b.get_position().x1
    fig.legend(handles, texts, loc="lower center", bbox_to_anchor=((x0 + x1) / 2, ax_a.get_position().y1 + 0.005),
               ncol=len(handles), frameon=False, handlelength=1.5, columnspacing=1.0, handletextpad=0.4,
               borderaxespad=0.0, borderpad=0.2)
    return fig


def plot_pdi_figure(summary, groups, added) -> plt.Figure:
    labels = lv._particle_labels(SIZES)
    fig, ax = plt.subplots(figsize=PDI_FIG_SIZE)
    ax.axhline(MAX_PDI, color=lv.COLOR_REFERENCE, linestyle=_DASH_THEORY, linewidth=1.2, zorder=1)
    for row in summary.itertuples(index=False):
        s = int(row.nominal_size_nm)
        _scatter_groups(ax, s, groups, added, pdi_pct)
        _mean_sd(ax, s, row.pdi_mean_pct, row.pdi_sd_pct)
    ax.set_xscale("log")
    ax.set_xlim(13, 1600)
    ax.set_ylim(0, 50)
    ax.xaxis.set_major_locator(mticker.FixedLocator(SIZES))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([labels[s].replace(" nm", "") for s in SIZES]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.set_xlabel("Particle size (nm)")
    ax.set_ylabel("Polydispersity index PDI (%)")
    handles, texts = _legend_handles(ax)
    handles.append(Line2D([], [], color=lv.COLOR_REFERENCE, linestyle=_DASH_THEORY, linewidth=1.2))
    texts.append(f"Exclusion limit {MAX_PDI:g} %")
    order = [0, 3, 1, 4, 2]  # matplotlib fills columns first; this reads row-wise in three columns
    handles, texts = [handles[i] for i in order], [texts[i] for i in order]
    fig.legend(handles, texts, loc="outside upper center", ncol=3, frameon=False, handlelength=1.2,
               columnspacing=0.8, handletextpad=0.3)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    data = load_litesizer_xlsx(REFERENCE_XLSX)
    groups = lv.group_accepted_measurements(data, SIZES, MAX_PDI)
    ref_mean = {s: float(np.mean([m.hydrodynamic_diameter_nm for m in ms])) for s, ms in groups.items()}
    added, log = load_additional(ref_mean)
    for m in data.measurements:
        if m.polydispersity_pct is not None and m.polydispersity_pct > MAX_PDI:
            log.append(dict(file=f"{REFERENCE_XLSX.name} [{m.start_time}]", step="PDI criterion",
                            parameter=f"MAX_PDI={MAX_PDI:g}", n_before=1, n_after=0,
                            reason=f"reference '{m.name}': PDI {m.polydispersity_pct:.1f} %"))
    summary = summarize(groups, added)
    replicates = lv.prepare_normalized_replicates(groups)
    lo, hi = lv.B_AXIS_LIM
    if ((added.dh_nm < lo) | (added.dh_nm > hi)).any():
        raise ValueError("Additional d_H outside panel B limits: " + added.loc[(added.dh_nm < lo) | (added.dh_nm > hi), "name"].str.cat(sep=", "))

    print("\nAdditional measurements plotted")
    print(added[["measurement_id", "start_local", "name", "size", "dh_nm", "pdi_pct", "T_C", "temperature_source"]]
          .round(1).to_string(index=False))
    print("\nExcluded")
    print(pd.DataFrame(log)[["step", "reason"]].to_string(index=False))
    print("\n" + summary.round(3).to_string(index=False))

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with plt.style.context(lv.STYLE_PATH):
        fig = plot_reference_figure(replicates, summary, groups, added)
        fig.savefig(SAVE_PATH / f"{FIG_STEM}.pdf", format="pdf")   # final size, no bbox_inches="tight"
        fig_pdi = plot_pdi_figure(summary, groups, added)
        fig_pdi.savefig(SAVE_PATH / f"{PDI_FIG_STEM}.pdf", format="pdf")
        if PREVIEW_PATH is not None:
            PREVIEW_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(PREVIEW_PATH / f"{FIG_STEM}_preview.png", dpi=200)
            fig_pdi.savefig(PREVIEW_PATH / f"{PDI_FIG_STEM}_preview.png", dpi=200)
    summary.to_csv(SAVE_PATH / f"{FIG_STEM}_statistics.csv", index=False)
    pd.DataFrame(log, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        SAVE_PATH / f"{FIG_STEM}_exclusions.csv", index=False, encoding="utf-8-sig")
    print(f"\nWritten to {SAVE_PATH}")


if __name__ == "__main__":
    main()
