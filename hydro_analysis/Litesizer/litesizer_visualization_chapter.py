"""
DLS particle-size and polydispersity figures for the results chapter (Abb35, Abb36).

Chapter variant of litesizer_visualization_all_measurements.py (appendix version,
not changed). With INCLUDE_ADDITIONAL = False only the three reference
measurements per size from REFERENCE_XLSX (Jan 2026, 20 °C) are shown; the
additional LiteSizer measurements stay in the appendix figure (E010). With
INCLUDE_ADDITIONAL = True the figures equal the appendix version.

Data handling is taken unchanged from the appendix script and
litesizer_visualization.py: group_accepted_measurements (PDI > 42 % excluded,
E007), panel A max-normalised intensity distributions, panel B d_H vs nominal
diameter with mean +- SD, PDI figure with the exclusion limit dashed.

Reads REFERENCE_XLSX (and, only with INCLUDE_ADDITIONAL = True, EXTENDED_CSV of the
appendix script); writes no pickle. Writes to SAVE_PATH (Styleguide v2, PDF, final size):
  dls_particle_size_reference_chapter.pdf
  dls_polydispersity_chapter.pdf
  dls_particle_size_reference_chapter_statistics.csv
  dls_particle_size_reference_chapter_exclusions.csv
No other script reads these outputs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import hydro_analysis.Litesizer.litesizer_visualization as lv
import hydro_analysis.Litesizer.litesizer_visualization_all_measurements as app
from hydro_analysis.Litesizer.litesizer_parser import load_litesizer_xlsx
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _DASH_THEORY, _add_log_minor_ticks

# ── Configuration ──────────────────────────────────────────────────────────────
INCLUDE_ADDITIONAL = False      # decided by user, 2026-10-14
REFERENCE_XLSX = app.REFERENCE_XLSX
SAVE_PATH = app.RESULTS / "Particle_Characterization"
FIG_STEM = "dls_particle_size_reference_chapter"
PDI_FIG_STEM = "dls_polydispersity_chapter"
SIZES = app.SIZES
MAX_PDI = app.MAX_PDI           # decided by user, 2026-10-06 (E007)

EMPTY_ADDITIONAL = pd.DataFrame(columns=["size", "T_C", "dh_nm", "pdi_pct", "D_um2_s"])


def _legend_handles(ax) -> tuple[list, list[str]]:
    """Reference points and mean +- SD; the additional-measurement markers only if shown."""
    handles, texts = app._legend_handles(ax)
    if INCLUDE_ADDITIONAL:
        return handles, texts
    return [handles[0], handles[3]], ["Reference, 20 °C", "Mean ± SD"]


def plot_panel_b(ax, summary, groups, added):
    lo, hi = lv.B_AXIS_LIM
    ax.plot([lo, hi], [lo, hi], color=lv.COLOR_REFERENCE, linestyle=_DASH_THEORY, linewidth=1.2, zorder=1)
    for row in summary.itertuples(index=False):
        s = int(row.nominal_size_nm)
        app._scatter_groups(ax, s, groups, added, app.dh_nm)
        app._mean_sd(ax, s, row.dls_diameter_mean_nm, row.dls_diameter_sd_nm)
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
    """Layout and legend of litesizer_visualization.plot_dls_reference_figure."""
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
    fig, ax = plt.subplots(figsize=app.PDI_FIG_SIZE)
    ax.axhline(MAX_PDI, color=lv.COLOR_REFERENCE, linestyle=_DASH_THEORY, linewidth=1.2, zorder=1)
    for row in summary.itertuples(index=False):
        s = int(row.nominal_size_nm)
        app._scatter_groups(ax, s, groups, added, app.pdi_pct)
        app._mean_sd(ax, s, row.pdi_mean_pct, row.pdi_sd_pct)
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
    fig.legend(handles, texts, loc="outside upper center", ncol=len(handles), frameon=False, handlelength=1.2,
               columnspacing=0.8, handletextpad=0.3)
    return fig


def main():
    data = load_litesizer_xlsx(REFERENCE_XLSX)
    groups = lv.group_accepted_measurements(data, SIZES, MAX_PDI)
    ref_mean = {s: float(np.mean([m.hydrodynamic_diameter_nm for m in ms])) for s, ms in groups.items()}
    if INCLUDE_ADDITIONAL:
        added, log = app.load_additional(ref_mean)
    else:
        added, log = EMPTY_ADDITIONAL.copy(), []
    for m in data.measurements:
        if m.polydispersity_pct is not None and m.polydispersity_pct > MAX_PDI:
            log.append(dict(file=f"{REFERENCE_XLSX.name} [{m.start_time}]", step="PDI criterion",
                            parameter=f"MAX_PDI={MAX_PDI:g}", n_before=1, n_after=0,
                            reason=f"reference '{m.name}': PDI {m.polydispersity_pct:.1f} %"))
    summary = app.summarize(groups, added)
    replicates = lv.prepare_normalized_replicates(groups)
    print(f"INCLUDE_ADDITIONAL = {INCLUDE_ADDITIONAL}; additional measurements plotted: {len(added)}")
    for s, ms in groups.items():
        print(f"  {s} nm: {len(ms)} reference measurements: "
              + "; ".join(f"{m.name} (d_H {m.hydrodynamic_diameter_nm:.1f} nm, PDI {m.polydispersity_pct:.1f} %)"
                          for m in ms))
    print("\n" + summary.round(3).to_string(index=False))

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with plt.style.context(lv.STYLE_PATH):
        fig = plot_reference_figure(replicates, summary, groups, added)
        fig.savefig(SAVE_PATH / f"{FIG_STEM}.pdf", format="pdf")   # final size, no bbox_inches="tight"
        fig_pdi = plot_pdi_figure(summary, groups, added)
        fig_pdi.savefig(SAVE_PATH / f"{PDI_FIG_STEM}.pdf", format="pdf")
    summary.to_csv(SAVE_PATH / f"{FIG_STEM}_statistics.csv", index=False)
    pd.DataFrame(log, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        SAVE_PATH / f"{FIG_STEM}_exclusions.csv", index=False, encoding="utf-8-sig")
    print(f"\nWritten to {SAVE_PATH}")


if __name__ == "__main__":
    main()
