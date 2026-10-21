"""
Zeta-potential figure for the appendix: the reference figure of litesizer_zeta_potential.py
extended by every additional zeta-potential measurement of the same particles (20-1000 nm).
The main-text figure (litesizer_zeta_potential.py) is not changed by this script.

Data
  Reference measurements: REFERENCE_XLSX (Zeta_repitition_allparticles.xlsx, Jan 2026, 20 °C),
  read with litesizer_zeta_potential.load_measurements, exactly as in the main figure.
  Additional measurements: zeta rows of the overview sheet in INVENTORY_XLSX that are not in
  REFERENCE_XLSX, written by litesizer_measurement_inventory.py (run that first). The nominal
  size is taken from the measurement name; rows without one of SIZES in the name are not
  plotted (user, 2026-10-06). Temperatures and buffers are pooled (user, 2026-10-06); the
  temperature is shown by the marker.
  ZETA_FOLDER (5 µm particles, Nov 2024) is read and every measurement in it is logged as not
  plotted: the 5 µm particles are not part of this figure (user, 2026-10-06).

Figure: same layout as the main figure (categorical nominal diameter, horizontal offsets for
visibility only, ±30 mV heuristic lines; saved as vector PDF); reference circles,
additional 20 °C diamonds, additional 25 °C triangles. No mean marker, as in the main figure.

Reads REFERENCE_XLSX, INVENTORY_XLSX and ZETA_FOLDER; writes no pickle. Writes to SAVE_PATH:
  zeta_potential_particle_stability_all_measurements.pdf
  zeta_potential_particle_stability_all_measurements_statistics.csv
  zeta_potential_particle_stability_all_measurements_values.csv
  zeta_potential_particle_stability_all_measurements_exclusions.csv
No other script reads these outputs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

import hydro_analysis.Litesizer.litesizer_zeta_potential as lzp
from hydro_analysis.Litesizer.litesizer_measurement_inventory import read_apkw
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS

# ── Configuration ──────────────────────────────────────────────────────────────
REFERENCE_XLSX = lzp.XLSX_PATH
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
INVENTORY_XLSX = RESULTS / "Particle_Characterization" / "DLS_measurement_inventory" / "dls_measurement_inventory.xlsx"
ZETA_FOLDER = Path(r"E:\Daten Promotion Sicherung\Zeta Potential")
SAVE_PATH = RESULTS / "_scratch" / "Zeta_potential_appendix"
FIG_STEM = "zeta_potential_particle_stability_all_measurements"

SIZES = [20, 50, 100, 200, 500, 1000]
REFERENCE_TEMPERATURE_C = 20.0   # target temperature of every reference sheet; checked at run time
FIG_SIZE = lzp.FIG_SIZE          # width class "narrow", as the main figure
MARKERS = {"reference": ("o", 10), 20.0: ("D", 9), 25.0: ("^", 14)}   # marker, scatter size s
LEGEND_TEXT = {"reference": "Reference, 20 °C", 20.0: "Additional, 20 °C", 25.0: "Additional, 25 °C"}


# ── Data ───────────────────────────────────────────────────────────────────────
def load_reference() -> pd.DataFrame:
    ref = lzp.load_measurements(REFERENCE_XLSX)
    if not (ref.temperature_C == REFERENCE_TEMPERATURE_C).all():
        raise ValueError(f"Reference temperatures differ from {REFERENCE_TEMPERATURE_C}: {ref.temperature_C.unique()}")
    return pd.DataFrame({"group": "reference", "nominal_size_nm": ref.nominal_size_nm.astype(int),
                         "name": ref.measurement_name, "start_local": "", "zeta_mV": ref.zeta_mV,
                         "conductivity_mS_cm": ref.conductivity_mS_cm, "T_C": ref.temperature_C,
                         "source": REFERENCE_XLSX.name + " [" + ref.source_sheet + "]"})


def load_additional(log: list[dict]) -> pd.DataFrame:
    if not INVENTORY_XLSX.is_file():
        raise FileNotFoundError(f"{INVENTORY_XLSX} missing: run litesizer_measurement_inventory.py first.")
    ov = pd.read_excel(INVENTORY_XLSX, sheet_name="overview")
    zeta = ov[(ov["mode"] == "zeta") & ~ov.in_reference_zeta_table]
    for _, r in zeta[~zeta.name_size_nm.isin(SIZES)].iterrows():
        log.append(dict(file=r.sources, step="nominal size", parameter="SIZES", n_before=1, n_after=0,
                        reason=f"{r.measurement_id} '{r.all_names}' ({r.zeta_mV:.1f} mV): no nominal size of SIZES in the name"))
    zeta = zeta[zeta.name_size_nm.isin(SIZES)]
    if zeta.T_C.isna().any():
        raise ValueError("Additional zeta measurement without temperature: " + ", ".join(zeta.loc[zeta.T_C.isna(), "name"]))
    if not zeta.T_C.isin([20.0, 25.0]).all():
        raise ValueError(f"Unexpected temperatures: {sorted(zeta.T_C.unique())}")
    return pd.DataFrame({"group": zeta.T_C, "nominal_size_nm": zeta.name_size_nm.astype(int),
                         "name": zeta.all_names, "start_local": zeta.start_local, "zeta_mV": zeta.zeta_mV,
                         "conductivity_mS_cm": zeta.conductivity_mS_cm, "T_C": zeta.T_C, "source": zeta.sources})


def log_zeta_folder(log: list[dict]) -> None:
    """Every file and measurement of ZETA_FOLDER gets a row; none of them is plotted."""
    reason = "5 µm particles, not part of this figure (user, 2026-10-06)"
    for p in sorted(x for x in ZETA_FOLDER.rglob("*") if x.is_file()):
        rel = str(p.relative_to(ZETA_FOLDER))
        if p.suffix == ".apkw":
            records, _ = read_apkw(p)
            for r in records:
                log.append(dict(file=f"{rel} [{r['source_part']}]", step="particle", parameter="5 µm",
                                n_before=1, n_after=0,
                                reason=f"{r['start_local']} '{r['name']}' ({r['mode']}"
                                       + (f", {r['zeta_mV']:.2f} mV" if r["mode"] == "zeta" else "") + f"): {reason}"))
        else:
            kind = {".xlsx": "export", ".xps": "report", ".backup": "backup"}.get(p.suffix, "file")
            if p.name.startswith("~$"):
                kind = "lock file"
            log.append(dict(file=rel, step="particle", parameter="5 µm", n_before=1, n_after=0,
                            reason=f"{kind} of the 5 µm measurements in the .apkw files: {reason}"))


def summarize(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in values.groupby("nominal_size_nm"):
        z = g.zeta_mV.to_numpy(dtype=float)
        sd = float(np.std(z, ddof=1)) if len(z) > 1 else np.nan
        cond = pd.to_numeric(g.conductivity_mS_cm, errors="coerce").dropna()
        rows.append({"nominal_size_nm": s, "n_measurements": len(z),
                     "n_reference": int((g.group == "reference").sum()),
                     "n_additional_20C": int((g.group == 20.0).sum()), "n_additional_25C": int((g.group == 25.0).sum()),
                     "zeta_mean_mV": z.mean(), "zeta_sd_mV": sd, "zeta_sem_mV": sd / np.sqrt(len(z)),
                     "zeta_min_mV": z.min(), "zeta_max_mV": z.max(),
                     "conductivity_min_mS_cm": cond.min(), "conductivity_max_mS_cm": cond.max()})
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────
def make_figure(values: pd.DataFrame, sizes: list[int]) -> plt.Figure:
    """Layout of litesizer_zeta_potential.make_figure; markers distinguish the groups."""
    with plt.style.context(lzp.STYLE_PATH):
        fig, ax = plt.subplots(figsize=FIG_SIZE, layout="constrained")
        ax.axhline(0, color="0.65", linewidth=0.6, zorder=0)
        low, high = min(0.0, values.zeta_mV.min()), max(0.0, values.zeta_mV.max())
        pad = max(4.0, (high - low) * 0.10)
        ax.set_ylim(low - pad, high + pad)
        for reference in (-30, 30):
            if low - pad < reference < high + pad:
                ax.axhline(reference, color="0.65", linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)
        for x, s in enumerate(sizes):
            base, dark = SIZE_COLORS[float(s)]
            g = values[values.nominal_size_nm == s].reset_index(drop=True)
            left = len(g) // 2
            offsets = np.concatenate((np.linspace(-0.21, -0.12, left), np.linspace(0.12, 0.21, len(g) - left)))
            offsets = np.random.default_rng(1729 + x).permutation(offsets)   # same scheme as the main figure
            for key, (marker, size) in MARKERS.items():
                sel = (g.group == key).to_numpy()
                ax.scatter(x + offsets[sel], g.zeta_mV[sel], s=size, marker=marker, facecolors=base, alpha=0.65,
                           edgecolors=dark, linewidths=0.45, zorder=3)
        ax.set_xticks(np.arange(len(sizes)), [f"{s:g}" for s in sizes])
        ax.set_xlim(-0.5, len(sizes) - 0.5)
        ax.set_xlabel("Nominal particle diameter (nm)")
        ax.set_ylabel(r"Zeta potential, $\zeta$ (mV)")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        handles = [Line2D([], [], marker=m, markersize=3.5, linestyle="None", markerfacecolor="#bbbbbb",
                          markeredgecolor="#555555", markeredgewidth=0.45, alpha=0.65) for m, _ in MARKERS.values()]
        ax.legend(handles, list(LEGEND_TEXT.values()), loc="lower left", frameon=False, handlelength=1.0,
                  borderaxespad=0.3, fontsize=7, labelspacing=0.3)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    log: list[dict] = []
    ref = load_reference()
    added = load_additional(log)
    log_zeta_folder(log)
    values = pd.concat([ref, added], ignore_index=True)
    sizes = [s for s in SIZES if s in set(values.nominal_size_nm)]
    summary = summarize(values)

    print("Additional measurements plotted")
    print(added[["start_local", "name", "nominal_size_nm", "zeta_mV", "conductivity_mS_cm", "T_C"]]
          .round(2).to_string(index=False))
    excl = pd.DataFrame(log, columns=["file", "step", "parameter", "n_before", "n_after", "reason"])
    print(f"\nNot plotted: {len(excl)} rows "
          f"({int(excl.reason.str.contains(r'\(zeta').sum())} zeta measurements in {ZETA_FOLDER.name}, "
          f"{int((excl.step == 'nominal size').sum())} without nominal size)")
    print("\n" + summary.round(2).to_string(index=False))

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    fig = make_figure(values, sizes)
    with plt.style.context(lzp.STYLE_PATH):
        fig.savefig(SAVE_PATH / f"{FIG_STEM}.pdf")
    plt.close(fig)
    summary.to_csv(SAVE_PATH / f"{FIG_STEM}_statistics.csv", index=False)
    values.to_csv(SAVE_PATH / f"{FIG_STEM}_values.csv", index=False, encoding="utf-8-sig")
    excl.to_csv(SAVE_PATH / f"{FIG_STEM}_exclusions.csv", index=False, encoding="utf-8-sig")
    print(f"\nWritten to {SAVE_PATH}")


if __name__ == "__main__":
    main()
