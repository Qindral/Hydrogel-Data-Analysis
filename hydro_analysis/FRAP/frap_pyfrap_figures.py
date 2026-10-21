"""
FRAP, PyFRAP alternative figures (E018, user 2026-10-08): the D figures of frap_depth_figures.py
with D from PyFRAP instead of the propagation method, and one figure comparing both methods.
Consumer stage: reads
  cache/frap_pyfrap_result_<version>.pkl  written by frap_pyfrap_fit.py (versions 2d, thin3d, slab03):
                                {"results": DataFrame with measurement_id, D_pyfrap_um2_s, qc_pyfrap, ...,
                                "curves": pinned data and fitted curves per measurement}
  cache/frap_v2_phase1.pkl      via frap_depth_figures.load_values (D_propagation, depth, QC), which
                                also reads the z position from the Leica MetaData XML (read-only).
Nothing is refitted. Measurements, depth assignment and exclusions (water_2) are those of
frap_depth_figures.py. D0 for PyFRAP = arithmetic mean of the QC-passed deep water PyFRAP values,
as for the propagation method (user, 2026-10-06). The propagation figures stay unchanged.

Figures (FIGURE_DIR/<version>/ for every version with a cache, PDF, each with a sidecar JSON):
  frap_pyfrap_d_vs_depth.pdf        (narrow) D_PyFRAP of every measurement, open symbols fail qc_pyfrap
  frap_pyfrap_deff_d0_vs_depth.pdf  (narrow) D_eff/D0 from PyFRAP, mean +- SD (ddof = 1) of the QC-passed
  frap_method_comparison.pdf        (full) A: D_PyFRAP against D_propagation per measurement, line y = x;
                                    B: D_eff/D0 mean +- SD of both methods side by side
and in FIGURE_DIR, all versions together (user, 2026-10-08: show the differences):
  frap_pyfrap_versions_comparison.pdf  (full) A: D, B: D_eff/D0, mean +- SD of the QC-passed measurements
                                    per group for propagation and every PyFRAP version; C: pinned data and
                                    PyFRAP fits of all versions for EXAMPLE_ID
Also writes frap_pyfrap_values.csv (per version folder), frap_pyfrap_versions_values.csv and the
exclusion CSVs. No other script reads them.
test_with_synthetic_result() runs the figures on invented PyFRAP values into a given folder; it is a
test only and never writes to FIGURE_DIR.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from hydro_analysis.FRAP.frap_depth_figures import (
    COLORS, D_COLUMN, EXCLUDE_IDS, FIG_SIZE, MARKERS, RATIO_SUB_OFFSET, STYLE_PATH, X_POS, _points, load_values)

# ── Configuration ──────────────────────────────────────────────────────────────
PYFRAP_CACHE = Path(__file__).resolve().parent / "cache" / "frap_pyfrap_result_{version}.pkl"
VERSIONS = {"2d": "PyFRAP 2D", "thin3d": "PyFRAP 1 µm", "slab03": "PyFRAP 0.3 µm in 1 µm"}
VERSION_FILL = {"propagation": "full", "2d": "bottom", "thin3d": "left", "slab03": "top"}
VERSION_OFFSET = {"propagation": -0.15, "2d": -0.05, "thin3d": 0.05, "slab03": 0.15}
VERSION_LINE = {"2d": "-", "thin3d": (0, (1, 1)), "slab03": (0, (5, 1, 1, 1))}
EXAMPLE_ID = "gel_B1_1"           # example for the fitted curves (chosen by Claude)
GROUP_X = {("Water", "deep"): 0, ("Hydrogel", "shallow"): 1, ("Hydrogel", "deep"): 2}
FIGURE_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\FRAP\FRAP_v2\figures_pyfrap")
FULL_SIZE = (6.30, 6.30 / 1.42 / 1.6)     # width class "full", two panels in one row
METHOD_LABELS = {"propagation": "Propagation", "pyfrap": "PyFRAP"}
METHOD_OFFSET = {"propagation": -0.06, "pyfrap": 0.06}   # panel B: fixed sub-position per method, not jitter
METHOD_FILL = {"propagation": "full", "pyfrap": "bottom"}  # panel B: filled = propagation, half-filled = PyFRAP
                                                         # (open symbols mean "fails QC" in panel A)


# ── Data ───────────────────────────────────────────────────────────────────────
def load_pyfrap(version: str) -> tuple[pd.DataFrame, dict]:
    path = Path(str(PYFRAP_CACHE).format(version=version))
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run frap_pyfrap_fit.py (PyFRAP environment) first.")
    cache = pickle.loads(path.read_bytes())
    res = cache["results"]
    missing = {"measurement_id", "D_pyfrap_um2_s", "qc_pyfrap"} - set(res.columns)
    if missing:
        raise ValueError(f"{path}: columns missing: {sorted(missing)}")
    return res, cache.get("curves", {})


def merge(df: pd.DataFrame, pyfrap: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Propagation table plus the PyFRAP columns; measurements without a PyFRAP value are logged and dropped."""
    log = []
    out = df.merge(pyfrap[["measurement_id", "D_pyfrap_um2_s", "qc_pyfrap"]], on="measurement_id", how="left")
    for r in out[out.D_pyfrap_um2_s.isna()].itertuples():
        log.append(dict(file=r.file, step="no PyFRAP value", parameter="D_pyfrap_um2_s", n_before=1, n_after=0,
                        reason=f"{r.measurement_id}: no PyFRAP D in the result cache"))
    out = out[out.D_pyfrap_um2_s.notna()].copy()
    out["qc_pyfrap"] = out.qc_pyfrap.astype(bool)
    ref = out[(out.sample_type == "Water") & (out.position == "deep") & out.qc_pyfrap]
    if ref.empty:
        raise ValueError("No QC-passed deep water measurement with a PyFRAP value for D0.")
    d0 = float(ref.D_pyfrap_um2_s.mean())
    out["Deff_D0_pyfrap"] = out.D_pyfrap_um2_s / d0
    out.attrs.update(df.attrs)
    out.attrs["D0_pyfrap"] = d0
    out.attrs["D0_pyfrap_ids"] = list(ref.measurement_id)
    return out, log


# ── Plotting ───────────────────────────────────────────────────────────────────
def save(fig, out_dir: Path, stem: str, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.pdf"
    fig.savefig(path, format="pdf")   # final size, no bbox_inches="tight"
    plt.close(fig)
    (out_dir / f"{stem}.json").write_text(json.dumps(
        {"figure": path.name, "created": f"{datetime.now():%Y-%m-%d %H:%M}",
         "caches": [str(PYFRAP_CACHE)], **meta}, indent=2, default=str), encoding="utf-8")


def group_stats(df, column, gate) -> dict:
    out = {}
    for (stype, pos), sub in df.groupby(["sample_type", "position"]):
        v = sub.loc[sub[gate], column]
        out[(stype, pos)] = {"n": len(v), "mean": float(v.mean()), "SD": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                             "measurements": list(sub.loc[sub[gate], "measurement_id"]),
                             "qc_failed_not_in_mean": list(sub.loc[~sub[gate], "measurement_id"])}
    return out


def _category_axis(ax):
    ax.set_xticks(list(X_POS.values()), list(X_POS.keys()))
    ax.set_xlim(-0.6, len(X_POS) - 0.4)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)


def _depth_handles():
    return [Line2D([], [], marker=MARKERS[p], ms=4, mfc="#bbbbbb", mec="#555555", mew=0.6, ls="none",
                   label=p.capitalize()) for p in ("shallow", "deep")]


def plot_d(df, out_dir):
    d = df.assign(qc_pass=df.qc_pyfrap)                  # _points draws open symbols for qc_pass == False
    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        handles = _points(ax, d, "D_pyfrap_um2_s")
        ax.set_ylim(0, 1.25 * d.D_pyfrap_um2_s.max())
        ax.set_ylabel(r"Diffusion coefficient $D_\mathrm{PyFRAP}$ (µm$^2$ s$^{-1}$)")
        ax.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, handletextpad=0.3,
                  columnspacing=0.8)
        save(fig, out_dir, "frap_pyfrap_d_vs_depth", {"value": "D_pyfrap_um2_s", "gate": "qc_pyfrap",
                                                       "left_out": list(EXCLUDE_IDS)})


def plot_ratio(df, out_dir) -> dict:
    stats = group_stats(df, "Deff_D0_pyfrap", "qc_pyfrap")
    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.axhline(1.0, color="0.65", lw=0.8, zorder=1)
        for (stype, pos), g in stats.items():
            base, dark = COLORS[stype]
            ax.errorbar(X_POS[stype] + RATIO_SUB_OFFSET[pos], g["mean"], yerr=g["SD"], fmt=MARKERS[pos], ms=4,
                        mfc=base, mec=dark, mew=0.6, ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=3)
        _category_axis(ax)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel(r"$D_\mathrm{eff}\,/\,D_0$ (PyFRAP)")
        ax.legend(handles=_depth_handles(), loc="center right", handletextpad=0.2, borderaxespad=0.3, handlelength=1.0)
        save(fig, out_dir, "frap_pyfrap_deff_d0_vs_depth", {
            "value": "D_pyfrap_um2_s / D0_pyfrap", "gate": "qc_pyfrap",
            "shown": "mean +- SD (ddof = 1) of the QC-passed measurements", "D0_um2_s": df.attrs["D0_pyfrap"],
            "D0_from": df.attrs["D0_pyfrap_ids"], "groups": {f"{k[0]} {k[1]}": v for k, v in stats.items()},
            "left_out": list(EXCLUDE_IDS)})
    return stats


def plot_comparison(df, out_dir) -> dict:
    stats = {"propagation": group_stats(df, "Deff_D0", "qc_pass"),
             "pyfrap": group_stats(df, "Deff_D0_pyfrap", "qc_pyfrap")}
    with plt.style.context(STYLE_PATH):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FULL_SIZE, layout="constrained")
        both = df.qc_pass & df.qc_pyfrap
        for (stype, pos), sub in df.groupby(["sample_type", "position"]):
            base, dark = COLORS[stype]
            for ok in (True, False):
                s = sub[both.loc[sub.index] == ok]
                ax_a.plot(s.D_um2_s, s.D_pyfrap_um2_s, MARKERS[pos], ms=4, mfc=base if ok else "none", mec=dark,
                          mew=0.6, ls="none", alpha=0.7, zorder=3)
        values = pd.concat([df.D_um2_s, df.D_pyfrap_um2_s])
        lim = (0.9 * values.min(), 1.1 * values.max())   # same range on both axes
        ax_a.plot(lim, lim, color="0.65", lw=0.8, zorder=1)
        ax_a.set_xlim(lim)
        ax_a.set_ylim(lim)
        ax_a.set_xlabel(r"$D_\mathrm{propagation}$ (µm$^2$ s$^{-1}$)")
        ax_a.set_ylabel(r"$D_\mathrm{PyFRAP}$ (µm$^2$ s$^{-1}$)")
        handles = [Line2D([], [], marker="o", ms=4, mfc=COLORS[t][0], mec=COLORS[t][1], mew=0.6, ls="none", label=t)
                   for t in X_POS] + _depth_handles()
        ax_a.legend(handles=handles, loc="best", handletextpad=0.2, borderaxespad=0.3, handlelength=1.0)

        ax_b.axhline(1.0, color="0.65", lw=0.8, zorder=1)
        for m, groups in stats.items():
            for (stype, pos), g in groups.items():
                base, dark = COLORS[stype]
                x = X_POS[stype] + RATIO_SUB_OFFSET[pos] + METHOD_OFFSET[m]
                ax_b.errorbar(x, g["mean"], yerr=g["SD"], fmt=MARKERS[pos], ms=4, mfc=base, fillstyle=METHOD_FILL[m], mfcalt="white",
                              mec=dark, mew=0.6, ecolor=dark, elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=3)
        _category_axis(ax_b)
        ax_b.set_ylim(0, 1.25)
        ax_b.set_ylabel(r"$D_\mathrm{eff}\,/\,D_0$")
        ax_b.legend(handles=[Line2D([], [], marker="o", ms=4, mfc="#bbbbbb", fillstyle=METHOD_FILL[m], mfcalt="white",
                                    mec="#555555", mew=0.6, ls="none", label=METHOD_LABELS[m]) for m in stats],
                    loc="lower left", handletextpad=0.2, borderaxespad=0.3, handlelength=1.0)
        for ax, label in ((ax_a, "A"), (ax_b, "B")):
            ax.text(-0.2, 1.02, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
        save(fig, out_dir, "frap_method_comparison", {
            "A": "D per measurement, x = propagation, y = PyFRAP, line y = x; open = fails qc_pass or qc_pyfrap",
            "B": "D_eff/D0 mean +- SD (ddof = 1) of the QC-passed measurements; filled = propagation, half-filled = PyFRAP",
            "D0_propagation_um2_s": df.attrs["D0"], "D0_pyfrap_um2_s": df.attrs["D0_pyfrap"],
            "groups": {m: {f"{k[0]} {k[1]}": v for k, v in g.items()} for m, g in stats.items()},
            "left_out": list(EXCLUDE_IDS)})
    return stats


def make_figures(df, out_dir: Path):
    plot_d(df, out_dir)
    plot_ratio(df, out_dir)
    return plot_comparison(df, out_dir)


def plot_versions(base, merged: dict, curves: dict, out_dir: Path) -> pd.DataFrame:
    """A: D, B: D_eff/D0 (mean +- SD per group) for propagation and every PyFRAP version; C: example fits."""
    rows = []
    for (stype, pos), x in GROUP_X.items():
        sub = base[(base.sample_type == stype) & (base.position == pos) & base.qc_pass]
        rows.append({"method": "propagation", "group": f"{stype} {pos}", "x": x, "n": len(sub),
                     "D_mean": sub.D_um2_s.mean(), "D_SD": sub.D_um2_s.std(ddof=1),
                     "Deff_mean": sub.Deff_D0.mean(), "Deff_SD": sub.Deff_D0.std(ddof=1)})
        for v, df in merged.items():
            s = df[(df.sample_type == stype) & (df.position == pos) & df.qc_pyfrap]
            rows.append({"method": v, "group": f"{stype} {pos}", "x": x, "n": len(s),
                         "D_mean": s.D_pyfrap_um2_s.mean(), "D_SD": s.D_pyfrap_um2_s.std(ddof=1),
                         "Deff_mean": s.Deff_D0_pyfrap.mean(), "Deff_SD": s.Deff_D0_pyfrap.std(ddof=1)})
    stats = pd.DataFrame(rows)
    labels = {"propagation": "Propagation", **VERSIONS}
    with plt.style.context(STYLE_PATH):
        fig = plt.figure(figsize=(6.30, 6.30 / 1.42), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        ax_a, ax_b, ax_c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
        for ax, col in ((ax_a, "D"), (ax_b, "Deff")):
            for r in stats.itertuples():
                stype = r.group.split()[0]
                base_c, dark = COLORS[stype]
                ax.errorbar(r.x + VERSION_OFFSET[r.method], getattr(r, f"{col}_mean"), yerr=getattr(r, f"{col}_SD"),
                            fmt=MARKERS[r.group.split()[1]], ms=4, mfc=base_c, fillstyle=VERSION_FILL[r.method],
                            mfcalt="white", mec=dark, mew=0.6, ecolor=dark, elinewidth=0.8, capsize=2.0,
                            capthick=0.8, zorder=3)
            ax.set_xticks(list(GROUP_X.values()), ["Water", "Hydrogel\nshallow", "Hydrogel\ndeep"])
            ax.set_xlim(-0.5, len(GROUP_X) - 0.5)
            ax.tick_params(axis="x", which="both", top=False, bottom=False)
            ax.set_ylim(bottom=0)
        ax_a.set_ylabel(r"$D$ (µm$^2$ s$^{-1}$)")
        ax_b.axhline(1.0, color="0.65", lw=0.8, zorder=1)
        ax_b.set_ylabel(r"$D_\mathrm{eff}\,/\,D_0$")
        ax_b.legend(handles=[Line2D([], [], marker="o", ms=4, mfc="#bbbbbb", fillstyle=VERSION_FILL[m], mfcalt="white",
                                    mec="#555555", mew=0.6, ls="none", label=labels[m]) for m in labels],
                    loc="lower left", fontsize=7, handletextpad=0.2, borderaxespad=0.3, handlelength=1.0)
        example = {v: c[EXAMPLE_ID] for v, c in curves.items() if EXAMPLE_ID in c}
        if example:
            first = next(iter(example.values()))
            ax_c.plot(first["t"], first["data_pinned"], "o", ms=3, mfc=COLORS["Hydrogel"][0], mec=COLORS["Hydrogel"][1],
                      mew=0.4, ls="none", alpha=0.7, label="Data (pinned)", zorder=2)
            for v, c in example.items():
                d = merged[v].set_index("measurement_id").D_pyfrap_um2_s.get(EXAMPLE_ID, np.nan)
                ax_c.plot(c["t"], c["fit_pinned"], color=COLORS["Hydrogel"][1], lw=1.5, ls=VERSION_LINE[v],
                          label=f"{VERSIONS[v]}, $D$ = {d:.1f} µm$^2$ s$^{{-1}}$", zorder=3)
            ax_c.set_xlabel("Time after bleach $t$ (s)")
            ax_c.set_ylabel("Pinned recovery")
            ax_c.legend(loc="lower right", fontsize=7)
        for ax, label in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
            ax.text(-0.08 if ax is ax_c else -0.2, 1.02, label, transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="bottom")
        save(fig, out_dir, "frap_pyfrap_versions_comparison", {
            "A_B": "mean +- SD (ddof = 1) of the QC-passed measurements per group; propagation: qc_propagation, "
                   "PyFRAP: qc_pyfrap; fill style = method", "C": f"pinned data and PyFRAP fits of {EXAMPLE_ID}",
            "versions": VERSIONS, "D0": {"propagation": base.attrs["D0"],
                                         **{v: df.attrs["D0_pyfrap"] for v, df in merged.items()}},
            "groups": stats.to_dict(orient="records"), "left_out": list(EXCLUDE_IDS)})
    return stats


# ── Test ───────────────────────────────────────────────────────────────────────
def test_with_synthetic_result(out_dir: Path, seed: int = 3):
    """Figures from invented PyFRAP values and curves into out_dir; test only, never FIGURE_DIR."""
    if Path(out_dir).resolve() == FIGURE_DIR.resolve():
        raise ValueError("Synthetic test figures must not go to FIGURE_DIR.")
    df, _, _ = load_values()
    rng = np.random.default_rng(seed)
    merged, curves = {}, {}
    t = np.linspace(0, 120, 80)
    for i, v in enumerate(VERSIONS):
        fake = pd.DataFrame({"measurement_id": df.measurement_id,
                             "D_pyfrap_um2_s": df.D_um2_s * rng.uniform(0.6 - 0.2 * i, 1.1, len(df)),
                             "qc_pyfrap": [True] * (len(df) - 1) + [False]})
        merged[v], _ = merge(df, fake)
        make_figures(merged[v], Path(out_dir) / v)
        curves[v] = {EXAMPLE_ID: {"t": t, "data_pinned": 1 - np.exp(-t / 20) + rng.normal(0, 0.02, len(t)),
                                  "fit_pinned": 1 - np.exp(-t / (20 + 5 * i))}}
    return plot_versions(df, merged, curves, Path(out_dir))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    df, log, _ = load_values()
    merged, curves, logs = {}, {}, list(log)
    for v in VERSIONS:
        if not Path(str(PYFRAP_CACHE).format(version=v)).is_file():
            print(f"{v}: no result cache, skipped")
            continue
        if not pickle.loads(Path(str(PYFRAP_CACHE).format(version=v)).read_bytes()).get("complete", True):
            print(f"{v}: result cache incomplete (fit still running), skipped")
            continue
        res, curves[v] = load_pyfrap(v)
        merged[v], log_merge = merge(df, res)
        logs += [{**row, "step": f"{v}: {row['step']}"} for row in log_merge]
        out_dir = FIGURE_DIR / v
        make_figures(merged[v], out_dir)
        print(f"{v}: D0 propagation {merged[v].attrs['D0']:.3f}, PyFRAP {merged[v].attrs['D0_pyfrap']:.3f} um2/s")
        cols = ["measurement_id", "sample_type", "position", "depth_um", "D_um2_s", "qc_pass", "Deff_D0",
                "D_pyfrap_um2_s", "qc_pyfrap", "Deff_D0_pyfrap"]
        merged[v][cols].to_csv(out_dir / "frap_pyfrap_values.csv", index=False)
    if not merged:
        raise FileNotFoundError("No PyFRAP result cache found: run frap_pyfrap_fit.py (PyFRAP environment) first.")
    stats = plot_versions(df, merged, curves, FIGURE_DIR)
    print(stats.round(3).to_string(index=False))
    stats.to_csv(FIGURE_DIR / "frap_pyfrap_versions_values.csv", index=False)
    pd.DataFrame(logs, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        FIGURE_DIR / "frap_pyfrap_values_exclusions.csv", index=False)
    print(f"Written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
