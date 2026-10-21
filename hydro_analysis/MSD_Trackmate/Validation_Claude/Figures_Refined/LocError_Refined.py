"""
Refined re-plots of Loc_Error_Analyse_immob_particle.py's three figures
(ratio-as-percent, trajectories overlay, sigma-vs-size), per the new
publication styling rules. Does NOT modify Loc_Error_Analyse_immob_particle.py,
raw data, or any cache -- pure consumer, reuses that script's own functions
by import (IMMOB_FOLDERS, _build_immob_datasets, _load_raw_trajectory_data,
_drop_tracks_exceeding_window, _scan_qualifying_mpp, _qualifying_groups,
_load_median_stepsize_nm, TRAJECTORY_WINDOW_NM) and reads its cached sigma_loc
table (cache/localization_error_immobilized.pkl) unchanged.

Only the trajectory positions (never cached by the original script) are
re-derived here, by calling the SAME functions the original main() calls for
that purpose, in the same order -- not a reimplementation, no new filtering
logic, no change to which tracks qualify.

Dissertation figures (Styleguide v2, PDF, SAVE_PATH):
  A_sigma_loc_combined_by_size.pdf   -- Abb04, narrow. Boxes: sigma_loc of the
                                         immobilised particles per (size, mpp).
                                         Diamonds: median of sigma_loc from the
                                         intercept of the buffer MSD fits
                                         (fit_results_MSD["sigma_loc_nm"] in
                                         cache/msd_d0_results.pkl, read only).
                                         Files with a negative intercept carry
                                         NaN there and are not averaged; "n/N"
                                         under each diamond gives files with a
                                         value / files of that size (E005 open).
  B_trajectories_overlay_by_size.pdf -- Abb06, full. Unchanged content; pixel
                                         size in the panel titles; trajectory
                                         lines rasterised.
  D_expected_locerror_by_size.csv    -- table D below.
Not used in the dissertation (written to SCRATCH_PATH, PNG):
  C_locerror_vs_stepsize_ratio_pct.png -- ratio as percent (not a dimensionless
                                         fraction), gray boxes, 100% reference
                                         line (replaces the ratio=1 line)
  D_expected_locerror_by_size.png/.csv -- table, one row per qualifying
                                         (size, mpp) group of A: n tracks,
                                         median + IQR and mean +- SD of
                                         sigma_xy_static_nm, median step size
                                         and median/step ratio (%)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Loc_Error_Analyse_immob_particle import (
    IMMOB_FOLDERS, CACHE_FILE, TRAJECTORY_WINDOW_NM,
    _build_immob_datasets, _load_raw_trajectory_data, _drop_tracks_exceeding_window,
    _scan_qualifying_mpp, _qualifying_groups, _load_median_stepsize_nm,
)

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
SAVE_PATH = RESULTS / "SPT"
SCRATCH_PATH = RESULTS / "_scratch" / "LocError_Refined"
CACHE_D0 = Path(__file__).resolve().parents[2] / "cache" / "msd_d0_results.pkl"
STYLE = Path(__file__).resolve().parents[3] / "thesis.mplstyle"
WIDTH_NARROW_IN, WIDTH_FULL_IN, RATIO = 4.72, 6.30, 1.42

COLOR_GRAY = "#808080"
COLOR_GRAY_DARK = "#4d4d4d"


def _load_pickle(path: Path, compute_script: str):
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_trajectory_by_group() -> dict[tuple[float, float], list[pd.DataFrame]]:
    """Reproduces Loc_Error_Analyse_immob_particle.py::main()'s trajectory_by_group
    construction exactly (same functions, same order) -- positions are not
    persisted in the sigma_loc cache, so this is the only way to get them
    without touching that script."""
    trajectory_by_group: dict[tuple[float, float], list[pd.DataFrame]] = {}
    for dataset_label, folder in IMMOB_FOLDERS.items():
        if not folder.exists():
            print(f"  [SKIP] Ordner nicht gefunden: {folder}")
            continue
        datasets = _build_immob_datasets(folder)
        for result in datasets.values():
            if result is None:
                continue
            mpp = result.get("mpp")
            base_name = result.get("base_name")
            xml_path, tif_path = result.get("xml_path"), result.get("tif_path")
            particle_size_nm = result.get("particle_size_nm")
            if mpp is None or xml_path is None or particle_size_nm is None:
                continue
            raw_traj = _load_raw_trajectory_data(Path(xml_path), tif_path, base_name)
            if raw_traj is not None and not raw_traj.empty:
                raw_traj, _ = _drop_tracks_exceeding_window(raw_traj, mpp, TRAJECTORY_WINDOW_NM, base_name)
            if raw_traj is not None and not raw_traj.empty:
                traj_df = raw_traj[["particle", "frame", "x", "y"]].copy()
                key = (float(particle_size_nm), round(float(mpp), 3))
                trajectory_by_group.setdefault(key, []).append(traj_df)
    return trajectory_by_group


# ── A: sigma_loc combined by size ───────────────────────────────────────────────

def buffer_intercept_sigma(d0_cache: dict) -> dict[float, dict]:
    """{nominal_nm: {"values": positive-intercept sigma_loc (nm), "n": files with a
    value, "N": files of that size}} from the buffer MSD fits, as stored."""
    out: dict[float, dict] = {}
    for rd in d0_cache.values():
        size = rd.get("particle_size_nm")
        fit = rd.get("fit_results_MSD")
        if size is None:
            continue
        entry = out.setdefault(float(size), {"values": [], "n": 0, "N": 0})
        entry["N"] += 1
        sigma = fit.get("sigma_loc_nm", np.nan) if isinstance(fit, dict) else np.nan
        if sigma is not None and np.isfinite(sigma):
            entry["values"].append(float(sigma))
            entry["n"] += 1
    return out


def plot_sigma_combined_by_size(df: pd.DataFrame, qualifying: list[tuple[float, float]],
                                 dls_labels: dict[float, int], intercept: dict[float, dict]) -> plt.Figure:
    labels = [f"{dls_labels.get(s, int(s))}\n{m:.3g} µm px$^{{-1}}$" for s, m in qualifying]
    groups = [df.loc[(df["particle_size_nm"] == s) & (df["mpp"] == m), "sigma_xy_static_nm"].dropna().to_numpy()
              for s, m in qualifying]
    positions = np.arange(len(qualifying), dtype=float)

    fig, ax = plt.subplots(figsize=(WIDTH_NARROW_IN, WIDTH_NARROW_IN / RATIO))
    bp = ax.boxplot(groups, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
                     manage_ticks=False)
    for box in bp["boxes"]:
        box.set_facecolor(COLOR_GRAY)
        box.set_edgecolor(COLOR_GRAY_DARK)
        box.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(COLOR_GRAY_DARK)
    for pos, (s, _m) in zip(positions, qualifying):
        entry = intercept.get(s, {"values": [], "n": 0, "N": 0})
        if entry["values"]:
            face, edge = SIZE_COLORS[s]
            ax.plot(pos, np.median(entry["values"]), marker="D", markersize=5, color=face,
                    markeredgecolor=edge, markeredgewidth=0.6, linestyle="none", zorder=4)
        ax.text(pos, 0.02, f"{entry['n']}/{entry['N']}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(-0.1 * ymax, 1.22 * ymax)   # headroom for the legend
    ax.set_ylabel(r"Localisation error $\sigma_\mathrm{loc}$ (nm)")
    ax.set_xlabel("Particle size (nm)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_GRAY, edgecolor=COLOR_GRAY_DARK, alpha=0.6,
                      label="Immobilised particles"),
        Line2D([0], [0], marker="D", markersize=5, color="0.6", markeredgecolor="0.3", linestyle="none",
               label="Buffer MSD intercept (median)"),
    ], loc="upper right")
    return fig


# ── B: trajectories overlay ──────────────────────────────────────────────────────

def plot_trajectories_by_size(by_group: dict[tuple[float, float], list[pd.DataFrame]],
                               qualifying: list[tuple[float, float]],
                               dls_labels: dict[float, int]) -> plt.Figure:
    n = len(qualifying)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_FULL_IN, nrows * WIDTH_FULL_IN / ncols * 0.95),
                              squeeze=False, sharex=True, sharey=True)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    half = TRAJECTORY_WINDOW_NM
    for idx, (ax, (size_nm, mpp)) in enumerate(zip(ax_flat, qualifying)):
        row, col = divmod(idx, ncols)
        for traj_df in by_group.get((size_nm, mpp), []):
            for _, g in traj_df.groupby("particle"):
                g = g.sort_values("frame")
                x_nm = g["x"].to_numpy(dtype=float) * mpp * 1000.0
                y_nm = g["y"].to_numpy(dtype=float) * mpp * 1000.0
                ax.plot(x_nm - x_nm[0], y_nm - y_nm[0], color="black", linewidth=0.5, alpha=0.5,
                        rasterized=True)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_xticks([-half, -half / 2, 0, half / 2, half])
        ax.set_yticks([-half, -half / 2, 0, half / 2, half])
        ax.set_aspect("equal")
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm ({mpp:.3g} µm px$^{{-1}}$)")
        if row == nrows - 1 or (row == nrows - 2 and (idx + ncols) >= n):
            ax.set_xlabel(r"$\Delta x$ (nm)")
        if col == 0:
            ax.set_ylabel(r"$\Delta y$ (nm)")
    return fig


# ── C: loc_err / step size ratio, as percent ────────────────────────────────────

def plot_ratio_percent(df: pd.DataFrame, qualifying: list[tuple[float, float]],
                        step_size_nm: dict[float, float], dls_labels: dict[float, int]) -> plt.Figure:
    """Same data/grouping as Loc_Error_Analyse_immob_particle.py::
    plot_locerror_vs_stepsize_ratio(), but expressed in percent (ratio*100)
    instead of a dimensionless fraction, and gray instead of colored."""
    sizes = sorted({s for s, _ in qualifying})
    mpp_by_size: dict[float, list[float]] = {}
    for s, m in qualifying:
        mpp_by_size.setdefault(s, []).append(m)

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    positions = np.arange(len(sizes), dtype=float)

    boxes = []
    n_no_reference = 0
    for i, s in enumerate(sizes):
        step = step_size_nm.get(s)
        if step is None or not np.isfinite(step) or step <= 0:
            n_no_reference += 1
            continue
        sub = df.loc[(df["particle_size_nm"] == s) & df["mpp"].isin(mpp_by_size[s]), "sigma_xy_static_nm"]
        ratios_pct = (sub / step * 100.0).dropna().to_numpy()
        if ratios_pct.size:
            boxes.append((positions[i], ratios_pct))
    if boxes:
        bp = ax.boxplot([r for _, r in boxes], positions=[p for p, _ in boxes], widths=0.5,
                         patch_artist=True, showfliers=False, manage_ticks=False)
        for box in bp["boxes"]:
            box.set_facecolor(COLOR_GRAY)
            box.set_edgecolor(COLOR_GRAY_DARK)
            box.set_alpha(0.6)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(COLOR_GRAY_DARK)

    ax.axhline(100.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{dls_labels.get(s, int(s))} nm" for s in sizes])
    ax.set_xlabel("Particle Size (nm)")
    ax.set_ylabel("Localization Error / Step Size Ratio (%)")
    ax.legend(handles=[Line2D([0], [0], color="black", linewidth=1.0, linestyle="--",
                              label="Sigma Log Error = Step Size (100%)")],
              loc="upper right", frameon=False)
    if n_no_reference:
        print(f"  [NOTE] {n_no_reference} Groesse(n) ohne Schrittweiten-Referenz -- kein Boxplot dort, "
              f"siehe Konsolenausgabe von iMSD_histograms.py fuer den Grund.")
    return fig


# ── D: expected localization error per particle size (table) ────────────────────

def build_locerror_table(df: pd.DataFrame, qualifying: list[tuple[float, float]],
                          step_size_nm: dict[float, float], dls_labels: dict[float, int]) -> pd.DataFrame:
    """One row per qualifying (size, mpp) group -- the same per-track
    sigma_xy_static_nm values shown as boxes in A, summarized. The median is
    the expected localization error of a single track of that size."""
    rows = []
    for s, m in qualifying:
        vals = df.loc[(df["particle_size_nm"] == s) & (df["mpp"] == m), "sigma_xy_static_nm"].dropna().to_numpy()
        if vals.size == 0:
            continue
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        step = step_size_nm.get(s, np.nan)
        ratio_pct = median / step * 100.0 if np.isfinite(step) and step > 0 else np.nan
        rows.append({
            "Particle Size DLS (nm)": dls_labels.get(s, int(s)),
            "Nominal Size (nm)": int(s),
            "mpp (um/px)": m,
            "n Tracks": int(vals.size),
            "Median Sigma Loc (nm)": median,
            "Q1 Sigma Loc (nm)": q1,
            "Q3 Sigma Loc (nm)": q3,
            "Mean Sigma Loc (nm)": float(np.mean(vals)),
            "SD Sigma Loc (nm)": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            "Median Step Size (nm)": step,
            "Median Sigma Loc / Step Size (%)": ratio_pct,
        })
    return pd.DataFrame(rows)


def render_locerror_table(table: pd.DataFrame) -> plt.Figure:
    def fmt(v: float, digits: int = 1) -> str:
        return f"{v:.{digits}f}" if np.isfinite(v) else "--"

    display = pd.DataFrame({
        "Particle Size (nm)\nDLS (Nominal)": [f"{d} ({n})" for d, n in zip(table["Particle Size DLS (nm)"], table["Nominal Size (nm)"])],
        "mpp (µm/px)": [f"{m:.3g}" for m in table["mpp (um/px)"]],
        "n Tracks": table["n Tracks"].astype(str),
        "Median σ_loc (nm)": [fmt(v) for v in table["Median Sigma Loc (nm)"]],
        "IQR (nm)": [f"{fmt(a)} – {fmt(b)}" for a, b in zip(table["Q1 Sigma Loc (nm)"], table["Q3 Sigma Loc (nm)"])],
        "Mean ± SD (nm)": [f"{fmt(a)} ± {fmt(b)}" for a, b in zip(table["Mean Sigma Loc (nm)"], table["SD Sigma Loc (nm)"])],
        "Median Step Size (nm)": [fmt(v) for v in table["Median Step Size (nm)"]],
        "σ_loc / Step Size (%)": [fmt(v) for v in table["Median Sigma Loc / Step Size (%)"]],
    })
    fig, ax = plt.subplots(figsize=(10.0, 0.4 * len(display) + 1.0))
    ax.axis("off")
    tbl = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)
    for col in range(len(display.columns)):
        tbl[0, col].set_height(tbl[0, col].get_height() * 1.4)
    for (row, _), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", wrap=True)
    fig.tight_layout()
    return fig


def main() -> None:
    df = _load_pickle(CACHE_FILE, "Loc_Error_Analyse_immob_particle.py")
    print(f"Cache geladen: {CACHE_FILE} ({len(df)} Tracks)")
    dls_labels = get_dls_labels()

    ref_mpp = _scan_qualifying_mpp()
    step_size_nm = _load_median_stepsize_nm()
    qualifying = _qualifying_groups(df, ref_mpp)
    print(f"{len(qualifying)} (size, mpp) Gruppen qualifizieren fuer die Abbildungen.")

    print("Rekonstruiere Trajektorien (gleiche Funktionen wie im Original-Skript, nicht gecacht)...")
    trajectory_by_group = build_trajectory_by_group()

    d0_cache = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    intercept = buffer_intercept_sigma(d0_cache)
    for size in sorted(intercept):
        v = intercept[size]
        med = f"{np.median(v['values']):.1f} nm" if v["values"] else "--"
        print(f"  Buffer MSD intercept {dls_labels.get(size, int(size))} nm: n/N = {v['n']}/{v['N']}, median {med}")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
    with plt.style.context(STYLE):
        for fig, filename in (
            (plot_sigma_combined_by_size(df, qualifying, dls_labels, intercept), "A_sigma_loc_combined_by_size.pdf"),
            (plot_trajectories_by_size(trajectory_by_group, qualifying, dls_labels),
             "B_trajectories_overlay_by_size.pdf"),
        ):
            fig.savefig(SAVE_PATH / filename)
            plt.close(fig)
            print(f"Plot gespeichert: {SAVE_PATH / filename}")
    with plt.rc_context(_RC):
        fig = plot_ratio_percent(df, qualifying, step_size_nm, dls_labels)
        safe_savefig(fig, SCRATCH_PATH / "C_locerror_vs_stepsize_ratio_pct.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

        table = build_locerror_table(df, qualifying, step_size_nm, dls_labels)
        print(table.to_string(index=False))
        fig = render_locerror_table(table)
        safe_savefig(fig, SCRATCH_PATH / "D_expected_locerror_by_size.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    csv_path = SAVE_PATH / "D_expected_locerror_by_size.csv"
    table.to_csv(csv_path, index=False)
    print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
