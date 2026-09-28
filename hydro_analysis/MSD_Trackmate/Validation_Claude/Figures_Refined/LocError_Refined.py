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

Outputs (own folder, PNG only, 600 dpi, English labels):
  A_sigma_loc_combined_by_size.png   -- y: "Sigma Log Error (nm)", x: "Particle
                                         Size (nm)" (DLS-labeled ticks)
  B_trajectories_overlay_by_size.png -- no suptitle, all-black uniform alpha,
                                         shared x/y limits+ticks+aspect,
                                         outer-only axis labels
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
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Loc_Error_Analyse_immob_particle import (
    IMMOB_FOLDERS, CACHE_FILE, TRAJECTORY_WINDOW_NM,
    _build_immob_datasets, _load_raw_trajectory_data, _drop_tracks_exceeding_window,
    _scan_qualifying_mpp, _qualifying_groups, _load_median_stepsize_nm,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "LocError"

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

def plot_sigma_combined_by_size(df: pd.DataFrame, qualifying: list[tuple[float, float]],
                                 dls_labels: dict[float, int]) -> plt.Figure:
    labels = [f"{dls_labels.get(s, int(s))} nm\n(mpp={m:.3g})" for s, m in qualifying]
    groups = [df.loc[(df["particle_size_nm"] == s) & (df["mpp"] == m), "sigma_xy_static_nm"].dropna().to_numpy()
              for s, m in qualifying]
    positions = np.arange(len(qualifying), dtype=float)

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    bp = ax.boxplot(groups, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
                     manage_ticks=False)
    for box in bp["boxes"]:
        box.set_facecolor(COLOR_GRAY)
        box.set_edgecolor(COLOR_GRAY_DARK)
        box.set_alpha(0.6)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(COLOR_GRAY_DARK)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Sigma Log Error (nm)")
    ax.set_xlabel("Particle Size (nm)")
    return fig


# ── B: trajectories overlay ──────────────────────────────────────────────────────

def plot_trajectories_by_size(by_group: dict[tuple[float, float], list[pd.DataFrame]],
                               qualifying: list[tuple[float, float]],
                               dls_labels: dict[float, int]) -> plt.Figure:
    n = len(qualifying)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2.5),
                              constrained_layout=True, squeeze=False, sharex=True, sharey=True)
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
                ax.plot(x_nm - x_nm[0], y_nm - y_nm[0], color="black", linewidth=0.6, alpha=0.5,
                        rasterized=True)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_xticks([-half, -half / 2, 0, half / 2, half])
        ax.set_yticks([-half, -half / 2, 0, half / 2, half])
        ax.set_aspect("equal")
        ax.set_title(f"{dls_labels.get(size_nm, int(size_nm))} nm", fontsize=10, fontweight="semibold")
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

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        for fig, filename in (
            (plot_sigma_combined_by_size(df, qualifying, dls_labels), "A_sigma_loc_combined_by_size.png"),
            (plot_trajectories_by_size(trajectory_by_group, qualifying, dls_labels),
             "B_trajectories_overlay_by_size.png"),
            (plot_ratio_percent(df, qualifying, step_size_nm, dls_labels), "C_locerror_vs_stepsize_ratio_pct.png"),
        ):
            safe_savefig(fig, SAVE_PATH / filename, dpi=600, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot gespeichert: {SAVE_PATH / filename}")

        table = build_locerror_table(df, qualifying, step_size_nm, dls_labels)
        print(table.to_string(index=False))
        fig = render_locerror_table(table)
        safe_savefig(fig, SAVE_PATH / "D_expected_locerror_by_size.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {SAVE_PATH / 'D_expected_locerror_by_size.png'}")
    csv_path = SAVE_PATH / "D_expected_locerror_by_size.csv"
    table.to_csv(csv_path, index=False)
    print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
