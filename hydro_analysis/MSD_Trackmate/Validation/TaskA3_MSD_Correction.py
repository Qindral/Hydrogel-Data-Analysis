"""
Localization-error-corrected MSD, refit D and n, compare to uncorrected.

Pure consumer: reads cache/msd_d0_results.pkl (water), cache/msd_20mg_files.pkl
(hydrogel), and cache/localization_error_immobilized.pkl. Never touches raw
XML/TIFF.

For 20 nm (35 nm-labeled) and 50 nm nominal particles in both conditions,
subtracts a single representative sigma_loc (the independently-measured
static-estimator pooled mean from Task A1, for that nominal size -- NOT a
per-mobile-file refit) from each file's full eMSD curve:

    MSD_corrected(tau) = MSD_measured(tau) - 4*sigma_loc^2

Every value of the corrected curve is kept and reported, including negative
ones -- never clipped. core.analysis.fit_powerlaw_with_errors() (reused
unmodified) needs log(y), which is undefined for y<=0, so ONLY the fit-call
input window excludes non-positive points; this exclusion is counted and
logged per file (how many of the fit_points lag times were dropped), it is
not a silent removal from the stored/reported data. If fewer than 2 points
remain positive, that file's corrected fit is skipped and recorded as such.

Run MSD_FromTrackmate_D0.py, MSD_FromTrackmate_20mg.py, and
TaskA_Immobilized_Compute.py first (or after any raw-data change) to
refresh all three inputs.

Shows each figure first (plt.show(), blocking); only after the window is
closed does it save into its own Auswertungsbilder\\MSD_Correction_LocError\\
subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.analysis import fit_powerlaw_with_errors
from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0 = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_20MG = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
CACHE_LOCERR = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "MSD_Correction_LocError"

TARGET_SIZES_NM = (20.0, 50.0)

COLOR_MEASURED, COLOR_MEASURED_DARK = "#3B6E8C", "#2A4F66"
COLOR_CORRECTED, COLOR_CORRECTED_DARK = "#da0000", "#990000"


def _load_pickle(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _representative_sigma_loc_um(locerr_df: pd.DataFrame, size_nm: float) -> float:
    sub = locerr_df[locerr_df["particle_size_nm"] == size_nm]
    if sub.empty:
        return np.nan
    vals = np.concatenate([sub["sigma_x_static_nm"].dropna().to_numpy(),
                            sub["sigma_y_static_nm"].dropna().to_numpy()])
    return float(np.mean(vals)) / 1000.0 if vals.size else np.nan


def correct_and_refit(fit: dict, sigma_loc_um: float) -> dict:
    """Returns a dict with the corrected series, refit result, and exclusion bookkeeping."""
    emsd = fit["emsd"]
    fit_points = fit["fit_points"]
    offset_um2 = 4.0 * sigma_loc_um ** 2

    corrected_full = emsd - offset_um2   # full curve, ALL values kept incl. negative

    window = corrected_full.iloc[0:fit_points]
    positive_window = window[window > 0]
    n_excluded = int(fit_points - len(positive_window))
    excluded_lags = window.index[window <= 0].to_numpy() if n_excluded else np.array([])

    corrected_fit = None
    if len(positive_window) >= 2:
        corrected_fit = fit_powerlaw_with_errors(positive_window, points=len(positive_window))

    return {
        "corrected_emsd_full": corrected_full,
        "offset_um2": offset_um2,
        "n_excluded_from_fit": n_excluded,
        "excluded_lag_times_s": excluded_lags,
        "corrected_fit": corrected_fit,
    }


def build_comparison_table(results: dict, condition: str, locerr_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    detail: dict[str, dict] = {}   # xml_path -> {"fit": ..., "correction": ...} for the plotting step
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        if size_nm not in TARGET_SIZES_NM:
            continue
        fit = r.get("fit_results_MSD")
        if fit is None or fit.get("emsd") is None:
            continue
        sigma_loc_um = _representative_sigma_loc_um(locerr_df, float(size_nm))
        if not np.isfinite(sigma_loc_um):
            print(f"  [SKIP] No immobilized-particle sigma_loc available for "
                  f"{size_nm:.0f} nm ({condition}) -- {r.get('base_name')}")
            continue

        correction = correct_and_refit(fit, sigma_loc_um)
        corrected_fit = correction["corrected_fit"]

        D_before = fit["D_um2_per_s"]
        n_before = fit["exponent"]
        D_after = (corrected_fit["A"][0] / 4.0) if corrected_fit is not None else np.nan
        n_after = corrected_fit["n"][0] if corrected_fit is not None else np.nan

        rows.append({
            "xml_path": r.get("xml_path"),
            "base_name": r.get("base_name"),
            "particle_size_nm": float(size_nm),
            "condition": condition,
            "num_tracks": r.get("num_tracks"),
            "sigma_loc_um": sigma_loc_um,
            "offset_um2": correction["offset_um2"],
            "n_excluded_from_fit": correction["n_excluded_from_fit"],
            "D_before": D_before,
            "D_after": D_after,
            "D_rel_change": (D_after - D_before) / D_before if np.isfinite(D_after) and D_before else np.nan,
            "n_before": n_before,
            "n_after": n_after,
            "n_rel_change": (n_after - n_before) / n_before if np.isfinite(n_after) and n_before else np.nan,
        })
        detail[r.get("xml_path")] = {"fit": fit, "correction": correction, "size_nm": float(size_nm),
                                      "condition": condition, "num_tracks": r.get("num_tracks")}
    return pd.DataFrame(rows), detail


def plot_msd_correction_example(detail_entry: dict, dls_labels: dict[float, int]) -> plt.Figure:
    fit = detail_entry["fit"]
    correction = detail_entry["correction"]
    emsd = fit["emsd"]
    corrected = correction["corrected_emsd_full"]
    fit_points = fit["fit_points"]
    corrected_fit = correction["corrected_fit"]

    # Zoom to a range that keeps the fit region legible -- the full multi-second
    # eMSD curve dwarfs the (deliberately small) localization-error offset, making
    # both curves and the fits indistinguishable if the whole curve is shown.
    view_points = min(len(emsd), fit_points * 6)

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    ax.plot(emsd.index[:view_points], emsd.values[:view_points], "o-", color=COLOR_MEASURED,
            markersize=4, linewidth=1.2, label="Measured MSD")
    ax.plot(corrected.index[:view_points], corrected.values[:view_points], "s-", color=COLOR_CORRECTED,
            markersize=4, linewidth=1.2, label="Corrected MSD (measured - 4σ²)")

    # Both fitted power-law curves (MSD = A*tau^n), over the same fit range.
    tau_fit = emsd.index.values[:fit_points]
    y_before_fit = fit["A"] * tau_fit ** fit["exponent"]
    ax.plot(tau_fit, y_before_fit, "--", color=COLOR_MEASURED_DARK, linewidth=1.8, label="Fit (measured)")
    if corrected_fit is not None:
        A_after, n_after = float(corrected_fit["A"][0]), float(corrected_fit["n"][0])
        y_after_fit = A_after * tau_fit ** n_after
        ax.plot(tau_fit, y_after_fit, "--", color=COLOR_CORRECTED_DARK, linewidth=1.8, label="Fit (corrected)")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(emsd.index[0], emsd.index[min(fit_points, len(emsd)) - 1], color="grey", alpha=0.15,
               label="Fit range")
    ax.set_xlim(0, emsd.index[view_points - 1])

    label = f"{dls_labels.get(detail_entry['size_nm'], int(detail_entry['size_nm']))} nm"
    ax.set_title(f"{label}, {detail_entry['condition']}  |  N={detail_entry['num_tracks']} tracks\n"
                 f"offset = 4σ² = {correction['offset_um2']:.4g} µm², "
                 f"{correction['n_excluded_from_fit']} of {fit_points} fit points excluded (≤0 after correction)",
                 fontsize=10)
    ax.set_xlabel("Lag time (s)")
    ax.set_ylabel(r"MSD (µm²)  --  linear axis (corrected values can be negative)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    return fig


def plot_paired_comparison(table: pd.DataFrame, dls_labels: dict[float, int]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.00), constrained_layout=True)
    groups = table.groupby(["particle_size_nm", "condition"])

    for ax, (col_before, col_after, ylabel) in zip(
        axes, [("D_before", "D_after", r"D (µm²/s)"), ("n_before", "n_after", r"n")]
    ):
        for (size_nm, condition), g in groups:
            label_nm = dls_labels.get(size_nm, int(size_nm))
            for _, row in g.iterrows():
                if not (np.isfinite(row[col_before]) and np.isfinite(row[col_after])):
                    continue
                ax.plot([0, 1], [row[col_before], row[col_after]], "o-", alpha=0.5, markersize=4,
                        color=COLOR_MEASURED if condition == "water" else COLOR_CORRECTED)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["before", "after"])
        ax.set_ylabel(ylabel)
    fig.suptitle("Localization-error correction: paired before/after per file "
                 "(blue = water, red = hydrogel)", fontsize=11, fontweight="semibold")
    return fig


def main() -> None:
    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    hydrogel_results = _load_pickle(CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    print(f"Cache geladen: {CACHE_20MG} ({len(hydrogel_results)} Dateien)")
    locerr_df = _load_pickle(CACHE_LOCERR, "TaskA_Immobilized_Compute.py")
    print(f"Cache geladen: {CACHE_LOCERR} ({len(locerr_df)} Tracks)")

    table_water, detail_water = build_comparison_table(d0_results, "water", locerr_df)
    table_gel, detail_gel = build_comparison_table(hydrogel_results, "hydrogel", locerr_df)
    table = pd.concat([table_water, table_gel], ignore_index=True)
    detail = {**detail_water, **detail_gel}

    print(table.to_string(index=False))

    dls_labels = get_dls_labels()
    with plt.rc_context(_RC):
        # Fig 2: one representative file per (size, condition) -- the one with most tracks
        for (size_nm, condition), g in table.groupby(["particle_size_nm", "condition"]):
            g_valid = g.dropna(subset=["num_tracks"])
            if g_valid.empty:
                continue
            best_xml = g_valid.loc[g_valid["num_tracks"].idxmax(), "xml_path"]
            fig = plot_msd_correction_example(detail[best_xml], dls_labels)
            plt.show()
            if SAVE_PATH is not None:
                SAVE_PATH.mkdir(parents=True, exist_ok=True)
                fname = f"msd_correction_{condition}_{int(size_nm)}nm.png"
                fig.savefig(SAVE_PATH / fname, dpi=600, bbox_inches="tight")
                print(f"Plot gespeichert: {SAVE_PATH / fname}")

        # Fig 3: paired D/n before-after
        fig3 = plot_paired_comparison(table, dls_labels)
        plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            fig3.savefig(SAVE_PATH / "d_n_paired_before_after.png", dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {SAVE_PATH / 'd_n_paired_before_after.png'}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "msd_correction_comparison.csv"
        table.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
