"""
MSD-intercept diagnostic vs. independently-measured localization error.

Pure consumer: reads cache/msd_d0_results.pkl (MSD_FromTrackmate_D0.py) and
cache/localization_error_immobilized.pkl (TaskA_Immobilized_Compute.py).
Never touches raw XML/TIFF.

For each water file, redoes the linear MSD fit MSD = slope*tau + b itself
from the already-cached fit_results_MSD["emsd"], over the SAME fit_points
already used by the primary power-law fit (fit_results_MSD["fit_points"]),
keeping the SIGNED intercept b (um^2). This is necessary because the
existing core.analysis.perform_msd_analysis() (unmodified here, per
CLAUDE.md) already computes this same linear fit internally but silently
converts a negative intercept to NaN in its own sigma_loc_nm field -- that
is insufficient for reporting whether negative intercepts occur, so this
script keeps the sign itself instead of reading that field back.

(hydro_analysis/MSD_Trackmate/Location_Error_From_MSD.py does a related
signed-intercept fit from raw XML, but hardcodes only the first 2 points
regardless of its own configured LOCATION_FIT_POINTS -- cited here as
related prior art, not reused, since it would not reproduce the primary
analysis's actual fit range.)

A negative b is reported as a negative intercept value -- it is NEVER
converted into "negative sigma_loc". sigma_from_intercept_nm is only
defined (non-NaN) where b > 0, and is a separate, fit-range-dependent
diagnostic quantity, not a replacement for the independently-measured
sigma_loc_immobilized_nm from Task A1.

Run MSD_FromTrackmate_D0.py and TaskA_Immobilized_Compute.py first (or
after any raw-data change) to refresh both inputs.

Shows the figure first (plt.show(), blocking); only after the window is
closed does it save into its own
Auswertungsbilder\\LocError_InterceptComparison\\ subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0 = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_LOCERR = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "LocError_InterceptComparison"

TARGET_SIZES_NM = (20.0, 50.0)   # 35nm/50nm headline; other sizes are controls, still computed

COLOR_B = "#3B6E8C"
COLOR_B_DARK = "#2A4F66"
COLOR_REF = "#da00bd"
COLOR_REF_DARK = "#9b5191"


def _load_pickle(path: Path, compute_script: str):
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _signed_intercept(emsd: pd.Series, fit_points: int) -> tuple[float, float]:
    """Redo the linear MSD fit over the same fit range, keep the signed intercept.
    Returns (slope_um2_per_s, intercept_b_um2), both NaN if not computable."""
    if emsd is None or emsd.empty:
        return np.nan, np.nan
    xs = emsd.index.values.astype(float)[:fit_points]
    ys = emsd.values.astype(float)[:fit_points]
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def build_table(d0_results: dict) -> pd.DataFrame:
    rows = []
    for r in d0_results.values():
        fit = r.get("fit_results_MSD")
        if fit is None:
            continue
        size_nm = r.get("particle_size_nm")
        emsd = fit.get("emsd")
        fit_points = fit.get("fit_points")
        if size_nm is None or emsd is None or fit_points is None:
            continue
        slope, intercept_b_um2 = _signed_intercept(emsd, fit_points)
        sigma_from_intercept_nm = (
            float(np.sqrt(intercept_b_um2 / 4.0) * 1000.0) if np.isfinite(intercept_b_um2) and intercept_b_um2 > 0
            else np.nan
        )
        # Redundancy check against core.analysis's own (sign-destroyed) field -- both are
        # the identical computation wherever both are finite; printed as a sanity check.
        core_sigma_loc_nm = fit.get("sigma_loc_nm", np.nan)
        rows.append({
            "xml_path": r.get("xml_path"),
            "base_name": r.get("base_name"),
            "particle_size_nm": float(size_nm),
            "num_tracks": r.get("num_tracks"),
            "fit_points": fit_points,
            "slope_um2_per_s": slope,
            "intercept_b_um2": intercept_b_um2,
            "sigma_from_intercept_nm": sigma_from_intercept_nm,
            "core_sigma_loc_nm": core_sigma_loc_nm,
        })
    return pd.DataFrame(rows)


def _representative_sigma_loc(locerr_df: pd.DataFrame, size_nm: float) -> float:
    """Pooled mean of x & y static-SD across both immobilized datasets, for one nominal size."""
    sub = locerr_df[locerr_df["particle_size_nm"] == size_nm]
    if sub.empty:
        return np.nan
    vals = np.concatenate([sub["sigma_x_static_nm"].dropna().to_numpy(),
                            sub["sigma_y_static_nm"].dropna().to_numpy()])
    return float(np.mean(vals)) if vals.size else np.nan


def plot_intercept_distribution(table: pd.DataFrame, locerr_df: pd.DataFrame,
                                 dls_labels: dict[float, int]) -> plt.Figure:
    sizes = sorted(table["particle_size_nm"].dropna().unique())
    n = len(sizes)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2),
                              constrained_layout=True, squeeze=False)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    for ax, size_nm in zip(ax_flat, sizes):
        sub = table[table["particle_size_nm"] == size_nm]
        b_vals = sub["intercept_b_um2"].dropna().to_numpy()
        n_neg = int((b_vals < 0).sum())

        ax.hist(b_vals, bins=15, color=COLOR_B, edgecolor=COLOR_B_DARK, alpha=0.8)
        ax.axvline(0, color="black", linewidth=1.0)

        sigma_loc_nm = _representative_sigma_loc(locerr_df, size_nm)
        if np.isfinite(sigma_loc_nm):
            b_expected_um2 = 4.0 * (sigma_loc_nm / 1000.0) ** 2
            ax.axvline(b_expected_um2, color=COLOR_REF_DARK, linewidth=1.5,
                       label=f"4σ² (immob.) = {b_expected_um2:.3g} µm²")
            ax.legend(fontsize=6, loc="upper right", frameon=False)

        label = f"{dls_labels.get(size_nm, int(size_nm))} nm"
        ax.set_title(f"{label}  |  N={len(b_vals)}, neg={n_neg}", fontsize=9)
        ax.set_xlabel(r"Intercept $b$ (µm²), signed")
        ax.set_ylabel("Count")

    fig.suptitle("MSD-fit intercept vs. independently-measured localization error",
                 fontsize=12, fontweight="semibold")
    return fig


def main() -> None:
    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    locerr_df = _load_pickle(CACHE_LOCERR, "TaskA_Immobilized_Compute.py")
    print(f"Cache geladen: {CACHE_LOCERR} ({len(locerr_df)} Tracks)")

    table = build_table(d0_results)

    max_diff = np.nanmax(np.abs(table["sigma_from_intercept_nm"] - table["core_sigma_loc_nm"]))
    print(f"Redundanz-Check (sigma_from_intercept_nm vs. core sigma_loc_nm): "
          f"max |diff| = {max_diff:.3g} nm (both derived from the same fit, should be ~0 where both finite)")

    for size_nm in sorted(table["particle_size_nm"].dropna().unique()):
        sub = table[table["particle_size_nm"] == size_nm]
        n_neg = int((sub["intercept_b_um2"] < 0).sum())
        print(f"  size={size_nm:>6.1f} nm: N={len(sub)}, negative intercepts={n_neg}")

    dls_labels = get_dls_labels()
    with plt.rc_context(_RC):
        fig = plot_intercept_distribution(table, locerr_df, dls_labels)
        plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            png_path = SAVE_PATH / "intercept_vs_sigma_loc.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {png_path}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "intercept_comparison_summary.csv"
        table.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
