"""
Estimate localization error from MSD intercept.

MSD(Δt) ≈ 4DΔt + 4σ²  ->  σ = sqrt(intercept / 4)

Uses core IO + MSD analysis to keep workflow consistent.
Scans a folder recursively for TrackMate XML files.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.io import build_datasets
from core.analysis import perform_msd_analysis, DEFAULT_MSD_FIT_POINTS

# -----------------------------
# Configuration
# -----------------------------

# Folder to scan (recursively) for TrackMate XML files
ROOT_FOLDER = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.28 _ Immobilized\Tracks")

# Number of MSD points for the MSD power-law fit (core analysis)
MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS

# Number of early MSD points for linear fit (intercept -> localization error)
LOCATION_FIT_POINTS = 6

# Per-file output only (no saving)
OUTPUT_CSV: Optional[Path] = None

# Plot results after processing
SHOW_PLOT = True

# Plot MSD/IMSD per file with y-intercept line
SHOW_MSD_PLOTS = True


def fit_location_error_from_emsd(emsd: pd.Series, points: int) -> Dict[str, Any]:
    """Fit MSD(Δt) = 4DΔt + 4σ² using early MSD points and return σ."""
    if emsd is None or emsd.empty:
        return {
            "n_points": 0,
            "fit_points_requested": int(points),
            "slope_um2_per_s": np.nan,
            "intercept_um2": np.nan,
            "D_um2_per_s": np.nan,
            "sigma2_um2": np.nan,
            "sigma_um": np.nan,
            "fit_reason": "emsd_empty",
        }

    n = min(int(points), len(emsd))
    if n < 2:
        return {
            "n_points": n,
            "fit_points_requested": int(points),
            "slope_um2_per_s": np.nan,
            "intercept_um2": np.nan,
            "D_um2_per_s": np.nan,
            "sigma2_um2": np.nan,
            "sigma_um": np.nan,
            "fit_reason": "too_few_points",
        }

    xs = emsd.index.values.astype(float)[:n]
    ys = emsd.values.astype(float)[:n]
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    n_used = len(xs)
    if n_used < 2:
        return {
            "n_points": n_used,
            "fit_points_requested": int(points),
            "slope_um2_per_s": np.nan,
            "intercept_um2": np.nan,
            "D_um2_per_s": np.nan,
            "sigma2_um2": np.nan,
            "sigma_um": np.nan,
            "fit_reason": "nonfinite_points",
        }

    slope, intercept = np.polyfit(xs[:2], ys[:2], 1)
    sigma2_um2 = intercept / 4.0
    sigma_um = np.sqrt(sigma2_um2) if sigma2_um2 >= 0 else np.nan
    D_um2_per_s = slope / 4.0

    return {
        "n_points": n_used,
        "fit_points_requested": int(points),
        "slope_um2_per_s": float(slope),
        "intercept_um2": float(intercept),
        "D_um2_per_s": float(D_um2_per_s),
        "sigma2_um2": float(sigma2_um2),
        "sigma_um": float(sigma_um) if np.isfinite(sigma_um) else np.nan,
        "fit_reason": None if np.isfinite(sigma_um) else ("negative_intercept" if intercept < 0 else "fit_nan"),
    }


def plot_msd_with_intercept(
    imsd: Optional[pd.DataFrame],
    emsd: Optional[pd.Series],
    loc: Dict[str, Any],
    base_name: str,
) -> None:
    if emsd is None or emsd.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))

    if isinstance(imsd, pd.DataFrame) and not imsd.empty:
        for col in imsd.columns:
            ax.plot(imsd.index.values, imsd[col].values, color="tab:gray", alpha=0.25, lw=0.8)

    ax.plot(emsd.index.values, emsd.values, "o-", color="tab:blue", label="EMSD")

    slope = loc.get("slope_um2_per_s", np.nan)
    intercept = loc.get("intercept_um2", np.nan)
    if np.isfinite(slope) and np.isfinite(intercept):
        n_fit = int(loc.get("fit_points_requested", loc.get("n_points", 0)))
        t_fit = emsd.index.values.astype(float)[:n_fit] if n_fit > 0 else emsd.index.values.astype(float)
        t_line = np.concatenate(([0.0], t_fit)) if len(t_fit) else np.array([0.0])
        y_line = slope * t_line + intercept
        ax.plot(t_line, y_line, "--", color="tab:red", label="Linear fit")
        ax.axhline(intercept, color="tab:orange", ls=":", lw=1.2, label="Intercept (4σ²)")

    ax.set_xlabel("Δt (s)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title(f"MSD + Intercept: {base_name}")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    datasets = build_datasets(ROOT_FOLDER)
    if not datasets:
        print("No XML files found for analysis.")
        return

    total = len(datasets)
    print(f"\nLocation error analysis for {total} files")
    print("=" * 70)

    plot_rows = []
    for idx, (_, result) in enumerate(datasets.items(), 1):
        base_name = result.get("base_name", "unknown")
        print(f"\n[{idx}/{total}] {base_name}")

        try:
            perform_msd_analysis(result, fit_points=MSD_FIT_POINTS)
            fit = result.get("fit_results_MSD") or {}
            emsd = fit.get("emsd")
            imsd = fit.get("imsd")
            loc = fit_location_error_from_emsd(emsd, points=LOCATION_FIT_POINTS)

            sigma_um = loc["sigma_um"]
            sigma_nm = sigma_um * 1000.0 if np.isfinite(sigma_um) else np.nan

            if np.isfinite(sigma_nm):
                print(f"  σ = {sigma_nm:.2f} nm")
            else:
                reason = loc.get("fit_reason") or "unknown"
                print(f"  σ = NaN (reason: {reason})")
            plot_rows.append({"base_name": base_name, "sigma_nm": sigma_nm})
            if SHOW_MSD_PLOTS:
                plot_msd_with_intercept(imsd, emsd, loc, base_name)
        except Exception as exc:
            print(f"  Failed: {exc}")

    if SHOW_PLOT and plot_rows:
        df = pd.DataFrame(plot_rows)
        df = df[np.isfinite(df["sigma_nm"])]
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(df))
            ax.plot(x, df["sigma_nm"].values, "o", color="tab:blue")
            ax.set_ylabel("Localization error σ (nm)")
            ax.set_xlabel("File")
            ax.set_title("Localization error from MSD intercept")
            if len(df) <= 20:
                ax.set_xticks(x)
                ax.set_xticklabels(df["base_name"].tolist(), rotation=45, ha="right")
            else:
                ax.set_xticks(x[:: max(1, len(df) // 10)])
                ax.set_xticklabels([str(i + 1) for i in x[:: max(1, len(df) // 10)]])
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
