"""
Normalize MSD diffusion coefficients: D_hydrogel / D_wasser.

Uses:
  - Pickle cache from MSD_FromTrackmate_20mg.py (hydrogel)
  - Pickle cache from MSD_FromTrackmate_D0.py (water)
  - DLS measurements from core.io.get_dls_reference_maps (water reference)
"""

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from core.io import get_dls_reference_maps

# -----------------------------
# Configuration
# -----------------------------
CACHE_20MG = Path(__file__).parent / "cache" / "msd_20mg_results.pkl"
CACHE_D0 = Path(__file__).parent / "cache" / "msd_d0_results.pkl"

SAVE_PATH = None
SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")

OUTPUT_CSV = "msd_ratio_hydrogel_vs_water.csv"
OUTPUT_CSV_DLS = "msd_ratio_hydrogel_vs_dls.csv"
OUTPUT_PNG_WATER = "msd_ratio_hydrogel_vs_water.png"
OUTPUT_PDF_WATER = "msd_ratio_hydrogel_vs_water.pdf"
OUTPUT_PNG_DLS = "msd_ratio_hydrogel_vs_dls.png"
OUTPUT_PDF_DLS = "msd_ratio_hydrogel_vs_dls.pdf"
RF_NM = 4.0
X_LIM_MIN = 0.1  # log-scale cannot include 0
X_LIM_MAX = 100.0
Y_LIM_MIN = 0.0
Y_LIM_MAX = 1.0
SHOW_INDIVIDUALS = False


def _load_cache(path: Path) -> dict:
    if not path.exists():
        print(f"Cache not found: {path}")
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_msd_by_size(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        size = r.get("particle_size_nm")
        d = r.get("D_MSD")
        if size is None or d is None:
            continue
        rows.append({"particle_size_nm": float(size), "D_MSD": float(d)})
    if not rows:
        return pd.DataFrame(columns=["particle_size_nm", "D_MSD"])
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["particle_size_nm", f"D_{label}", f"D_{label}_std", f"n_{label}"])
    g = df.groupby("particle_size_nm")["D_MSD"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(
        columns={
            "mean": f"D_{label}",
            "std": f"D_{label}_std",
            "count": f"n_{label}",
        }
    )
    return out


def _ratio_with_error(d_num: float, d_den: float, err_num: float, err_den: float) -> tuple[float, float]:
    if not np.isfinite(d_num) or not np.isfinite(d_den) or d_den == 0:
        return np.nan, np.nan
    ratio = d_num / d_den
    if not np.isfinite(err_num) or not np.isfinite(err_den) or d_num == 0 or d_den == 0:
        return ratio, np.nan
    rel = np.sqrt((err_num / d_num) ** 2 + (err_den / d_den) ** 2)
    return ratio, ratio * rel


def _map_size(size: float, size_override: dict[float, float] | None) -> float:
    if size_override and size in size_override:
        return float(size_override[size])
    return float(size)


def _size_err(size: float, size_err: dict[float, float] | None) -> float:
    if size_err and size in size_err:
        return float(size_err[size])
    return 0.0


def plot_ratio_with_individuals(
    df_individual: pd.DataFrame,
    df_avg: pd.DataFrame,
    ratio_col: str,
    ratio_err_col: str,
    size_override: dict[float, float] | None,
    size_err: dict[float, float] | None,
    title: str,
    out_png: Path | None,
    out_pdf: Path | None,
    fit_line: dict | None = None,
    show_individuals: bool = True,
) -> None:
    if df_individual.empty and df_avg.empty:
        print(f"No data to plot: {title}")
        return

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

    # Individual file ratios
    if show_individuals and not df_individual.empty:
        for size in sorted(df_individual["particle_size_nm"].unique()):
            subset = df_individual[df_individual["particle_size_nm"] == size]
            x = _map_size(float(size), size_override)
            vals = subset[ratio_col].dropna().values
            if len(vals) == 0:
                continue
            ax.scatter([x] * len(vals), vals, s=40, alpha=0.5, color="#0000da", zorder=2)

    # Average ratios with error bars
    if not df_avg.empty:
        xs = [_map_size(float(s), size_override) for s in df_avg["particle_size_nm"].values]
        ys = df_avg[ratio_col].values
        yerr = df_avg[ratio_err_col].values
        xerr = [(_size_err(float(s), size_err) if _size_err(float(s), size_err) > 0.1 else 0.0)
                for s in df_avg["particle_size_nm"].values]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            xerr=xerr,
            fmt="o",
            markersize=10,
            markerfacecolor="#0000da",
            markeredgecolor="#000099",
            markeredgewidth=1.2,
            ecolor="#000099",
            elinewidth=1.6,
            capsize=4.0,
            capthick=1.2,
            zorder=5,
            label="Mean ± SD",
        )

    ax.set_xscale("log")
    ax.set_xlim(X_LIM_MIN, X_LIM_MAX)
    ax.set_ylim(Y_LIM_MIN, Y_LIM_MAX)
    ax.set_xlabel("Particle Size (nm)")
    ax.set_ylabel("D_hydrogel / D_reference")
    ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.grid(True, alpha=0.2)
    if fit_line is not None:
        ax.plot(fit_line["x"], fit_line["y"], color="black", linewidth=2.0, label=fit_line["label"])
    ax.legend(loc="best", fontsize=9, frameon=False)

    plt.show()

    if out_png is not None:
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    hydrogel = _load_cache(CACHE_20MG)
    water = _load_cache(CACHE_D0)

    df_h = _extract_msd_by_size(hydrogel)
    df_w = _extract_msd_by_size(water)

    agg_h = _aggregate(df_h, "hydrogel")
    agg_w = _aggregate(df_w, "water")

    if agg_h.empty:
        print("No hydrogel MSD data found.")
        return

    merged = agg_h.merge(agg_w, on="particle_size_nm", how="left")

    ratios = []
    for _, row in merged.iterrows():
        r, r_err = _ratio_with_error(
            row.get("D_hydrogel", np.nan),
            row.get("D_water", np.nan),
            row.get("D_hydrogel_std", np.nan),
            row.get("D_water_std", np.nan),
        )
        ratios.append((r, r_err))
    merged["ratio_hydrogel_over_water"] = [r for r, _ in ratios]
    merged["ratio_hydrogel_over_water_err"] = [e for _, e in ratios]

    # DLS normalization (hydrogel / DLS-water)
    dls_maps = get_dls_reference_maps()
    dls_D = dls_maps["dls_D_um2_per_s"]
    dls_err = dls_maps["dls_D_err_um2_per_s"]

    dls_rows = []
    for _, row in agg_h.iterrows():
        size = float(row["particle_size_nm"])
        d_h = row["D_hydrogel"]
        d_h_err = row["D_hydrogel_std"]
        d_dls = dls_D.get(size, np.nan)
        d_dls_err = dls_err.get(size, np.nan)
        ratio, ratio_err = _ratio_with_error(d_h, d_dls, d_h_err, d_dls_err)
        dls_rows.append(
            {
                "particle_size_nm": size,
                "D_hydrogel": d_h,
                "D_hydrogel_std": d_h_err,
                "D_dls": d_dls,
                "D_dls_err": d_dls_err,
                "ratio_hydrogel_over_dls": ratio,
                "ratio_hydrogel_over_dls_err": ratio_err,
            }
        )
    dls_df = pd.DataFrame(dls_rows)

    print("\nHydrogel vs Water (MSD) ratios:")
    print(merged.sort_values("particle_size_nm").to_string(index=False))

    print("\nHydrogel vs DLS ratios:")
    print(dls_df.sort_values("particle_size_nm").to_string(index=False))

    if SAVE_PATH is not None:
        out_dir = Path(SAVE_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_dir / OUTPUT_CSV, index=False)
        dls_df.to_csv(out_dir / OUTPUT_CSV_DLS, index=False)
        print(f"\nSaved: {out_dir / OUTPUT_CSV}")
        print(f"Saved: {out_dir / OUTPUT_CSV_DLS}")

    # Individual file ratios (hydrogel / water mean)
    water_mean_map = agg_w.set_index("particle_size_nm")["D_water"].to_dict()
    df_h_ratio = df_h.copy()
    df_h_ratio["ratio_hydrogel_over_water"] = df_h_ratio["D_MSD"] / df_h_ratio["particle_size_nm"].map(water_mean_map)
    df_h_ratio = df_h_ratio.dropna(subset=["ratio_hydrogel_over_water"])

    # Individual file ratios (hydrogel / DLS)
    df_h_ratio_dls = df_h.copy()
    df_h_ratio_dls["ratio_hydrogel_over_dls"] = df_h_ratio_dls["D_MSD"] / df_h_ratio_dls["particle_size_nm"].map(dls_D)
    df_h_ratio_dls = df_h_ratio_dls.dropna(subset=["ratio_hydrogel_over_dls"])

    size_override = dls_maps["size_override_nm"]
    size_err = dls_maps["size_err_nm"]

    # Fit Amsden model to hydrogel/water averages
    fit_line = None
    if not merged.empty:
        x_sizes = merged["particle_size_nm"].values.astype(float)
        x_diam = np.array([_map_size(s, size_override) for s in x_sizes], dtype=float)
        y_ratio = merged["ratio_hydrogel_over_water"].values.astype(float)
        y_err = merged["ratio_hydrogel_over_water_err"].values.astype(float)

        mask = np.isfinite(x_diam) & np.isfinite(y_ratio) & (x_diam > 0) & (y_ratio > 0)
        mask &= (x_diam <= X_LIM_MAX)

        x_fit = x_diam[mask] / 2.0  # rs = radius
        y_fit = y_ratio[mask]
        y_sigma = y_err[mask] if y_err.size == y_ratio.size else None

        if x_fit.size >= 2:
            def amsden_model(rs_nm, k, phi_v_nm, rf_nm=RF_NM):
                denom = (k * phi_v_nm + 2.0 * rf_nm)
                return np.exp(-np.pi * ((rs_nm + rf_nm) / denom) ** 2)

            try:
                p0 = [1.0, 50.0]
                bounds = ([1e-6, 1e-6], [1e6, 1e6])
                popt, _ = curve_fit(
                    lambda r, k, phi: amsden_model(r, k, phi),
                    x_fit,
                    y_fit,
                    p0=p0,
                    sigma=y_sigma if y_sigma is not None and np.all(np.isfinite(y_sigma)) else None,
                    absolute_sigma=False,
                    bounds=bounds,
                    maxfev=10000,
                )
                k_fit, phi_fit = popt
                x_line = np.logspace(np.log10(max(X_LIM_MIN, 0.1)), np.log10(X_LIM_MAX), 200)
                y_line = amsden_model(x_line / 2.0, k_fit, phi_fit)
                fit_line = {
                    "x": x_line,
                    "y": y_line,
                    "label": f"Amsden fit (k={k_fit:.3g}, φᵛ={phi_fit:.3g} nm)",
                }
            except Exception as e:
                print(f"Amsden fit failed: {e}")

    # Plot hydrogel vs water
    out_png = Path(SAVE_PATH) / OUTPUT_PNG_WATER if SAVE_PATH is not None else None
    out_pdf = Path(SAVE_PATH) / OUTPUT_PDF_WATER if SAVE_PATH is not None else None
    plot_ratio_with_individuals(
        df_h_ratio,
        merged,
        "ratio_hydrogel_over_water",
        "ratio_hydrogel_over_water_err",
        size_override,
        size_err,
        "Hydrogel / Water (MSD)",
        out_png,
        out_pdf,
        fit_line=fit_line,
        show_individuals=SHOW_INDIVIDUALS,
    )

    # Plot hydrogel vs DLS
    out_png_dls = Path(SAVE_PATH) / OUTPUT_PNG_DLS if SAVE_PATH is not None else None
    out_pdf_dls = Path(SAVE_PATH) / OUTPUT_PDF_DLS if SAVE_PATH is not None else None
    plot_ratio_with_individuals(
        df_h_ratio_dls,
        dls_df,
        "ratio_hydrogel_over_dls",
        "ratio_hydrogel_over_dls_err",
        size_override,
        size_err,
        "Hydrogel / DLS",
        out_png_dls,
        out_pdf_dls,
        show_individuals=SHOW_INDIVIDUALS,
    )


if __name__ == "__main__":
    main()
