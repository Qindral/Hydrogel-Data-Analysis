"""
Final integration table: Validation aspect | Measurement | Main result |
Impact on SPT interpretation.

Pure consumer: reads the CSV/pickle outputs of Tasks A, B, and C. Never
touches raw XML/TIFF or the msd_*.pkl caches. Must be run last, after every
other script in this folder.

The "Impact on SPT interpretation" column is AUTHORED PROSE seeded from the
numeric results in the other columns -- it is not itself a further
computation. Review and edit those cells before using this table in the
dissertation; this script only assembles and formats what the other tasks
already measured.

Shows the table first (plt.show() on a rendered table figure, blocking);
only after the window is closed does it save into its own
Auswertungsbilder\\Validation_Summary\\ subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

# ── Configuration ──────────────────────────────────────────────────────────────
BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
LOCERR_PKL = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"
INTERCEPT_CSV = BASE / "LocError_InterceptComparison" / "intercept_comparison_summary.csv"
CORRECTION_CSV = BASE / "MSD_Correction_LocError" / "msd_correction_comparison.csv"
LENGTH_CORR_CSV = BASE / "TrackLengthBias_Scatter" / "track_length_correlation_stats.csv"
SENSITIVITY_CSV = BASE / "MinTrackLength_Sensitivity" / "min_track_length_sensitivity_table.csv"
DETECT_CSV = BASE / "Detectability_Composite" / "detectability_selection_table.csv"
SAVE_PATH: Path | None = BASE / "Validation_Summary"


def _require(path: Path, producer: str):
    if not path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {path}\nBitte zuerst {producer} ausführen.")


def build_summary() -> pd.DataFrame:
    _require(LOCERR_PKL, "TaskA_Immobilized_Compute.py")
    _require(INTERCEPT_CSV, "TaskA2_A4_Intercept_Comparison.py")
    _require(CORRECTION_CSV, "TaskA3_MSD_Correction.py")
    _require(LENGTH_CORR_CSV, "TaskB2_LengthBias_Scatter.py")
    _require(SENSITIVITY_CSV, "TaskB3_MinTrackLength_Sensitivity.py")
    _require(DETECT_CSV, "TaskC_Detectability_Figure.py")

    with open(LOCERR_PKL, "rb") as f:
        locerr_df = pickle.load(f)
    intercept_df = pd.read_csv(INTERCEPT_CSV)
    correction_df = pd.read_csv(CORRECTION_CSV)
    length_corr_df = pd.read_csv(LENGTH_CORR_CSV)
    sensitivity_df = pd.read_csv(SENSITIVITY_CSV)
    detect_df = pd.read_csv(DETECT_CSV)

    sigma_35 = locerr_df.loc[locerr_df["particle_size_nm"] == 20.0,
                              ["sigma_x_static_nm", "sigma_y_static_nm"]].to_numpy().mean()
    n_neg_total = int((intercept_df["intercept_b_um2"] < 0).sum())
    n_files_total = len(intercept_df)

    d_change = correction_df["D_rel_change"].dropna()
    n_change = correction_df["n_rel_change"].dropna()

    max_abs_corr = length_corr_df.loc[
        length_corr_df["particle_size_nm"].isin([20.0, 50.0]), "pearson_r"
    ].abs().max()

    sens_35 = sensitivity_df[(sensitivity_df["particle_size_nm"] == 20.0)]
    d_median_range = (
        f"{sens_35['D_median'].min():.3g}-{sens_35['D_median'].max():.3g} µm²/s"
        if not sens_35.empty else "n/a"
    )

    challenging_row = detect_df[detect_df["role"] == "challenging"]
    good_row = detect_df[detect_df["role"] == "good"]
    challenging_snr = float(challenging_row["snr"].iloc[0]) if not challenging_row.empty else np.nan
    good_snr = float(good_row["snr"].iloc[0]) if not good_row.empty else np.nan

    rows = [
        {
            "Validation aspect": "Localization error",
            "Measurement": "sigma_loc (immobilized, static estimator, 35 nm)",
            "Main result": f"sigma_loc ~ {sigma_35:.0f} nm (pooled x/y mean, 35 nm)",
            "Impact on SPT interpretation": "REVIEW: compare to the particle's own size / typical step "
                "size for this dataset before concluding materiality.",
        },
        {
            "Validation aspect": "Localization correction",
            "Measurement": "delta D / delta n after 4*sigma_loc^2 subtraction (35/50 nm)",
            "Main result": f"D changes by {d_change.median()*100:.1f}% (median), "
                f"n changes by {n_change.median()*100:.1f}% (median), N={len(correction_df)} files",
            "Impact on SPT interpretation": "REVIEW: state whether this changes the n<1 conclusion.",
        },
        {
            "Validation aspect": "MSD-fit intercept diagnostic",
            "Measurement": "signed intercept b vs. independently-measured sigma_loc",
            "Main result": f"{n_neg_total} of {n_files_total} files have a NEGATIVE fitted intercept "
                "(never reinterpreted as negative sigma_loc)",
            "Impact on SPT interpretation": "REVIEW: note the intercept's fit-range sensitivity as a "
                "caveat, not as an independent sigma_loc estimate.",
        },
        {
            "Validation aspect": "Track-length dependence",
            "Measurement": "Pearson r of per-track D/n vs. log10(track length), 35/50 nm",
            "Main result": f"max |Pearson r| = {max_abs_corr:.2f} across 35/50 nm, water & hydrogel",
            "Impact on SPT interpretation": "REVIEW: weak correlations support that ensemble D/n are not "
                "dominated by track-length artifacts, but state the caveat about mean vs. median "
                "sensitivity to short-track outliers noted in TaskB3.",
        },
        {
            "Validation aspect": "Track-length robustness",
            "Measurement": "ensemble D_median (35 nm) across min-length thresholds 5-50 frames",
            "Main result": f"D_median range: {d_median_range}",
            "Impact on SPT interpretation": "REVIEW: state whether this range is small relative to the "
                "condition-to-condition (water vs. hydrogel) difference in D.",
        },
        {
            "Validation aspect": "Particle detectability",
            "Measurement": "SNR of the 'very good' vs. 'challenging' selected 35 nm detections",
            "Main result": f"good SNR = {good_snr:.2g}, challenging SNR = {challenging_snr:.2g}",
            "Impact on SPT interpretation": "REVIEW: state whether the challenging example is "
                "quantitatively distinguishable from the background SNR distribution.",
        },
    ]
    return pd.DataFrame(rows)


def render_table_figure(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.0, 0.6 * len(df) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_text_props(wrap=True)
        if row == 0:
            cell.set_text_props(fontweight="bold")
    fig.tight_layout()
    return fig


def main() -> None:
    summary = build_summary()
    print(summary.to_string(index=False))

    with plt.rc_context(_RC):
        fig = render_table_figure(summary)
        plt.show()
        if SAVE_PATH is not None:
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            fig.savefig(SAVE_PATH / "validation_summary_table.png", dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {SAVE_PATH / 'validation_summary_table.png'}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "validation_summary_table.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
