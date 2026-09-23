"""
Final integration table: Validation aspect | Measurement | Main result |
Impact on SPT interpretation.

Pure consumer: reads cache/localization_error_immobilized.pkl
(Loc_Error_Analyse_immob_particle.py), cache/imsd_median_stepsize_by_size.pkl
(iMSD_histograms.py), the two CSVs written by Correlations.py (per-file
loc_err correction, track-length-bin comparison), fittingpoints_robustness.csv
(Fittingpoints_robustness.py) and snr_summary_by_size.csv (SNR_Analysis.py).
Never touches raw XML/TIFF or the msd_*.pkl caches. Must be run last, after
every other script in this folder.

The "Localization error" row is now one row per particle size and includes
sigma_loc divided by the median real diffusive step size at that size --
same ratio, same median-step-size source (Wasser+Hydrogel pooled per size)
as Loc_Error_Analyse_immob_particle.py::plot_locerror_vs_stepsize_ratio.

No intercept-diagnostic row anymore (Correlations.py dropped that plot on
request; see obsolete/TaskA2_A4_Intercept_Comparison.py directly if needed).
No separate "track-length robustness" row anymore either --
TaskB1_TrackLengthBias_Compute.py / TaskB3_MinTrackLength_Sensitivity.py
(a cruder per-track-average approximation, not a real ensemble refit) were
moved to obsolete/, fully superseded by Correlations.py's track-length-bin
comparison (see "Track-length dependence" below).

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
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig

# ── Configuration ──────────────────────────────────────────────────────────────
BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
LOCERR_PKL = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"
STEPSIZE_PKL = Path(__file__).parent.parent / "cache" / "imsd_median_stepsize_by_size.pkl"
# Correlations.py schreibt beide CSVs in einen gemeinsamen Ordner.
CORRECTION_CSV = BASE / "Correlations" / "msd_correction_comparison.csv"
LENGTH_BIN_CSV = BASE / "Correlations" / "length_bin_comparison.csv"
FITTINGPOINTS_CSV = BASE / "Fittingpoints_Robustness" / "fittingpoints_robustness.csv"
SNR_CSV = BASE / "SNR_Analysis" / "snr_summary_by_size.csv"
SAVE_PATH: Path | None = BASE / "Validation_Summary"


def _require(path: Path, producer: str):
    if not path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {path}\nBitte zuerst {producer} ausführen.")


def build_summary() -> pd.DataFrame:
    _require(LOCERR_PKL, "Loc_Error_Analyse_immob_particle.py")
    _require(STEPSIZE_PKL, "iMSD_histograms.py")
    _require(CORRECTION_CSV, "Correlations.py")
    _require(LENGTH_BIN_CSV, "Correlations.py")
    _require(FITTINGPOINTS_CSV, "Fittingpoints_robustness.py")
    _require(SNR_CSV, "SNR_Analysis.py")

    with open(LOCERR_PKL, "rb") as f:
        locerr_df = pickle.load(f)
    with open(STEPSIZE_PKL, "rb") as f:
        median_stepsize_um: dict[float, float] = pickle.load(f)
    correction_df = pd.read_csv(CORRECTION_CSV)
    length_bin_df = pd.read_csv(LENGTH_BIN_CSV)
    fittingpoints_df = pd.read_csv(FITTINGPOINTS_CSV)
    snr_df = pd.read_csv(SNR_CSV)

    d_change = correction_df["D_rel_change"].dropna()
    n_change = correction_df["n_rel_change"].dropna()

    bin_pivot_d = length_bin_df.pivot_table(index=["particle_size_nm", "condition"],
                                             columns="length_bin", values="D_um2_per_s", aggfunc="median")
    bin_pivot_n = length_bin_df.pivot_table(index=["particle_size_nm", "condition"],
                                             columns="length_bin", values="exponent", aggfunc="median")
    bin_d_rel_range = ((bin_pivot_d.max(axis=1) - bin_pivot_d.min(axis=1)) / bin_pivot_d.min(axis=1)).max()
    bin_n_abs_range = (bin_pivot_n.max(axis=1) - bin_pivot_n.min(axis=1)).max()

    fp_d_pivot = fittingpoints_df.pivot_table(index=["partikelgröße_nm", "bedingung"],
                                               columns="fittingpoints", values="D_median")
    fp_n_pivot = fittingpoints_df.pivot_table(index=["partikelgröße_nm", "bedingung"],
                                               columns="fittingpoints", values="n_median")
    fp_d_rel_range = ((fp_d_pivot.max(axis=1) - fp_d_pivot.min(axis=1)) / fp_d_pivot.min(axis=1)).max()
    fp_n_abs_range = (fp_n_pivot.max(axis=1) - fp_n_pivot.min(axis=1)).max()

    rows = []
    for size_nm in sorted(locerr_df["particle_size_nm"].dropna().unique()):
        sub = locerr_df[locerr_df["particle_size_nm"] == size_nm]
        sigma_nm = float(sub[["sigma_x_static_nm", "sigma_y_static_nm"]].to_numpy().mean())
        step_um = median_stepsize_um.get(float(size_nm))
        step_nm = step_um * 1000.0 if step_um is not None else np.nan
        ratio = sigma_nm / step_nm if np.isfinite(step_nm) and step_nm > 0 else np.nan
        main_result = (
            f"sigma_loc = {sigma_nm:.0f} nm, mediane Schrittweite = {step_nm:.0f} nm, "
            f"Verhaeltnis = {ratio:.2f}"
            if np.isfinite(ratio)
            else f"sigma_loc = {sigma_nm:.0f} nm, keine Schrittweiten-Referenz fuer diese Groesse"
        )
        rows.append({
            "Validation aspect": "Localization error",
            "Measurement": f"{size_nm:.0f} nm: sigma_loc (immobilized, static estimator, pooled x/y mean) "
                "/ mediane Schrittweite (Wasser+Hydrogel gepoolt, siehe iMSD_histograms.py)",
            "Main result": main_result,
            "Impact on SPT interpretation": "REVIEW: Verhaeltnis nahe/ueber 1 bedeutet rauschlimitierte "
                "statt bewegungslimitierte Messung bei dieser Groesse (gleiche Kennzahl wie "
                "Loc_Error_Analyse_immob_particle.py::plot_locerror_vs_stepsize_ratio).",
        })
    rows += [
        {
            "Validation aspect": "Localization correction",
            "Measurement": "delta D / delta n after 4*sigma_loc^2 subtraction, per file/eMSD end-result "
                "(35/50 nm)",
            "Main result": f"D changes by {d_change.median()*100:.1f}% (median), "
                f"n changes by {n_change.median()*100:.1f}% (median), N={len(correction_df)} files",
            "Impact on SPT interpretation": "REVIEW: state whether this changes the n<1 conclusion.",
        },
        {
            "Validation aspect": "Track-length dependence",
            "Measurement": "median ensemble D/n per file, across track-length bins "
                "(10-20/20-40/40-70/70+ frames), water and hydrogel, per particle size",
            "Main result": f"max relative D_median range across bins = {bin_d_rel_range*100:.1f}%, "
                f"max n_median range across bins = {bin_n_abs_range:.2f}",
            "Impact on SPT interpretation": "REVIEW: state whether files agree well within a bin (real "
                "physics) or scatter widely (processing artifact), and whether D/n trend systematically "
                "from short to long tracks.",
        },
        {
            "Validation aspect": "Fitting-points robustness",
            "Measurement": "per-track D_median/n_median across 4-8 MSD fit points, per (size, condition)",
            "Main result": f"max relative D_median range = {fp_d_rel_range*100:.1f}%, "
                f"max n_median range = {fp_n_abs_range:.2f} (across all size/condition groups)",
            "Impact on SPT interpretation": "REVIEW: state whether the default fit_points="
                "DEFAULT_MSD_FIT_POINTS=6 sits in a stable region of this range or near an edge where "
                "D/n are still changing quickly with the number of fit points.",
        },
    ]
    for _, g in snr_df.sort_values(["particle_size_nm", "condition"]).iterrows():
        rows.append({
            "Validation aspect": "SNR / Detektierbarkeit",
            "Measurement": f"{g['condition']}, {g['particle_size_nm']:.0f} nm, N={int(g['n'])} Partikel "
                "(Peak/Hintergrund-SNR, siehe SNR_Analysis.py)",
            "Main result": f"Median SNR = {g['median']:.1f} (IQR {g['q25']:.1f}-{g['q75']:.1f})",
            "Impact on SPT interpretation": "REVIEW: SNR nahe/unter ~2-3 deutet auf rauschlimitierte "
                "Detektion bei dieser Groesse/Bedingung hin, nicht nur bewegungslimitierte Messung.",
        })
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
            safe_savefig(fig, SAVE_PATH / "validation_summary_table.png", dpi=600, bbox_inches="tight")
            print(f"Plot gespeichert: {SAVE_PATH / 'validation_summary_table.png'}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = SAVE_PATH / "validation_summary_table.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
