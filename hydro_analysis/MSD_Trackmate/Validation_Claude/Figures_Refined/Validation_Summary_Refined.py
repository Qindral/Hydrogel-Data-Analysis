"""
Refined re-issue of Validation_Summary.py's integration table, extended
with the additional coverage the refined figures now provide. Does NOT
modify Validation_Summary.py or any cache/CSV -- pure consumer/recomputer,
reuses that script's data sources unchanged plus two new pieces of
analysis that only exist in the refined scripts:

  - Localization correction, ALL 6 particle sizes instead of just 20/50 nm.
    Validation_Summary.py's row reads msd_correction_comparison.csv, which
    Correlations.py only ever computes for TARGET_SIZES_NM=(20, 50) (the
    only sizes it originally trusted a sigma_loc reference for). Item 9's
    investigation (Correlations_Refined.py::build_correction_table_all_
    sizes) found that the immobilized dataset now actually has a valid
    sigma_loc reference for all 6 sizes, so this table recomputes the same
    correction (correct_and_refit, imported unchanged from Correlations.py)
    for every size and reports one row per size instead of one pooled row.
  - Theoretical PSF vs. measured PSF, per size (new row, not present in the
    original table at all). Reuses PSF_Refined.py's per-size formula
    (diffraction-limited width combined in quadrature with the particle's
    own DLS diameter) against the already-computed psf_fits.csv.

Everything else (localization error/step-size ratio, track-length
dependence, fitting-points robustness, SNR) is read from the SAME
already-computed sources as Validation_Summary.py -- no new computation,
since the refined figures only changed how these are DISPLAYED, not the
underlying numbers.

Shows nothing interactively (headless, unlike Validation_Summary.py, which
blocks on plt.show() -- this one is meant to be re-run silently after
building the refined figures). Saves the table figure and CSV into its own
Figures_Refined/Validation_Summary/ folder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import (
    CACHE_D0, CACHE_20MG, CACHE_LOCERR, _load_pickle,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.PSF_Analysis import DIFFRACTION_LIMITED_SIZES_NM
from hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.Correlations_Refined import (
    build_correction_table_all_sizes,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.PSF_Refined import (
    FWHM_FACTOR, theoretical_fwhm_nm,
)

# ── Configuration ──────────────────────────────────────────────────────────────
BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
LOCERR_PKL = Path(__file__).parent.parent.parent / "cache" / "localization_error_immobilized.pkl"
STEPSIZE_PKL = Path(__file__).parent.parent.parent / "cache" / "imsd_median_stepsize_by_size.pkl"
LENGTH_BIN_CSV = BASE / "Correlations" / "length_bin_comparison.csv"
FITTINGPOINTS_CSV = BASE / "Fittingpoints_Robustness" / "fittingpoints_robustness.csv"
SNR_CSV = BASE / "SNR_Analysis" / "snr_summary_by_size.csv"
PSF_CSV = BASE / "PSF_Analysis" / "psf_fits.csv"
SAVE_PATH = BASE / "Figures_Refined" / "Validation_Summary"


def _require(path: Path, producer: str):
    if not path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {path}\nBitte zuerst {producer} ausführen.")


def build_summary() -> pd.DataFrame:
    _require(LOCERR_PKL, "Loc_Error_Analyse_immob_particle.py")
    _require(STEPSIZE_PKL, "iMSD_histograms.py")
    _require(LENGTH_BIN_CSV, "Correlations.py")
    _require(FITTINGPOINTS_CSV, "Fittingpoints_robustness.py")
    _require(SNR_CSV, "SNR_Analysis.py")
    _require(PSF_CSV, "PSF_Analysis.py")

    with open(LOCERR_PKL, "rb") as f:
        locerr_df = pickle.load(f)
    with open(STEPSIZE_PKL, "rb") as f:
        median_stepsize_um: dict[float, float] = pickle.load(f)
    length_bin_df = pd.read_csv(LENGTH_BIN_CSV)
    fittingpoints_df = pd.read_csv(FITTINGPOINTS_CSV)
    snr_df = pd.read_csv(SNR_CSV)
    psf_df = pd.read_csv(PSF_CSV)
    dls_labels = get_dls_labels()

    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    hydrogel_results = _load_pickle(CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    correction_all = build_correction_table_all_sizes(d0_results, hydrogel_results, locerr_df)

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
        real_nm = dls_labels.get(size_nm, int(size_nm))
        main_result = (
            f"sigma_loc = {sigma_nm:.0f} nm, mediane Schrittweite = {step_nm:.0f} nm, "
            f"Verhaeltnis = {ratio:.2f}"
            if np.isfinite(ratio)
            else f"sigma_loc = {sigma_nm:.0f} nm, keine Schrittweiten-Referenz fuer diese Groesse"
        )
        rows.append({
            "Validation aspect": "Localization error",
            "Measurement": f"{real_nm} nm: sigma_loc (immobilized, static estimator, pooled x/y mean) "
                "/ mediane Schrittweite (Wasser+Hydrogel gepoolt)",
            "Main result": main_result,
            "Impact on SPT interpretation": "REVIEW: Verhaeltnis nahe/ueber 1 bedeutet rauschlimitierte "
                "statt bewegungslimitierte Messung bei dieser Groesse.",
        })

    for size_nm in sorted(correction_all["particle_size_nm"].dropna().unique()):
        sub = correction_all[correction_all["particle_size_nm"] == size_nm]
        real_nm = dls_labels.get(size_nm, int(size_nm))
        with_ref = sub[sub["has_reference"]]
        d_change = ((with_ref["D_after"] - with_ref["D_before"]) / with_ref["D_before"]).dropna()
        n_change = ((with_ref["n_after"] - with_ref["n_before"]) / with_ref["n_before"]).dropna()
        if len(with_ref) and len(d_change):
            main_result = (f"D changes by {d_change.median()*100:.1f}% (median), "
                            f"n changes by {n_change.median()*100:.1f}% (median), N={len(with_ref)} files")
        else:
            main_result = f"no sigma_loc reference for this size, N={len(sub)} files without correction"
        rows.append({
            "Validation aspect": "Localization correction (all sizes, refined)",
            "Measurement": f"{real_nm} nm: delta D / delta n after 4*sigma_loc^2 subtraction, "
                "per file/eMSD end-result (water+hydrogel pooled)",
            "Main result": main_result,
            "Impact on SPT interpretation": "REVIEW: state whether this changes the n<1 conclusion at "
                "this size; previously only 35/50 nm had this correction, now all sizes with a valid "
                "sigma_loc reference do.",
        })

    rows += [
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
                "(Peak/Hintergrund-SNR)",
            "Main result": f"Median SNR = {g['median']:.1f} (IQR {g['q25']:.1f}-{g['q75']:.1f})",
            "Impact on SPT interpretation": "REVIEW: SNR nahe/unter ~2-3 deutet auf rauschlimitierte "
                "Detektion bei dieser Groesse/Bedingung hin, nicht nur bewegungslimitierte Messung.",
        })

    for size_nm in sorted(psf_df["particle_size_nm"].dropna().unique()):
        sub = psf_df[psf_df["particle_size_nm"] == size_nm]
        measured_fwhm_nm = float((FWHM_FACTOR * sub["sigma_fit_nm"]).median())
        real_nm = float(dls_labels.get(size_nm, size_nm))
        theory_fwhm_nm = theoretical_fwhm_nm(real_nm)
        rel_dev = (measured_fwhm_nm - theory_fwhm_nm) / theory_fwhm_nm
        confirmed = " (wavelength confirmed)" if size_nm in DIFFRACTION_LIMITED_SIZES_NM else \
            " (wavelength NOT independently confirmed for this size)"
        rows.append({
            "Validation aspect": "PSF vs. theory (refined, new)",
            "Measurement": f"{int(round(real_nm))} nm: median measured FWHM ({FWHM_FACTOR:.4f}*sigma_fit, "
                f"N={len(sub)} tracks) vs. theoretical sqrt(diffraction_FWHM^2 + diameter^2){confirmed}",
            "Main result": f"measured = {measured_fwhm_nm:.0f} nm, theoretical = {theory_fwhm_nm:.0f} nm, "
                f"deviation = {rel_dev*100:+.1f}%",
            "Impact on SPT interpretation": "REVIEW: large positive deviation may indicate motion blur, "
                "multiple/overlapping emitters, or an out-of-focus population at this size; large negative "
                "deviation may indicate an overly aggressive width-outlier cut.",
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

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        fig = render_table_figure(summary)
        safe_savefig(fig, SAVE_PATH / "validation_summary_table_refined.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {SAVE_PATH / 'validation_summary_table_refined.png'}")

    csv_path = SAVE_PATH / "validation_summary_table_refined.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Tabelle gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
