"""
Refined re-plot of PSF_Analysis.py's psf_scatter_vs_theory.png, per item 11
of the refined-figures spec: the theoretical PSF reference is drawn at
EVERY displayed particle size, not only the smallest
(DIFFRACTION_LIMITED_SIZES_NM=(20.0,) in the original), and now varies
per size instead of being one flat constant line.

Follow-up correction from user feedback: the reported theoretical AND
measured quantity must be the FWHM (full width at half maximum), not
"2*sigma" -- for a Gaussian, FWHM = 2*sqrt(2*ln(2))*sigma ~= 2.3548*sigma,
which is larger than 2*sigma (the diameter-like quantity this script used
before). Both sides of the comparison are converted with the same factor
(FWHM_FACTOR below) so the plot compares like with like:

    FWHM_diffraction_nm = FWHM_FACTOR * calculate_theoretical_psf_sigma(...)
    FWHM_eff_nm(size)   = sqrt(FWHM_diffraction_nm**2 + size_nm**2)
    FWHM_measured_nm    = FWHM_FACTOR * sigma_fit_nm   (per track, from psf_fits.csv)

The quadrature combination itself (confirmed correct by the user) is
unchanged: the observed image is approximately the convolution of the
diffraction-limited PSF with the particle's own extent, and convolving two
Gaussian-like profiles combines their widths in quadrature; taking the
particle's own physical diameter as its Gaussian-equivalent FWHM
contribution (rather than converting it through its own sigma) matches the
reference values the user provided. size_nm is the real, already-measured
DLS particle diameter -- no new or unconfirmed particle-/dye-specific
constant is introduced. The one substantive caveat, carried over verbatim
from PSF_Analysis.py's own docstring, still applies to the diffraction
term: WAVELENGTH_NM=575.0 is only confirmed for the smallest (Nile-Red)
bead; applying the same wavelength to the other, larger sizes' diffraction
term assumes the same dye emission, which is not independently confirmed
for them -- stated explicitly in the legend and console output.

mpp check (per user request): sigma_fit_nm in psf_fits.csv is already
converted from the per-frame pixel fit to nm using each MOVIE's own mpp
(PSF_Analysis.py::process_movie: sigma_x_nm = sx_px * mpp * 1000, not a
single global mpp), so the expected monotonic growth of FWHM with particle
size is already correctly scaled on the measured side; confirmed here by
inspecting psf_fits.csv directly (mpp varies 0.15/0.30 across movies/sizes,
never a single fixed value). This script only re-derives sigma_fit_nm ->
FWHM_measured_nm (a fixed per-row multiplication), it never re-touches
pixel values or mpp itself.

Axis label changed from "Particle Size (DLS, diameter)" to the more
universal "Particle diameter (nm)" per user request.

Does NOT modify PSF_Analysis.py or any cache/CSV -- pure consumer, reads
the existing psf_fits.csv and imports WAVELENGTH_NM, NUMERICAL_APERTURE,
COLOR_FIT, COLOR_XML, COLOR_THEORY unchanged.

Output (own folder, PNG, 600 dpi): psf_scatter_vs_theory_all_sizes.png.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.core.physics import calculate_theoretical_psf_sigma
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.PSF_Analysis import (
    WAVELENGTH_NM, NUMERICAL_APERTURE, DIFFRACTION_LIMITED_SIZES_NM,
    COLOR_FIT, COLOR_XML, COLOR_THEORY,
)

# ── Configuration ──────────────────────────────────────────────────────────────
PSF_CSV = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "PSF_Analysis" / "psf_fits.csv"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "PSF_Analysis"

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))   # FWHM = FWHM_FACTOR * sigma, for a Gaussian


def theoretical_fwhm_nm(size_nm: float) -> float:
    """FWHM_eff(size) = sqrt(FWHM_diffraction^2 + size_nm^2), see module docstring."""
    diffraction_fwhm_nm = FWHM_FACTOR * calculate_theoretical_psf_sigma(WAVELENGTH_NM, NUMERICAL_APERTURE)
    return float(np.hypot(diffraction_fwhm_nm, size_nm))


def plot_psf_scatter_theory_all_sizes(df: pd.DataFrame, dls_labels: dict[float, int]) -> plt.Figure:
    sizes = sorted(df["particle_size_nm"].dropna().unique())
    width = 0.35
    rng = np.random.default_rng(0)
    fig, ax_left = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    ax_right = ax_left.twinx()
    positions = np.arange(len(sizes))

    for x0, size in zip(positions, sizes):
        sub = df.loc[df["particle_size_nm"] == size]
        psf_fwhm_nm = FWHM_FACTOR * sub["sigma_fit_nm"].to_numpy()
        fitting_diameter_px = 2.0 * sub["sigma_xml_px"].to_numpy()

        jitter_l = (rng.random(len(psf_fwhm_nm)) - 1.0) * width * 0.9
        ax_left.scatter(x0 + jitter_l, psf_fwhm_nm, s=12, alpha=0.55,
                        facecolor=COLOR_FIT, edgecolor="none", zorder=3)
        if len(psf_fwhm_nm):
            mean_val = float(np.mean(psf_fwhm_nm))
            ax_left.plot([x0 - width, x0], [mean_val, mean_val], color="black", linewidth=1.6, zorder=4)

        jitter_r = rng.random(len(fitting_diameter_px)) * width * 0.9
        ax_right.scatter(x0 + jitter_r, fitting_diameter_px, s=12, alpha=0.55,
                         facecolor=COLOR_XML, edgecolor="none", zorder=3)
        if len(fitting_diameter_px):
            mean_val = float(np.mean(fitting_diameter_px))
            ax_right.plot([x0, x0 + width], [mean_val, mean_val], color="black", linewidth=1.6, zorder=4)

    # Theory: diffraction-limited FWHM combined in quadrature with each
    # size's own physical (DLS) diameter -- varies per size, drawn at EVERY
    # size position, not only DIFFRACTION_LIMITED_SIZES_NM.
    for x0, size in zip(positions, sizes):
        real_size_nm = float(dls_labels.get(size, size))
        theory_fwhm_nm = theoretical_fwhm_nm(real_size_nm)
        ax_left.plot([x0 - width, x0], [theory_fwhm_nm, theory_fwhm_nm], color=COLOR_THEORY,
                    linewidth=1.6, linestyle=(0, (4, 3)), zorder=5)

    ax_left.set_xticks(positions)
    ax_left.set_xticklabels([f"{dls_labels.get(s, int(s))} nm" for s in sizes])
    ax_left.set_xlabel("Particle diameter (nm)")
    ax_left.set_ylabel("PSF FWHM (nm)", color=COLOR_FIT)
    ax_right.set_ylabel("Fitting pixel diameter (px)", color=COLOR_XML)
    ax_left.tick_params(axis="y", colors=COLOR_FIT)
    ax_right.tick_params(axis="y", colors=COLOR_XML)
    ax_left.set_xlim(-0.7, len(sizes) - 0.3)

    confirmed_label = f"{DIFFRACTION_LIMITED_SIZES_NM[0]:.0f} nm" if DIFFRACTION_LIMITED_SIZES_NM else "smallest size"
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_FIT, markersize=6,
               label="PSF FWHM (nm, left, per track)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_XML, markersize=6,
               label="Fitting pixel diameter (px, right, per track)"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Mean"),
        Line2D([0], [0], color=COLOR_THEORY, linewidth=1.6, linestyle=(0, (4, 3)),
               label=f"Theoretical FWHM = sqrt(diffraction FWHM$^2$ + particle diameter$^2$) "
                     f"({WAVELENGTH_NM:.0f} nm, NA={NUMERICAL_APERTURE}); wavelength confirmed only "
                     f"for the {confirmed_label} bead, applied to all sizes' diffraction term"),
    ]
    ax_left.legend(handles=legend_handles, loc="upper left", fontsize=6.5, frameon=False)
    return fig


def main() -> None:
    if not PSF_CSV.exists():
        raise FileNotFoundError(f"Nicht gefunden: {PSF_CSV}\nBitte zuerst PSF_Analysis.py ausführen.")
    df = pd.read_csv(PSF_CSV)
    dls_labels = get_dls_labels()

    diffraction_fwhm_nm = FWHM_FACTOR * calculate_theoretical_psf_sigma(WAVELENGTH_NM, NUMERICAL_APERTURE)
    print(f"Diffraction-limited term: FWHM = {diffraction_fwhm_nm:.1f} nm (= {FWHM_FACTOR:.4f} * sigma), "
          f"at wavelength={WAVELENGTH_NM:.0f} nm, NA={NUMERICAL_APERTURE} (confirmed only for the "
          f"{DIFFRACTION_LIMITED_SIZES_NM} bead; applied to all sizes' diffraction term, which assumes "
          "the same dye emission for the larger, unconfirmed beads).")
    print("Per-size theoretical FWHM = sqrt(diffraction_FWHM^2 + particle_diameter^2) "
          "(measured side uses the same FWHM_FACTOR * sigma_fit_nm conversion for a like-for-like comparison):")
    for size_nm in sorted(df["particle_size_nm"].dropna().unique()):
        real_size_nm = float(dls_labels.get(size_nm, size_nm))
        theory_fwhm_nm = theoretical_fwhm_nm(real_size_nm)
        print(f"  {real_size_nm:.0f} nm: theoretical FWHM = {theory_fwhm_nm:.1f} nm")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        fig = plot_psf_scatter_theory_all_sizes(df, dls_labels)
        fig_path = SAVE_PATH / "psf_scatter_vs_theory_all_sizes.png"
        safe_savefig(fig, fig_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {fig_path}")


if __name__ == "__main__":
    main()
