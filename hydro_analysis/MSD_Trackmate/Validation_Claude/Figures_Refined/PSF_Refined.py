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
    FWHM_measured_nm    = FWHM_FACTOR * sigma_fit     (per track, see Input below)

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

Input: cache/particle_size_snr.pkl (Particle_Size_SNR_Compute.py --mode
all), table "tracks", condition "immobilized" only -- only for immobilized
particles is the measured width free of motion blur and therefore comparable
with the diffraction-limited theory. This replaces the former psf_fits.csv
input (PSF_Analysis.py), which was limited to 200 tracks x 20 frames per
size. Per track: FWHM_measured_nm = median of the accepted per-detection 2D
Gaussian fits (mean of x and y, converted to nm with each movie's own mpp);
tracks with fewer than MIN_ACCEPTED_FITS accepted fits are left out.
Detection diameter = 2 * TrackMate detection radius, converted to nm with
each movie's own mpp; all values are shown in nm on one axis.

Axis label changed from "Particle Size (DLS, diameter)" to the more
universal "Particle diameter (nm)" per user request.

Pure consumer; imports WAVELENGTH_NM, NUMERICAL_APERTURE, COLOR_FIT,
COLOR_XML, COLOR_THEORY from PSF_Analysis.py unchanged.

Output (own folder, PNG, 600 dpi): psf_scatter_vs_theory_all_sizes.png.
"""
from __future__ import annotations

import argparse
import pickle
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
PSS_CACHE = Path(__file__).resolve().parents[2] / "cache" / "particle_size_snr.pkl"
CONDITION = "immobilized"
MIN_ACCEPTED_FITS = 3

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Figures_Refined" / "PSF_Analysis"

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))   # FWHM = FWHM_FACTOR * sigma, for a Gaussian


def theoretical_fwhm_nm(size_nm: float) -> float:
    """FWHM_eff(size) = sqrt(FWHM_diffraction^2 + size_nm^2), see module docstring."""
    diffraction_fwhm_nm = FWHM_FACTOR * calculate_theoretical_psf_sigma(WAVELENGTH_NM, NUMERICAL_APERTURE)
    return float(np.hypot(diffraction_fwhm_nm, size_nm))


def plot_psf_scatter_theory_all_sizes(df: pd.DataFrame, dls_labels: dict[float, int]) -> plt.Figure:
    """Per track, both in nm on one axis: measured FWHM (left of each size
    position) and TrackMate detection diameter 2 R converted with the movie's
    own mpp (right); theory as dashed line over the FWHM column."""
    sizes = sorted(df["particle_size_nm"].dropna().unique())
    width = 0.35
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    positions = np.arange(len(sizes))

    for x0, size in zip(positions, sizes):
        sub = df.loc[df["particle_size_nm"] == size]
        for values, color, side in ((sub["fwhm_nm"].to_numpy(), COLOR_FIT, -1),
                                    (sub["detection_diameter_nm"].to_numpy(), COLOR_XML, +1)):
            if not len(values):
                continue
            jitter = rng.random(len(values)) * width * 0.9 * side
            ax.scatter(x0 + jitter, values, s=12, alpha=0.55, facecolor=color, edgecolor="none", zorder=3)
            mean_val = float(np.mean(values))
            ax.plot(sorted([x0, x0 + side * width]), [mean_val, mean_val], color="black", linewidth=1.6, zorder=4)
        theory = theoretical_fwhm_nm(float(dls_labels.get(size, size)))
        ax.plot([x0 - width, x0], [theory, theory], color=COLOR_THEORY, linewidth=1.6,
                linestyle=(0, (4, 3)), zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{dls_labels.get(s, int(s))} nm" for s in sizes])
    ax.set_xlabel("Particle diameter (nm)")
    ax.set_ylabel("Width (nm)")
    ax.set_xlim(-0.7, len(sizes) - 0.3)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_FIT, markersize=6,
               label="Measured FWHM (2D Gaussian, per track)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_XML, markersize=6,
               label="TrackMate detection diameter 2R (per track)"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Mean"),
        Line2D([0], [0], color=COLOR_THEORY, linewidth=1.6, linestyle=(0, (4, 3)),
               label=f"Theory: FWHM$_{{PSF}}$ ({WAVELENGTH_NM:.0f} nm, NA {NUMERICAL_APERTURE}) "
                     "combined with particle diameter"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, frameon=False)
    return fig


def load_tracks(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {path}\nBitte zuerst Particle_Size_SNR_Compute.py --mode all ausführen.")
    with open(path, "rb") as f:
        tracks = pickle.load(f)["tracks"]
    selected = tracks[(tracks["condition"] == CONDITION) & (tracks["n_fit_accepted"] >= MIN_ACCEPTED_FITS)]
    print(f"{len(selected)} {CONDITION} tracks with >= {MIN_ACCEPTED_FITS} accepted fits "
          f"(of {int((tracks['condition'] == CONDITION).sum())})")
    return selected.copy()


def main(pss_cache: Path = PSS_CACHE, output_dir: Path = SAVE_PATH) -> None:
    df = load_tracks(Path(pss_cache))
    dls_labels = get_dls_labels()

    diffraction_fwhm_nm = FWHM_FACTOR * calculate_theoretical_psf_sigma(WAVELENGTH_NM, NUMERICAL_APERTURE)
    print(f"Diffraction-limited term: FWHM = {diffraction_fwhm_nm:.1f} nm (= {FWHM_FACTOR:.4f} * sigma), "
          f"at wavelength={WAVELENGTH_NM:.0f} nm, NA={NUMERICAL_APERTURE} (confirmed only for the "
          f"{DIFFRACTION_LIMITED_SIZES_NM} bead; applied to all sizes' diffraction term, which assumes "
          "the same dye emission for the larger, unconfirmed beads).")
    print("Per-size theoretical FWHM = sqrt(diffraction_FWHM^2 + particle_diameter^2):")
    for size_nm in sorted(df["particle_size_nm"].dropna().unique()):
        real_size_nm = float(dls_labels.get(size_nm, size_nm))
        measured = df.loc[df["particle_size_nm"] == size_nm, "fwhm_nm"]
        print(f"  {real_size_nm:.0f} nm: theoretical FWHM = {theoretical_fwhm_nm(real_size_nm):.1f} nm, "
              f"measured median = {measured.median():.1f} nm (n = {len(measured)} tracks)")

    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        fig = plot_psf_scatter_theory_all_sizes(df, dls_labels)
        fig_path = output_dir / "psf_scatter_vs_theory_all_sizes.png"
        safe_savefig(fig, fig_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PSS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=SAVE_PATH)
    args = parser.parse_args()
    main(args.input, args.output_dir)
