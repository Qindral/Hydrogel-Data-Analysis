"""
New PSF-analysis figure: overlaid per-size histograms on a SINGLE shared
axis (all six particle sizes together, per user request), rather than
PSF_Refined.py's per-size scatter grid (which places each size at its own
x position). Both variants below share the same nanometre x-axis and the
same theoretical diffraction-limited FWHM per size, drawn as a vertical
dashed line in that size's own colour.

Two output PNGs:
  1. psf_fwhm_histogram_by_size.png -- measured FWHM_measured_nm =
     FWHM_FACTOR * sigma_fit_nm (per track; PSF_Refined.py's own
     definition, unchanged, already in nm via each movie's own mpp).
  2. psf_fitting_diameter_nm_histogram_by_size.png -- PSF_Refined.py's
     "fitting pixel diameter" quantity (2 * sigma_xml_px, the raw
     TrackMate-detector radius, never itself a PSF fit) converted from
     pixels to nanometres using each row's OWN mpp column already present
     in psf_fits.csv (mpp varies per movie/size -- see PSF_Refined.py's mpp
     docstring note): fitting_diameter_nm = 2 * sigma_xml_px * mpp * 1000.
     Because this quantity is a configured detection radius, not a PSF fit,
     it is not multiplied by FWHM_FACTOR; it is shown on the same nm axis
     and against the same theoretical FWHM lines purely so the two are
     visually comparable, per user request.

Histograms are drawn as unfilled `density=True` step outlines (not raw
Counts): the six size groups have very different track counts (16 to 135
in the current psf_fits.csv), so a Counts histogram would make the
low-n groups nearly invisible next to the high-n ones on one shared axis.
This differs deliberately from the "Counts, not density" rule in
_size_grid_utils.py, which was chosen for a different case (one panel per
size, always Water vs. Hydrogel at comparable n) -- not applicable here
since all six sizes are overlaid on one axis with widely differing n.
Density-step overlays for exactly this situation already have a precedent
in Compare_Tracks_vs_RawAnalysis.py.

Colour palette: SIZE_COLORS is imported unchanged from Correlations.py
(Style_guide.txt's 8-colour Jet-derived accent palette, largest=1000 nm/
index 1/blue ... smallest=20 nm/index 6/orange) -- the same mapping already
reused by Fittingpoints_Refined.py, rather than duplicating the six hex
pairs again here.

Does NOT modify PSF_Analysis.py, PSF_Refined.py, Correlations.py, or any
cache/CSV -- pure consumer: reads the existing psf_fits.csv and imports
theoretical_fwhm_nm/FWHM_FACTOR from PSF_Refined.py and WAVELENGTH_NM/
NUMERICAL_APERTURE/DIFFRACTION_LIMITED_SIZES_NM from PSF_Analysis.py
unchanged.

Outputs (own folder, PNG, 600 dpi): psf_fwhm_histogram_by_size.png,
psf_fitting_diameter_nm_histogram_by_size.png.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig
from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS
from hydro_analysis.MSD_Trackmate.Validation_Claude.PSF_Analysis import (
    WAVELENGTH_NM, NUMERICAL_APERTURE, DIFFRACTION_LIMITED_SIZES_NM,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.PSF_Refined import (
    FWHM_FACTOR, theoretical_fwhm_nm, PSF_CSV, SAVE_PATH,
)

# ── Configuration ──────────────────────────────────────────────────────────────
N_BINS = 40


def plot_overlaid_histogram_by_size(df: pd.DataFrame, value_nm: pd.Series, xlabel: str,
                                     dls_labels: dict[float, int]) -> plt.Figure:
    """One shared axis, one step-outline density histogram per particle
    size, plus a size-coloured vertical dashed line for that size's
    theoretical FWHM (see module docstring)."""
    sizes = sorted(df["particle_size_nm"].dropna().unique())
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)

    finite = value_nm[np.isfinite(value_nm)]
    bin_edges = np.linspace(0.0, float(finite.max()) * 1.05, N_BINS + 1)

    for size in sizes:
        face, edge = SIZE_COLORS.get(size, ("#999999", "#555555"))
        vals = value_nm[df["particle_size_nm"] == size].dropna().to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        label = f"{dls_labels.get(size, int(size))} nm (n={vals.size})"
        ax.hist(vals, bins=bin_edges, density=True, histtype="step",
                 linewidth=1.7, color=edge, label=label, zorder=3)

    for size in sizes:
        _, edge = SIZE_COLORS.get(size, ("#999999", "#555555"))
        real_size_nm = float(dls_labels.get(size, size))
        theory_val = theoretical_fwhm_nm(real_size_nm)
        ax.axvline(theory_val, color=edge, linewidth=1.4, linestyle=(0, (4, 3)), zorder=4)

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    confirmed_label = f"{DIFFRACTION_LIMITED_SIZES_NM[0]:.0f} nm" if DIFFRACTION_LIMITED_SIZES_NM else "smallest size"
    theory_handle = Line2D([0], [0], color="black", linewidth=1.4, linestyle=(0, (4, 3)),
                            label="Theoretical FWHM per size (dashed, size-matched colour)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [theory_handle], loc="upper right", fontsize=6.5, frameon=False)
    ax.text(0.0, -0.135,
            f"Theoretical FWHM: diffraction ({WAVELENGTH_NM:.0f} nm, NA={NUMERICAL_APERTURE}) + particle "
            f"diameter in quadrature; wavelength confirmed only for the {confirmed_label} bead, applied "
            "to all sizes' diffraction term.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color="#444444", clip_on=False)
    return fig


def main() -> None:
    if not PSF_CSV.exists():
        raise FileNotFoundError(f"Nicht gefunden: {PSF_CSV}\nBitte zuerst PSF_Analysis.py ausfuehren.")
    df = pd.read_csv(PSF_CSV)
    dls_labels = get_dls_labels()

    fwhm_measured_nm = FWHM_FACTOR * df["sigma_fit_nm"]
    fitting_diameter_nm = 2.0 * df["sigma_xml_px"] * df["mpp"] * 1000.0

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        fig = plot_overlaid_histogram_by_size(df, fwhm_measured_nm, "Measured PSF FWHM (nm)", dls_labels)
        fig_path = SAVE_PATH / "psf_fwhm_histogram_by_size.png"
        safe_savefig(fig, fig_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {fig_path}")

        fig = plot_overlaid_histogram_by_size(df, fitting_diameter_nm, "Fitting pixel diameter (nm)", dls_labels)
        fig_path = SAVE_PATH / "psf_fitting_diameter_nm_histogram_by_size.png"
        safe_savefig(fig, fig_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {fig_path}")


if __name__ == "__main__":
    main()
