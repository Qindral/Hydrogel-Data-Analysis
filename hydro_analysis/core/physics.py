"""Shared physical constants and formulas."""

import numpy as np

# Physical constants
TEMPERATURE_K = 293.15
VISCOSITY_PA_S = 0.001002
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K


def calculate_theoretical_diffusion(
    particle_size_nm: float,
    temperature: float = TEMPERATURE_K,
    viscosity: float = VISCOSITY_PA_S,
) -> float:
    """
    Calculate theoretical diffusion coefficient using Stokes-Einstein equation.

    D = k_B * T / (6 * pi * eta * r)

    Returns diffusion coefficient in um^2/s.
    """
    radius_m = ((particle_size_nm * 1e-9)/ 2) 
    D_m2_per_s = BOLTZMANN_CONSTANT * temperature / (6 * np.pi * viscosity * radius_m)
    D_um2_per_s = D_m2_per_s * 1e12
    return D_um2_per_s


def calculate_theoretical_psf_sigma(wavelength_nm: float, numerical_aperture: float) -> float:
    """
    Gaussian-approximation of a diffraction-limited widefield/epifluorescence
    point-spread function: sigma = 0.21 * wavelength / NA (the standard
    relation giving the Gaussian sigma that best matches an Airy PSF for a
    well-corrected microscope; see e.g. Zhang et al. 2007, Applied Optics,
    "Gaussian approximations of fluorescence microscope point-spread
    function models").

    Returns sigma in the same units as wavelength_nm (nm).
    """
    return 0.21 * wavelength_nm / numerical_aperture
