"""
Unified TrackMate XML Analysis Script

Processes TrackMate XML files and performs both:
1. MSD (Mean Squared Displacement) analysis
2. Step size (displacement) diffusion analysis

Uses centralized functions from core.io, core.analysis, and core.visualization.

Usage:
    python Unified_XML_Analysis.py

Author: Jonas
Date: 2026-01-23
"""

from pathlib import Path
from typing import Dict

from matplotlib import pyplot as plt
import trackpy as tp

import pandas as pd
import numpy as np

# Import core functions
from core.io import read_trackmate_xml, find_rec_tif_files, parse_rec_file
from core.analysis import (
    perform_msd_analysis,
    calculate_theoretical_diffusion,
)
from core.visualization import plot_msd_results, plot_stepsize_results


# ============================================================================
# CONFIGURATION
# ============================================================================

# Analysis parameters
MIN_TRACK_LENGTH = 30
MIN_TRACK_LENGTH_STEPSIZE = 10
MSD_FIT_POINTS = 6
STEP_INTERVAL = 1

# Quality criteria
MIN_EXPONENT = 0.85
MAX_SIGMA_RATIO = 1.5
MAX_MEAN_SIGMA_RATIO = 0.3

# Physical constants
TEMPERATURE_K = 293.15
VISCOSITY_PA_S = 0.001002

def main():
    path = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks\1000_nm_08_Tracks.xml")
    path = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\Tracks\1000nm_water_01_Tracks.xml")
    
    rec_tif = find_rec_tif_files(path)
    meta_data = parse_rec_file(rec_tif['rec_file'])
    tracks = read_trackmate_xml(Path(path))
    mpp = meta_data.get('mpp')  # microns per pixel
    fps = meta_data.get('fps')   # frames per second
    print(f"Metadata: mpp={mpp} µm/px, fps={fps} Hz")
    msd_results = perform_msd_analysis(tracks,mpp,fps,7)

    """trackpy"""
    imsd = tp.imsd(tracks, mpp=mpp, fps=fps)
    emsd = tp.emsd(tracks, mpp=mpp, fps=fps)
    # print(emsd.iloc[:7])
    fit = tp.utils.fit_powerlaw(emsd.iloc[:7])
    n = fit.n.iloc[0]
    A = fit.A.iloc[0]
    D_tp = A / 4.0  # 2D diffusion coefficient



  
    theoretical_D = calculate_theoretical_diffusion(
        particle_size_nm=1002,
        temperature=TEMPERATURE_K,
        viscosity=VISCOSITY_PA_S
    )

    plt.plot(msd_results['emsd'].index, msd_results['emsd'], linewidth=2, label='eMSD')
    plt.plot(msd_results['imsd'].index, msd_results['imsd'],color = "black" , alpha = 0.3, linewidth=1)#, label='iMSD')
    # plt.plot(msd_results['emsd'].index, msd_results['D_um2_per_s'] * 4 * msd_results['emsd'].index, label='MSD Fit')
    plt.plot(msd_results['emsd'].index, msd_results['fit_result']['A'][0]  * msd_results['emsd'].index**msd_results['fit_result']['n'][0], label='Power-law Fit')
    plt.plot(msd_results['emsd'].index, theoretical_D * 2 * msd_results['emsd'].index, label='Theoretical MSD')
    plt.ylabel('MSD (µm²/s)')
    plt.xlabel('Lag Time (s)')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('MSD Analysis')
    plt.legend()
    plt.show()

    plot_msd_results(msd_results,particle_size = 1000, save_path=None)
    print(f"TrackPy Diffusion Coefficient: {D_tp:.4f} µm²/s")
    print(f"MSD Analysis Diffusion Coefficient: {msd_results['D_um2_per_s']:.4f} µm²/s")
    print(f"Theoretical Diffusion Coefficient: {theoretical_D:.4f} µm²/s")

    '''Stepsize Analysis'''




'''    
    return {
        'method': 'MSD',
        'D_um2_per_s': D,
        'D_error': D_err,
        'exponent': n,
        'exponent_error': n_err,
        'n_particles': tracks['particle'].nunique(),
        'n_detections': len(tracks),
        'imsd': imsd,
        'emsd': emsd,
        'fit_result': fit_result
    }'''

if __name__ == "__main__":
    main()