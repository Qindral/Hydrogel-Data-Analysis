"""Core functionality for hydrogel analysis."""

# IO functions
from .io import (
    find_rec_tif_files,
    parse_rec_file,
    read_trackmate_xml,
    check_text_encoding,
    scan_xml_folder,
    build_datasets,
    compare_xml,
)

# Analysis functions
from .analysis import (
    compute_step_size_diffusion,
    fit_gaussian_diffusion_stepsize,
    fit_powerlaw_with_errors,
    perform_msd_analysis,
    calculate_theoretical_diffusion,
)

# Visualization functions
from .visualization import (
    plot_msd_results,
    plot_stepsize_results,
    plot_trajectories,
    plot_theory_comparison,
    plot_diffusion_comparison,
)

__all__ = [
    # IO functions
    'find_rec_tif_files',
    'parse_rec_file',
    'read_trackmate_xml',
    'check_text_encoding',
    'scan_xml_folder',
    'build_datasets',
    'compare_xml',
    
    # Analysis functions
    'compute_step_size_diffusion',
    'fit_gaussian_diffusion_stepsize',
    'compute_theoretical_diffusion',
    'fit_powerlaw_with_errors',
    'perform_msd_analysis',
    'calculate_theoretical_diffusion',
    
    # Visualization functions
    'plot_msd_results',
    'plot_stepsize_results',
    'plot_trajectories',
    'plot_theory_comparison',
    'plot_diffusion_comparison',
    'compute_step_size_diffusion',
    'compute_theoretical_diffusion',
]

