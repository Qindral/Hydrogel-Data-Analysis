"""Core functionality for hydrogel analysis."""

# IO functions
from .io import (
    find_rec_tif_files,
    parse_rec_file,
    read_trackmate_xml,
    check_text_encoding,
    scan_xml_folder,
    build_datasets,
    collect_files_from_list,
    collect_all_files_by_particle_size,
    single_file_data,
    compare_xml,
)

# Analysis functions
from .analysis import (
    calculate_step_sizes,
    fit_gaussian_diffusion_1d,
    diffusion_2d_from_1d_fits,
    fit_powerlaw_with_errors,
    perform_msd_analysis,
    perform_stepsize_analysis,
    calculate_theoretical_diffusion,
    analyze_multiple_files,
    DEFAULT_MSD_FIT_POINTS,
    DEFAULT_STEP_INTERVAL,
)

# Visualization functions
from .visualization import (
    plot_msd_results,
    plot_stepsize_results,
    plot_trajectories,
    plot_theory_comparison,
    plot_diffusion_comparison,
    plot_step_size_overlay,
    plot_dx_dy_distributions,
)

__all__ = [
    # IO functions
    'find_rec_tif_files',
    'parse_rec_file',
    'read_trackmate_xml',
    'check_text_encoding',
    'scan_xml_folder',
    'build_datasets',
    'collect_files_from_list',
    'collect_all_files_by_particle_size',
    'single_file_data',
    'compare_xml',
    
    # Analysis functions
    'calculate_step_sizes',
    'fit_gaussian_diffusion_1d',
    'diffusion_2d_from_1d_fits',
    'fit_powerlaw_with_errors',
    'perform_msd_analysis',
    'perform_stepsize_analysis',
    'calculate_theoretical_diffusion',
    'analyze_multiple_files',
    'DEFAULT_MSD_FIT_POINTS',
    'DEFAULT_STEP_INTERVAL',
    
    # Visualization functions
    'plot_msd_results',
    'plot_stepsize_results',
    'plot_trajectories',
    'plot_theory_comparison',
    'plot_diffusion_comparison',
    'plot_step_size_overlay',
    'plot_dx_dy_distributions',
]

