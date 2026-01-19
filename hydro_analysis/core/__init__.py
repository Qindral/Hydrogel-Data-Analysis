"""Core functionality for hydrogel analysis."""

from .io import (
    DatasetIndex,
    DatasetFiles,
    parse_pco_camware_rec_text,
    parse_pco_camware_rec_file,
    canonical_base,
    processed_level,
    xml_variant_from_path,
    build_datasets,
    load_index,
)
from .analysis import (
    compute_step_size_diffusion, 
    compute_theoretical_diffusion,
    fit_powerlaw_with_errors,
    read_trackmate_xml,
    calculate_step_sizes,
    calculate_diffusion_from_steps,
)
from .visualization import ResultsAggregator

__all__ = [
    # Core classes
    'DatasetIndex',
    'DatasetFiles',
    'ResultsAggregator',
    
    # Main functions
    'load_index',
    'build_datasets',
    
    # Metadata parsing
    'parse_pco_camware_rec_text',
    'parse_pco_camware_rec_file',
    
    # Utilities
    'canonical_base',
    'processed_level',
    'xml_variant_from_path',
    
    # Analysis - from MSD_FromTrackmate_D0.py
    'fit_powerlaw_with_errors',
    'read_trackmate_xml',
    
    # Analysis - from Schrittweiten_methode_D0.py
    'calculate_step_sizes',
    'calculate_diffusion_from_steps',
    'compute_step_size_diffusion',
    'compute_theoretical_diffusion',
]

