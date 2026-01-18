"""Core functionality for hydrogel analysis."""

from .io import TrackLoader, find_dataset_files, find_xml_files, extract_particle_size_from_path, DatasetFiles
from .analysis import compute_step_size_diffusion, compute_theoretical_diffusion
from .visualization import ResultsAggregator

__all__ = [
    'TrackLoader',
    'find_dataset_files',
    'find_xml_files',
    'extract_particle_size_from_path',
    'DatasetFiles',
    'compute_step_size_diffusion',
    'compute_theoretical_diffusion',
    'ResultsAggregator',
]
