"""
Step Size Diffusion Analysis - Refactored Version

Flexible script that accepts either a single XML file or a folder path.
Uses core modules for consistency and supports parallel processing.

Usage:
    # Single file
    path = r"E:\PhD Data Analysis\file.xml"
    
    # Folder (finds all *Tracks.xml recursively)
    path = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung"
    
    # Then run: python Schrittweiten_methode_D0_refactored.py
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Core modules
from core import (
    TrackLoader,
    find_dataset_files,
    extract_particle_size_from_path,
    compute_step_size_diffusion,
    compute_theoretical_diffusion,
    ResultsAggregator,
    DatasetFiles
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# INPUT: Set your path here (file or folder)
path = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung"

# OUTPUT: Where to save results
output_dir = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\trackmate_MSD_results")

# PROCESSING OPTIONS
parallel = True  # Use multiple CPU cores?
max_workers = 8  # Number of parallel workers (None = use all CPUs)
show_progress = True  # Show progress bar?

# ANALYSIS PARAMETERS
step_interval = 6  # Use every nth step (1=all, 6=every 6th)
min_track_length = 10  # Minimum detections per track
max_sigma_ratio = 1.5  # Quality: max σ_x/σ_y ratio
max_mean_sigma_ratio = 0.3  # Quality: max |mean|/σ ratio

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_single_xml(dataset: DatasetFiles) -> Dict[str, Any]:
    """Process a single dataset (XML with optional TIF and REC)."""
    xml_path = dataset.xml_path
    particle_size = extract_particle_size_from_path(xml_path)
    
    tracks = TrackLoader.from_trackmate_xml(
        xml_path,
        min_length=min_track_length,
        rec_path=dataset.rec_path
    )
    
    if tracks.empty:
        return {
            'file': xml_path.name,
            'particle_size_nm': particle_size,
            'n_particles': 0,
            'n_steps': 0,
            'D_measured': None,
            'D_error': None,
            'quality_ok': False,
            'tif_found': dataset.tif_path is not None,
            'rec_found': dataset.rec_path is not None,
            'status': 'no_tracks'
        }
    
    result = compute_step_size_diffusion(
        tracks,
        step_interval=step_interval,
        max_sigma_ratio=max_sigma_ratio,
        max_mean_sigma_ratio=max_mean_sigma_ratio
    )
    
    result.update({
        'file': xml_path.name,
        'particle_size_nm': particle_size,
        'n_particles': tracks['particle'].nunique(),
        'mpp': tracks.attrs['mpp'],
        'fps': tracks.attrs['fps'],
        'mode': tracks.attrs['mode'],
        'tif_found': dataset.tif_path is not None,
        'rec_found': dataset.rec_path is not None,
        'status': 'success'
    })
    
    if particle_size is not None:
        result['D_theory'] = compute_theoretical_diffusion(particle_size)
    
    return result


def process_sequential(datasets: List[DatasetFiles]) -> List[Dict[str, Any]]:
    """Process datasets sequentially with progress bar."""
    results = []
    iterator = tqdm(datasets, desc="Processing") if show_progress else datasets
    
    for dataset in iterator:
        try:
            results.append(process_single_xml(dataset))
        except Exception as e:
            logger.error(f"Failed {dataset.xml_path.name}: {e}")
    
    return results


def process_parallel(datasets: List[DatasetFiles]) -> List[Dict[str, Any]]:
    """Process datasets in parallel with progress bar."""
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_xml, ds): ds for ds in datasets}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(datasets), desc="Processing")
        
        for future in iterator:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Processing error: {e}")
    
    return results


def aggregate_by_particle_size(aggregator: ResultsAggregator):
    """Print summary statistics grouped by particle size."""
    df = aggregator.get_dataframe()
    
    if df.empty:
        logger.warning("No results to aggregate")
        return
    
    logger.info("\n" + "="*70)
    logger.info("SUMMARY BY PARTICLE SIZE")
    if df.empty:
        return
    
    print("\nSUMMARY BY PARTICLE SIZE")
    print("="*60)
    
    for size in sorted(df['particle_size_nm'].dropna().unique()):
        subset = df[df['particle_size_nm'] == size]
        valid = subset[subset['D_measured'].notna()]
        
        if valid.empty:
            continue
        
        D_mean = valid['D_measured'].mean()
        D_std = valid['D_measured'].std()
        D_theory = valid['D_theory'].iloc[0] if 'D_theory' in valid else None
        quality_pass = valid['quality_ok'].sum()
        
        print(f"\n{size:.0f} nm ({len(subset)} files, {quality_pass} passed QC):")
        print(f"  D_measured = {D_mean:.4f} ± {D_std:.4f} µm²/s")
        if D_theory:
            ratio = D_mean / D_theory
            print
    """Main execution function."""
    logger.info("="*70)
    logger.info("Step Size Diffusion Analysis (Refactored)")
    logger.info("="*70)
    input_path = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find datasets (XML + optional TIF/REC)
    datasets = find_dataset_files(input_path)
    
    if not datasets:
        print("No XML files found")
        return
    
    print(f"Found {len(datasets)} datasets")
    matched_tif = sum(1 for ds in datasets if ds.tif_path)
    matched_rec = sum(1 for ds in datasets if ds.rec_path)
    print(f"Matched: {matched_tif} TIF files, {matched_rec} REC files")
    
    # Group by particle size
    by_size = {}
    for ds in datasets:
        size = extract_particle_size_from_path(ds.xml_path)
        if size:
            by_size.setdefault(size, []).append(ds)
    
    if by_size:
        print(f"Particle sizes: {sorted(by_size.keys())}")
        for size in sorted(by_size.keys()):
            print(f"  {size:.0f} nm: {len(by_size[size])} files")
    
    # Process
    if parallel and len(datasets) > 1:
        results = process_parallel(datasets)
    else:
        results = process_sequential(datasets)
    
    # Aggregate
    aggregator = ResultsAggregator()
    for result in results:
        aggregator.add(**result)
    
    # Save
    results_csv = output_dir / "stepsize_analysis_individual_results.csv"
    aggregator.save_summary(results_csv)
    
    aggregate_by_particle_size(aggregator)
    
    # Plots
    fig1 = aggregator.plot_size_vs_diffusion(show_theory=True, title="Step Size Diffusion Analysis")
    fig1.savefig(output_dir / "diffusion_vs_size.png", dpi=300, bbox_inches='tight')
    
    fig2 = aggregator.plot_quality_summary()
    fig2.savefig(output_dir / "quality_summary.png", dpi=300, bbox_inches='tight')
    
    print(f"\nComplete: {sum(1 for r in results if r['status'] == 'success')}/{len(results)} successful")
    print(f"Results saved to:
    main()
