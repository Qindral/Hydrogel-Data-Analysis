"""
TrackMate XML Particle Counter and Step Size Diffusion Analysis - 20mg Hydrogel

This script processes TrackMate-generated XML files to count particles and calculate
diffusion coefficients using the step size (displacement) method for particles in 20mg hydrogel.

Main features:
- Scans directory structure for particle size folders with Tracks subfolders
- Counts total particles per particle size from TrackMate XML files
- Calculates diffusion coefficients from step size distributions
- Exports summary statistics and comparisons with theory to CSV

Method: Diffusion coefficient from displacement distributions
For each particle track, calculate frame-to-frame displacements in x and y:
    dx_i = x_{i+1} - x_i
    dy_i = y_{i+1} - y_i
Fit Gaussian distributions to dx and dy separately to extract variance σ².
For 2D Brownian motion: σ² = 2*D*dt, therefore:
    D = σ² / (2 * dt)
where dt is the time interval between frames.

Author: Jonas
Date: 2026-01-13
"""

# Import all functions and constants from the D0 version
import sys
from pathlib import Path

# Add parent directory to path to import from Schrittweiten_methode_D0
sys.path.insert(0, str(Path(__file__).parent))

from Schrittweiten_methode_D0 import (
    # Utility functions
    extract_particle_size_from_path,
    parse_rec_file,
    get_mpp_from_fps_and_size,
    get_mpp_from_fps,
    # XML parsing
    extract_image_dimensions_from_xml,
    read_trackmate_xml,
    collect_all_files_by_particle_size,
    find_tracks_in_particle_folders,
    # Analysis functions
    calculate_step_sizes,
    calculate_diffusion_from_steps,
    analyze_single_file,
    analyze_all_files,
    combine_by_particle_size,
    count_particles_per_size,
    # Plotting functions
    plot_step_size_distributions,
    plot_step_size_overlay,
    plot_dx_dy_distributions,
    plot_diffusion_comparison,
    print_comparison_table,
    calculate_theoretical_D,
    # Constants
    BOLTZMANN_CONSTANT,
    TEMPERATURE,
    WATER_VISCOSITY,
    DEFAULT_MPP,
    DEFAULT_FPS,
    STEP_INTERVAL,
    MIN_TRACK_LENGTH,
    MAX_SIGMA_RATIO,
    MAX_MEAN_SIGMA_RATIO
)

# ============================================================================
# CONFIGURATION - 20MG SPECIFIC PATHS
# ============================================================================

# Directory paths for 20mg measurements
ROOT_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")
SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\trackmate_MSD_results")

# Create output directory if it doesn't exist
SAVE_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for step size diffusion analysis in 20mg hydrogel.
    
    This function:
    1. Scans for XML track files and extracts FPS from .rec files
    2. Counts particles per particle size
    3. Calculates diffusion coefficients using step size method
    4. Creates plots and exports results
    """
    print("=" * 70)
    print("TrackMate Step Size Diffusion Analysis - 20mg Hydrogel")
    print("=" * 70)
    print(f"\nRoot directory: {ROOT_PATH}")
    print(f"Save directory: {SAVE_PATH}")
    print(f"Step interval: {STEP_INTERVAL} (using every {STEP_INTERVAL}{'st' if STEP_INTERVAL == 1 else 'rd' if STEP_INTERVAL == 3 else 'th'} step)")
    print(f"Minimum track length: {MIN_TRACK_LENGTH} points")
    print(f"Quality thresholds: sigma_ratio <= {MAX_SIGMA_RATIO}, mean/sigma <= {MAX_MEAN_SIGMA_RATIO}\n")
    
    # Step 1: Collect XML files with FPS data
    print("=" * 70)
    print("Step 1: Scanning for XML track files and calibration data...")
    print("=" * 70)
    
    files_df = collect_all_files_by_particle_size(ROOT_PATH)
    
    if files_df.empty:
        print("\nERROR: No XML files found!")
        print("Please check that the ROOT_PATH is correct and contains particle size folders with Tracks/ subfolders.")
        return
    
    print(f"\n[OK] Found {len(files_df)} XML files across {files_df['particle_size_nm'].nunique()} particle sizes")
    
    # Save XML file listing
    output_files_csv = SAVE_PATH / "xml_file_associations_20mg.csv"
    files_df.to_csv(output_files_csv, index=False)
    print(f"[OK] XML file associations saved to: {output_files_csv}")
    
    # Display mode statistics
    import pandas as pd
    xml_with_fps = files_df[files_df['fps'].notna()]
    
    if not xml_with_fps.empty:
        print("\n" + "=" * 70)
        print("MODE DETECTION STATISTICS")
        print("=" * 70)
        
        # Group by mode
        mode_20 = xml_with_fps[xml_with_fps['mode'] == '20 FPS']
        mode_60 = xml_with_fps[xml_with_fps['mode'] == '60 FPS']
        mode_unknown = xml_with_fps[xml_with_fps['mode'] == 'Unknown']
        
        print(f"\nTotal XML files: {len(xml_with_fps)}")
        print(f"  • 20 FPS mode: {len(mode_20)} files ({len(mode_20)/len(xml_with_fps)*100:.1f}%)")
        print(f"  • 60 FPS mode: {len(mode_60)} files ({len(mode_60)/len(xml_with_fps)*100:.1f}%)")
        if len(mode_unknown) > 0:
            print(f"  • Unknown mode: {len(mode_unknown)} files ({len(mode_unknown)/len(xml_with_fps)*100:.1f}%)")
        
        # Detailed breakdown by particle size
        print("\n" + "-" * 70)
        print("BREAKDOWN BY PARTICLE SIZE")
        print("-" * 70)
        
        for particle_size in sorted(xml_with_fps['particle_size_nm'].unique()):
            size_files = xml_with_fps[xml_with_fps['particle_size_nm'] == particle_size]
            print(f"\n{particle_size:.0f} nm ({len(size_files)} XML files):")
            for _, row in size_files.iterrows():
                size_str = f"{row['x_max']}×{row['y_max']}" if pd.notna(row['x_max']) else "unknown"
                print(f"  • {row['xml_name']}: {row['mode']} ({size_str}, {row['mpp']} µm/px)")
    else:
        print("\nWarning: No calibration data found - will use default values")
    
    # Step 2: Count particles
    print("\n" + "=" * 70)
    print("Step 2: Counting particles from XML files...")
    print("=" * 70 + "\n")
    
    # Count particles per size
    particle_counts = []
    for particle_size in sorted(files_df['particle_size_nm'].unique()):
        size_xmls = files_df[files_df['particle_size_nm'] == particle_size]
        total_particles = 0
        
        print(f"{particle_size:.0f} nm:")
        for _, row in size_xmls.iterrows():
            xml_path = Path(row['xml_path'])
            df = read_trackmate_xml(xml_path)
            if df is not None and not df.empty:
                num_particles = df['particle'].nunique()
                total_particles += num_particles
                print(f"  • {row['xml_name']}: {num_particles} particles")
        
        particle_counts.append({
            'particle_size_nm': particle_size,
            'num_xml_files': len(size_xmls),
            'total_particles': total_particles
        })
        
        print(f"  → Total: {total_particles} particles from {len(size_xmls)} files\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(particle_counts)
    
    # Print summary table
    print("=" * 70)
    print("PARTICLE COUNT SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    
    # Save particle count summary
    output_summary_csv = SAVE_PATH / "particle_count_summary_20mg.csv"
    summary_df.to_csv(output_summary_csv, index=False)
    print(f"\n[OK] Particle count summary saved to: {output_summary_csv}")
    
    # Step 3: Calculate diffusion coefficients from step sizes
    print("\n" + "=" * 70)
    print("Step 3: Calculating diffusion coefficients from step sizes...")
    print("=" * 70)
    
    results_df = analyze_all_files(files_df)
    
    if not results_df.empty:
        # Save individual file results
        output_results_csv = SAVE_PATH / "hydrogel_20mg_stepsize_analysis_results.csv"
        results_df.to_csv(output_results_csv, index=False)
        print(f"\n[OK] Individual file results saved to: {output_results_csv}")
        
        # Step 4: Combine results by particle size
        print("\n" + "=" * 70)
        print("Step 4: Combining results by particle size...")
        print("=" * 70)
        
        combined_df = combine_by_particle_size(results_df)
        
        # Save combined results
        output_combined_csv = SAVE_PATH / "hydrogel_20mg_diffusion_coefficients_stepsize.csv"
        combined_df.to_csv(output_combined_csv, index=False)
        print(f"\n[OK] Combined results saved to: {output_combined_csv}")
        
        # Print comparison table
        print_comparison_table(combined_df)
        
        # Step 5: Create visualizations
        print("\n" + "=" * 70)
        print("Step 5: Creating visualizations...")
        print("=" * 70)
        
        # Plot diffusion comparison
        plot_diffusion_comparison(combined_df, results_df, SAVE_PATH)
        
        # Plot individual file distributions
        plot_step_size_distributions(results_df, SAVE_PATH)
        plot_dx_dy_distributions(results_df, SAVE_PATH)
        plot_step_size_overlay(results_df, SAVE_PATH)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nAll results saved to: {SAVE_PATH}")
        print("\nGenerated files:")
        print(f"  • xml_file_associations_20mg.csv")
        print(f"  • particle_count_summary_20mg.csv")
        print(f"  • hydrogel_20mg_stepsize_analysis_results.csv")
        print(f"  • hydrogel_20mg_diffusion_coefficients_stepsize.csv")
        print(f"  • diffusion_comparison_stepsize_individual.png")
        print(f"  • Individual step size distribution plots")
        print(f"  • Individual dx/dy distribution plots")
        print(f"  • Step size overlay plot")
    else:
        print("\nERROR: No valid analysis results - check that XML files contain valid track data")


if __name__ == "__main__":
    main()
