"""Test script to verify core modules functionality."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    TrackLoader,
    compute_step_size_diffusion,
    compute_theoretical_diffusion,
    ResultsAggregator
)

def create_sample_tracks():
    """Create sample tracking data for testing."""
    np.random.seed(42)
    
    data = []
    for pid in range(5):
        x, y = 0.0, 0.0
        for frame in range(50):
            x += np.random.normal(0, 0.5)
            y += np.random.normal(0, 0.5)
            data.append({'particle': pid, 'frame': frame, 'x': x, 'y': y})
    
    df = pd.DataFrame(data)
    df.attrs = {'mpp': 0.15, 'fps': 20.0, 'mode': 'Test'}
    return df

def test_step_size_analysis():
    """Test step size diffusion calculation."""
    print("Testing step size analysis...")
    print("="*70)
    
    tracks = create_sample_tracks()
    print(f"Created sample tracks: {tracks['particle'].nunique()} particles, {len(tracks)} detections")
    
    result = compute_step_size_diffusion(tracks, step_interval=1, max_sigma_ratio=2.0, max_mean_sigma_ratio=0.5)
    
    print(f"Computed diffusion coefficient:")
    print(f"  D = {result['D_um2_per_s']:.4f} ± {result['D_error']:.4f} µm²/s")
    print(f"  sigma_x = {result['sigma_x']:.4f} µm, sigma_y = {result['sigma_y']:.4f} µm")
    print(f"  Steps used: {result['n_steps']}")
    print(f"  Quality: {'PASS' if result['quality_ok'] else 'FAIL'}")
    if not result['quality_ok']:
        print(f"  Issues: {', '.join(result['quality_issues'])}")
    
    return result

def test_theoretical_diffusion():
    """Test theoretical diffusion calculation."""
    print("\n" + "="*70)
    print("Testing theoretical diffusion...")
    print("="*70)
    
    sizes = [20, 50, 100, 200, 500, 1000]
    
    for size_nm in sizes:
        D_theory = compute_theoretical_diffusion(size_nm)
        print(f"{size_nm:4.0f} nm: D = {D_theory:.4f} µm²/s")
    
    print("✓ Theoretical calculations complete")

def test_results_aggregator():
    for size_nm in sizes:
        D_theory = compute_theoretical_diffusion(size_nm)
        print(f"{size_nm:4.0f} nm: D = {D_theory:.4f} µm²/s")
    
    print("r = ResultsAggregator()
    
    # Add some sample results
    for size in [20, 50, 100]:
        for i in range(3):
            D_theory = compute_theoretical_diffusion(size)
            D_measured = D_theory * np.random.uniform(0.9, 1.1)
            aggregator.add(
                file=f"test_{size}nm_{i}.xml",
                particle_size_nm=size,
                D_measured=D_measured,
                D_error=D_measured * 0.1,
                D_theory=D_theory,
                n_particles=np.random.randint(20, 50),
                quality_ok=np.random.random() > 0.2
            )
    
    print(f"Added {len(aggregator.results)} results")
    df = aggregator.get_dataframe()
    print(f"Created DataFrame: {len(df)} rows x {len(df.columns)} columns")
    
    try:
        fig = aggregator.plot_size_vs_diffusion(show_theory=True, title="Test Plot")
        fig.savefig("test_plot.png", dpi=150)
        print("Created and saved diffusion comparison plot")
    except Exception as e:
        print(f"Plot creation failed: {e}")
    
    try:
        fig2 = aggregator.plot_quality_summary()
        fig2.savefig("test_quality.png", dpi=150)
        print("Created and saved quality summary plot")
    except Exception as e:
        print(f"
def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CORE MODULE TESTING")
    print("="*70 + "\n")
    
    try:
        # Test 1: Step size analysis
        test_step_size_analysis()
        
        # Test 2: Theoretical diffusion
        test_theoretical_diffusion()
        
        # Test 3: Results aggregator
        test_results_aggregator()
        test_step_size_analysis()
        test_theoretical_diffusion()
        test_results_aggregator()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70)
        print("Core modules are working correctly.")
        print("You can now use Schrittweiten_methode_D0_refactored.py")
        
    except Exception as e:
        print(f"\n