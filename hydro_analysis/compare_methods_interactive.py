"""
Interactive Method Comparison Tool

Compare MSD and Step Size methods for a single TrackMate XML file with adjustable parameters.
Allows dynamic adjustment of:
- Step interval (1, 2, 3, ...)
- MSD fit points
- Quality filters

Author: Jonas
Date: 2026-01-13
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pathlib import Path
from scipy import stats
import trackpy as tp

# Import functions from existing modules
import sys
sys.path.append(str(Path(__file__).parent))

from Schrittweiten_methode_D0 import (
    read_trackmate_xml,
    calculate_step_sizes,
    calculate_theoretical_D,
    BOLTZMANN_CONSTANT,
    TEMPERATURE,
    WATER_VISCOSITY
)

from MSD_FromTrackmate_D0 import (
    calculate_imsd_for_file,
    fit_powerlaw_with_errors
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Example file path - adjust as needed
EXAMPLE_FILE = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\Tracks\1000 nm_4_Tracks.xml")

# Default parameters
DEFAULT_MPP = 0.15  # µm/px
DEFAULT_FPS = 20  # frames/s
DEFAULT_PARTICLE_SIZE = 200  # nm
DEFAULT_STEP_INTERVAL = 1
DEFAULT_FIT_POINTS = 6
DEFAULT_MIN_EXPONENT = 0.85

# DLS measurements (nm²/ms -> convert to µm²/s by /1000)
DLS_MEASUREMENTS = {
    20: 12.38750325 * 1e3 / 1000.0,   # µm²/s
    50: 8.201969711 * 1e3 / 1000.0,
    100: 4.139082033 * 1e3 / 1000.0,
    200: 1.745323167 * 1e3 / 1000.0,
    500: 0.621773811 * 1e3 / 1000.0,
    1000: 0.356862091 * 1e3 / 1000.0
}


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_step_size_method(xml_path: Path, mpp: float, fps: float, step_interval: int = 1):
    """
    Analyze using step size (Gaussian fit) method.
    
    Returns:
        dict with D, D_std, sigma_x, sigma_y, quality metrics
    """
    df = read_trackmate_xml(xml_path)
    if df is None or df.empty:
        return None
    
    step_df = calculate_step_sizes(df, step_interval=step_interval)
    if step_df.empty:
        return None
    
    # Convert to nm
    dx_nm = step_df['dx'].values * mpp * 1000.0
    dy_nm = step_df['dy'].values * mpp * 1000.0
    
    # Fit Gaussians
    mu_x, sigma_x = stats.norm.fit(dx_nm)
    mu_y, sigma_y = stats.norm.fit(dy_nm)
    
    # Calculate dt with frame interval
    frame_interval = step_df['frame_interval'].iloc[0] if 'frame_interval' in step_df.columns else 1
    dt = (1000.0 / fps) * frame_interval  # ms
    
    # Calculate D in nm²/ms, then convert to µm²/s
    D_x = (sigma_x**2) / (2.0 * dt)
    D_y = (sigma_y**2) / (2.0 * dt)
    D = (D_x + D_y) / 2.0 / 1000.0  # nm²/ms → µm²/s
    D_std = np.abs(D_x - D_y) / 2.0 / 1000.0
    
    # Quality metrics
    sigma_ratio = max(sigma_x, sigma_y) / min(sigma_x, sigma_y)
    mean_x_ratio = abs(mu_x) / sigma_x if sigma_x > 0 else 0
    mean_y_ratio = abs(mu_y) / sigma_y if sigma_y > 0 else 0
    
    return {
        'method': 'Step Size (Gaussian)',
        'D': D,
        'D_std': D_std,
        'D_x': D_x / 1000.0,  # nm²/ms → µm²/s
        'D_y': D_y / 1000.0,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'mu_x': mu_x,
        'mu_y': mu_y,
        'sigma_ratio': sigma_ratio,
        'mean_x_ratio': mean_x_ratio,
        'mean_y_ratio': mean_y_ratio,
        'num_steps': len(dx_nm),
        'dx_nm': dx_nm,
        'dy_nm': dy_nm,
        'step_interval': step_interval
    }


def analyze_msd_method(xml_path: Path, mpp: float, fps: float, fit_points: int = 6, min_exponent: float = 0.85):
    """
    Analyze using MSD power-law fit method.
    
    Returns:
        dict with D, D_std, exponent, quality metrics
    """
    # Calculate iMSD
    imsd = calculate_imsd_for_file(xml_path, mpp, fps)
    if imsd is None or imsd.empty:
        return None
    
    # Filter particles by exponent
    valid_particles = []
    particle_exponents = []
    
    for particle_id in imsd.columns:
        particle_msd = imsd[particle_id].dropna()
        
        if len(particle_msd) < fit_points:
            continue
        
        try:
            particle_params = fit_powerlaw_with_errors(particle_msd, points=fit_points, plot=False)
            n_particle = float(particle_params.n[0])
            
            if n_particle >= min_exponent:
                valid_particles.append(particle_id)
                particle_exponents.append(n_particle)
        except Exception:
            continue
    
    if not valid_particles:
        return None
    
    # Calculate ensemble MSD for valid particles
    filtered_imsd = imsd[valid_particles]
    ensemble_msd = filtered_imsd.mean(axis=1)
    
    # Fit power-law
    try:
        tp.quiet()
        params = fit_powerlaw_with_errors(ensemble_msd, points=fit_points, plot=False)
        A = float(params.A[0])
        A_err = float(params.A_err[0])
        n = float(params.n[0])
        n_err = float(params.n_err[0])
        
        # Calculate D: MSD = 4*D*t => D = A/4, convert nm²/ms → µm²/s
        D = A / 4.0 / 1000.0
        D_err = A_err / 4.0 / 1000.0
        
        return {
            'method': 'MSD (Power-law)',
            'D': D,
            'D_std': D_err,
            'A': A,
            'A_err': A_err,
            'n': n,
            'n_err': n_err,
            'num_particles': len(valid_particles),
            'total_particles': len(imsd.columns),
            'mean_exponent': np.mean(particle_exponents),
            'std_exponent': np.std(particle_exponents),
            'imsd': filtered_imsd,
            'ensemble_msd': ensemble_msd,
            'fit_points': fit_points,
            'min_exponent': min_exponent
        }
    except Exception as e:
        print(f"Error in MSD fitting: {e}")
        return None


def extract_particle_size_from_path(xml_path: Path) -> float:
    """
    Extract particle size from file path or filename.
    
    Expected patterns:
    - "200 nm 3x.xml" -> 200
    - "1000 nm_4_Tracks.xml" -> 1000
    - "50nm_..." -> 50
    - "/200nm/..." (in path) -> 200
    - "/1000 nm/..." (in path) -> 1000
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        Particle size in nm, or DEFAULT_PARTICLE_SIZE if not found
    """
    import re
    
    # Check filename first
    filename = xml_path.name
    
    # Pattern: number followed by optional space and "nm"
    # Examples: "200 nm", "200nm", "1000 nm", "50nm"
    pattern = r'(\d+)\s*nm'
    
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        size = float(match.group(1))
        print(f"  Extracted particle size from filename: {size} nm")
        return size
    
    # Check parent directories
    for parent in xml_path.parents:
        match = re.search(pattern, parent.name, re.IGNORECASE)
        if match:
            size = float(match.group(1))
            print(f"  Extracted particle size from path: {size} nm")
            return size
    
    print(f"  WARNING: Could not extract particle size, using default: {DEFAULT_PARTICLE_SIZE} nm")
    return DEFAULT_PARTICLE_SIZE


# ============================================================================
# VISUALIZATION
# ============================================================================

class InteractiveComparison:
    """Interactive comparison tool with parameter controls."""
    
    def __init__(self, xml_path: Path, mpp: float, fps: float, particle_size: float = None):
        self.xml_path = xml_path
        self.mpp = mpp
        self.fps = fps
        
        # Auto-extract particle size if not provided
        if particle_size is None:
            self.particle_size = extract_particle_size_from_path(xml_path)
        else:
            self.particle_size = particle_size
        
        # Current parameters
        self.step_interval = DEFAULT_STEP_INTERVAL
        self.fit_points = DEFAULT_FIT_POINTS
        self.min_exponent = DEFAULT_MIN_EXPONENT
        
        # Create figure
        self.fig = plt.figure(figsize=(18, 10))
        self.setup_layout()
        self.setup_controls()
        
        # Initial analysis
        self.update_analysis()
        
    def setup_layout(self):
        """Setup figure layout with subplots."""
        # Main plots
        self.ax_dx = plt.subplot2grid((4, 3), (0, 0))  # dx distribution
        self.ax_dy = plt.subplot2grid((4, 3), (0, 1))  # dy distribution
        self.ax_msd = plt.subplot2grid((4, 3), (0, 2), rowspan=2)  # MSD plot
        
        self.ax_comparison = plt.subplot2grid((4, 3), (1, 0), colspan=2)  # D comparison plot
        
        self.ax_summary = plt.subplot2grid((4, 3), (2, 0), colspan=3)  # Summary table
        self.ax_summary.axis('off')
        
        # Controls area
        self.ax_controls = plt.subplot2grid((4, 3), (3, 0), colspan=3)
        self.ax_controls.axis('off')
        
    def setup_controls(self):
        """Setup interactive controls."""
        # Step interval slider
        ax_step = plt.axes([0.15, 0.10, 0.2, 0.025])
        self.slider_step = Slider(ax_step, 'Step Interval', 1, 10, valinit=self.step_interval, valstep=1)
        self.slider_step.on_changed(self.on_step_change)
        
        # Fit points slider
        ax_fit = plt.axes([0.15, 0.06, 0.2, 0.025])
        self.slider_fit = Slider(ax_fit, 'MSD Fit Points', 3, 15, valinit=self.fit_points, valstep=1)
        self.slider_fit.on_changed(self.on_fit_change)
        
        # Min exponent slider
        ax_exp = plt.axes([0.15, 0.02, 0.2, 0.025])
        self.slider_exp = Slider(ax_exp, 'Min Exponent', 0.5, 1.2, valinit=self.min_exponent, valfmt='%.2f')
        self.slider_exp.on_changed(self.on_exp_change)
        
        # Update button
        ax_button = plt.axes([0.45, 0.06, 0.1, 0.04])
        self.button = Button(ax_button, 'Update')
        self.button.on_clicked(self.on_update)
        
    def on_step_change(self, val):
        self.step_interval = int(val)
        
    def on_fit_change(self, val):
        self.fit_points = int(val)
        
    def on_exp_change(self, val):
        self.min_exponent = val
        
    def on_update(self, event):
        self.update_analysis()
        
    def update_analysis(self):
        """Run both analyses and update plots."""
        # Clear axes
        self.ax_dx.clear()
        self.ax_dy.clear()
        self.ax_msd.clear()
        self.ax_summary.clear()
        self.ax_summary.axis('off')
        
        print("\n" + "="*70)
        print(f"Analyzing: {self.xml_path.name}")
        print(f"Parameters: step_interval={self.step_interval}, fit_points={self.fit_points}, min_exponent={self.min_exponent:.2f}")
        print("="*70)
        
        # Step Size Method
        step_result = analyze_step_size_method(
            self.xml_path, self.mpp, self.fps, self.step_interval
        )
        
        # MSD Method
        msd_result = analyze_msd_method(
            self.xml_path, self.mpp, self.fps, self.fit_points, self.min_exponent
        )
        
        # Plot dx distribution with Gaussian fit
        if step_result:
            dx = step_result['dx_nm']
            dy = step_result['dy_nm']
            
            # dx histogram and fit
            self.ax_dx.hist(dx, bins=50, density=True, alpha=0.7, color='cornflowerblue', edgecolor='black')
            x_range = np.linspace(dx.min(), dx.max(), 200)
            gaussian_x = stats.norm.pdf(x_range, step_result['mu_x'], step_result['sigma_x'])
            self.ax_dx.plot(x_range, gaussian_x, 'r-', linewidth=2, 
                           label=f'Gaussian: μ={step_result["mu_x"]:.1f}, σ={step_result["sigma_x"]:.1f} nm')
            self.ax_dx.axvline(0, color='black', linestyle='--', alpha=0.5)
            self.ax_dx.set_xlabel('dx [nm]', fontsize=10)
            self.ax_dx.set_ylabel('Density', fontsize=10)
            self.ax_dx.set_title(f'dx Distribution (interval={self.step_interval})', fontsize=11, fontweight='bold')
            self.ax_dx.legend(fontsize=8)
            self.ax_dx.grid(True, alpha=0.3)
            
            # dy histogram and fit
            self.ax_dy.hist(dy, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
            y_range = np.linspace(dy.min(), dy.max(), 200)
            gaussian_y = stats.norm.pdf(y_range, step_result['mu_y'], step_result['sigma_y'])
            self.ax_dy.plot(y_range, gaussian_y, 'r-', linewidth=2,
                           label=f'Gaussian: μ={step_result["mu_y"]:.1f}, σ={step_result["sigma_y"]:.1f} nm')
            self.ax_dy.axvline(0, color='black', linestyle='--', alpha=0.5)
            self.ax_dy.set_xlabel('dy [nm]', fontsize=10)
            self.ax_dy.set_ylabel('Density', fontsize=10)
            self.ax_dy.set_title(f'dy Distribution (interval={self.step_interval})', fontsize=11, fontweight='bold')
            self.ax_dy.legend(fontsize=8)
            self.ax_dy.grid(True, alpha=0.3)
        
        # Plot MSD with fit
        if msd_result:
            ensemble_msd = msd_result['ensemble_msd']
            imsd = msd_result['imsd']
            
            # Individual MSDs (light gray)
            for col in imsd.columns[:min(50, len(imsd.columns))]:  # Max 50 for visibility
                self.ax_msd.plot(imsd.index, imsd[col], 'k-', alpha=0.05, linewidth=0.5)
            
            # Ensemble MSD
            self.ax_msd.plot(ensemble_msd.index, ensemble_msd, 'o', markersize=5, color='blue', label='Ensemble MSD')
            
            # Fitting range
            self.ax_msd.plot(ensemble_msd.iloc[0:self.fit_points].index, 
                            ensemble_msd.iloc[0:self.fit_points], 'o', 
                            markersize=4, color='red', label='Fitting range')
            
            # Power-law fit
            fit_x = ensemble_msd.iloc[0:self.fit_points].index
            fit_y = msd_result['A'] * np.array(fit_x) ** msd_result['n']
            self.ax_msd.plot(fit_x, fit_y, 'g--', linewidth=2, 
                            label=f'Fit: A={msd_result["A"]:.2e}, n={msd_result["n"]:.3f}')
            
            # Theoretical
            D_theory = calculate_theoretical_D(self.particle_size)
            theory_y = 4 * D_theory * np.array(fit_x)
            self.ax_msd.plot(fit_x, theory_y, 'purple', linestyle='--', linewidth=2,
                            label=f'Theory: D={D_theory:.2e}')
            
            self.ax_msd.set_xscale('log')
            self.ax_msd.set_yscale('log')
            self.ax_msd.set_xlabel('Lag time [s]', fontsize=10)
            self.ax_msd.set_ylabel('MSD [µm²]', fontsize=10)
            self.ax_msd.set_title(f'MSD Analysis ({msd_result["num_particles"]}/{msd_result["total_particles"]} particles, n≥{self.min_exponent:.2f})', 
                                 fontsize=11, fontweight='bold')
            self.ax_msd.legend(fontsize=8)
            self.ax_msd.grid(True, alpha=0.3, which='both')
        
        # Comparison plot: D_measured vs D_theory with error bars
        self.ax_comparison.clear()
        D_theory = calculate_theoretical_D(self.particle_size) / 1000.0  # Convert to µm²/s
        
        # Get DLS value if available
        D_dls = DLS_MEASUREMENTS.get(self.particle_size, None)
        
        methods = []
        D_values = []
        D_errors = []
        colors = []
        
        if step_result:
            methods.append('Step Size\n(Gaussian)')
            D_values.append(step_result['D'])
            D_errors.append(step_result['D_std'])
            # Color code by quality
            if step_result.get('quality_flag') == 'good':
                colors.append('green')
            elif step_result.get('quality_flag') == 'anisotropic':
                colors.append('orange')
            else:
                colors.append('red')
        
        if msd_result:
            methods.append('MSD\n(Power-law)')
            D_values.append(msd_result['D'])
            D_errors.append(msd_result['D_std'])
            # Color code by exponent
            if msd_result['n'] > 0.95:
                colors.append('green')
            elif msd_result['n'] > 0.85:
                colors.append('orange')
            else:
                colors.append('red')
        
        if methods:
            x_pos = np.arange(len(methods))
            
            # Plot measured values with error bars
            for i, (method, D, D_err, color) in enumerate(zip(methods, D_values, D_errors, colors)):
                self.ax_comparison.errorbar(i, D, yerr=D_err, fmt='o', markersize=12, 
                                           color=color, ecolor=color, elinewidth=2, 
                                           capsize=5, capthick=2, alpha=0.7)
            
            # Plot theory line
            self.ax_comparison.axhline(D_theory, color='black', linestyle='--', linewidth=2, 
                                      label=f'Theory: {D_theory:.2e} µm²/s', alpha=0.8)
            
            # Plot DLS value if available
            if D_dls is not None:
                self.ax_comparison.axhline(D_dls, color='red', linestyle=':', linewidth=2, 
                                          label=f'DLS: {D_dls:.2e} µm²/s', alpha=0.8)
                # Add star marker at x=-0.5 for visibility
                self.ax_comparison.plot(-0.5, D_dls, '*', markersize=20, color='red', 
                                       markeredgewidth=2, markeredgecolor='darkred', zorder=10)
            
            # Shaded region for ±10% from theory
            self.ax_comparison.axhspan(D_theory * 0.9, D_theory * 1.1, 
                                      color='lightgray', alpha=0.3, label='±10% range')
            
            # Formatting
            self.ax_comparison.set_xticks(x_pos)
            self.ax_comparison.set_xticklabels(methods, fontsize=9)
            self.ax_comparison.set_xlim(-0.7, len(methods) - 0.3)  # Extended for DLS marker
            self.ax_comparison.set_ylabel('D [µm²/s]', fontsize=10)
            self.ax_comparison.set_title('Comparison: Measured D ± uncertainty vs Theory', 
                                        fontsize=11, fontweight='bold')
            self.ax_comparison.legend(fontsize=8, loc='best')
            self.ax_comparison.grid(True, alpha=0.3, axis='y')
            
            # Add deviation labels
            for i, (D, D_err) in enumerate(zip(D_values, D_errors)):
                deviation = (D - D_theory) / D_theory * 100
                # Calculate if theory is within error bars
                within_error = abs(D - D_theory) <= D_err
                marker = '✓' if within_error else '✗'
                self.ax_comparison.text(i, D + D_err + 0.02 * D_theory, 
                                       f'{marker} {deviation:+.1f}%', 
                                       ha='center', va='bottom', fontsize=8,
                                       color='green' if within_error else 'red',
                                       fontweight='bold')
        
        # Summary table
        
        summary_text = f"File: {self.xml_path.name}\n"
        summary_text += f"Particle Size: {self.particle_size} nm | Theory: D = {D_theory:.2e} µm²/s"
        if D_dls is not None:
            summary_text += f" | DLS: D = {D_dls:.2e} µm²/s"
        summary_text += "\n\n"
        
        if step_result:
            summary_text += f"STEP SIZE METHOD (interval={self.step_interval}):\n"
            summary_text += f"  D = {step_result['D']:.4e} ± {step_result['D_std']:.2e} µm²/s\n"
            summary_text += f"  D_x = {step_result['D_x']:.4e}, D_y = {step_result['D_y']:.4e}\n"
            summary_text += f"  σ_x = {step_result['sigma_x']:.2f} nm, σ_y = {step_result['sigma_y']:.2f} nm (ratio={step_result['sigma_ratio']:.2f})\n"
            summary_text += f"  μ_x = {step_result['mu_x']:.2f} nm, μ_y = {step_result['mu_y']:.2f} nm\n"
            summary_text += f"  Steps: {step_result['num_steps']}\n"
            deviation_step = (step_result['D'] - D_theory) / D_theory * 100
            sigma_deviation = abs(step_result['D'] - D_theory) / step_result['D_std'] if step_result['D_std'] > 0 else 0
            summary_text += f"  Deviation from theory: {deviation_step:+.1f}% ({sigma_deviation:.1f}σ)\n"
            summary_text += f"  Theory within error bars: {'YES' if abs(step_result['D'] - D_theory) <= step_result['D_std'] else 'NO'}\n"
            if D_dls is not None:
                deviation_dls = (step_result['D'] - D_dls) / D_dls * 100
                sigma_deviation_dls = abs(step_result['D'] - D_dls) / step_result['D_std'] if step_result['D_std'] > 0 else 0
                summary_text += f"  Deviation from DLS: {deviation_dls:+.1f}% ({sigma_deviation_dls:.1f}σ)\n"
            summary_text += "\n"
        
        if msd_result:
            summary_text += f"MSD METHOD (fit points={self.fit_points}, min_n={self.min_exponent:.2f}):\n"
            summary_text += f"  D = {msd_result['D']:.4e} ± {msd_result['D_std']:.2e} µm²/s\n"
            summary_text += f"  A = {msd_result['A']:.4e} ± {msd_result['A_err']:.2e}\n"
            summary_text += f"  n = {msd_result['n']:.3f} ± {msd_result['n_err']:.3f}\n"
            summary_text += f"  Particles: {msd_result['num_particles']}/{msd_result['total_particles']} (mean n={msd_result['mean_exponent']:.3f}±{msd_result['std_exponent']:.3f})\n"
            deviation_msd = (msd_result['D'] - D_theory) / D_theory * 100
            sigma_deviation = abs(msd_result['D'] - D_theory) / msd_result['D_std'] if msd_result['D_std'] > 0 else 0
            summary_text += f"  Deviation from theory: {deviation_msd:+.1f}% ({sigma_deviation:.1f}σ)\n"
            summary_text += f"  Theory within error bars: {'YES' if abs(msd_result['D'] - D_theory) <= msd_result['D_std'] else 'NO'}\n"
            if D_dls is not None:
                deviation_dls = (msd_result['D'] - D_dls) / D_dls * 100
                sigma_deviation_dls = abs(msd_result['D'] - D_dls) / msd_result['D_std'] if msd_result['D_std'] > 0 else 0
                summary_text += f"  Deviation from DLS: {deviation_dls:+.1f}% ({sigma_deviation_dls:.1f}σ)\n"
            summary_text += "\n"
        
        if step_result and msd_result:
            diff = abs(step_result['D'] - msd_result['D']) / ((step_result['D'] + msd_result['D'])/2) * 100
            summary_text += f"COMPARISON:\n"
            summary_text += f"  Difference between methods: {diff:.1f}%"
        
        self.ax_summary.text(0.05, 0.95, summary_text, transform=self.ax_summary.transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        self.fig.canvas.draw_idle()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run interactive comparison tool."""
    print("=" * 70)
    print("Interactive Method Comparison Tool")
    print("=" * 70)
    
    # Check if example file exists
    if not EXAMPLE_FILE.exists():
        print(f"\nERROR: Example file not found: {EXAMPLE_FILE}")
        print("\nPlease edit the EXAMPLE_FILE path in this script to point to your XML file.")
        print("You can also provide path as command line argument:")
        print(f"  python {Path(__file__).name} <path_to_xml_file> [mpp] [fps] [particle_size_nm]")
        return
    
    # Parse command line arguments if provided
    xml_path = EXAMPLE_FILE
    mpp = DEFAULT_MPP
    fps = DEFAULT_FPS
    particle_size = None  # Will be auto-extracted
    
    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
        if len(sys.argv) > 2:
            mpp = float(sys.argv[2])
        if len(sys.argv) > 3:
            fps = float(sys.argv[3])
        if len(sys.argv) > 4:
            particle_size = float(sys.argv[4])  # Override auto-extraction
    
    print(f"\nAnalyzing file: {xml_path}")
    print(f"Parameters: mpp={mpp} µm/px, fps={fps} Hz")
    if particle_size is None:
        print(f"Particle size will be auto-extracted from file path")
    else:
        print(f"Particle size: {particle_size} nm (manually specified)")
    print("\nUse sliders to adjust:")
    print("  - Step Interval: Compare 1, 2, 3, ... frame intervals")
    print("  - MSD Fit Points: Number of points for power-law fit")
    print("  - Min Exponent: Filter threshold for free diffusion\n")
    
    # Create interactive tool
    tool = InteractiveComparison(xml_path, mpp, fps, particle_size)
    plt.show()


if __name__ == "__main__":
    main()
