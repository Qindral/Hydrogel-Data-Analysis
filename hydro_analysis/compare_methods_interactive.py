"""
Interactive Method Comparison Tool

Compare MSD and Step Size methods for a single TrackMate XML file with adjustable parameters.
Allows dynamic adjustment of:
- Step interval (1, 2, 3, ...)
- MSD fit points
- Exponent filters (min/max)
- Track length filters (min/max)

Author: Jonas
Date: 2026-01-13
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import stats
import trackpy as tp

# Import from core modules
from core.io import extract_particle_size_from_path, single_file_data
from core.analysis import (
    calculate_step_sizes,
    fit_gaussian_diffusion_1d,
    diffusion_2d_from_1d_fits,
    fit_powerlaw_with_errors,
)
from core.physics import calculate_theoretical_diffusion


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default parameters
DEFAULT_PARTICLE_SIZE = 200  # nm
DEFAULT_STEP_INTERVAL = 1
DEFAULT_FIT_POINTS = 6
DEFAULT_MIN_EXPONENT = 0.85
DEFAULT_MAX_EXPONENT = 1.15
DEFAULT_MIN_TRACK_LENGTH = 10
DEFAULT_MAX_TRACK_LENGTH = 500

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

def filter_tracks_by_length(tracks_df, min_length: int, max_length: int):
    """Filter tracks by their length (number of frames)."""
    if tracks_df is None or tracks_df.empty:
        return tracks_df, 0, 0

    # Count frames per particle
    track_lengths = tracks_df.groupby('particle').size()
    total_tracks = len(track_lengths)

    # Filter by length
    valid_particles = track_lengths[(track_lengths >= min_length) & (track_lengths <= max_length)].index
    filtered_df = tracks_df[tracks_df['particle'].isin(valid_particles)].copy()

    return filtered_df, len(valid_particles), total_tracks


def analyze_step_size_method(
    result_dict: dict,
    step_interval: int = 1,
    min_track_length: int = 10,
    max_track_length: int = 500
) -> dict | None:
    """
    Analyze using step size (Gaussian fit) method with track length filtering.

    Args:
        result_dict: Standardized result dictionary from io.single_file_data
        step_interval: Frame interval for step calculation
        min_track_length: Minimum track length in frames
        max_track_length: Maximum track length in frames

    Returns:
        dict with D, D_std, sigma_x, sigma_y, quality metrics
    """
    tracks_df = result_dict.get('tracks_df')
    if tracks_df is None or tracks_df.empty:
        return None

    mpp = result_dict['mpp']
    fps = result_dict['fps']

    # Filter tracks by length
    filtered_tracks, num_valid, num_total = filter_tracks_by_length(
        tracks_df, min_track_length, max_track_length
    )

    if filtered_tracks.empty:
        return None

    # Calculate step sizes using core function
    step_df = calculate_step_sizes(filtered_tracks, step_interval=step_interval)
    if step_df.empty:
        return None

    # Fit Gaussian to each axis using core functions
    fit_x = fit_gaussian_diffusion_1d(
        step_df['dx'], mpp, fps, frame_interval=step_interval, axis='x'
    )
    fit_y = fit_gaussian_diffusion_1d(
        step_df['dy'], mpp, fps, frame_interval=step_interval, axis='y'
    )

    # Combine 2D result using core function
    fit_2d = diffusion_2d_from_1d_fits(fit_x, fit_y)

    # Convert steps to nm for plotting
    dx_nm = step_df['dx'].values * mpp * 1000.0
    dy_nm = step_df['dy'].values * mpp * 1000.0

    return {
        'method': 'Step Size (Gaussian)',
        'D': fit_2d['D_um2_per_s'],
        'D_std': fit_2d['D_dir_disagreement_um2_per_s'],
        'D_x': fit_2d['D_x_um2_per_s'],
        'D_y': fit_2d['D_y_um2_per_s'],
        'sigma_x': fit_2d['sigma_x_nm'],
        'sigma_y': fit_2d['sigma_y_nm'],
        'mu_x': fit_2d['mu_x_nm'],
        'mu_y': fit_2d['mu_y_nm'],
        'sigma_ratio': fit_2d['sigma_ratio'],
        'mean_x_ratio': fit_2d['mean_x_ratio'],
        'mean_y_ratio': fit_2d['mean_y_ratio'],
        'quality_flag': fit_2d['quality_flag'],
        'num_steps': len(dx_nm),
        'num_tracks': num_valid,
        'total_tracks': num_total,
        'dx_nm': dx_nm,
        'dy_nm': dy_nm,
        'step_interval': step_interval
    }


def analyze_msd_method(
    result_dict: dict,
    fit_points: int = 6,
    min_exponent: float = 0.85,
    max_exponent: float = 1.15,
    min_track_length: int = 10,
    max_track_length: int = 500
) -> dict | None:
    """
    Analyze using MSD power-law fit method with particle filtering.

    Args:
        result_dict: Standardized result dictionary from io.single_file_data
        fit_points: Number of points for power-law fit
        min_exponent: Minimum exponent threshold for filtering particles
        max_exponent: Maximum exponent threshold for filtering particles
        min_track_length: Minimum track length in frames
        max_track_length: Maximum track length in frames

    Returns:
        dict with D, D_std, exponent, quality metrics
    """
    tracks_df = result_dict.get('tracks_df')
    if tracks_df is None or tracks_df.empty:
        return None

    mpp = result_dict['mpp']
    fps = result_dict['fps']

    # Filter tracks by length
    filtered_tracks, num_length_valid, num_total = filter_tracks_by_length(
        tracks_df, min_track_length, max_track_length
    )

    if filtered_tracks.empty:
        return None

    # Calculate iMSD using trackpy
    tp.quiet()
    try:
        imsd = tp.imsd(filtered_tracks, mpp=mpp, fps=fps)
    except Exception as e:
        print(f"Error calculating iMSD: {e}")
        return None

    if imsd is None or imsd.empty:
        return None

    # Filter particles by exponent
    valid_particles = []
    particle_exponents = []
    rejected_low = 0
    rejected_high = 0

    for particle_id in imsd.columns:
        particle_msd = imsd[particle_id].dropna()

        if len(particle_msd) < fit_points:
            continue

        try:
            particle_params = fit_powerlaw_with_errors(particle_msd, points=fit_points)
            n_particle = float(particle_params['n'][0])

            if n_particle < min_exponent:
                rejected_low += 1
            elif n_particle > max_exponent:
                rejected_high += 1
            else:
                valid_particles.append(particle_id)
                particle_exponents.append(n_particle)
        except Exception:
            continue

    if not valid_particles:
        return None

    # Calculate ensemble MSD for valid particles
    filtered_imsd = imsd[valid_particles]
    ensemble_msd = filtered_imsd.mean(axis=1)

    # Fit power-law using core function
    try:
        params = fit_powerlaw_with_errors(ensemble_msd, points=fit_points)
        A = float(params['A'][0])
        A_err = float(params['A_err'][0])
        n = float(params['n'][0])
        n_err = float(params['n_err'][0])

        # Calculate D: MSD = 4*D*t => D = A/4
        D = A / 4.0
        D_err = A_err / 4.0

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
            'tracks_after_length_filter': num_length_valid,
            'total_tracks': num_total,
            'rejected_low_n': rejected_low,
            'rejected_high_n': rejected_high,
            'mean_exponent': np.mean(particle_exponents),
            'std_exponent': np.std(particle_exponents),
            'imsd': filtered_imsd,
            'ensemble_msd': ensemble_msd,
            'fit_points': fit_points,
            'min_exponent': min_exponent,
            'max_exponent': max_exponent
        }
    except Exception as e:
        print(f"Error in MSD fitting: {e}")
        return None


# ============================================================================
# VISUALIZATION
# ============================================================================

class InteractiveComparison:
    """Interactive comparison tool with parameter controls."""

    def __init__(self, xml_path: Path, particle_size: float = None):
        self.xml_path = xml_path

        # Load data using core io module
        self.result_dict = single_file_data(xml_path)
        if self.result_dict is None:
            raise ValueError(f"Could not load data from {xml_path}")

        self.mpp = self.result_dict['mpp']
        self.fps = self.result_dict['fps']

        # Get track statistics for slider ranges
        tracks_df = self.result_dict.get('tracks_df')
        if tracks_df is not None and not tracks_df.empty:
            track_lengths = tracks_df.groupby('particle').size()
            self.max_possible_track_length = int(track_lengths.max())
            self.min_possible_track_length = int(track_lengths.min())
            self.num_total_tracks = len(track_lengths)
        else:
            self.max_possible_track_length = 500
            self.min_possible_track_length = 1
            self.num_total_tracks = 0

        # Determine particle size with robust extraction
        if particle_size is not None:
            self.particle_size = particle_size
        elif self.result_dict.get('particle_size_nm') is not None:
            self.particle_size = self.result_dict['particle_size_nm']
        else:
            self.particle_size = self._extract_particle_size(xml_path)

        print(f"  Particle size: {self.particle_size} nm")
        print(f"  mpp: {self.mpp} µm/px, fps: {self.fps} Hz")
        print(f"  Total tracks: {self.num_total_tracks}")
        print(f"  Track lengths: {self.min_possible_track_length} - {self.max_possible_track_length} frames")

        # Current parameters
        self.step_interval = DEFAULT_STEP_INTERVAL
        self.fit_points = DEFAULT_FIT_POINTS
        self.min_exponent = DEFAULT_MIN_EXPONENT
        self.max_exponent = DEFAULT_MAX_EXPONENT
        self.min_track_length = max(DEFAULT_MIN_TRACK_LENGTH, self.min_possible_track_length)
        self.max_track_length = min(DEFAULT_MAX_TRACK_LENGTH, self.max_possible_track_length)

        # Create figure
        self.fig = plt.figure(figsize=(22, 15))
        self.setup_layout()
        self.setup_controls()

        # Initial analysis
        self.update_analysis()

    def _extract_particle_size(self, xml_path: Path) -> float:
        """Extract particle size from filename or parent directories."""
        import re
        pattern = r'(\d+)\s*nm'

        # Try filename first
        match = re.search(pattern, xml_path.name, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # Try parent directories (up to 3 levels)
        for parent in list(xml_path.parents)[:3]:
            match = re.search(pattern, parent.name, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # Fallback to core function
        size = extract_particle_size_from_path(xml_path.parent)
        if size is not None:
            return size

        print(f"  WARNING: Could not extract particle size, using default: {DEFAULT_PARTICLE_SIZE} nm")
        return DEFAULT_PARTICLE_SIZE

    def setup_layout(self):
        """Setup figure layout with subplots using GridSpec for better control."""
        from matplotlib.gridspec import GridSpec

        # Create GridSpec: 3 rows, 3 columns
        gs = GridSpec(3, 3, figure=self.fig,
                      height_ratios=[1, 1, 0.6],
                      width_ratios=[1, 1, 1.2],
                      hspace=0.30, wspace=0.25,
                      left=0.05, right=0.98, top=0.93, bottom=0.22)

        # Row 0: dx distribution, dy distribution
        self.ax_dx = self.fig.add_subplot(gs[0, 0])
        self.ax_dy = self.fig.add_subplot(gs[0, 1])

        # MSD plot spans rows 0-1, column 2
        self.ax_msd = self.fig.add_subplot(gs[0:2, 2])

        # Row 1: Comparison plot spans columns 0-1
        self.ax_comparison = self.fig.add_subplot(gs[1, 0:2])

        # Row 2: Summary text (full width)
        self.ax_summary = self.fig.add_subplot(gs[2, :])
        self.ax_summary.axis('off')

    def setup_controls(self):
        """Setup interactive controls in the bottom area."""
        slider_height = 0.015
        slider_width = 0.18

        # Left column: Step Size parameters
        col1_x = 0.08

        ax_step = plt.axes([col1_x, 0.14, slider_width, slider_height])
        self.slider_step = Slider(ax_step, 'Step Interval', 1, 10,
                                  valinit=self.step_interval, valstep=1)
        self.slider_step.on_changed(self.on_param_change)

        # Middle column: MSD parameters
        col2_x = 0.35

        ax_fit = plt.axes([col2_x, 0.14, slider_width, slider_height])
        self.slider_fit = Slider(ax_fit, 'MSD Fit Points', 3, 20,
                                 valinit=self.fit_points, valstep=1)
        self.slider_fit.on_changed(self.on_param_change)

        ax_min_exp = plt.axes([col2_x, 0.10, slider_width, slider_height])
        self.slider_min_exp = Slider(ax_min_exp, 'n_min', 0.0, 2.0,
                                     valinit=self.min_exponent, valfmt='%.2f')
        self.slider_min_exp.on_changed(self.on_param_change)

        ax_max_exp = plt.axes([col2_x, 0.06, slider_width, slider_height])
        self.slider_max_exp = Slider(ax_max_exp, 'n_max', 0.0, 2.0,
                                     valinit=self.max_exponent, valfmt='%.2f')
        self.slider_max_exp.on_changed(self.on_param_change)

        # Right column: Track length filters
        col3_x = 0.62

        max_len_slider = min(self.max_possible_track_length, 1000)
        ax_min_len = plt.axes([col3_x, 0.14, slider_width, slider_height])
        self.slider_min_len = Slider(ax_min_len, 'Track min', 1, max_len_slider,
                                     valinit=self.min_track_length, valstep=1)
        self.slider_min_len.on_changed(self.on_param_change)

        ax_max_len = plt.axes([col3_x, 0.10, slider_width, slider_height])
        self.slider_max_len = Slider(ax_max_len, 'Track max', 1, max_len_slider,
                                     valinit=self.max_track_length, valstep=1)
        self.slider_max_len.on_changed(self.on_param_change)

        # Update button
        ax_button = plt.axes([col3_x + 0.02, 0.03, 0.12, 0.04])
        self.button = Button(ax_button, 'Update Analysis')
        self.button.on_clicked(self.on_update)

        # Add section labels
        self.fig.text(col1_x + slider_width/2, 0.175, 'STEP SIZE',
                     ha='center', fontsize=11, fontweight='bold')
        self.fig.text(col2_x + slider_width/2, 0.175, 'MSD FILTER',
                     ha='center', fontsize=11, fontweight='bold')
        self.fig.text(col3_x + slider_width/2, 0.175, 'TRACK LENGTH',
                     ha='center', fontsize=11, fontweight='bold')

    def on_param_change(self, _val):
        """Update parameters from sliders (but don't re-analyze yet)."""
        self.step_interval = int(self.slider_step.val)
        self.fit_points = int(self.slider_fit.val)
        self.min_exponent = self.slider_min_exp.val
        self.max_exponent = self.slider_max_exp.val
        self.min_track_length = int(self.slider_min_len.val)
        self.max_track_length = int(self.slider_max_len.val)

        # Ensure min <= max for track length
        if self.min_track_length > self.max_track_length:
            self.max_track_length = self.min_track_length

        # Ensure min <= max for exponent
        if self.min_exponent > self.max_exponent:
            self.max_exponent = self.min_exponent

    def on_update(self, _event):
        self.update_analysis()

    def update_analysis(self):
        """Run both analyses and update plots."""
        # Clear axes
        self.ax_dx.clear()
        self.ax_dy.clear()
        self.ax_msd.clear()
        self.ax_comparison.clear()
        self.ax_summary.clear()
        self.ax_summary.axis('off')

        print("\n" + "="*70)
        print(f"Analyzing: {self.xml_path.name}")
        print(f"Step interval: {self.step_interval}")
        print(f"MSD fit points: {self.fit_points}, n: [{self.min_exponent:.2f}, {self.max_exponent:.2f}]")
        print(f"Track length: [{self.min_track_length}, {self.max_track_length}]")
        print("="*70)

        # Step Size Method
        step_result = analyze_step_size_method(
            self.result_dict,
            self.step_interval,
            self.min_track_length,
            self.max_track_length
        )

        # MSD Method
        msd_result = analyze_msd_method(
            self.result_dict,
            self.fit_points,
            self.min_exponent,
            self.max_exponent,
            self.min_track_length,
            self.max_track_length
        )

        # Plot dx distribution with Gaussian fit
        if step_result:
            dx = step_result['dx_nm']
            dy = step_result['dy_nm']

            # dx histogram and fit
            self.ax_dx.hist(dx, bins=50, density=True, alpha=0.7, color='cornflowerblue', edgecolor='black', linewidth=0.5)
            x_range = np.linspace(dx.min(), dx.max(), 200)
            gaussian_x = stats.norm.pdf(x_range, step_result['mu_x'], step_result['sigma_x'])
            self.ax_dx.plot(x_range, gaussian_x, 'r-', linewidth=2.5,
                           label=f'μ={step_result["mu_x"]:.1f}, σ={step_result["sigma_x"]:.1f} nm')
            self.ax_dx.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
            self.ax_dx.set_xlabel('dx [nm]', fontsize=12)
            self.ax_dx.set_ylabel('Density', fontsize=12)
            self.ax_dx.set_title(f'dx Distribution (Δt={self.step_interval}, {step_result["num_tracks"]}/{step_result["total_tracks"]} tracks)',
                                fontsize=13, fontweight='bold')
            self.ax_dx.legend(fontsize=10, loc='upper right')
            self.ax_dx.grid(True, alpha=0.3)
            self.ax_dx.tick_params(labelsize=10)

            # dy histogram and fit
            self.ax_dy.hist(dy, bins=50, density=True, alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.5)
            y_range = np.linspace(dy.min(), dy.max(), 200)
            gaussian_y = stats.norm.pdf(y_range, step_result['mu_y'], step_result['sigma_y'])
            self.ax_dy.plot(y_range, gaussian_y, 'r-', linewidth=2.5,
                           label=f'μ={step_result["mu_y"]:.1f}, σ={step_result["sigma_y"]:.1f} nm')
            self.ax_dy.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
            self.ax_dy.set_xlabel('dy [nm]', fontsize=12)
            self.ax_dy.set_ylabel('Density', fontsize=12)
            self.ax_dy.set_title(f'dy Distribution (Δt={self.step_interval}, {step_result["num_steps"]} steps)',
                                fontsize=13, fontweight='bold')
            self.ax_dy.legend(fontsize=10, loc='upper right')
            self.ax_dy.grid(True, alpha=0.3)
            self.ax_dy.tick_params(labelsize=10)

        # Plot MSD with fit
        if msd_result:
            ensemble_msd = msd_result['ensemble_msd']
            imsd = msd_result['imsd']

            # Individual MSDs (light gray)
            for col in imsd.columns[:min(50, len(imsd.columns))]:
                self.ax_msd.plot(imsd.index, imsd[col], 'k-', alpha=0.05, linewidth=0.5)

            # Ensemble MSD
            self.ax_msd.plot(ensemble_msd.index, ensemble_msd, 'o', markersize=6, color='blue', label='Ensemble MSD')

            # Fitting range
            self.ax_msd.plot(ensemble_msd.iloc[0:self.fit_points].index,
                            ensemble_msd.iloc[0:self.fit_points], 'o',
                            markersize=5, color='red', label='Fitting range')

            # Power-law fit
            fit_x = ensemble_msd.iloc[0:self.fit_points].index
            fit_y = msd_result['A'] * np.array(fit_x) ** msd_result['n']
            self.ax_msd.plot(fit_x, fit_y, 'g--', linewidth=2.5,
                            label=f'Fit: A={msd_result["A"]:.2e}, n={msd_result["n"]:.3f}')

            # Theoretical line using core physics function
            D_theory = calculate_theoretical_diffusion(self.particle_size)
            theory_y = 4 * D_theory * np.array(fit_x)
            self.ax_msd.plot(fit_x, theory_y, 'purple', linestyle='--', linewidth=2.5,
                            label=f'Theory: D={D_theory:.2e}')

            self.ax_msd.set_xscale('log')
            self.ax_msd.set_yscale('log')
            self.ax_msd.set_xlabel('Lag time [s]', fontsize=12)
            self.ax_msd.set_ylabel('MSD [µm²]', fontsize=12)

            # Title with filter info
            title_str = f'MSD Analysis\n'
            title_str += f'{msd_result["num_particles"]}/{msd_result["total_particles"]} particles '
            title_str += f'(n: {self.min_exponent:.2f}-{self.max_exponent:.2f})'
            self.ax_msd.set_title(title_str, fontsize=13, fontweight='bold')
            self.ax_msd.legend(fontsize=9, loc='upper left')
            self.ax_msd.grid(True, alpha=0.3, which='both')
            self.ax_msd.tick_params(labelsize=10)

        # Comparison plot
        D_theory = calculate_theoretical_diffusion(self.particle_size)
        D_dls = DLS_MEASUREMENTS.get(self.particle_size, None)

        methods = []
        D_values = []
        D_errors = []
        colors = []

        if step_result:
            methods.append('Step Size\n(Gaussian)')
            D_values.append(step_result['D'])
            D_errors.append(step_result['D_std'])
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
            if msd_result['n'] > 0.95:
                colors.append('green')
            elif msd_result['n'] > 0.85:
                colors.append('orange')
            else:
                colors.append('red')

        if methods:
            x_pos = np.arange(len(methods))

            for i, (_method, D, D_err, color) in enumerate(zip(methods, D_values, D_errors, colors)):
                self.ax_comparison.errorbar(i, D, yerr=D_err, fmt='o', markersize=14,
                                           color=color, ecolor=color, elinewidth=2.5,
                                           capsize=6, capthick=2.5, alpha=0.8)

            self.ax_comparison.axhline(D_theory, color='black', linestyle='--', linewidth=2.5,
                                      label=f'Theory: {D_theory:.2e} µm²/s', alpha=0.8)

            if D_dls is not None:
                self.ax_comparison.axhline(D_dls, color='red', linestyle=':', linewidth=2.5,
                                          label=f'DLS: {D_dls:.2e} µm²/s', alpha=0.8)
                self.ax_comparison.plot(-0.5, D_dls, '*', markersize=22, color='red',
                                       markeredgewidth=2, markeredgecolor='darkred', zorder=10)

            self.ax_comparison.axhspan(D_theory * 0.9, D_theory * 1.1,
                                      color='lightgray', alpha=0.3, label='±10% range')

            self.ax_comparison.set_xticks(x_pos)
            self.ax_comparison.set_xticklabels(methods, fontsize=11)
            self.ax_comparison.set_xlim(-0.7, len(methods) - 0.3)
            self.ax_comparison.set_ylabel('D [µm²/s]', fontsize=12)
            self.ax_comparison.set_title('Comparison: Measured D vs Theory',
                                        fontsize=13, fontweight='bold')
            self.ax_comparison.legend(fontsize=10, loc='best')
            self.ax_comparison.grid(True, alpha=0.3, axis='y')
            self.ax_comparison.tick_params(labelsize=10)

            for i, (D, D_err) in enumerate(zip(D_values, D_errors)):
                deviation = (D - D_theory) / D_theory * 100
                within_error = abs(D - D_theory) <= D_err
                marker = '+' if within_error else 'x'
                self.ax_comparison.text(i, D + D_err + 0.02 * D_theory,
                                       f'{marker} {deviation:+.1f}%',
                                       ha='center', va='bottom', fontsize=10,
                                       color='green' if within_error else 'red',
                                       fontweight='bold')

        # Summary table - split into two columns
        col1_text = f"FILE: {self.xml_path.name}\n"
        col1_text += f"Particle: {self.particle_size:.0f} nm | mpp: {self.mpp} µm/px | fps: {self.fps} Hz\n"
        col1_text += f"Theory: D = {D_theory:.4e} µm²/s"
        if D_dls:
            col1_text += f" | DLS: {D_dls:.4e} µm²/s"
        col1_text += f"\nTrack filter: [{self.min_track_length}, {self.max_track_length}] frames\n\n"

        if step_result:
            col1_text += f"STEP SIZE (Δt={self.step_interval}):\n"
            col1_text += f"  D = {step_result['D']:.4e} ± {step_result['D_std']:.2e} µm²/s\n"
            col1_text += f"  σ_x={step_result['sigma_x']:.1f}, σ_y={step_result['sigma_y']:.1f} nm\n"
            col1_text += f"  Tracks: {step_result['num_tracks']}/{step_result['total_tracks']}, Steps: {step_result['num_steps']}\n"
            col1_text += f"  Quality: {step_result['quality_flag']}\n"
            dev = (step_result['D'] - D_theory) / D_theory * 100
            col1_text += f"  Deviation: {dev:+.1f}%\n"

        col2_text = ""
        if msd_result:
            col2_text += f"MSD (fit={self.fit_points}pts, n:[{self.min_exponent:.2f},{self.max_exponent:.2f}]):\n"
            col2_text += f"  D = {msd_result['D']:.4e} ± {msd_result['D_std']:.2e} µm²/s\n"
            col2_text += f"  n = {msd_result['n']:.3f} ± {msd_result['n_err']:.3f}\n"
            col2_text += f"  Particles: {msd_result['num_particles']}/{msd_result['total_particles']}\n"
            col2_text += f"  Rejected: n<{self.min_exponent:.2f}: {msd_result['rejected_low_n']}, n>{self.max_exponent:.2f}: {msd_result['rejected_high_n']}\n"
            col2_text += f"  Mean n: {msd_result['mean_exponent']:.3f} ± {msd_result['std_exponent']:.3f}\n"
            dev = (msd_result['D'] - D_theory) / D_theory * 100
            col2_text += f"  Deviation: {dev:+.1f}%\n"

        if step_result and msd_result:
            diff = abs(step_result['D'] - msd_result['D']) / ((step_result['D'] + msd_result['D'])/2) * 100
            col2_text += f"\nMETHOD DIFF: {diff:.1f}%"

        self.ax_summary.text(0.02, 0.95, col1_text, transform=self.ax_summary.transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))

        self.ax_summary.text(0.52, 0.95, col2_text, transform=self.ax_summary.transAxes,
                            fontsize=9, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4))

        # Main title
        self.fig.suptitle(f'Interactive Method Comparison - {self.particle_size:.0f} nm particles',
                         fontsize=16, fontweight='bold', y=0.97)

        self.fig.canvas.draw_idle()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run interactive comparison tool."""
    print("=" * 70)
    print("Interactive Method Comparison Tool")
    print("=" * 70)

    # Example file path - adjust as needed
    EXAMPLE_FILE = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\Tracks\1000 nm_4_Tracks.xml")

    # Parse command line arguments if provided
    xml_path = EXAMPLE_FILE
    particle_size = None

    if len(sys.argv) > 1:
        xml_path = Path(sys.argv[1])
        if len(sys.argv) > 2:
            particle_size = float(sys.argv[2])

    if not xml_path.exists():
        print(f"\nERROR: File not found: {xml_path}")
        print(f"\nUsage: python {Path(__file__).name} <path_to_xml_file> [particle_size_nm]")
        return

    print(f"\nAnalyzing file: {xml_path}")
    print("\nInteractive filters:")
    print("  - Step Interval: Frame interval for step size calculation")
    print("  - MSD Fit Points: Number of lag times for power-law fit")
    print("  - n_min / n_max: Exponent range for particle filtering")
    print("  - Track min/max: Track length filter in frames\n")

    tool = InteractiveComparison(xml_path, particle_size)
    plt.show()


if __name__ == "__main__":
    main()
