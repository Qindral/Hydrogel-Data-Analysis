"""Visualization functions that return figures instead of saving."""

from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ResultsAggregator:
    """Collect results from multiple analyses before plotting."""
    
    def __init__(self):
        self.results: List[Dict] = []
    
    def add(self, **kwargs):
        """Add a single result."""
        self.results.append(kwargs)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy analysis."""
        return pd.DataFrame(self.results)
    
    def save_summary(self, output_path: Path):
        """Save CSV summary."""
        df = self.get_dataframe()
        df.to_csv(output_path, index=False)
    
    def plot_size_vs_diffusion(
        self,
        show_theory: bool = True,
        title: Optional[str] = None
    ) -> plt.Figure:
        """Create D vs particle size comparison plot.
        
        Args:
            show_theory: Whether to show theoretical curve
            title: Optional plot title
        
        Returns:
            matplotlib Figure object (not saved)
        """
        df = self.get_dataframe()
        
        if df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by particle size
        for size in sorted(df['particle_size_nm'].unique()):
            subset = df[df['particle_size_nm'] == size]
            
            # Plot individual measurements
            ax.errorbar(
                [size] * len(subset),
                subset['D_measured'],
                yerr=subset.get('D_error', 0),
                fmt='o',
                alpha=0.6,
                markersize=8,
                label=f'{size:.0f} nm ({len(subset)} files)'
            )
        
        # Theory line
        if show_theory and 'D_theory' in df.columns:
            sizes = sorted(df['particle_size_nm'].unique())
            theory_vals = [df[df['particle_size_nm'] == s]['D_theory'].iloc[0] for s in sizes]
            ax.plot(sizes, theory_vals, 'k--', label='Theory (Stokes-Einstein)', lw=2)
        
        ax.set_xlabel('Particle Size (nm)', fontsize=12)
        ax.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        if title:
            ax.set_title(title, fontsize=14)
        
        fig.tight_layout()
        return fig
    
    def plot_quality_summary(self) -> plt.Figure:
        """Create quality control summary plot."""
        df = self.get_dataframe()
        
        if df.empty or 'quality_ok' not in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No quality data available', ha='center', va='center')
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Quality pass rate by particle size
        quality_by_size = df.groupby('particle_size_nm')['quality_ok'].agg(['sum', 'count'])
        quality_by_size['pass_rate'] = 100 * quality_by_size['sum'] / quality_by_size['count']
        
        ax1.bar(quality_by_size.index, quality_by_size['pass_rate'], alpha=0.7, edgecolor='black')
        ax1.axhline(y=80, color='r', linestyle='--', label='80% threshold')
        ax1.set_xlabel('Particle Size (nm)', fontsize=12)
        ax1.set_ylabel('Quality Pass Rate (%)', fontsize=12)
        ax1.set_title('Data Quality by Particle Size', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Overall quality distribution
        quality_counts = df['quality_ok'].value_counts()
        labels = ['Passed', 'Failed']
        sizes = [quality_counts.get(True, 0), quality_counts.get(False, 0)]
        colors = ['#90EE90', '#FFB6C6']
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Quality Distribution', fontsize=14)
        
        fig.tight_layout()
        return fig
