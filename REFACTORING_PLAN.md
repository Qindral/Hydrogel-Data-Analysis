# Hydrogel Data Analysis - Refactoring Plan

## Executive Summary

This document outlines a comprehensive restructuring plan to consolidate duplicate code, improve robustness, enable flexible processing modes, implement parallelization, and create a cleaner results aggregation system.

## Current Problems Identified

1. **Code Duplication**: `fit_powerlaw_with_errors()`, `load_tracks_xml()`, `subtract_background()` duplicated across 4-7 files
2. **Inconsistent Processing**: Each script has its own loop/batch logic
3. **Excessive Prints**: Hard to extract useful information when processing many files
4. **No Parallelization**: Sequential processing of independent files
5. **Immediate Figure Saving**: No way to aggregate results before saving
6. **Rigid Structure**: Scripts hardcoded for specific workflows

## Proposed Architecture

```
hydro_analysis/
├── core/
│   ├── __init__.py
│   ├── io.py                    # TIFF/XML loading, metadata extraction
│   ├── tracking.py              # Trackpy spot detection, TrackMate parsing
│   ├── analysis.py              # MSD calculation, power-law fitting, step size
│   ├── visualization.py         # Plotting functions (return figures, don't save)
│   └── utils.py                 # Background subtraction, calibration helpers
│
├── workflows/
│   ├── __init__.py
│   ├── base.py                  # BaseWorkflow class with common patterns
│   ├── msd_workflow.py          # MSD-specific workflows
│   ├── stepsize_workflow.py     # Step size diffusion workflows
│   └── sem_workflow.py          # SEM particle analysis workflows
│
├── cli/
│   ├── __init__.py
│   └── process.py               # Unified CLI with subcommands
│
├── legacy/                       # Move old scripts here
│   ├── MSD_Trackpy.py
│   ├── Trackpy_MSD_v1.py
│   └── ... (other deprecated files)
│
├── data_loader.py               # Keep existing - already good
├── metadata.py                  # Keep existing - already good
├── Particle_Parameter_Tuner.py # Keep - interactive tool
├── sem_particle_viewer.py       # Keep - interactive GUI
└── ...
```

## Phase 1: Core Module Consolidation

### 1.1 Create `core/io.py` - Unified I/O Functions

**Purpose**: Single source of truth for loading data

```python
"""Unified I/O functions for hydrogel analysis."""
from pathlib import Path
from typing import Optional, Dict, Any, List
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import tifffile
from .metadata import DatasetMetadata

class TrackLoader:
    """Unified track loading from XML or trackpy."""
    
    @staticmethod
    def from_trackmate_xml(
        xml_path: Path,
        mpp: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> pd.DataFrame:
        """Load tracks from TrackMate XML with automatic metadata extraction.
        
        Returns DataFrame with columns: ['particle', 'frame', 'x', 'y']
        and attrs: {'mpp': float, 'fps': float}
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract metadata from XML if not provided
        if mpp is None:
            space_unit = root.get('spaceUnits', 'pixel')
            # Parse from XML or default
            mpp = 0.15  # fallback
            
        if fps is None:
            frame_interval = float(root.get('frameInterval', '1.0'))
            time_unit = root.get('timeUnits', 'frame')
            fps = 1.0 / frame_interval if frame_interval > 0 else 1.0
        
        # Parse tracks
        data = []
        for pid, particle in enumerate(root.findall('particle')):
            for detection in particle.findall('detection'):
                data.append({
                    'particle': pid,
                    'frame': int(float(detection.get('t'))),
                    'x': float(detection.get('x')),
                    'y': float(detection.get('y')),
                })
        
        df = pd.DataFrame(data)
        df.attrs['mpp'] = mpp
        df.attrs['fps'] = fps
        return df.sort_values(['particle', 'frame']).reset_index(drop=True)
    
    @staticmethod
    def filter_tracks(df: pd.DataFrame, min_length: int = 30) -> pd.DataFrame:
        """Filter tracks by minimum length."""
        counts = df.groupby('particle').size()
        valid = counts[counts >= min_length].index
        return df[df['particle'].isin(valid)].reset_index(drop=True)

def find_xml_files(
    root_path: Path,
    pattern: str = "**/*Tracks.xml"
) -> List[Path]:
    """Recursively find TrackMate XML files."""
    return sorted(root_path.glob(pattern))
```

### 1.2 Create `core/analysis.py` - Unified Analysis Functions

```python
"""Unified analysis functions for MSD and diffusion."""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy.optimize import curve_fit

def fit_powerlaw_with_errors(
    em_series: pd.Series,
    points: int = 10,
) -> SimpleNamespace:
    """Fit power-law y = A * x^n with error estimates.
    
    Single implementation replacing all duplicates.
    """
    xs = em_series.iloc[:points].index.values.astype(float)
    ys = em_series.iloc[:points].values.astype(float)
    
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    if mask.sum() < 3:
        return SimpleNamespace(
            A=np.array([np.nan]), n=np.array([np.nan]),
            A_err=np.array([np.nan]), n_err=np.array([np.nan])
        )
    
    lx = np.log(xs[mask])
    ly = np.log(ys[mask])
    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
    
    n_fit, logA_fit = float(coeffs[0]), float(coeffs[1])
    se = np.sqrt(np.diag(cov))
    
    A_fit = np.exp(logA_fit)
    A_err = A_fit * float(se[1])
    
    return SimpleNamespace(
        A=np.array([A_fit]),
        n=np.array([n_fit]),
        A_err=np.array([A_err]),
        n_err=np.array([se[0]]),
        cov=cov
    )

def compute_msd(
    tracks: pd.DataFrame,
    max_lagtime: Optional[int] = None,
) -> pd.DataFrame:
    """Compute ensemble MSD from tracks.
    
    Wrapper around trackpy with consistent interface.
    """
    import trackpy as tp
    
    msd = tp.imsd(tracks, mpp=tracks.attrs.get('mpp', 1.0),
                  fps=tracks.attrs.get('fps', 1.0), max_lagtime=max_lagtime)
    return msd

def compute_diffusion_from_msd(
    msd: pd.DataFrame,
    fit_points: int = 6,
) -> Tuple[float, float, float, float]:
    """Extract diffusion coefficient from MSD.
    
    Returns: (D_µm2_per_s, D_error, exponent, exponent_error)
    """
    em = msd.mean(axis=1)
    fit = fit_powerlaw_with_errors(em, points=fit_points)
    
    # D = A / 4 for 2D diffusion
    D = fit.A[0] / 4.0
    D_err = fit.A_err[0] / 4.0
    
    return D, D_err, fit.n[0], fit.n_err[0]

def compute_step_size_diffusion(
    tracks: pd.DataFrame,
) -> Tuple[float, float]:
    """Compute diffusion from displacement distributions.
    
    Method: Fit Gaussians to dx, dy distributions.
    D = σ² / (2 * dt)
    """
    dt = 1.0 / tracks.attrs.get('fps', 1.0)
    mpp = tracks.attrs.get('mpp', 1.0)
    
    # Calculate displacements
    displacements = []
    for pid in tracks['particle'].unique():
        track = tracks[tracks['particle'] == pid].sort_values('frame')
        dx = np.diff(track['x'].values) * mpp
        dy = np.diff(track['y'].values) * mpp
        displacements.extend(dx)
        displacements.extend(dy)
    
    displacements = np.array(displacements)
    sigma_sq = np.var(displacements)
    sigma_sq_err = sigma_sq / np.sqrt(len(displacements))
    
    D = sigma_sq / (2 * dt)
    D_err = sigma_sq_err / (2 * dt)
    
    return D, D_err
```

### 1.3 Create `core/utils.py` - Utility Functions

```python
"""Utility functions for image processing."""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def subtract_background(
    image: np.ndarray,
    method: str = 'rolling_ball',
    radius: int = 50,
    sigma: float = 1.0
) -> np.ndarray:
    """Unified background subtraction.
    
    Args:
        image: Input image
        method: 'rolling_ball', 'gaussian', or 'none'
        radius: Radius for rolling ball
        sigma: Sigma for Gaussian blur
    """
    if method == 'rolling_ball':
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2, radius * 2)
        )
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cv2.subtract(image, background)
    
    elif method == 'gaussian':
        background = gaussian_filter(image, sigma=sigma)
        return np.clip(image - background, 0, None)
    
    else:
        return image
```

### 1.4 Create `core/visualization.py` - Deferred Plotting

```python
"""Visualization functions that return figures instead of saving."""
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
    
    def plot_size_vs_diffusion(
        self,
        show_theory: bool = True,
        title: Optional[str] = None
    ) -> plt.Figure:
        """Create comparison plot (returns figure, doesn't save)."""
        df = self.get_dataframe()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by particle size
        for size in df['particle_size_nm'].unique():
            subset = df[df['particle_size_nm'] == size]
            ax.errorbar(
                [size] * len(subset),
                subset['D_measured'],
                yerr=subset['D_error'],
                fmt='o',
                alpha=0.6,
                label=f'{size} nm'
            )
        
        if show_theory and 'D_theory' in df.columns:
            sizes = df['particle_size_nm'].unique()
            theory = df.groupby('particle_size_nm')['D_theory'].first()
            ax.plot(sizes, theory.values, 'k--', label='Theory', lw=2)
        
        ax.set_xlabel('Particle Size (nm)')
        ax.set_ylabel('Diffusion Coefficient (µm²/s)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title)
        
        fig.tight_layout()
        return fig
    
    def save_summary(self, output_path: Path):
        """Save CSV summary."""
        df = self.get_dataframe()
        df.to_csv(output_path, index=False)
```

## Phase 2: Workflow Base Classes

### 2.1 Create `workflows/base.py` - Processing Patterns

```python
"""Base workflow with flexible processing modes."""
from pathlib import Path
from typing import List, Optional, Callable, Any, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for processing modes."""
    mode: str = 'single'  # 'single', 'batch', 'folder'
    parallel: bool = True
    max_workers: Optional[int] = None
    show_progress: bool = True
    log_level: str = 'INFO'

class BaseWorkflow:
    """Base class for all analysis workflows."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging instead of prints."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def process_single(self, input_path: Path) -> Dict[str, Any]:
        """Process a single file - IMPLEMENT IN SUBCLASS."""
        raise NotImplementedError
    
    def process_batch(self, input_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple files with optional parallelization."""
        if self.config.parallel and len(input_paths) > 1:
            return self._process_parallel(input_paths)
        else:
            return self._process_sequential(input_paths)
    
    def process_folder(
        self,
        folder: Path,
        pattern: str = "**/*.xml"
    ) -> List[Dict[str, Any]]:
        """Process all matching files in folder."""
        files = list(folder.glob(pattern))
        logger.info(f"Found {len(files)} files matching '{pattern}'")
        return self.process_batch(files)
    
    def _process_parallel(self, paths: List[Path]) -> List[Dict[str, Any]]:
        """Parallel processing with progress bar."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(self.process_single, p): p for p in paths}
            
            iterator = as_completed(futures)
            if self.config.show_progress:
                iterator = tqdm(iterator, total=len(paths), desc="Processing")
            
            for future in iterator:
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
        
        return results
    
    def _process_sequential(self, paths: List[Path]) -> List[Dict[str, Any]]:
        """Sequential processing with progress bar."""
        results = []
        
        iterator = paths
        if self.config.show_progress:
            iterator = tqdm(paths, desc="Processing")
        
        for path in iterator:
            try:
                result = self.process_single(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
        
        return results
    
    def run(
        self,
        input: Path,
        pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Auto-detect mode and run appropriate processing."""
        if input.is_file():
            logger.info(f"Single file mode: {input}")
            return [self.process_single(input)]
        
        elif input.is_dir():
            logger.info(f"Folder mode: {input}")
            pattern = pattern or self._default_pattern()
            return self.process_folder(input, pattern)
        
        else:
            raise ValueError(f"Input path does not exist: {input}")
    
    def _default_pattern(self) -> str:
        """Default glob pattern - override in subclass."""
        return "**/*.xml"
```

### 2.2 Create `workflows/msd_workflow.py` - MSD Implementation

```python
"""MSD analysis workflow."""
from pathlib import Path
from typing import Dict, Any
import logging
from .base import BaseWorkflow, ProcessingConfig
from ..core.io import TrackLoader
from ..core.analysis import compute_msd, compute_diffusion_from_msd
from ..core.visualization import ResultsAggregator

logger = logging.getLogger(__name__)

class MSDWorkflow(BaseWorkflow):
    """MSD analysis from TrackMate XML files."""
    
    def __init__(
        self,
        config: ProcessingConfig,
        mpp: float = 0.15,
        fps: float = 20.0,
        min_track_length: int = 30,
        fit_points: int = 6
    ):
        super().__init__(config)
        self.mpp = mpp
        self.fps = fps
        self.min_track_length = min_track_length
        self.fit_points = fit_points
        self.aggregator = ResultsAggregator()
    
    def process_single(self, xml_path: Path) -> Dict[str, Any]:
        """Process single TrackMate XML file."""
        logger.info(f"Processing {xml_path.name}")
        
        # Load tracks
        tracks = TrackLoader.from_trackmate_xml(
            xml_path, mpp=self.mpp, fps=self.fps
        )
        
        # Filter short tracks
        tracks = TrackLoader.filter_tracks(tracks, self.min_track_length)
        
        if len(tracks) == 0:
            logger.warning(f"No valid tracks in {xml_path.name}")
            return {'file': xml_path.name, 'status': 'no_tracks'}
        
        # Compute MSD
        msd = compute_msd(tracks)
        
        # Extract diffusion coefficient
        D, D_err, n, n_err = compute_diffusion_from_msd(msd, self.fit_points)
        
        result = {
            'file': xml_path.name,
            'n_particles': tracks['particle'].nunique(),
            'n_detections': len(tracks),
            'D_measured': D,
            'D_error': D_err,
            'exponent': n,
            'exponent_error': n_err,
            'status': 'success'
        }
        
        # Add to aggregator
        self.aggregator.add(**result)
        
        logger.info(
            f"  → {result['n_particles']} particles, "
            f"D = {D:.4f} ± {D_err:.4f} µm²/s"
        )
        
        return result
```

## Phase 3: Unified CLI

### 3.1 Create `cli/process.py` - Command Line Interface

```python
"""Unified command-line interface for all workflows."""
import argparse
from pathlib import Path
import sys
from ..workflows.base import ProcessingConfig
from ..workflows.msd_workflow import MSDWorkflow
from ..workflows.stepsize_workflow import StepsizeWorkflow

def create_parser():
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description='Hydrogel Data Analysis Pipeline'
    )
    
    # Global options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # MSD analysis
    msd_parser = subparsers.add_parser('msd', help='MSD analysis from TrackMate XML')
    msd_parser.add_argument('input', type=Path, help='Input file or folder')
    msd_parser.add_argument('--output', type=Path, required=True,
                           help='Output directory for results')
    msd_parser.add_argument('--mpp', type=float, default=0.15,
                           help='Micrometers per pixel')
    msd_parser.add_argument('--fps', type=float, default=20.0,
                           help='Frames per second')
    msd_parser.add_argument('--min-length', type=int, default=30,
                           help='Minimum track length')
    msd_parser.add_argument('--fit-points', type=int, default=6,
                           help='Points for power-law fitting')
    
    # Step size analysis
    step_parser = subparsers.add_parser('stepsize', help='Step size diffusion analysis')
    step_parser.add_argument('input', type=Path, help='Input file or folder')
    step_parser.add_argument('--output', type=Path, required=True)
    step_parser.add_argument('--mpp', type=float, default=0.15)
    step_parser.add_argument('--fps', type=float, default=20.0)
    
    return parser

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Create config
    config = ProcessingConfig(
        parallel=args.parallel,
        max_workers=args.workers,
        log_level=args.log_level
    )
    
    # Run appropriate workflow
    if args.command == 'msd':
        workflow = MSDWorkflow(
            config=config,
            mpp=args.mpp,
            fps=args.fps,
            min_track_length=args.min_length,
            fit_points=args.fit_points
        )
        results = workflow.run(args.input)
        
        # Save results
        args.output.mkdir(parents=True, exist_ok=True)
        workflow.aggregator.save_summary(args.output / 'summary.csv')
        
        # Create and save figure
        fig = workflow.aggregator.plot_size_vs_diffusion(title='MSD Analysis Results')
        fig.savefig(args.output / 'comparison.png', dpi=300)
        
        print(f"\nResults saved to {args.output}")
        print(f"Processed {len(results)} files successfully")
    
    elif args.command == 'stepsize':
        # Similar pattern...
        pass

if __name__ == '__main__':
    main()
```

## Phase 4: Migration Strategy

### Step-by-Step Migration

1. **Week 1: Core Modules**
   - Create `core/` directory structure
   - Extract and consolidate common functions
   - Write unit tests for core functions
   - Update imports in 1-2 existing scripts to use core

2. **Week 2: Workflow Framework**
   - Create `workflows/base.py`
   - Implement `MSDWorkflow`
   - Test parallel processing
   - Migrate `MSD_FromTrackmate_D0.py` logic

3. **Week 3: Additional Workflows**
   - Implement `StepsizeWorkflow`
   - Implement `SEMWorkflow`
   - Migrate remaining analysis scripts

4. **Week 4: CLI and Polish**
   - Create unified CLI
   - Add comprehensive logging
   - Create migration guide
   - Move old scripts to `legacy/`

## Usage Examples

### Before (Current Approach)
```bash
# Must edit script to change paths
python hydro_analysis/MSD_FromTrackmate_D0.py
```

### After (Unified CLI)
```bash
# Single file
python -m hydro_analysis.cli.process msd single_file.xml --output results/

# Folder with parallel processing
python -m hydro_analysis.cli.process msd /path/to/data/ \
    --output results/ \
    --parallel \
    --workers 8 \
    --mpp 0.15 \
    --fps 20

# Step size analysis
python -m hydro_analysis.cli.process stepsize /path/to/data/ \
    --output results/stepsize/ \
    --parallel
```

### Python API (After)
```python
from hydro_analysis.workflows import MSDWorkflow, ProcessingConfig
from hydro_analysis.core.visualization import ResultsAggregator
from pathlib import Path

# Configure
config = ProcessingConfig(parallel=True, max_workers=4)
workflow = MSDWorkflow(config, mpp=0.15, fps=20.0)

# Process
results = workflow.run(Path("data/"))

# Get aggregated results
df = workflow.aggregator.get_dataframe()
print(df.groupby('particle_size_nm')['D_measured'].mean())

# Save figure
fig = workflow.aggregator.plot_size_vs_diffusion()
fig.savefig('results/final_comparison.png', dpi=300)
```

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Code reuse** | 7 copies of `fit_powerlaw` | 1 centralized version |
| **Processing modes** | Hardcoded per script | Automatic single/batch/folder |
| **Parallelization** | None | Built-in with progress bars |
| **Logging** | 100+ print statements | Structured logging with levels |
| **Result aggregation** | Save each figure immediately | Collect, then save once |
| **CLI** | Edit Python files | Clean command-line interface |
| **Testability** | Hard to test | Core functions easily testable |

## Next Steps

1. Review this plan and provide feedback
2. I'll create the core modules first
3. We'll test with one existing script
4. Gradually migrate remaining scripts
5. Update documentation

Would you like me to proceed with implementing Phase 1 (Core Modules)?
