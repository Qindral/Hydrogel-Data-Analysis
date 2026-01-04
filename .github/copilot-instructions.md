# AI Coding Agent Instructions for Hydrogel-Data-Analysis

## Project Overview
**Hydrogel-Data-Analysis** is a python image analysis plugin for hydrogel particle dynamics research. It combines TIFF image loading with particle tracking (trackpy) and mean-squared displacement (MSD) analysis to characterize particle motion in hydrogel systems.

### Core Domains
- **Image I/O**: Load TIFF files with OME, Leica, and PCO metadata extraction
- **Metadata Management**: Standardize instrument metadata into unified `DatasetMetadata` objects
- **Particle Tracking**: Use trackpy for spot detection and trajectory linking
- **Dynamics Analysis**: Compute MSD and fit power-law exponents to measure diffusion
- **UI/Visualization**: napari plugin with Qt-based info and display panels

## Critical Architecture Patterns

### 1. Data Pipeline: LoadedDataset → MSD Analysis
```
TIFF → DatasetLoader → LoadedDataset (data + DatasetMetadata)
                    ↓
              napari Viewer (Raw/Filtered layers)
                    ↓
          trackpy_msd.main() or pytrackmate_MSD_XML()
                    ↓
              MSD DataFrame & plots
```

**Key Files**:
- [data_loader.py](data_loader.py): `DatasetLoader` class handles TIFF parsing
- [metadata.py](metadata.py): `DatasetMetadata` dataclass + utilities
- [trackpy_msd.py](trackpy_msd.py): Unified MSD computation pipeline
- [napari_plugin.py](napari_plugin.py): Entry points for napari integration

### 2. Metadata Extraction Strategy
The loader applies **cascading metadata sources**:
1. Basic TIFF tags (Artist, Software, DateTime)
2. OME XML (pixel size, timestamps, channel names) — supports multiple OME versions
3. Leica-specific tags (stage position, Z-step)
4. PCO image metadata
5. Sidecar XML files from Data directory (pytrackmate outputs)

Each source fills missing fields; later sources don't overwrite. Example: `_populate_metadata_from_ome()` → `_populate_metadata_from_leica()` → `_populate_metadata_from_standard_tags()` → `_populate_metadata_from_sidecars()`.

**Convention**: Always preserve raw XML as `DatasetMetadata.raw_metadata` for debugging.

### 3. Dataset Kind Classification
`DatasetMetadata.infer_kind()` heuristically labels data as "FRAP", "SPT", or "FULL" based on:
- Keywords in notes + software field (case-insensitive)
- Axis counts: FRAP/bleach → "FRAP"; sparse C + T>100 → "SPT"; Z>1 + T>1 → "FULL"

This hints at analysis strategy without requiring explicit user input.

### 4. Trackpy Integration & MSD Computation
**Parameters** (from `trackpy_msd.main()`):
- `diameter`, `distance`, `minmass`: Spot detection (trackpy.locate)
- `mpp` (µm/px), `fps`: Calibration (from metadata)
- `smooth`, `radius`: Optional background subtraction (Gaussian or rolling-ball)

**Output**:
- Trajectories as pandas DataFrame with columns: `particle`, `frame`, `x`, `y`
- MSD computed via trackpy's built-in; power-law fitting via `fit_powerlaw_with_errors()`

**Files**:
- [MSD_Trackpy_clean.py](MSD_Trackpy_clean.py): Cleaner version; prefer this
- [pytrackmate_MSD_XML.py](pytrackmate_MSD_XML.py): Loads from external TrackMate XML
- [MSD_Trackpy.py](MSD_Trackpy.py): Older version; legacy

### 5. Qt/Napari Integration
- **DisplayPanel** ([_qt.py](\_qt.py)): Colormap, sigma filter, scalebar toggles
- **InfoPanel** ([_qt.py](\_qt.py)): Read-only metadata display (calibration, kind, axes)
- **Docking**: `open_dataset_dialog()` creates "Raw" and "Filtered" image layers; attaches panels to viewer

**Scale Setting**: `_update_scale()` applies physical units (µm/px, µm) to napari layers based on metadata.

## Developer Workflows

### Running Tests
No formal test suite yet. Use:
```bash
python -m pytest hydro_analysis/ -v
```

### Loading a Dataset Interactively
```python
from hydro_analysis.data_loader import DatasetLoader
from pathlib import Path

path = Path("Data/2025_09_01_16_03_50--FRAP/FRAP 006/FRAP Pb1 Series24/*.tif")
loader = DatasetLoader(path)
loaded = loader.load()
print(loaded.metadata.infer_kind())
print(loaded.metadata.as_dict())
```

### Running MSD Analysis
```python
from hydro_analysis.trackpy_msd import main

main(
    tif_path="Data/.../image.tif",
    diameter=9, distance=10, minmass=420,
    mpp=0.065, fps=1.0,  # Get from metadata
    plot=True, smooth=True, radius=50
)
```

### Napari Plugin Launch
```bash
napari
# Then use Plugins → Hydro Analysis → Open Dataset…
```

## Code Conventions

### 1. Dataclasses & Type Hints
All metadata is stored in `@dataclass` objects (e.g., `DatasetMetadata`, `TimeSummary`). Always use type hints; default to `Optional[Type]` for nullable fields.

**Example**:
```python
@dataclass
class DatasetMetadata:
    path: Path
    px_size_xy_um: Optional[float] = None
    timestamps: List[float] = field(default_factory=list)
```

### 2. Axes Convention
- Axes string follows ImageIO/OME: "CZYX", "TCZYX", etc.
- Map axes to dimensions via `dict(zip(metadata.axes, metadata.shape))`.
- X, Y are spatial; Z is depth; T is time; C is channel.

**Example from metadata.py**:
```python
@property
def width_height(self) -> Tuple[int, int]:
    axes_to_dim = dict(zip(self.axes, self.shape))
    return axes_to_dim.get("X", 0), axes_to_dim.get("Y", 0)
```

### 3. Error Handling
- XML parsing: Wrap in try-except; return silently if malformed (don't fail the whole load).
- File I/O: Raise `FileNotFoundError` early; use `Path.exists()` checks.
- Metadata defaults: Always provide sensible fallbacks (e.g., `dt = 1.0` if frameInterval absent).

### 4. Calibration from Metadata
- Pixel size from OME `PhysicalSizeX/Y` (µm/px); fallback to Leica tags.
- Z-step from OME `PhysicalSizeZ` or Leica ScanInfo.
- Time from OME `Plane.DeltaT` or root `frameInterval` (seconds).
- Stage position (mm) from Leica acquisition metadata.

### 5. Trackpy DataFrame Schema
After `load_tracks_xml()` or spot detection:
```
columns: ['particle', 'frame', 'x', 'y']
attrs:
  'mpp': float (µm/px)
  'fps': float (frames/second)
```

Trackpy's MSD functions rely on these attributes; always set them.

## Integration Points

### External Data: PyTrackMate XML
[pytrackmate_MSD_XML.py](pytrackmate_MSD_XML.py) parses TrackMate outputs with frame interval and spatial unit metadata. Converts to trackpy-compatible DataFrame.

### PCO Image Files
[frequency_check_Pco.py](frequency_check_Pco.py) reads PCO-specific TIFF tags. Integrated into `_populate_metadata_from_standard_tags()`.

### Napari Layers & Metadata Storage
Metadata attached to napari layers as `raw_layer.metadata["metadata"] = DatasetMetadata(...)`. Allows UI panels to query calibration without disk I/O.

## Common Pitfalls

1. **OME Namespace**: Schemas vary (2015-01, 2016-06). Always check both; don't assume fixed URI.
2. **Background Subtraction**: Rolling-ball radius must match (2×radius) not radius alone for cv2 kernels.
3. **Frame Indexing**: XML frame indices are 0-based; ensure consistency with pandas DataFrames.
4. **Missing Metadata**: Gracefully handle absent calibration; infer_kind() uses heuristics, not guarantees.
5. **Qt Thread Safety**: Use `@thread_worker` for long-running trackpy operations to avoid blocking napari UI.

## File Organization
```
hydro_analysis/
  __init__.py              # Module exports (DatasetLoader, DatasetMetadata)
  data_loader.py           # TIFF I/O & metadata extraction
  metadata.py              # DatasetMetadata + utilities
  napari_plugin.py         # napari entry points & dialogs
  _qt.py                   # InfoPanel, DisplayPanel widgets
  trackpy_msd.py           # Main MSD analysis pipeline
  MSD_Trackpy_clean.py     # Cleaner trackpy wrapper (prefer)
  pytrackmate_MSD_XML.py   # TrackMate XML parsing
  plots.py                 # Visualization helpers
  Particle_Parameter_Tuner.py  # Interactive parameter UI
  frequency_check_Pco.py   # PCO-specific metadata
  Stepsize.py              # Step/displacement analysis
```

Data stored in `Data/` with instrument-dated subdirectories (e.g., `2025_09_01_16_03_50--FRAP/`).

