# Unified Data Loading - Documentation

## Overview

The `hydro_analysis.core.io` module provides unified functions to load and combine track data from multiple sources. This eliminates code duplication and provides flexible data loading patterns.

## Key Features

✅ **Automatic file matching**: Matches TIF, REC, and XML files by intelligent name parsing  
✅ **Metadata extraction**: Extracts FPS, MPP, and dimensions from REC files and XML  
✅ **Multiple XMLs per dataset**: Automatically combines tracks from multiple TrackMate XML files  
✅ **Folder-based loading**: Load all datasets from a folder structure recursively  
✅ **Particle size grouping**: Automatically group datasets by particle size extracted from folder names  
✅ **Flexible combining**: Combine tracks with proper particle ID renumbering  

## Quick Start Examples

### 1. Load a Single Dataset with Multiple XMLs

```python
from hydro_analysis.core import find_dataset_files, load_all_tracks_from_dataset

# Find datasets in a folder
datasets = find_dataset_files(Path("Data/my_experiment"))

# Load first dataset (combines all its XMLs)
tracks = load_all_tracks_from_dataset(
    dataset=datasets[0],
    min_length=30,      # Minimum track length
    combine_xmls=True   # Combine multiple XMLs into one DataFrame
)

print(f"Loaded {tracks['particle'].nunique()} particles")
print(f"MPP: {tracks.attrs['mpp']} µm/px")
print(f"FPS: {tracks.attrs['fps']} fps")
```

### 2. Load All Datasets from a Folder

```python
from hydro_analysis.core import load_all_datasets_from_folder

# Load all datasets, returns dict: dataset_name → tracks
datasets_dict = load_all_datasets_from_folder(
    root_path=Path("Data/my_experiment"),
    min_length=30,
    combine_per_dataset=True  # Combine XMLs within each dataset
)

# Iterate over loaded datasets
for name, tracks in datasets_dict.items():
    print(f"{name}: {tracks['particle'].nunique()} particles")
```

### 3. Combine All Datasets into One DataFrame

```python
from hydro_analysis.core import load_and_combine_all_datasets

# Load and combine everything
all_tracks = load_and_combine_all_datasets(
    root_path=Path("Data/my_experiment"),
    min_length=30,
    add_dataset_column=True  # Add 'dataset' column to track source
)

print(f"Total: {all_tracks['particle'].nunique()} particles")
print(all_tracks.groupby('dataset')['particle'].nunique())
```

### 4. Group by Particle Size

```python
from hydro_analysis.core import group_datasets_by_particle_size

# Automatically group by particle size from folder names
# Works with: "50nm", "100_nm", "200 nm" folder names
size_groups = group_datasets_by_particle_size(
    root_path=Path("Data/my_experiment"),
    min_length=30
)

# Dictionary: particle_size_nm → combined tracks
for size, tracks in sorted(size_groups.items()):
    print(f"{size} nm: {tracks['particle'].nunique()} particles")
```

## File Matching System

### Base Name Extraction

The system intelligently extracts base names from filenames:

- **Removes prefixes**: `Resultof`, `processed_`, `preprocessed_`
- **Removes suffixes**: `_Tracks`, `_processed`, `_preprocessed`
- **Preserves**: spaces, underscores, numbers like `_1`, `_01`

Examples:
```
"50 nm_2.tif"                    → "50 nm_2"
"processed_50 nm_2.tif"          → "50 nm_2"
"Resultof50 nm_2_Tracks.xml"     → "50 nm_2"
```

### File Matching Rules

For each dataset, the system finds:

1. **Base TIF**: Original TIF file (e.g., `50 nm_2.tif`)
2. **REC file**: `basename.tif.rec` or `basename.rec`
3. **Processed TIFs**: In `processed/` subfolder with matching base name
4. **XML files**: In same folder or `Tracks/` subfolder with matching base name

### Folder Structures

The system handles multiple folder structures:

#### Structure A: Flat folder
```
experiment/
  ├── 50nm_1.tif
  ├── 50nm_1.tif.rec
  ├── 50nm_1_Tracks.xml
  ├── 50nm_2.tif
  └── ...
```

#### Structure B: Organized with subfolders
```
experiment/
  ├── 50nm_1.tif
  ├── 50nm_1.tif.rec
  ├── processed/
  │   └── processed_50nm_1.tif
  └── Tracks/
      └── 50nm_1_Tracks.xml
```

#### Structure C: Particle size folders
```
experiment/
  ├── 50nm/
  │   ├── measurement_1.tif
  │   ├── measurement_1.rec
  │   └── Tracks/
  │       └── measurement_1_Tracks.xml
  └── 100nm/
      └── ...
```

## Metadata Extraction

### Priority Order

Metadata is extracted with this priority:

1. **Manual override** (mpp/fps parameters)
2. **REC file** (exposure time → FPS, dimensions → MPP)
3. **XML file** (frameInterval → FPS)
4. **Dimension detection** (image size → MPP)

### MPP (Micrometers per Pixel)

Known mappings:
```python
200 × 150 px  → 0.30 µm/px  (60 FPS mode)
400 × 300 px  → 0.15 µm/px  (20 FPS mode)
696 × 520 px  → 0.149 µm/px (full sensor)
```

### FPS (Frames per Second)

Extracted from:
- REC file: `FPS = 1000 / (exposure_ms + delay_ms)`
- XML file: `FPS = 1 / frameInterval`

## DataFrame Schema

All track DataFrames have this structure:

### Columns
```python
['particle', 'frame', 'x', 'y']
```

Optional:
```python
['dataset']  # If add_dataset_column=True
```

### Attributes (`.attrs`)
```python
{
    'mpp': float,           # Micrometers per pixel
    'fps': float,           # Frames per second
    'mode': str,            # Detection mode (e.g., "60 FPS", "20 FPS")
    'xml_source': str,      # Source XML filename
    'dataset_name': str     # Dataset base name
}
```

## Advanced Usage

### Combine Multiple Track DataFrames

```python
from hydro_analysis.core import combine_track_dataframes

# Manually combine track DataFrames with renumbering
track_list = [tracks1, tracks2, tracks3]
combined = combine_track_dataframes(
    track_list,
    preserve_attrs=True  # Keep attrs from first DataFrame
)

# Particle IDs are renumbered consecutively
print(combined['particle'].max())  # Sequential IDs
```

### Filter After Loading

```python
# Load with minimal filtering
tracks = load_all_tracks_from_dataset(
    dataset=dataset,
    min_length=10  # Permissive initial filter
)

# Apply stricter filtering
track_lengths = tracks.groupby('particle').size()
valid_particles = track_lengths[track_lengths >= 50].index
tracks_filtered = tracks[tracks['particle'].isin(valid_particles)]
```

### Manual Metadata Override

```python
# Override automatic metadata detection
tracks = load_all_tracks_from_dataset(
    dataset=dataset,
    mpp=0.149,      # Force this MPP
    fps=60.0,       # Force this FPS
    min_length=30
)
```

### Process by Particle Size

```python
from hydro_analysis.core import group_datasets_by_particle_size
from hydro_analysis.core.analysis import compute_step_size_diffusion

# Load grouped by size
size_groups = group_datasets_by_particle_size(
    root_path=Path("Data/experiment"),
    min_length=30
)

# Analyze each size
results = []
for size_nm, tracks in sorted(size_groups.items()):
    D, D_err = compute_step_size_diffusion(tracks)
    
    results.append({
        'particle_size_nm': size_nm,
        'n_particles': tracks['particle'].nunique(),
        'D_measured': D,
        'D_error': D_err
    })

import pandas as pd
df = pd.DataFrame(results)
print(df)
```

## Integration with Existing Code

### Replace Old Loading Code

**Before:**
```python
# Old duplicated code
tree = ET.parse(xml_path)
root = tree.getroot()
data = []
for pid, p in enumerate(root.findall('particle')):
    for det in p.findall('detection'):
        data.append({...})
df = pd.DataFrame(data)
df.attrs['mpp'] = 0.15
df.attrs['fps'] = 20.0
```

**After:**
```python
from hydro_analysis.core import TrackLoader

tracks = TrackLoader.from_trackmate_xml(
    xml_path=xml_path,
    min_length=30
)
# mpp, fps automatically extracted!
```

### Batch Processing Pattern

**Before:**
```python
# Manual loop with prints
for xml in xml_files:
    print(f"Processing {xml}")
    tree = ET.parse(xml)
    # ... duplicate parsing code ...
    results.append(...)
```

**After:**
```python
from hydro_analysis.core import load_all_datasets_from_folder
import logging

logging.basicConfig(level=logging.INFO)

# Automatic, parallel-ready
datasets = load_all_datasets_from_folder(
    root_path=Path("Data/experiment"),
    min_length=30
)
# Structured logging, no prints
```

## Error Handling

The unified loader handles errors gracefully:

- **Missing files**: Skips datasets without REC files, logs warning
- **Malformed XML**: Catches parse errors, continues with other files
- **Empty tracks**: Returns empty DataFrame with correct schema
- **Missing metadata**: Falls back to sensible defaults

```python
import logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings/errors

tracks = load_and_combine_all_datasets(
    root_path=Path("Data/experiment"),
    min_length=30
)
# Continues even if some files fail
```

## Performance Considerations

### Memory Efficiency

For large datasets:
```python
# Load one at a time
datasets = find_dataset_files(root_path)
for dataset in datasets:
    tracks = load_all_tracks_from_dataset(dataset, min_length=30)
    # Process tracks
    # ... computation ...
    del tracks  # Free memory
```

### Parallel Processing Ready

The unified loader prepares data for parallel processing:
```python
from concurrent.futures import ProcessPoolExecutor

def process_dataset(dataset):
    tracks = load_all_tracks_from_dataset(dataset, min_length=30)
    # ... your analysis ...
    return result

datasets = find_dataset_files(root_path)

with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_dataset, datasets))
```

## Summary

The unified data loading system provides:

1. ✅ **Single source of truth** - No more duplicate code
2. ✅ **Flexible patterns** - Single file, batch, or full folder
3. ✅ **Automatic metadata** - No manual calibration needed
4. ✅ **Robust matching** - Intelligent file name parsing
5. ✅ **Error handling** - Graceful degradation
6. ✅ **Ready for refactoring** - Clean API for workflows

Use these functions as building blocks for analysis workflows!
