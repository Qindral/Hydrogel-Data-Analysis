# Core I/O Module - Repair and Extension Summary

## Date: 2026-01-18

## What Was Repaired

### 1. Indentation Errors Fixed

**Problem**: The `find_dataset_files()` function had multiple indentation errors:
- Line 216: `base_name = _extract_base_name(base_tif.stem)` - incorrect indentation
- Lines 223, 235, 243: Multiple nested blocks with wrong indentation
- Line 261: `datasets.extend(_search_folder(subfolder, search_subfolders=False))` - incorrect indentation

**Solution**: Fixed all indentation to properly nest:
- `for base_tif in base_tifs:` loop properly indented
- `for rec_name in ...` loop properly indented inside parent loop
- `if processed_folder.exists():` block properly indented
- All nested structures now correctly aligned

### 2. Logic Error in Legacy Function

**Problem**: `find_xml_files()` tried to access `ds.xml_path` (singular) when datasets have `ds.xml_paths` (plural list)

**Solution**: Changed to collect all XMLs from all datasets:
```python
all_xmls = []
for ds in datasets:
    all_xmls.extend(ds.xml_paths)
return sorted(set(all_xmls))
```

### 3. Empty Folder Handling

**Problem**: Code checked for special subfolders even when no subfolders existed

**Solution**: Added early return for folders without subfolders:
```python
if not subfolders:
    return _search_folder(root_path, search_subfolders=False)
```

## What Was Extended

### New Unified Data Loading Functions

Added comprehensive functions to unify all data loading across the codebase:

#### 1. `load_all_tracks_from_dataset()`
**Purpose**: Load and optionally combine all XMLs from a single dataset

**Features**:
- Loads all XMLs associated with a dataset
- Automatically extracts metadata from REC files
- Combines multiple XMLs with proper particle ID renumbering
- Handles errors gracefully (skips failed XMLs)
- Preserves source information in DataFrame attributes

**Example**:
```python
tracks = load_all_tracks_from_dataset(
    dataset=dataset,
    mpp=None,        # Auto-extract from REC
    fps=None,        # Auto-extract from REC
    min_length=30,   # Filter short tracks
    combine_xmls=True  # Combine into single DataFrame
)
```

#### 2. `combine_track_dataframes()`
**Purpose**: Combine multiple track DataFrames with renumbered particle IDs

**Features**:
- Renumbers particle IDs consecutively across DataFrames
- Preserves DataFrame attributes from first DataFrame
- Handles empty DataFrames gracefully
- Maintains sorted order (particle, frame)

**Example**:
```python
combined = combine_track_dataframes(
    track_list=[tracks1, tracks2, tracks3],
    preserve_attrs=True
)
```

#### 3. `load_all_datasets_from_folder()`
**Purpose**: Load all datasets from a folder structure

**Features**:
- Recursively finds all datasets in folder structure
- Loads each dataset independently
- Returns dictionary mapping dataset_name → tracks
- Structured logging for progress tracking
- Error resilient (continues if individual datasets fail)

**Example**:
```python
datasets_dict = load_all_datasets_from_folder(
    root_path=Path("Data/experiment"),
    mpp=None,
    fps=None,
    min_length=30,
    combine_per_dataset=True
)
```

#### 4. `load_and_combine_all_datasets()`
**Purpose**: Load all datasets and combine into single DataFrame

**Features**:
- Loads all datasets from folder
- Combines into single DataFrame with renumbered particles
- Optionally adds 'dataset' column to track source
- Preserves metadata from first dataset
- Provides summary statistics in logs

**Example**:
```python
all_tracks = load_and_combine_all_datasets(
    root_path=Path("Data/experiment"),
    min_length=30,
    add_dataset_column=True  # Add source tracking
)
```

#### 5. `group_datasets_by_particle_size()`
**Purpose**: Automatically group datasets by particle size

**Features**:
- Extracts particle size from folder names (e.g., "50nm", "100_nm")
- Groups datasets by particle size
- Combines tracks within each size group
- Returns dict: particle_size_nm → combined tracks
- Useful for size-dependent analysis

**Example**:
```python
size_groups = group_datasets_by_particle_size(
    root_path=Path("Data/experiment"),
    min_length=30
)

for size_nm, tracks in sorted(size_groups.items()):
    print(f"{size_nm} nm: {tracks['particle'].nunique()} particles")
```

### Enhanced Logging

Added structured logging throughout:
- **INFO level**: Progress updates, file counts, particle counts
- **WARNING level**: Missing files, no tracks found
- **ERROR level**: XML parsing failures, I/O errors

All logging uses Python's `logging` module instead of print statements:
```python
logger.info(f"Found {len(datasets)} datasets in {root_path}")
logger.warning(f"No XML files found for dataset {dataset.base_name}")
logger.error(f"Failed to load {xml_path.name}: {e}")
```

### Updated Module Exports

Extended `core/__init__.py` to export all new functions:
```python
__all__ = [
    # Core classes
    'TrackLoader',
    'DatasetFiles',
    'ResultsAggregator',
    
    # Finding and organizing
    'find_dataset_files',
    'find_xml_files',
    'extract_particle_size_from_path',
    
    # Unified loading functions (NEW)
    'load_all_tracks_from_dataset',
    'load_all_datasets_from_folder',
    'load_and_combine_all_datasets',
    'group_datasets_by_particle_size',
    'combine_track_dataframes',
    
    # Analysis
    'compute_step_size_diffusion',
    'compute_theoretical_diffusion',
]
```

## Documentation Created

### 1. Test Script (`test_unified_loading.py`)
Comprehensive test script demonstrating:
- Finding datasets
- Loading single dataset
- Loading all datasets from folder
- Combining all data
- Grouping by particle size

### 2. User Guide (`UNIFIED_LOADING_GUIDE.md`)
Complete documentation including:
- Quick start examples
- File matching system explanation
- Folder structure handling
- DataFrame schema
- Advanced usage patterns
- Integration with existing code
- Error handling
- Performance considerations

## Benefits of Extensions

### 1. Eliminates Code Duplication
**Before**: Each script had its own XML loading logic (duplicated 4-7 times)
**After**: Single source of truth in `core/io.py`

### 2. Flexible Loading Patterns
- Load single file
- Load single dataset (multiple XMLs)
- Load all datasets in folder
- Combine everything
- Group by particle size

### 3. Automatic Metadata Extraction
No need to manually specify mpp/fps:
```python
# Before
tracks = load_tracks(xml, mpp=0.15, fps=20.0)

# After
tracks = TrackLoader.from_trackmate_xml(xml)  # Auto-extracted!
```

### 4. Robust File Matching
Handles various naming conventions:
- `50 nm_2.tif` / `50 nm_2_Tracks.xml`
- `processed_50 nm_2.tif`
- `Resultof50 nm_2_Tracks.xml`

### 5. Error Resilience
Continues processing even if individual files fail:
```python
# Loads successfully even if some XMLs are corrupted
datasets = load_all_datasets_from_folder(root_path)
```

### 6. Ready for Parallel Processing
Functions designed for easy parallelization:
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_dataset, datasets))
```

## Usage in Existing Scripts

### Migration Pattern

**Old Code** (e.g., in `MSD_FromTrackmate_D0.py`):
```python
# Manual XML parsing (60+ lines)
tree = ET.parse(xml_path)
root = tree.getroot()
data = []
for pid, particle in enumerate(root.findall('particle')):
    for detection in particle.findall('detection'):
        data.append({...})
df = pd.DataFrame(data)
df.attrs['mpp'] = 0.15
# ... more manual setup ...
```

**New Code** (using unified loader):
```python
from hydro_analysis.core import load_all_datasets_from_folder

# One line!
datasets = load_all_datasets_from_folder(
    root_path=ROOT_PATH,
    min_length=MIN_TRACK_LENGTH
)
```

### Example Integration

For particle size analysis:
```python
from hydro_analysis.core import group_datasets_by_particle_size
from hydro_analysis.core.analysis import compute_step_size_diffusion

# Group by size
size_groups = group_datasets_by_particle_size(
    root_path=Path("Data/SPT_2025"),
    min_length=30
)

# Analyze each size
results = []
for size_nm, tracks in sorted(size_groups.items()):
    D, D_err = compute_step_size_diffusion(tracks)
    results.append({
        'size': size_nm,
        'D': D,
        'D_err': D_err,
        'n_particles': tracks['particle'].nunique()
    })
```

## Testing Status

✅ **Code compiles**: No syntax errors
✅ **Test script created**: Comprehensive test suite
✅ **Error handling**: Graceful degradation tested
✅ **Documentation**: Complete user guide written
✅ **Module exports**: All functions accessible

## Next Steps

### Immediate
1. ✅ Fix indentation errors (DONE)
2. ✅ Add unified loading functions (DONE)
3. ✅ Create test script (DONE)
4. ✅ Write documentation (DONE)

### Future Integration
1. Update `MSD_FromTrackmate_D0.py` to use unified loading
2. Update `Schrittweiten_methode_D0.py` to use unified loading
3. Create workflow base classes using these functions
4. Add parallel processing support
5. Implement result aggregation system

## Files Modified/Created

### Modified
- ✅ `hydro_analysis/core/io.py` - Fixed + extended (446 → 685 lines)
- ✅ `hydro_analysis/core/__init__.py` - Added exports

### Created
- ✅ `hydro_analysis/test_unified_loading.py` - Test suite
- ✅ `UNIFIED_LOADING_GUIDE.md` - User documentation
- ✅ `CORE_IO_REPAIR_SUMMARY.md` - This file

## Summary

The `core/io.py` module has been **repaired** (indentation + logic errors) and **significantly extended** with unified data loading functions. The new functions eliminate code duplication, provide flexible loading patterns, and are ready for integration into existing analysis scripts. Complete documentation and tests are provided.

**Status**: ✅ Ready for use in refactored workflows
