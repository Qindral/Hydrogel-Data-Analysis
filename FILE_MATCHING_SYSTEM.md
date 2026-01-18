# File Matching System

## Overview

The refactored code now includes a comprehensive file matching system that handles the three types of data files:
- **`.xml`** - TrackMate tracking data from ImageJ
- **`.tif`** - Image pixel arrays  
- **`.rec`** - Image metadata

## Key Features

### 1. Automatic File Matching

The new `find_dataset_files()` function automatically finds and matches related files:

```python
from core import find_dataset_files

datasets = find_dataset_files(Path("your/folder"))

for ds in datasets:
    print(f"XML: {ds.xml_path}")
    print(f"TIF: {ds.tif_path}")  # None if not found
    print(f"REC: {ds.rec_path}")  # None if not found
```

### 2. Handles Multiple Folder Structures

**Case 1: All files in same folder**
```
folder/
  sample.xml
  sample.tif
  sample.rec
```

**Case 2: XML in Tracks subfolder**
```
folder/
  sample.tif
  sample.rec
  Tracks/
    sample.xml
```

### 3. Handles `_processed` Suffix

The system automatically matches files with or without the `_processed` suffix:

**Example 1: XML has suffix, TIF doesn't**
```
folder/
  sample.tif
  sample_processed.xml
```
→ **Matches correctly** (base_name = "sample")

**Example 2: TIF has suffix, XML doesn't**
```
folder/
  sample.xml
  sample_processed.tif
```
→ **Matches correctly** (base_name = "sample")

**Example 3: Both have suffix**
```
folder/
  sample_processed.xml
  sample_processed.tif
```
→ **Matches correctly** (base_name = "sample")

## DatasetFiles Structure

```python
@dataclass
class DatasetFiles:
    xml_path: Path              # Always present (required)
    tif_path: Optional[Path]    # None if not found
    rec_path: Optional[Path]    # None if not found
    base_name: str              # Common name without _processed
```

## Usage Examples

### Single File
```python
from core import find_dataset_files

datasets = find_dataset_files(Path("data/sample.xml"))
# Returns 1 dataset with sample.xml and any matching TIF/REC
```

### Folder
```python
datasets = find_dataset_files(Path("data/my_experiment"))
# Finds all XMLs in folder and Tracks/ subfolder
# Matches each XML with corresponding TIF and REC
```

### Check What Was Found
```python
datasets = find_dataset_files(Path("data/"))

print(f"Found {len(datasets)} datasets")
matched_tif = sum(1 for ds in datasets if ds.tif_path)
matched_rec = sum(1 for ds in datasets if ds.rec_path)
print(f"Matched: {matched_tif} TIF files, {matched_rec} REC files")
```

## Testing

Run the test script to verify matching works:

```bash
python hydro_analysis/test_file_matching.py
```

This will scan your data folder and show which files were matched.

## Integration with Analysis

The refactored `Schrittweiten_methode_D0_refactored.py` now:
1. Uses `find_dataset_files()` instead of `find_xml_files()`
2. Reports how many TIF and REC files were matched
3. Stores `tif_found` and `rec_found` flags in results CSV

This prepares the codebase for future functionality that loads and processes TIF/REC files alongside the XML tracking data.

## Code Changes Summary

### New Files
- `core/io.py`: Added `DatasetFiles` dataclass and `find_dataset_files()` function
- `test_file_matching.py`: Test script to verify file matching

### Modified Files
- `core/__init__.py`: Exports `DatasetFiles` and `find_dataset_files`
- `Schrittweiten_methode_D0_refactored.py`: Uses new file matching system

### Logging Reduction
- Changed log level from `INFO` to `WARNING` (only shows errors)
- Replaced `logger.info()` calls with `print()` for user-facing messages
- Reduced output from ~30 log messages to ~5 essential messages
- Result: Clean, readable output focused on important information
