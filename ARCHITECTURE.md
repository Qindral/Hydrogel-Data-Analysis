# Architecture Overview

## Current Architecture (Before Refactoring)

```
┌─────────────────────────────────────────────────────────────┐
│  Each Script is a Complete Standalone Program               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  MSD_FromTrackmate_D0.py (1474 lines)                      │
│  ├── Duplicate: fit_powerlaw_with_errors()                 │
│  ├── Duplicate: load_tracks_xml()                          │
│  ├── Hardcoded paths                                        │
│  ├── 50+ print statements                                   │
│  └── Sequential processing only                             │
│                                                              │
│  MSD_FromTrackmate_20mg.py (1388 lines)                    │
│  ├── Duplicate: fit_powerlaw_with_errors()                 │
│  ├── Duplicate: load_tracks_xml()                          │
│  ├── Hardcoded paths                                        │
│  ├── 50+ print statements                                   │
│  └── Sequential processing only                             │
│                                                              │
│  Schrittweiten_methode_D0.py (1418 lines)                  │
│  ├── Similar duplicate functions                            │
│  ├── Hardcoded paths                                        │
│  ├── 60+ print statements                                   │
│  └── Sequential processing only                             │
│                                                              │
│  pytrackmate_MSD_XML.py (533 lines)                        │
│  ├── Duplicate: fit_powerlaw_with_errors()                 │
│  ├── Hardcoded file list                                    │
│  └── Sequential processing                                  │
│                                                              │
│  trackpy_msd.py (685 lines)                                │
│  ├── Duplicate: fit_powerlaw_with_errors() (2 times!)      │
│  ├── Duplicate: subtract_background()                       │
│  └── Single file only                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Problems:
❌ ~500 lines of duplicate code across files
❌ Can't easily switch between single/batch/folder modes
❌ No parallelization (waste of multi-core CPU)
❌ Print statements create unusable output with many files
❌ Figures saved immediately (can't aggregate/compare)
❌ Hard to test or maintain
```

## Proposed Architecture (After Refactoring)

```
┌─────────────────────────────────────────────────────────────┐
│  Modular Architecture with Reusable Components              │
└─────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────┐
│                    CORE MODULES                            │
│  (Shared functions used by all workflows)                  │
├────────────────┬────────────────┬──────────────────────────┤
│  core/io.py    │ core/analysis  │ core/visualization.py   │
│                │     .py        │                          │
│ TrackLoader    │ fit_powerlaw   │ ResultsAggregator       │
│ find_xml_files │ compute_msd    │ plot_size_vs_diff()     │
│ load_metadata  │ compute_diff   │ plot_msd_comparison()   │
│                │ step_size_D    │ (return figures)        │
└────────────────┴────────────────┴──────────────────────────┘
                          ↑
                          │ Import from core
                          │
┌─────────────────────────────────────────────────────────────┐
│                  WORKFLOW LAYER                             │
│  (Implements processing patterns)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  workflows/base.py (BaseWorkflow)                          │
│  ├── Auto-detect: single file / batch / folder             │
│  ├── Parallel processing (ProcessPoolExecutor)             │
│  ├── Progress bars (tqdm)                                   │
│  ├── Structured logging                                     │
│  └── Error handling                                         │
│                                                              │
│  workflows/msd_workflow.py (MSDWorkflow)                   │
│  ├── Inherits: BaseWorkflow                                │
│  ├── Implements: process_single()                          │
│  └── Uses: core.io, core.analysis                          │
│                                                              │
│  workflows/stepsize_workflow.py (StepsizeWorkflow)         │
│  ├── Inherits: BaseWorkflow                                │
│  ├── Implements: process_single()                          │
│  └── Uses: core.io, core.analysis                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          ↑
                          │
┌─────────────────────────────────────────────────────────────┐
│                  USER INTERFACE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  cli/process.py (Command Line)                             │
│  ├── Subcommands: msd, stepsize, sem                       │
│  ├── Arguments: --parallel, --workers, --mpp, --fps        │
│  └── Output: CSV summary + aggregated figures              │
│                                                              │
│  Python API                                                 │
│  ├── from workflows import MSDWorkflow                      │
│  ├── workflow = MSDWorkflow(config)                        │
│  └── results = workflow.run(path)                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Benefits:
✅ Single source of truth for all common functions
✅ Automatic single/batch/folder mode detection
✅ Built-in parallelization with progress tracking
✅ Clean logging (no print spam)
✅ Deferred visualization (aggregate before saving)
✅ Easy to test and maintain
✅ Consistent interface across all analyses
```

## Data Flow Example: MSD Analysis

### Before (Current)
```
User edits script paths
   ↓
Script runs (sequential)
   ↓
Print to console (messy)
   ↓
Save each figure immediately
   ↓
Manual CSV creation
```

### After (Refactored)
```
User runs command:
  $ python -m hydro_analysis.cli.process msd data/ --output results/ --parallel
                ↓
         CLI parses arguments
                ↓
      Creates MSDWorkflow instance
                ↓
   Workflow detects folder mode → scans for XML files
                ↓
         ┌──────────────────────────────────┐
         │   Parallel Processing            │
         ├──────────────────────────────────┤
         │ Worker 1: file_001.xml           │
         │   → core.io.TrackLoader()        │
         │   → core.analysis.compute_msd()  │
         │   → Result added to aggregator   │
         │                                   │
         │ Worker 2: file_002.xml           │
         │   → ...                           │
         │                                   │
         │ Worker N: file_103.xml           │
         │   → ...                           │
         └──────────────────────────────────┘
                ↓
         [Progress: 103/103]
                ↓
   ResultsAggregator has all data
                ↓
   Generate comparison figure (once)
                ↓
   Save CSV summary (once)
                ↓
   Done! Clean output:
     - results/summary.csv
     - results/comparison.png
```

## Processing Modes Example

### Single File
```bash
python -m hydro_analysis.cli.process msd single.xml --output results/
```
```
→ Loads single.xml
→ Computes MSD
→ Saves results/single_msd.png
→ Saves results/summary.csv (1 row)
```

### Batch Processing
```bash
python -m hydro_analysis.cli.process msd data/ --output results/ --parallel
```
```
→ Finds all *Tracks.xml in data/
→ Processes in parallel (8 workers)
→ Shows progress bar
→ Aggregates all results
→ Saves results/comparison.png (all data)
→ Saves results/summary.csv (N rows)
```

### Folder Processing with Pattern
```bash
python -m hydro_analysis.cli.process msd data/ \
    --pattern "**/*20mg*Tracks.xml" \
    --output results/20mg/ \
    --parallel
```
```
→ Finds only files matching pattern
→ Processes in parallel
→ Saves to results/20mg/
```

## Function Consolidation Example

### Before: Duplicated in 7 files
```python
# In MSD_FromTrackmate_D0.py
def fit_powerlaw_with_errors(em_series, points=10):
    xs = em_series.iloc[:points].index.values.astype(float)
    ys = em_series.iloc[:points].values.astype(float)
    # ... 30 lines ...
    return SimpleNamespace(A=..., n=...)

# In MSD_FromTrackmate_20mg.py
def fit_powerlaw_with_errors(em_series, points=10):
    xs = em_series.iloc[:points].index.values.astype(float)
    ys = em_series.iloc[:points].values.astype(float)
    # ... 30 lines (identical!) ...
    return SimpleNamespace(A=..., n=...)

# In pytrackmate_MSD_XML.py
# ... same function again ...

# In trackpy_msd.py
# ... and again (twice!) ...
```

### After: Single Implementation
```python
# In core/analysis.py
def fit_powerlaw_with_errors(em_series, points=10):
    """Fit power-law with error estimates.
    
    Used by all MSD analysis workflows.
    """
    xs = em_series.iloc[:points].index.values.astype(float)
    ys = em_series.iloc[:points].values.astype(float)
    # ... 30 lines (once!) ...
    return SimpleNamespace(A=..., n=...)

# In all other files:
from hydro_analysis.core.analysis import fit_powerlaw_with_errors
```

**Result**: 
- ~200 lines of duplicate code → 30 lines
- Fix a bug once, fixes everywhere
- Easy to test in isolation
- Clear documentation in one place

## Migration Path

```
Phase 1: Core Modules
├── Create core/io.py
├── Create core/analysis.py
├── Create core/utils.py
└── Extract all duplicate functions
    Status: ⏳ Ready to implement

Phase 2: Workflow Framework
├── Create workflows/base.py
├── Create workflows/msd_workflow.py
├── Test parallel processing
└── Migrate one existing script
    Status: ⏳ Waiting for Phase 1

Phase 3: Additional Workflows
├── Create workflows/stepsize_workflow.py
├── Create workflows/sem_workflow.py
└── Migrate remaining scripts
    Status: ⏳ Waiting for Phase 2

Phase 4: CLI and Polish
├── Create cli/process.py
├── Add comprehensive logging
├── Move old scripts to legacy/
└── Update documentation
    Status: ⏳ Waiting for Phase 3

Result: Clean, maintainable, fast codebase
└── Status: 🎯 Ready when you are!
```

## Estimated Time Investment

| Task | Time | Value |
|------|------|-------|
| Create core modules | 2-3 hours | High - eliminates 500+ lines duplication |
| Workflow framework | 2-3 hours | High - enables parallel processing |
| Migrate existing scripts | 2-3 hours | Medium - preserves functionality |
| CLI and polish | 1-2 hours | Medium - better UX |
| **Total** | **7-11 hours** | **Huge - faster, cleaner, maintainable** |

**ROI**: Every future analysis will be faster to write and run!
