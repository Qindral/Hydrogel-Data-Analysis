# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hydrogel-Data-Analysis is a Python toolkit for a PhD project analyzing hydrogel/nanoparticle dynamics: single-particle tracking (SPT) diffusion (MSD + step-size methods), FRAP recovery, SEM particle sizing, LiteSizer (DLS) reference measurements, and trajectory/microscopy visualization. It is a personal research codebase, not a library with a stable public API — scripts are run directly by the author to produce figures for a dissertation.

## Environment & Commands

- Python 3.12 venv at `.venv\`. Activate before running anything: `.venv\Scripts\Activate.ps1` (or `.venv/Scripts/activate` in bash).
- The package is installed editable (`hydro_analysis.egg-info` present), so `from hydro_analysis.core...` and `from hydro_analysis.MSD_Trackmate...` imports work from anywhere. Reinstall with `pip install -e .` if imports break.
- `requirements.txt` at repo root is the canonical dependency list. **`numpy` must stay `<2.4`** — newer numpy breaks the pinned `numba`/`llvmlite` versions used for `trackpy`.
- No build step, no linter/formatter config, and no test suite exists in this repo (pytest is only listed as a dependency, unused so far). Don't invent lint/test commands that aren't there.
- Scripts are run directly, not via a CLI: `python hydro_analysis/MSD_Trackmate/MSD_D0_overview.py`, etc. Most have a `main()` guarded by `if __name__ == "__main__":` and hardcoded input paths at the top (see Data Handling below) — running one script processes one specific dataset, there is no generic "run the pipeline" entry point.

## Repository Structure

```
hydro_analysis/
  core/                  # Shared, domain-agnostic primitives — see "core/ contract" below
  MSD_Trackmate/          # SPT diffusion analysis (MSD + step-size methods), largest domain
    old skripts/          # Superseded, do not build on these
  FRAP/                   # FRAP recovery analysis (Leica SP8 confocal data)
  SEM_Particles/           # SEM image particle segmentation/sizing (watershed, Qt viewer)
  SEM_Data/                # SEM metadata inventory tooling
  Litesizer/               # DLS (LiteSizer) reference measurement parsing/visualization
  Trajectory/              # Trajectory overlay/figure generation on microscopy frames
  Parameter_Check/         # Interactive trackpy parameter tuning GUI, instrument metadata checks
  Visualisation/           # Ad-hoc demo/candidate-frame viewers
  Fotos_Labor/              # Lab photo EXIF/metadata inventory
  3D_Visualisation/         # (currently empty scaffold)
Style_guide.txt            # Figure style bible (German) — see "Figures" below
```

Raw microscopy/instrument data (TIFF stacks, TrackMate XML, `.rec` files) is **not in this repo**. It lives on external drives, referenced by absolute Windows paths hardcoded near the top of each script (e.g. `E:\PhD Data Analysis\SPT 2025 II\...`, `H:\Daten Promotion Sicherung\...`). Editing a script's analysis often means editing these path/folder dictionaries, not the analysis logic.

## `core/` contract — read this before adding shared code

`core/` is split into four modules with a strict, self-declared non-overlap. Each module's docstring states what it owns and explicitly what it must never contain:

- `core/io.py` — data extraction and parsing only (TrackMate XML, `.rec` calibration files, filename/path parsing, folder scanning, DLS reference cache). *"Here wont be stored any further analysis methods or visualization methods."*
- `core/analysis.py` — MSD and step-size diffusion calculations. *"Here wont be stored any data loading or visualization methods."*
- `core/physics.py` — physical constants and formulas (Stokes-Einstein).
- `core/visualization.py` — functions that build and save `matplotlib` figures. *"Here wont be stored any data loading or analysis methods."*

Follow this separation for any new shared code: loading/parsing goes in `io.py`, numeric analysis in `analysis.py`, plotting in `visualization.py`. `core/__init__.py` re-exports the public surface of all three — add new shared functions to its imports/`__all__` too.

**Do not change existing functions in `core/` (or other widely-imported functions such as `single_file_data`, `remove_edge_artifacts`, `perform_msd_analysis`, `fit_powerlaw_with_errors`) unless the user has explicitly asked for that function to be changed.** Many downstream scripts depend on their exact current behavior and output schema; prefer adding a new function or a new script over silently altering shared logic.

## Data handling: compute vs. read-only scripts

Nearly every analysis domain follows the same two-stage pattern, and scripts are explicit in their own docstrings about which stage they are:

1. **Compute stage** (reads raw data, writes a cache): loads TrackMate XML / TIFF / `.rec` files via `core.io.single_file_data()` / `build_datasets()`, runs `core.analysis.perform_msd_analysis()` / `perform_stepsize_analysis()`, and pickles the result to `hydro_analysis/<Domain>/cache/*.pkl` (e.g. `MSD_FromTrackmate_20mg.py` → `cache/msd_20mg_files.pkl`, `cache/msd_20mg_result.pkl`). These scripts **always recompute from raw files and unconditionally overwrite** their pickle — they never read their own cache back to skip work.
2. **Consumer stage** (reads the cache only, never touches raw data): scripts like `MSD_Diffusion_vs_Size_20mg_WeightedAvg.py` load an existing pickle and raise `FileNotFoundError` with an instruction to run the compute script first if it's missing. These scripts only plot or further aggregate already-independent per-file results — they must never re-pool raw trajectories or refit anything themselves.

A script's own docstring header always states which stage it is, which pickle(s) it reads/writes, and which other scripts depend on its output (e.g. `MSD_FromTrackmate_20mg.py`'s docstring lists every downstream consumer by filename). When adding a new analysis, follow this same pattern and document the read/write relationship in the new script's header the same way — do not assume it's implicit from the filename.

Key standardized data structure: the **`result_dict`** (built by `core.io.single_file_data()` / `_build_result_dict()`), with fixed keys `tracks_df`, `mpp`, `fps`, `particle_size_nm`, `num_tracks`, `D_MSD`, `fit_results_MSD`, `D_step`, `fit_results_step`, etc. Most `core.analysis` and `core.visualization` functions take this dict (or a `dict[xml_path, result_dict]` collection) as input — match this shape for new analyses rather than inventing a parallel structure.

`core.io.remove_edge_artifacts()` is applied automatically inside `single_file_data()` for every script that loads tracks this way — near-border detections (within 3% of the frame edge) are dropped and trajectories split at the gap, because TrackMate occasionally mis-links spurious near-edge detections. This is silent, load-bearing behavior; don't re-implement or bypass it per-script.

## Figures

`hydro_analysis/Style_guide.txt` (German) is the canonical figure style spec — read it before writing new plotting code. Key rules:

- **PNG only, 600 dpi, never PDF/SVG.** ([[no-pdf-export]] feedback memory)
- Standard figure size `(7.15, 5.00)` inch, ratio 1.43:1, `constrained_layout=True` or `tight_layout()`.
- Font: Open Source Sans in-figure; ticks inward on all four sides (`direction="in"`, `top=True`, `right=True`); no grid by default.
- Fixed 8-color "Jet-derived" accent palette with `base`/`dark`/`bright` variants per index (defined in the style guide and reused as literal hex constants across plotting scripts, e.g. `COLOR_MEASURED_BASE = '#0000da'`). Reuse these exact hex values for new plots in the same family rather than picking new colors.
- Scatter marker size 26, `elinewidth=1.2`, `capsize=3.0`, fit line width 2.2 — see `core/visualization.py::plot_diffusion_comparison` / `plot_theory_comparison` for the reference implementation, and `Style_guide.txt` §12 for the dataset-vs-particle-size (log-log) plot convention specifically (shared `_RC` rcParams dict, defined once in `MSD_Trackmate/MSD_per_file_publication.py` and imported by sibling scripts — reuse it rather than redefining rcParams).
- Both `core/visualization.py` functions and many domain scripts build figures directly with `matplotlib` (rather than routing everything through `core/visualization.py`) — style constants are frequently duplicated as local module-level constants per script instead of imported, so when editing a plot's look, check whether the script defines its own `COLOR_*`/`_RC` constants before assuming it calls into `core/visualization.py`.
- Standard save location for evaluation-script plots: `E:\...\Auswertungsbilder` ([[project_save_path_auswertung]] project memory) — an external path, not inside the repo.

## Script header convention

Nearly every script opens with a module docstring, before any imports, that states in prose:
- what the script computes or plots, in domain terms (e.g. "D₀ overview — individual per-file power-law fits + 20 mg/mL C16 eMSD overlay");
- which pickled cache(s) it reads and/or writes, and by exact filename;
- which other scripts consume its output, or which script must be run first to produce its input;
- any non-obvious data-cleaning step applied upstream (e.g. the edge-artifact filtering note above) that the reader needs to know about to interpret results correctly.

Match this convention for new or edited scripts: prose docstring header first, then a `# ── Configuration ──` block of module-level constants (paths, thresholds, plot toggles), then functions, then a `main()` guarded by `if __name__ == "__main__":`. Comments and print statements elsewhere should stay minimal — only where they explain a non-obvious decision, no emojis, no exclamation marks, formal tone (this is an explicit convention already followed in `core/io.py` and `core/analysis.py`).

## Legacy code

`hydro_analysis/MSD_Trackmate/old skripts/` holds superseded versions (e.g. `trackpy_msd.py`, `Trackpy_MSD_v1.py`) kept for reference only — do not extend or import from these; use the current `core/`-based scripts instead.
