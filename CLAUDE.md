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
  thesis.mplstyle           # Matplotlib rcParams of the figure style guide (§14)
  Style_guide_old_obsolete.txt  # Superseded v1 style guide, do not follow
Styleguide_Figures_Dissertation.md  # Figure style guide v2 (German, repo root) — see "Figures" below
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

`Styleguide_Figures_Dissertation.md` (repo root, German, v2) is the canonical figure style spec, matched to the LaTeX layout of the dissertation — read it before writing or restyling plotting code. `hydro_analysis/Style_guide_old_obsolete.txt` is the superseded v1 (PNG only, 7.15 × 5.00 in); do not follow it. Key rules:

- **Golden rule: create every figure at exactly its printed size and embed it unscaled.** Only four width classes, each paired with a fixed LaTeX width: `full` 6.30 in (`width=\linewidth`), `narrow` 4.72 in (`0.75\linewidth`), `half` 3.07 in (subfigure `0.49\linewidth`), `third` 2.01 in (subfigure `0.32\linewidth`); height = width / 1.42. Multi-panel figures with shared axes are one Python figure in `full`, whose height may deviate from 1.42:1.
- **Export: plots as PDF (vector), microscopy images and crops as PNG at 600 dpi**; very dense plots as PDF with `rasterized=True` data layers. **Never `bbox_inches="tight"`** — it changes the figure size and therefore the printed font size; keep all labels inside the fixed figure via constrained layout or explicit axes positions. File names descriptive, lowercase, no spaces or umlauts (e.g. `emsd_hyd_35nm_surface.pdf`). ([[no-pdf-export]] feedback memory records this v2 rule.)
- **rcParams live in `hydro_analysis/thesis.mplstyle`** (style guide §14): Open Sans (mathtext too), axis labels 9 pt, tick labels and legend 8 pt, panel labels (A, B, …) 10 pt bold, ticks inward on all four sides (major 3.5 / minor 2.0 pt), spines 0.8 pt, no grid, no titles. Apply with `plt.style.context(<path to thesis.mplstyle>)` resolved from `__file__`, and save inside that context. Reference implementation: `Litesizer/litesizer_visualization.py`.
- Lines and markers: data line 1.2 pt (1.0 with many series), fit line 1.5 pt in the `dark` colour, theory line black 1.2 pt dashed `(0, (4, 3))`; markersize 4 (scatter `s≈16`), markeredgewidth 0.6; error bars `elinewidth=0.8`, `capsize=2.0`, `capthick=0.8`; overlapping points alpha 0.5–0.7, points exactly at their real x value (no jitter).
- Colours: 8-colour Jet-derived palette with `base`/`dark`/`bright` variants, plus muted category colours (A `#3B8C8C`, B `#D98C3D`, single series `#3B6E8C`, DLS reference `#da00bd`) — see style guide §11 for all hex values and roles (data `base`, fit/model `dark`).
- **Particle sizes have one fixed colour across all figures**: `SIZE_COLORS` in `MSD_Trackmate/Validation_Claude/Correlations.py`, `{nominal_nm: (base, dark)}`, 20 nm orange → 1000 nm blue (style guide §11, "Partikelgrößen"). Import it, never copy it; `base` for fills and marker faces, `dark` for lines, edges and error bars. Legends and size axes use the DLS labels from `core.io.get_dls_labels()` (35/50/100/240/560/1370 nm), not the nominal sizes.
- All figure text in English; axis labels as `Name symbol (unit)`, e.g. `Diffusion coefficient D (µm² s⁻¹)`.
- **Most existing scripts predate v2**: they use the `_RC` dict from `MSD_Trackmate/MSD_per_file_publication.py` (which also still provides `_DASH_THEORY` and `_add_log_minor_ticks()`), 7.15 × 5.00 in figures and PNG export with `bbox_inches="tight"`. Migrate a script to v2 only when asked; new or restyled figures follow v2. Style constants are frequently duplicated as local module-level constants per script (`COLOR_*`, `_RC`) rather than routed through `core/visualization.py`, so check the script itself before editing a plot's look.
- Save all figures and accompanying result workbooks to `E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data`. This user-specified destination supersedes the former `Auswertungsbilder` convention. Do not save results inside the GitHub repository or beside the raw data.

## Script header convention

Nearly every script opens with a module docstring, before any imports, that states in prose:
- what the script computes or plots, in domain terms (e.g. "D₀ overview — individual per-file power-law fits + 20 mg/mL C16 eMSD overlay");
- which pickled cache(s) it reads and/or writes, and by exact filename;
- which other scripts consume its output, or which script must be run first to produce its input;
- any non-obvious data-cleaning step applied upstream (e.g. the edge-artifact filtering note above) that the reader needs to know about to interpret results correctly.

Match this convention for new or edited scripts: prose docstring header first, then a `# ── Configuration ──` block of module-level constants (paths, thresholds, plot toggles), then functions, then a `main()` guarded by `if __name__ == "__main__":`. Comments and print statements elsewhere should stay minimal — only where they explain a non-obvious decision, no emojis, no exclamation marks, formal tone (this is an explicit convention already followed in `core/io.py` and `core/analysis.py`).

## Legacy code

`hydro_analysis/MSD_Trackmate/old skripts/` holds superseded versions (e.g. `trackpy_msd.py`, `Trackpy_MSD_v1.py`) kept for reference only — do not extend or import from these; use the current `core/`-based scripts instead.
