# AGENTS.md / CLAUDE.md

Guidance for AI coding agents (Claude Code, Codex, Copilot) working in this repository.
`AGENTS.md` and `CLAUDE.md` are identical on purpose so that every tool reads the same rules.
When one is edited, copy the change to the other.

## 0. Rules that override everything else

These rules exist because of concrete past failures: files dropped by a filter nobody noticed,
fallback values for pixel size and frame interval, features reported as done that were never
implemented, results produced without using the requested tool.

### 0.1 Data integrity

1. Raw data is read-only. Never modify, move, rename or delete `.tif`, `.rec`, `.xml`, `.lif`
   or instrument exports (`.xlsx`, `.csv` from LiteSizer, rheometer, microscope).
2. No measurement, file, track, frame, pixel or data point may be excluded, filtered, clipped,
   smoothed, drift-corrected, resampled, subsampled, re-weighted or otherwise altered unless
   the step is listed as approved in 0.2, or the user asked for it in the current task.
   If you think a filter or correction is needed: stop, describe it, ask. Do not apply it
   "for robustness".
3. Every exclusion and correction that is applied must be counted and written to disk, not only
   printed. Write an exclusion log next to the results (`<name>_exclusions.csv` with columns
   `file, step, parameter, n_before, n_after, reason`). A file that is skipped (missing
   calibration, failed fit, fps rule) gets a row too.
4. No fallbacks or defaults for physical parameters (pixel size, frame interval, exposure,
   temperature, viscosity, particle size, radii). If a value is missing, raise an error that
   names the file and the missing field. No `try/except` that swallows a file.
5. No hardcoded frame indices, ROI positions or thresholds inside functions. Every threshold
   lives in the `# ── Configuration ──` block with a comment stating who decided it:
   `# decided by user, YYYY-MM-DD` or `# PROPOSED, not approved`.
6. Never invent, simulate or interpolate measurement values. Synthetic data is allowed only in
   clearly named test functions and is never written to the results folder.
7. Random subsampling only on explicit request, with a fixed seed, and stated in the report.
8. If a load finds fewer files than the folder contains, that is a finding to report with the
   file names. It is never something to work around quietly.

### 0.2 Registry of data-handling steps

Anything that is not in this table is not approved. Rows marked TO CONFIRM are in the code today
but there is no record that the user decided them: keep their behaviour unchanged, do not copy
them into new scripts, and list them in your end-of-task report whenever they affected a result.

| Step | Where | Parameter | Status |
|---|---|---|---|
| Minimum track length | `core.analysis.MIN_TRACK_LENGTH`, `tp.filter_stubs` | 10 frames | approved by user |
| Detection filter | TrackMate | DoG only, no LoG | approved by user, 2026-09-21 |
| FRAP geometry | FRAP pipeline | nominal bleach ROI is never a geometric input | approved by user |
| FRAP noFITC controls | `FRAP/measurements.csv` (`include`), `frap_analysis_v2.py` | nofitc_2 and nofitc_3 excluded (FITC clearly present), not loaded, logged in `nofitc_controls_exclusions.csv` and the figure exclusion CSVs | approved by user, 2026-10-27 |
| Immobilised-particle analysis | `Validation_Claude/Loc_Error_*`, `PSF_*` | blurry frames and aggregates excluded | approved by user, 2026-09-21; the numeric thresholds are TO CONFIRM |
| MSD fit range | `core.analysis.DEFAULT_MSD_FIT_POINTS` | first 6 lag points (validation runs used 4, not yet repeated with 6) | approved by user, 2026-10-27 (E004 A) |
| Edge-artifact removal | `core.io.remove_edge_artifacts`, applied inside `single_file_data` | detections within 3 % of the border dropped, tracks split | approved by user, 2026-10-06 (E001 A) |
| Minimum frame rate | `MSD_FromTrackmate_D0.py`, `MSD_FromTrackmate_20mg.py` | nominal 20/50/100 nm files below 40 Hz skipped, written to the exclusion log (`msd_pipeline.MIN_FPS_SMALL_SIZES`, water and hydrogel) | approved by user, 2026-10-27 (E003 A) |
| Drift subtraction | `core.analysis.perform_msd_analysis` (> 50, older scripts), `perform_msd_analysis_blur_locerror` via `msd_pipeline.DRIFT_MIN_TRACKS_PER_FRAME` | applied if tracks/frames >= 30 | approved by user, 2026-10-27 (E009 B); the > 50 rule in `perform_msd_analysis` is unchanged |
| Weighting of per-file results | `core.analysis.weighted_average_per_size`, `msd_pipeline.WEIGHT_KEY` | weight = `num_tracks_fit`, tracks after the 10-frame filter (water and hydrogel) | approved by user, 2026-10-27 (E002 A) |
| Localisation error from intercept | `core.analysis.perform_msd_analysis` | reported sigma = immobilised particles; intercept only as diagnostic (`sigma_loc_intercept_nm`, `msd_pipeline`) | approved by user, 2026-10-27 (E005 A) |
| MSD fit with localisation error and motion blur | `core.analysis.perform_msd_analysis_blur_locerror`, `MSD_Trackmate/msd_pipeline.py` (water and hydrogel) | MSD = 4K/[(n+1)(n+2) t_E²]·[(τ+t_E)^(n+2) + |τ−t_E|^(n+2) − 2τ^(n+2) − 2t_E^(n+2)] + 4σ² (Savin & Doyle 2005, Backlund et al. 2015; K stored as `D_um2_per_s`, µm² s⁻ⁿ; `msd_pipeline.MSD_MODEL`) over 6 lags, t_E = exposure from .rec, σ = mean `sigma_xy_static_nm` of immobilised particles, fit in log space; σ taken at 0.3 µm/px for nominal 20/50/100 nm and at 0.15 µm/px for 200/500/1000 nm, whatever the file's pixel size (`SIGMA_MPP_BY_SIZE`, deviations logged as `sigma_loc_mpp`); per file the localisation error variance of the moving sample from a straight line through the 6 lags plus (4/3) D t_E is reported beside the static σ² and the dynamic error of dissertation eq 4.6, δ_dyn = δ_sta √(1 + D_lin t_E / s₀²) with s₀ = 250 nm (`PSF_SIGMA_S0_NM`) (output only) | approved by user, 2026-10-27 (E039 C, E040 A, E042 B, E043, E046 B; E041 replaced) |
| One analysis per movie | `msd_pipeline.ALTERNATIVE_MARKERS` | grouped over all folders of a size; re-analyses (Resultof, filtered_, _processed, _var, _v2) and byte-identical copies are listed in the per-file table only; original (first folder) kept | approved by user, 2026-10-22 / 2026-10-26 (E031) |
| DLS polydispersity | `Litesizer/*` | measurements with PDI > 42 % excluded | approved by user, 2026-10-06 (E007 A) |
| Rheology QC | `Rheology/Rheo_Hyd20_Compute.py` | runs with `axial_force` (> 0.5 N), `presheared`, `cracks`, `group_outlier` (factor 2) and FAIL runs left out of `qc_filtered` statistics | approved by user, 2026-10-08 (E008, E025) |
| Rheology point exclusion | `Rheology/Rheo_Hyd20_Compute.py`, `POINT_CRITERIA` | points with inertia ratio > 1 or torque < 5e-8 N m left out of fits (kept in tables, shown grey) | approved by user, 2026-10-08 (E026) |
| Rheology key figures | `Rheology/rheo_hyd20_key_figures.py` | only qc_filtered runs; 95 % t confidence interval over specimens; mesh and pore sizes pool specimens younger than 10 d; figure 1: 4 d specimen and 9 d amplitude sweep drawn at random (seed 20261008); pore error = 95 % CI and fibril diameter uniform 9-13 nm (1.96 x 4/sqrt(12) nm) in quadrature | approved by user, 2026-10-08 (E025, E027, E028) |
| SEM platinum coating | `SEM_Particles/sem_vs_dls_size_distribution.py`, `PT_COATING_NM` | 2 × 1.3 nm subtracted from every SEM diameter in the SEM–DLS comparison (Abb33), logged per size | approved by user, 2026-10-14 |
| Per-track MSD fits | `Validation_Claude/TaskB1_TrackLengthBias_Compute.py`, `Validation_Claude/Fittingpoints_robustness.py` (Abb14–15, Abb17–19) | each iMSD column fitted with the pipeline model (E046), σ and t_E of the file; tracks with NaN/≤ 0 in the window or a fit that does not converge are left out and logged (`track_length_bias_exclusions.csv`, `fittingpoints_exclusions.csv`) | approved by user, 2026-10-27 (E045 A, E046 B) |
| SPT variant without correction | `MSD_Trackmate/run_spt_uncorrected_powerlaw.py` | same pipeline and figure scripts, ensemble and per-track fits as plain power law 4Dτ^n over 6 lags (`MSD_MODEL_POWERLAW`); caches in `cache/uncorrected_powerlaw/`, outputs in `SPT/uncorrected_powerlaw/` and `_scratch/uncorrected_powerlaw/`; not run there: Abb03, `_uncorrected_fit` eMSD version, LocError figures | requested by user, 2026-10-27 |
| Amsden fit of D_eff/D0 | `MSD_Trackmate/MSD_Deff_over_D0_vs_Size_20mg_Amsden.py`, `R_F_MODE`, `FIBRIL_DIAMETER_NM`, `R_F_BOUNDS_NM` | Amsden 2022 eq 3, exp[-π((r_s + r_f)/(ξ + 2r_f))²], ξ (mesh size) fitted by weighted least squares with absolute errors; r_f fixed at 1.5 nm; one band, ±1σ absolute; earlier runs: r_f = 6 nm and r_f free within 3-15 nm (on the upper bound); per-file variant `MSD_Deff_over_D0_vs_Size_20mg_Amsden_PerFile.py`: every file one point, weight = `num_tracks_fit`, covariance scaled by the weighted residual variance, no error bars, marker area ∝ √N | approved by user, 2026-10-26 (E036 A, E037 C, E038 A); per-file variant 2026-10-27 |
| Autofluorescence decay | `Auto_fluorecence/autofluorescence_analysis.py` | dark level subtracted, I_0 = frame after the transition frame, transition frames not used, t1/2 = first frame with I_norm <= 0.5, all switch-ons; pause rise = I_0 after pause / frame (switch-off − 2) − 1 on raw counts; B1 wavelength 514 nm (not in .rec) | approved by user, 2026-10-07 (E011, E012) |

Each TO CONFIRM row has an entry in `ENTSCHEIDUNGEN.md` (see 0.8). When the user confirms or
rejects a row, update the status here with the date.

### 0.3 Decisions belong to the user

You implement, run, check and report. You do not decide:
- which method, model or estimator is used;
- thresholds, fit ranges, inclusion and exclusion criteria;
- what a result means physically, or what it says about the experiment.

If a task cannot be completed without such a decision, stop and ask, with the options and what
each would change. Reports contain numbers and what was done. No interpretation unless asked.

### 0.4 Scope

- Do what was asked and nothing else. One task answers one question. No additional figures,
  analyses, refactorings or "while I was there" fixes. Ideas go into the report under
  "Suggestions, not done".
- Do not change existing scripts unless the user names the script. Creating a new script is fine.
- When a task touches an existing analysis, first report what the existing code does (inputs,
  filters, aggregation, error bars) and which assumptions you would have to make. Continue only
  if no assumption is needed; otherwise wait.
- If the user says "only improve the prompt" or "only explain", do not touch code.
- Use the tool or module the user names (for example HyPoPy). Do not substitute your own
  implementation and present its numbers as if they came from that tool.

### 0.5 Verification before you report

- New analysis logic is first run on synthetic data with a known answer. Report recovered
  versus true value. Only then run it on real data.
- Distinguish in every report between implemented, run, and verified. Never describe a
  feature as present if it was not implemented and executed in this task.
- After every run, state per condition: files found, files loaded, files excluded (with reason),
  tracks before and after each filter.

### 0.6 Where things go

- New scripts are placed by question inside the domain folder. Do not create folders named after
  an AI tool or a version (`Validation_Claude`, `Validation_GPT`, `_old`, `_new`, `_Refined`);
  the existing ones stay as they are.
- Figures, tables and workbooks go to
  `E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data`
  in a subfolder per topic. Exploratory output goes to a `_scratch` subfolder there. Nothing is
  saved inside the repository or beside the raw data.
- Formats and styling follow `Styleguide_Figures_Dissertation.md`. Read it before plotting.
- Scripts run from the file itself (`if __name__ == "__main__":`), without console arguments.

### 0.7 End-of-task report

Every task ends with these five points, in this order:
1. Done: files created or changed, with paths.
2. Data used: files and tracks per condition.
3. Exclusions and corrections applied, with counts, including TO CONFIRM rows from 0.2.
4. Assumptions made.
5. Not done, open questions, suggestions.

### 0.8 Open decisions

`ENTSCHEIDUNGEN.md` (repo root, German) is the single list of decisions that belong to the user.
A desktop widget reads it, so keep its format exactly.

When a decision from 0.3 comes up during a task:
1. Stop at that point. Continue only with parts that do not depend on it.
2. Add an entry to `ENTSCHEIDUNGEN.md` with the next free number:
   `## E0NN · OFFEN · YYYY-MM-DD`, then `**Frage:**` (one sentence), `**Stelle:**` (file and
   function), `**Wirkung heute:**` (what the current behaviour does to the data, with counts),
   the options as `- A: ... → effect on the result`, and `**Entscheidung:** offen`.
3. Never set an entry to `ENTSCHIEDEN` yourself and never pick an option. Only the user decides;
   when they do, record their choice and the date in the entry and update the row in 0.2.
4. End your answer with this block, after the report from 0.7:

```
ENTSCHEIDUNG NÖTIG (n)
- E0NN: question in one sentence
  A: ... → changes ...
  B: ... → changes ...
```

   Give no recommendation unless the user asks for one. If nothing is open from this task,
   write `Keine Entscheidung offen.`

Before starting a task, read `ENTSCHEIDUNGEN.md`. If the task depends on an entry that is still
`OFFEN`, say so first and ask whether to wait or to proceed with the current behaviour.

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
- `core/physics.py` — physical constants and formulas (e.g. Stokes-Einstein).
- `core/visualization.py` — functions that build and save `matplotlib` figures. *"Here wont be stored any data loading or analysis methods."*

Follow this separation for any new shared code: loading/parsing goes in `io.py`, numeric analysis in `analysis.py`, plotting in `visualization.py`. `core/__init__.py` re-exports the public surface of all three — add new shared functions to its imports/`__all__` too.

**Do not change existing functions in `core/` (or other widely-imported functions such as `single_file_data`, `remove_edge_artifacts`, `perform_msd_analysis`, `fit_powerlaw_with_errors`) unless the user has explicitly asked for that function to be changed.** Many downstream scripts depend on their exact current behavior and output schema; prefer adding a new function or a new script over silently altering shared logic.

## Data handling: compute vs. read-only scripts

Nearly every analysis domain follows the same two-stage pattern, and scripts are explicit in their own docstrings about which stage they are:
Do not alter any original datafiles including: .tif .ref .xml .lif files. Those files must always remain in their original state. 
1. **Compute stage** (reads raw data, writes a cache): loads TrackMate XML / TIFF / `.rec` files via `core.io.single_file_data()` / `build_datasets()`, runs `core.analysis.perform_msd_analysis()` / `perform_stepsize_analysis()`, and pickles the result to `hydro_analysis/<Domain>/cache/*.pkl` (e.g. `MSD_FromTrackmate_20mg.py` → `cache/msd_20mg_files.pkl`, `cache/msd_20mg_result.pkl`). These scripts **always recompute from raw files and unconditionally overwrite** their pickle — they never read their own cache back to skip work.
2. **Consumer stage** (reads the cache only, never touches raw data): scripts like `MSD_Diffusion_vs_Size_20mg_WeightedAvg.py` load an existing pickle and raise `FileNotFoundError` with an instruction to run the compute script first if it's missing. These scripts only plot or further aggregate already-independent per-file results — they must never re-pool raw trajectories or refit anything themselves.

A script's own docstring header always states which stage it is, which pickle(s) it reads/writes, and which other scripts depend on its output (e.g. `MSD_FromTrackmate_20mg.py`'s docstring lists every downstream consumer by filename). When adding a new analysis, follow this same pattern and document the read/write relationship in the new script's header the same way — do not assume it's implicit from the filename.

Key standardized data structure: the **`result_dict`** (built by `core.io.single_file_data()` / `_build_result_dict()`), with fixed keys `tracks_df`, `mpp`, `fps`, `particle_size_nm`, `num_tracks`, `D_MSD`, `fit_results_MSD`, `D_step`, `fit_results_step`, etc. Most `core.analysis` and `core.visualization` functions take this dict (or a `dict[xml_path, result_dict]` collection) as input — match this shape for new analyses rather than inventing a parallel structure.

`core.io.remove_edge_artifacts()` is applied automatically inside `single_file_data()` for every script that loads tracks this way — near-border detections (within 3% of the frame edge) are dropped and trajectories split at the gap, because TrackMate occasionally mis-links spurious near-edge detections. This runs without any per-script call and downstream results depend on it; don't re-implement or bypass it per-script. Its approval status is tracked in the registry in 0.2.

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
- Save all figures and accompanying result workbooks to `E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data`. Since 2026-10-21 it holds one folder per topic (`SPT`, `FRAP`, `Rheology`, `Particle_Characterization`, `Autofluorescence`, `Laser_Characterization`, `AFM_FluidFM`, `SEM_Pore_Analysis`): the figures used in the dissertation, their tables and the `AbbNN_*.txt` comment at the top, pipeline data in named subfolders, unused figures in `<topic>/Obsolete`; exploratory output in `_scratch`. `HyPoPy Results`, `Hydrogel_Simulation` and `3D_Print_Hydrogel` belong to other repositories and stay where they are. This user-specified destination supersedes the former `Auswertungsbilder` convention. Do not save results inside the GitHub repository or beside the raw data.

## Script header convention

Nearly every script opens with a module docstring, before any imports, that states in prose:
- what the script computes or plots, in domain terms (e.g. "D₀ overview — individual per-file power-law fits + 20 mg/mL C16 eMSD overlay");
- which pickled cache(s) it reads and/or writes, and by exact filename;
- which other scripts consume its output, or which script must be run first to produce its input;
- any non-obvious data-cleaning step applied upstream (e.g. the edge-artifact filtering note above) that the reader needs to know about to interpret results correctly.

Match this convention for new or edited scripts: prose docstring header first, then a `# ── Configuration ──` block of module-level constants (paths, thresholds, plot toggles), then functions, then a `main()` guarded by `if __name__ == "__main__":`. Comments and print statements elsewhere should stay minimal — only where they explain a non-obvious decision, no emojis, no exclamation marks, formal tone (this is an explicit convention already followed in `core/io.py` and `core/analysis.py`).

## Legacy code

`hydro_analysis/MSD_Trackmate/old skripts/` holds superseded versions (e.g. `trackpy_msd.py`, `Trackpy_MSD_v1.py`) kept for reference only — do not extend or import from these; use the current `core/`-based scripts instead.
