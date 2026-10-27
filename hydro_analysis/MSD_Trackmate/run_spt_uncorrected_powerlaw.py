"""
SPT analysis, variant without localisation-error and motion-blur correction
("uncorrected power law"; user, 2026-10-27).

Runs the same compute and figure scripts as the corrected analysis, unchanged,
with two differences only:
  - the ensemble fit of every file is the plain power law MSD = 4 D tau^n over
    the same 6 lags (core.analysis.MSD_MODEL_POWERLAW, fit_powerlaw_plain, i.e.
    fit_powerlaw_with_errors in log space); the per-track fits follow the model
    of the file (core.analysis.fit_msd_like_ensemble), so they are plain power
    laws as well;
  - every cache and output path is redirected to a VARIANT subfolder: the
    caches listed in VARIANT_CACHE_NAMES to cache/<VARIANT>/, everything under
    RESULTS/SPT to RESULTS/SPT/<VARIANT>/, everything under RESULTS/_scratch to
    RESULTS/_scratch/<VARIANT>/. The corrected caches and figures are not touched.
All other steps are those of msd_pipeline.py and AGENTS.md section 0.2 (one
analysis per movie, fps rule, edge-artifact removal, drift rule, 10-frame
filter, weighting by tracks >= 10 frames, DLS D0). The fixed sigma and t_E are
still read and written to the tables (diagnostic columns), but do not enter
any fit.

Not run in this variant, because they compare or describe the correction itself
and would be empty or misleading here: the "_uncorrected_fit" eMSD version
(identical to the main version), Abb03 (before/after correction) and the
localisation-error figures (LocError_Refined, independent of the fit model).
From Correlations_Refined.py only Abb07 (D vs n) is drawn.

Paths that are inputs, not outputs, are not redirected (KEEP_ATTRS).
Writes VARIANT_INFO_FILE in the variant folder with this description and the
run date.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import os
import sys
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
VARIANT = "uncorrected_powerlaw"   # decided by user, 2026-10-27
CACHE_DIR = Path(__file__).resolve().parent / "cache"
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
VARIANT_CACHE_NAMES = {
    "msd_20mg_files.pkl", "msd_20mg_result.pkl", "msd_d0_results.pkl", "msd_d0_result.pkl",
    "track_length_bias_d0.pkl", "track_length_bias_20mg.pkl",
}
KEEP_ATTRS = {"LENGTH_BIN_CSV", "CSV_PATH"}   # inputs read by the figure scripts
EMSD_VARIANTS = {"fit": "", "imsd": "_imsd"}   # "_uncorrected_fit" would repeat the main version
VARIANT_INFO_FILE = RESULTS / "SPT" / VARIANT / "variant_info.txt"

# (module, run main()) in the order of the corrected analysis
MODULES = [
    "hydro_analysis.MSD_Trackmate.MSD_FromTrackmate_20mg",
    "hydro_analysis.MSD_Trackmate.MSD_FromTrackmate_D0",
    "hydro_analysis.MSD_Trackmate.MSD_per_file_20mg",
    "hydro_analysis.MSD_Trackmate.MSD_perfile_D0",
    "hydro_analysis.MSD_Trackmate.Validation_Claude.TaskB1_TrackLengthBias_Compute",
    "hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.iMSD_histograms_Refined",
    "hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.Fittingpoints_Refined",
    "hydro_analysis.MSD_Trackmate.MSD_AnomalousExponent_vs_Size_20mg_WeightedAvg",
    "hydro_analysis.MSD_Trackmate.Dissertation_Figures.Deff_AnomalousExponent_vs_Size",
    "hydro_analysis.MSD_Trackmate.MSD_Deff_over_D0_vs_Size_20mg_WeightedAvg",
    "hydro_analysis.MSD_Trackmate.MSD_Diffusion_vs_Size_20mg_WeightedAvg",
    "hydro_analysis.MSD_Trackmate.MSD_Diffusion_vs_Size_20mg",
    "hydro_analysis.MSD_Trackmate.Dissertation_Figures.Deff_vs_TrackCount",
    "hydro_analysis.MSD_Trackmate.MSD_Diffusion_vs_Size_D0",
    "hydro_analysis.MSD_Trackmate.MSD_Deff_over_D0_vs_Size_20mg_Amsden",
    "hydro_analysis.MSD_Trackmate.MSD_Deff_over_D0_vs_Size_20mg_Amsden_PerFile",
]
CORRELATIONS_REFINED = "hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.Correlations_Refined"


def _norm(p: Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def variant_path(p: Path) -> Path:
    """Redirect a cache or output path to the variant folder; other paths unchanged."""
    s = _norm(p)
    if p.name in VARIANT_CACHE_NAMES and _norm(p.parent) == _norm(CACHE_DIR):
        return CACHE_DIR / VARIANT / p.name
    for root in (RESULTS / "SPT", RESULTS / "_scratch"):
        r = _norm(root)
        if s == r or s.startswith(r + os.sep):
            rel = os.path.relpath(os.path.abspath(str(p)), os.path.abspath(str(root)))
            if rel.split(os.sep)[0] == VARIANT:
                return p
            return root / VARIANT / ("" if rel == "." else rel)
    return p


def redirect_module_paths() -> list[tuple[str, str, Path, Path]]:
    """Replace every module-level Path of the loaded MSD_Trackmate modules by variant_path()."""
    changed = []
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("hydro_analysis.MSD_Trackmate") or mod is None:
            continue
        for attr, value in list(vars(mod).items()):
            if attr in KEEP_ATTRS or not isinstance(value, Path):
                continue
            new = variant_path(value)
            if new != value:
                setattr(mod, attr, new)
                changed.append((mod_name, attr, value, new))
    return changed


def run_d_vs_n_only() -> None:
    """Abb07 of Correlations_Refined.py (main() would also draw Abb03, the correction itself)."""
    import matplotlib.pyplot as plt
    from hydro_analysis.core.io import get_dls_labels
    c = sys.modules[CORRELATIONS_REFINED]
    d0 = c._load_pickle(c.CACHE_D0, "MSD_FromTrackmate_D0.py")
    gel = c._load_pickle(c.CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    c.SAVE_PATH.mkdir(parents=True, exist_ok=True)
    c.SCRATCH_PATH.mkdir(parents=True, exist_ok=True)
    with plt.style.context(str(c.STYLE_FILE)):
        table = c.build_file_standard_table(d0, gel)
        for d_upper, suffix in ((c.D_AXIS_UPPER_MAIN, ""), (c.D_AXIS_UPPER_ZOOM, "_zoom_0to4")):
            fig, _ = c.plot_d_vs_n_refined(table, get_dls_labels(), d_upper, c.N_AXIS_RANGE_D_VS_N)
            c._save(fig, f"C_D_vs_n_scatter_refined{suffix}", dissertation=(suffix == ""))


def main() -> None:
    for name in MODULES + [CORRELATIONS_REFINED]:
        importlib.import_module(name)
    from hydro_analysis.core.analysis import MSD_MODEL_POWERLAW
    pipeline = sys.modules["hydro_analysis.MSD_Trackmate.msd_pipeline"]
    pipeline.MSD_MODEL = MSD_MODEL_POWERLAW
    sys.modules["hydro_analysis.MSD_Trackmate.MSD_perfile_D0"].VARIANTS = dict(EMSD_VARIANTS)

    changed = redirect_module_paths()
    print(f"Variante {VARIANT}: {len(changed)} Pfade umgeleitet, Modell {pipeline.MSD_MODEL}")
    for mod_name, attr, old, new in changed:
        print(f"  {mod_name.split('.')[-1]}.{attr}: {new}")
    (CACHE_DIR / VARIANT).mkdir(parents=True, exist_ok=True)
    VARIANT_INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    VARIANT_INFO_FILE.write_text(
        f"Run {_dt.datetime.now():%Y-%m-%d %H:%M} by run_spt_uncorrected_powerlaw.py\n" + (__doc__ or ""),
        encoding="utf-8")

    for name in MODULES:
        print(f"\n════ {name.split('.')[-1]} ({VARIANT}) ════")
        sys.modules[name].main()
    print(f"\n════ Correlations_Refined, Abb07 only ({VARIANT}) ════")
    run_d_vs_n_only()


if __name__ == "__main__":
    main()
