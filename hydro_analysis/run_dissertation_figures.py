"""
Shortcut: regenerate all dissertation figures from the data as they are.

Runs every figure script of the selection (SPT_ausgewaehlt_Abbildungen.docx) one after the other,
in figure order, each in its own process without plot windows (MPLBACKEND=Agg). No compute-stage
script is started: no MSD_FromTrackmate_*, frap_analysis_v2, Particle_Size_SNR_Compute,
Rheo_Hyd20_Compute or similar. The figure scripts read the existing caches in hydro_analysis/*/cache
and, where the figure needs them, image frames, metadata and instrument exports (read only). To
recompute the caches, run the compute scripts first, then this file.

Not part of this run because their scripts compute from raw data or still write outside the topic
folders: Abb13 (Dissertation_Figures/DLS_Diffusion_vs_Size.py), Abb28 (Trajectory/Hydrogel_Best_Frames.py),
Abb29 (Laser_Intensity/Laser_beam_norm_detector.py), Abb30 (SEM_Particles/sem_example_cutout.py),
FluidFM (AFM_Agnes/afm_force_distance_curve.py, Schematics/schematic_force_soft_surface_brush.py).

The two SEM pore figures come from the HyPoPy repository and run with its own virtual environment
(HYPOPY_PYTHON); if that interpreter is missing, those steps are skipped and logged as skipped.

Writes, per run, to RESULTS/_scratch/run_dissertation_figures/<date_time>/: one log file per step
(stdout and stderr of the script) and run_summary.csv (step, figures, script, status, seconds).
Every script writes its figures into its topic folder as before. Run from the file itself.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
HYPOPY_REPO = REPO.parent / "HyPoPy"
HYPOPY_PYTHON = HYPOPY_REPO / ".venv" / "Scripts" / "python.exe"
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
LOG_ROOT = RESULTS / "_scratch" / "run_dissertation_figures"
ONLY: list[str] | None = None      # e.g. ["Abb17-19", "Abb25-27"] to run only these steps; None = all
STOP_ON_ERROR = False              # False: continue with the next step and report failures at the end

# (step label, figures, repository, script relative to the repository, arguments)
STEPS = [
    ("Abb01-02", "SPT frames buffer / hydrogel", "main", "hydro_analysis/Trajectory/Random_Condition_Frames.py", []),
    ("Abb03+07", "location error correction, D vs n", "main",
     "hydro_analysis/MSD_Trackmate/Validation_Claude/Figures_Refined/Correlations_Refined.py", []),
    ("Abb04+06", "localisation error, immobilised trajectories", "main",
     "hydro_analysis/MSD_Trackmate/Validation_Claude/Figures_Refined/LocError_Refined.py", []),
    ("Abb05", "anomalous exponent, weighted average", "main",
     "hydro_analysis/MSD_Trackmate/MSD_AnomalousExponent_vs_Size_20mg_WeightedAvg.py", []),
    ("Abb08", "individual anomalous exponents", "main",
     "hydro_analysis/MSD_Trackmate/Dissertation_Figures/Deff_AnomalousExponent_vs_Size.py", []),
    ("Abb09", "D_eff / D0", "main", "hydro_analysis/MSD_Trackmate/MSD_Deff_over_D0_vs_Size_20mg_WeightedAvg.py", []),
    ("Abb10", "hydrogel D, weighted average", "main",
     "hydro_analysis/MSD_Trackmate/MSD_Diffusion_vs_Size_20mg_WeightedAvg.py", []),
    ("Abb11", "individual hydrogel D", "main", "hydro_analysis/MSD_Trackmate/MSD_Diffusion_vs_Size_20mg.py", []),
    ("Abb11b", "D vs number of tracks", "main",
     "hydro_analysis/MSD_Trackmate/Dissertation_Figures/Deff_vs_TrackCount.py", []),
    ("Abb12", "individual buffer D0 with inset", "main", "hydro_analysis/MSD_Trackmate/MSD_Diffusion_vs_Size_D0.py", []),
    ("Abb14-15", "MSD fitting points", "main",
     "hydro_analysis/MSD_Trackmate/Validation_Claude/Figures_Refined/Fittingpoints_Refined.py", []),
    ("Abb16", "DoG example", "main",
     "hydro_analysis/MSD_Trackmate/Validation_Claude/Figures_Refined/SNR_Hydrogel_Example.py", []),
    ("Abb17-19", "n, step size and track length histograms", "main",
     "hydro_analysis/MSD_Trackmate/Validation_Claude/Figures_Refined/iMSD_histograms_Refined.py", []),
    ("Abb20", "particle FWHM", "main", "hydro_analysis/MSD_Trackmate/Validation_Claude/Figures_Refined/PSF_Refined.py", []),
    ("Abb21-22", "SNR", "main", "hydro_analysis/MSD_Trackmate/Validation_Claude/Figures_Refined/SNR_Refined.py", []),
    ("Abb23", "eMSD buffer", "main", "hydro_analysis/MSD_Trackmate/MSD_perfile_D0.py", []),
    ("Abb24", "eMSD hydrogel", "main", "hydro_analysis/MSD_Trackmate/MSD_per_file_20mg.py", []),
    ("Abb25-27", "FRAP recovery, D, D_eff/D0", "main", "hydro_analysis/FRAP/frap_depth_figures.py", []),
    ("FRAP raw", "FRAP raw intensity (appendix)", "main", "hydro_analysis/FRAP/frap_raw_intensity_comparison.py", []),
    ("Abb31-32", "SEM Feret histograms", "main", "hydro_analysis/SEM_Particles/sem_feret_size_histogram.py", []),
    ("Abb33", "SEM vs DLS", "main", "hydro_analysis/SEM_Particles/sem_vs_dls_size_distribution.py", []),
    ("Abb34", "zeta potential", "main", "hydro_analysis/Litesizer/litesizer_zeta_potential.py", []),
    ("Abb35-36", "DLS size and PDI (chapter)", "main", "hydro_analysis/Litesizer/litesizer_visualization_chapter.py", []),
    ("DLS appendix", "DLS size and PDI, all measurements", "main",
     "hydro_analysis/Litesizer/litesizer_visualization_all_measurements.py", []),
    ("Abb38-40", "rheology", "main", "hydro_analysis/Rheology/rheo_hyd20_key_figures.py", []),
    ("Abb41", "SEM pore size comparison", "hypopy", "software/batch_figure.py", []),
    ("Abb42-44", "SEM examples and threshold schematic", "hypopy", "software/batch_crops.py", ["--dissertation"]),
    ("Autofluorescence", "decay and switch QC", "main", "hydro_analysis/Auto_fluorecence/autofluorescence_analysis.py", []),
]


def run_step(label: str, repo: str, script: str, args: list[str], log_dir: Path) -> tuple[str, float]:
    """Run one figure script in its own process; returns (status, seconds)."""
    if repo == "hypopy":
        python, cwd = HYPOPY_PYTHON, HYPOPY_REPO
        if not python.exists():
            (log_dir / f"{label}.log").write_text(f"skipped: {python} not found\n", encoding="utf-8")
            return "skipped (HyPoPy environment not found)", 0.0
    else:
        python, cwd = Path(sys.executable), REPO
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    start = time.perf_counter()
    proc = subprocess.run([str(python), str(cwd / script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - start
    (log_dir / f"{label}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    return ("ok" if proc.returncode == 0 else f"error (exit {proc.returncode})"), seconds


def main() -> None:
    log_dir = LOG_ROOT / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    steps = [s for s in STEPS if ONLY is None or s[0] in ONLY]
    rows = []
    for i, (label, figures, repo, script, args) in enumerate(steps, 1):
        print(f"[{i:2d}/{len(steps)}] {label:16s} {figures} ...", end=" ", flush=True)
        status, seconds = run_step(label, repo, script, args, log_dir)
        print(f"{status}, {seconds:.0f} s", flush=True)
        rows.append({"step": label, "figures": figures, "script": script, "status": status, "seconds": round(seconds, 1)})
        if STOP_ON_ERROR and status.startswith("error"):
            break
    with open(log_dir / "run_summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [r for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} steps ok. Logs: {log_dir}")
    for r in failed:
        print(f"  {r['step']}: {r['status']} -> {log_dir / (r['step'] + '.log')}")
    sys.exit(1 if any(r["status"].startswith("error") for r in failed) else 0)


if __name__ == "__main__":
    main()
