"""
Shared SPT MSD pipeline for every condition (water and hydrogel): one set of
functions, parameters and uncertainties (user, 2026-10-27, E040 A). Not run on
its own; the compute scripts MSD_FromTrackmate_D0.py (water) and
MSD_FromTrackmate_20mg.py (20 mg/mL hydrogel) only name their folders, .rec
assignments and outputs and call run_pipeline().

Data handling per file, in this order (decisions in ENTSCHEIDUNGEN.md):
  - E031: one analysis per movie. All XML files of one particle size (over all
    its folders) are grouped by movie: the name without "_Tracks" and without
    the re-analysis markers in ALTERNATIVE_MARKERS. The original analysis
    enters the pickles; re-analyses and byte-identical copies in a second
    folder are computed, but only listed in the per-file table ("alternative
    analysis" / "identical copy"). A movie with no unique original, or with
    copies that are not byte-identical, raises an error.
  - E006 A: a file without .rec calibration (fps, mpp) or without exposure
    time stops the run with an error naming the file. A .rec whose name
    differs from the movie is assigned per compute script (rec_overrides).
  - Edge-artifact removal (E001 A): core.io.single_file_data() ->
    remove_edge_artifacts(), detections within 3 % of the border dropped and
    trajectories split at the gap.
  - E003 A: for nominal 20/50/100 nm files below 40 Hz are skipped.
  - E009 B: drift subtraction (tp.subtract_drift) when tracks / frames >= 30.
  - 10-frame filter (tp.filter_stubs), approved.
  - E039 C, E046, E004 A, E005 A: the eMSD of the first 6 lags is fitted with
    MSD = 4K/[(n+1)(n+2) t_E^2] [(tau+t_E)^(n+2) + |tau-t_E|^(n+2) - 2 tau^(n+2)
    - 2 t_E^(n+2)] + 4 sigma^2 (Savin & Doyle 2005, Backlund et al. 2015;
    core.analysis.perform_msd_analysis_blur_locerror with MSD_MODEL; K is stored
    as D_um2_per_s, unit um^2 s^-n; for n = 1 the model equals the former
    4 D (tau - t_E/3) + 4 sigma^2), t_E = exposure time from the .rec,
    sigma = mean sigma_xy_static_nm of the immobilised particles of the same
    nominal size, at the pixel size given per size in SIGMA_MPP_BY_SIZE
    (0.3 um/px up to nominal 100 nm, 0.15 um/px above; E042), from
    cache/localization_error_immobilized.pkl
    (Validation_Claude/Loc_Error_Analyse_immob_particle.py). A file whose own
    pixel size differs gets a row in the exclusion log. Missing pairs raise.
  - E042: per file the localisation error variance of the moving sample is
    reported beside the static one (core.analysis.locerror_from_msd_slope,
    straight line through the same 6 eMSD lags, (4/3) D t_E added to the
    intercept); output only, it does not enter the fit.
  - E043: per file the dynamic localisation error after dissertation eq 4.6,
    delta_dyn = delta_sta * sqrt(1 + D t_E / s0^2) (core.physics.
    calculate_dynamic_localization_error), delta_sta = the fixed sigma above,
    D = slope / 4 of the same straight line (D_lin), s0 = PSF_SIGMA_S0_NM;
    output only, it does not enter the fit.
  - E002 A: the per-size average (core.analysis.weighted_average_per_size)
    weights each file by its tracks with >= 10 frames (num_tracks_fit); error
    = weighted standard error with Kish effective sample size.

Before the real data, test_blur_locerror_fit_synthetic() fits model-generated
eMSDs, simulated Brownian trajectories and simulated fractional Brownian
trajectories (n = SYNTH_FBM_N), both with motion blur (exposure averaged over
sub-steps) and localisation noise, and prints recovered vs. true values;
nothing synthetic is saved.

Per run, run_pipeline() writes the per-file pickle, the per-size pickle and,
in its table folder, <prefix>_exclusions.csv (every exclusion and correction
with counts), <prefix>_per_file.csv (all analyses, including alternative ones)
and <prefix>_weighted_average.csv.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import trackpy as tp

from hydro_analysis.core.physics import calculate_dynamic_localization_error
from hydro_analysis.core.io import (
    single_file_data, parse_rec_file, read_trackmate_xml, remove_edge_artifacts,
    extract_particle_size_from_path, _build_result_dict,
)
from hydro_analysis.core.analysis import (
    perform_msd_analysis_blur_locerror, fit_powerlaw_blur_locerror_general, msd_blur_locerror_general,
    locerror_from_msd_slope, MSD_MODEL_GENERAL,
    DEFAULT_MSD_FIT_POINTS, MIN_TRACK_LENGTH, weighted_average_per_size,
)

# ── Configuration ──────────────────────────────────────────────────────────────
MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS  # 6 lags, decided by user, 2026-10-27 (E004 A)

# Fit model with motion blur and localisation error for any n (Savin & Doyle 2005,
# Backlund et al. 2015). Decided by user, 2026-10-27 (E046; replaces 4 D (tau - t_E/3)^n + 4 sigma^2).
MSD_MODEL = MSD_MODEL_GENERAL

# Minimum frame rate for the small (DLS 35-100 nm / nominal 20-100 nm) particles;
# 200/500/1000 nm have no fps requirement. Decided by user, 2026-10-27 (E003 A).
MIN_FPS_SMALL_SIZES = 40.0
SMALL_SIZES_NM = (20.0, 50.0, 100.0)

# Drift subtraction when (tracks / frames) >= this value. Decided by user, 2026-10-27 (E009 B).
DRIFT_MIN_TRACKS_PER_FRAME = 30.0

# Weight of the per-file values in the average per size: tracks with >= 10 frames.
# Decided by user, 2026-10-27 (E002 A).
WEIGHT_KEY = "num_tracks_fit"

# Fixed localisation error per (nominal size, mpp): mean sigma_xy_static_nm of the
# immobilised particles. Decided by user, 2026-10-27 (E039 C, E005 A).
LOCERR_CACHE = Path(__file__).parent / "cache" / "localization_error_immobilized.pkl"

# Pixel size (um/px) of the immobilised data used per nominal size, whatever the pixel size
# of the SPT file: 0.3 up to 100 nm, 0.15 above. Decided by user, 2026-10-27 (E042; replaces E041).
SIGMA_MPP_BY_SIZE = {20.0: 0.3, 50.0: 0.3, 100.0: 0.3, 200.0: 0.15, 500.0: 0.15, 1000.0: 0.15}

# PSF width s0 in delta_dyn = delta_sta * sqrt(1 + D t_E / s0^2), all sizes and media.
# Decided by user, 2026-10-27 (E043).
PSF_SIGMA_S0_NM = 250.0

# Name markers of a re-analysis of the same movie (the markers core.io.find_rec_tif_files strips,
# plus "_v2", user, 2026-10-27). One analysis per movie, the original is kept (E031).
ALTERNATIVE_MARKERS = ("Resultof", "filtered_", "_processed", "_var", "_v2")

# Synthetic test (test_blur_locerror_fit_synthetic only)
SYNTH_SEED = 20261027
SYNTH_FPS = 60.0
SYNTH_D_UM2_S = 1.5
SYNTH_SIGMA_NM = 118.8
SYNTH_N_TRACKS = 400
SYNTH_N_FRAMES = 60
SYNTH_SUBSTEPS = 50
SYNTH_FBM_N = 0.6           # fractional Brownian motion, exponent n
SYNTH_FBM_TRACKS = 600
SYNTH_FBM_FRAMES = 30
SYNTH_FBM_SUBSTEPS = 40     # 10 sub-steps biased K by about -4 % at n = 0.6 (discrete exposure average)


def load_sigma_table() -> dict[tuple[float, float], float]:
    """{(nominal size nm, mpp): mean sigma_xy_static_nm} of the immobilised particles."""
    if not LOCERR_CACHE.exists():
        raise FileNotFoundError(
            f"Kein Cache gefunden: {LOCERR_CACHE}\n"
            "Bitte zuerst Validation_Claude/Loc_Error_Analyse_immob_particle.py ausführen."
        )
    with open(LOCERR_CACHE, "rb") as f:
        df = pickle.load(f)
    grouped = df.groupby(["particle_size_nm", "mpp"])["sigma_xy_static_nm"].mean()
    return {(float(s), round(float(m), 4)): float(v) for (s, m), v in grouped.items()}


def _sigma_for(size_nm: float, mpp: float, sigma_table: dict, name: str, exclusions: list) -> float:
    if float(size_nm) not in SIGMA_MPP_BY_SIZE:
        raise KeyError(f"No pixel size for the localisation error of size {size_nm} nm in SIGMA_MPP_BY_SIZE ({name})")
    key = (float(size_nm), SIGMA_MPP_BY_SIZE[float(size_nm)])
    if key not in sigma_table:
        raise KeyError(f"No immobilised-particle sigma for size {size_nm} nm, mpp {key[1]} ({name})")
    if round(float(mpp), 4) != key[1]:
        exclusions.append(_exclusion(name, "sigma_loc_mpp", f"file mpp {round(float(mpp), 4)} -> sigma of mpp {key[1]}",
                                     "", "", f"sigma {sigma_table[key]:.1f} nm of the immobilised particles at "
                                     f"{key[1]} um/px used (E042)"))
    return sigma_table[key]


def _movie_key(xml_path: Path) -> str:
    stem = xml_path.stem.replace("_Tracks", "")
    for marker in ALTERNATIVE_MARKERS:
        stem = stem.replace(marker, "")
    return stem


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def split_alternative_analyses(xml_files: list[Path]) -> tuple[list[Path], dict[Path, tuple[Path, str]]]:
    """E031: (kept analyses, {other analysis: (kept analysis of the same movie, role)}).
    xml_files in folder order; of byte-identical copies the first one is kept."""
    groups: dict[str, list[Path]] = {}
    for x in xml_files:
        groups.setdefault(_movie_key(x), []).append(x)
    kept, others = [], {}
    for key, paths in groups.items():
        if len(paths) == 1:
            kept.append(paths[0])
            continue
        originals = [p for p in paths if not any(m in p.stem for m in ALTERNATIVE_MARKERS)]
        if not originals:
            raise ValueError(f"E031: no original analysis for movie '{key}': {[str(p) for p in paths]}")
        if len({_md5(p) for p in originals}) != 1:
            raise ValueError(f"E031: several different original analyses for movie '{key}': {[str(p) for p in originals]}")
        keep = originals[0]
        kept.append(keep)
        for p in paths:
            if p is keep:
                continue
            others[p] = (keep, "identical copy" if p in originals else "alternative analysis")
    return kept, others


def _exclusion(file: str, step: str, parameter: str, n_before, n_after, reason: str) -> dict:
    return {"file": file, "step": step, "parameter": parameter,
            "n_before": n_before, "n_after": n_after, "reason": reason}


def _file_data_with_rec(xml_path: Path, rec_path: Path) -> dict:
    """single_file_data() with a given .rec (rec_overrides): same parsing and edge-artifact removal."""
    if not rec_path.exists():
        raise FileNotFoundError(f"E006: assigned .rec not found: {rec_path} ({xml_path.name})")
    rec = parse_rec_file(rec_path)
    if rec.get("fps") is None or rec.get("mpp") is None or not rec.get("size_x") or not rec.get("size_y"):
        raise ValueError(f"E006: incomplete calibration in {rec_path} ({xml_path.name}): {rec}")
    tracks_df, stats = remove_edge_artifacts(read_trackmate_xml(xml_path), rec["size_x"], rec["size_y"])
    return _build_result_dict(xml_path=xml_path, tracks_df=tracks_df, mpp=rec["mpp"], fps=rec["fps"],
                              particle_size_nm=extract_particle_size_from_path(xml_path), tif_path=None,
                              rec_path=rec_path, edge_filter_stats=stats)


def analyse_file(xml_path: Path, size_nm: float, sigma_table: dict, exclusions: list,
                 rec_overrides: dict[str, Path]) -> dict | None:
    """Load, filter and fit one file. Returns the result_dict, or None if skipped by the fps rule."""
    name = xml_path.name
    if name in rec_overrides:
        rd = _file_data_with_rec(xml_path, rec_overrides[name])
        exclusions.append(_exclusion(name, "calibration_source", rec_overrides[name].name, "", "",
                                     ".rec with a different name, assigned in the compute script (user)"))
    else:
        rd = single_file_data(xml_path)
    if rd is None:
        raise ValueError(f"E006: missing .rec calibration (fps/mpp) for {xml_path}")
    n_raw = rd["num_tracks"] - rd.get("edge_filter_splits", 0)
    exclusions.append(_exclusion(
        name, "edge_artifacts", "3 % border", n_raw, rd["num_tracks"],
        f"{rd.get('edge_filter_removed', 0)} detections removed, {rd.get('edge_filter_splits', 0)} "
        "extra tracks from splits (counts are tracks; E001 A)"))

    fps = rd["fps"]
    if size_nm in SMALL_SIZES_NM and fps < MIN_FPS_SMALL_SIZES:
        exclusions.append(_exclusion(name, "min_frame_rate", f"fps {fps:.2f} < {MIN_FPS_SMALL_SIZES}",
                                     1, 0, "file skipped (E003 A)"))
        print(f"  [SKIP fps={fps:.2f}<{MIN_FPS_SMALL_SIZES}] {name}")
        return None

    rec = parse_rec_file(Path(rd["rec_path"]))
    if rec.get("exposure_ms") is None:
        raise ValueError(f"E006: no exposure time in {rd['rec_path']} ({name})")
    sigma_nm = _sigma_for(size_nm, rd["mpp"], sigma_table, name, exclusions)

    rd["particle_size_nm"] = size_nm if rd.get("particle_size_nm") is None else rd["particle_size_nm"]
    perform_msd_analysis_blur_locerror(rd, sigma_nm, rec["exposure_ms"],
                                       DRIFT_MIN_TRACKS_PER_FRAME, fit_points=MSD_FIT_POINTS, model=MSD_MODEL)
    rd["exposure_ms"] = rec["exposure_ms"]
    fit = rd["fit_results_MSD"]
    fit["psf_sigma_s0_nm"] = PSF_SIGMA_S0_NM
    fit["delta_dyn_nm"] = calculate_dynamic_localization_error(
        sigma_nm, fit["D_lin_um2_per_s"], rec["exposure_ms"] / 1000.0, PSF_SIGMA_S0_NM)
    fit["delta_dyn2_nm2"] = fit["delta_dyn_nm"] ** 2
    exclusions.append(_exclusion(name, "min_track_length", f"{MIN_TRACK_LENGTH} frames",
                                 rd["num_tracks"], rd["num_tracks_fit"], "tp.filter_stubs (approved)"))
    if rd["drift_subtracted"]:
        exclusions.append(_exclusion(name, "drift_subtraction", f"tracks/frame >= {DRIFT_MIN_TRACKS_PER_FRAME}",
                                     rd["num_tracks_fit"], rd["num_tracks_fit"], "tp.subtract_drift (E009 B)"))
    return rd


def compute_results(xml_folders: dict[float, list[Path]], rec_overrides: dict[str, Path],
                    sigma_table: dict) -> tuple[dict, dict, list, list]:
    """MSD analysis for all XML files: (results, other analyses, exclusion rows, folder rows)."""
    results, others, exclusions, folders_found = {}, {}, [], []

    for size_nm, folders in xml_folders.items():
        xml_files = []
        for folder in folders:
            if not folder.exists():
                print(f"WARNING: folder not found: {folder}")
                folders_found.append({"size_nm": size_nm, "folder": str(folder), "n_xml": 0, "exists": False})
                continue
            found = sorted(folder.glob("*.xml"))
            folders_found.append({"size_nm": size_nm, "folder": str(folder), "n_xml": len(found), "exists": True})
            xml_files += found
        kept, other_map = split_alternative_analyses(xml_files)
        for xml_path in kept:
            rd = analyse_file(xml_path, size_nm, sigma_table, exclusions, rec_overrides)
            if rd is not None:
                results[rd["xml_path"]] = rd
        for other_path, (original, role) in other_map.items():
            exclusions.append(_exclusion(str(other_path), "duplicate_analysis", f"kept {original}", 1, 0,
                                         f"E031: one analysis per movie, original kept; listed as {role} "
                                         "in the per-file table"))
            rd = analyse_file(other_path, size_nm, sigma_table, [], rec_overrides)
            if rd is not None:
                rd["alternative_of"] = str(original)
                rd["role"] = f"{role} (E031)"
                others[rd["xml_path"]] = rd

    print(f"Analyse abgeschlossen: {len(results)} Dateien ausgewertet, {len(others)} weitere Analysen derselben Filme")
    return results, others, exclusions, folders_found


def _per_file_row(rd: dict, role: str) -> dict:
    fit = rd["fit_results_MSD"]
    return {
        "xml_path": rd["xml_path"], "base_name": rd["base_name"], "role": role,
        "alternative_of": rd.get("alternative_of", ""),
        "particle_size_nm": rd["particle_size_nm"], "fps": rd["fps"], "mpp_um_per_px": rd["mpp"],
        "exposure_ms": rd["exposure_ms"], "num_tracks": rd["num_tracks"], "num_tracks_fit": rd["num_tracks_fit"],
        "tracks_per_frame": rd["tracks_per_frame"], "drift_subtracted": rd["drift_subtracted"],
        "sigma_loc_fixed_nm": fit["sigma_loc_nm"],
        "D_um2_per_s": fit["D_um2_per_s"], "D_error": fit["D_error"],
        "exponent": fit["exponent"], "exponent_error": fit["exponent_error"], "r_squared_log": fit["r_squared"],
        "D_uncorrected": fit["D_uncorrected"], "exponent_uncorrected": fit["exponent_uncorrected"],
        "sigma_loc_intercept_nm": fit["sigma_loc_intercept_nm"],
        "D_lin_um2_per_s": fit["D_lin_um2_per_s"],
        "msd_offset_var_nm2": fit["msd_offset_var_nm2"],
        "sigma2_loc_dynamic_nm2": fit["sigma2_loc_dynamic_nm2"],
        "sigma2_loc_static_nm2": fit["sigma2_loc_static_nm2"],
        "sigma2_dynamic_over_static": fit["sigma2_loc_dynamic_nm2"] / fit["sigma2_loc_static_nm2"],
        "psf_sigma_s0_nm": fit["psf_sigma_s0_nm"],
        "delta_dyn_eq46_nm": fit["delta_dyn_nm"],
        "delta_dyn2_eq46_nm2": fit["delta_dyn2_nm2"],
        "4sigma2_over_msd_lag1": fit["offset_over_msd_lag1"],
        "n_fit_lags_msd_below_4sigma2": fit["n_lags_msd_below_offset"],
    }


def sigma2_dynamic_per_size(results: dict, size_nm: float) -> dict:
    """Dynamic vs. static localisation error variance of one size: per-file values,
    weighted by WEIGHT_KEY (as the per-size average, E002 A), plus min and max."""
    rds = [r for r in results.values() if r["particle_size_nm"] == size_nm]
    w = np.array([r[WEIGHT_KEY] for r in rds], dtype=float)
    dyn = np.array([r["fit_results_MSD"]["sigma2_loc_dynamic_nm2"] for r in rds])
    stat = np.array([r["fit_results_MSD"]["sigma2_loc_static_nm2"] for r in rds])
    eq46 = np.array([r["fit_results_MSD"]["delta_dyn2_nm2"] for r in rds])
    return {"delta_dyn2_eq46_nm2_weighted": float(np.average(eq46, weights=w)),
            "delta_dyn2_eq46_nm2_min": float(eq46.min()), "delta_dyn2_eq46_nm2_max": float(eq46.max()),
            "sigma2_loc_dynamic_nm2_weighted": float(np.average(dyn, weights=w)),
            "sigma2_loc_dynamic_nm2_min": float(dyn.min()), "sigma2_loc_dynamic_nm2_max": float(dyn.max()),
            "sigma2_loc_static_nm2_weighted": float(np.average(stat, weights=w))}


def write_tables(table_path: Path, prefix: str, results: dict, others: dict, exclusions: list,
                 avg_results: dict) -> pd.DataFrame:
    table_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exclusions, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        table_path / f"{prefix}_exclusions.csv", index=False, encoding="utf-8-sig")
    rows = [_per_file_row(r, "used") for r in results.values()]
    rows += [_per_file_row(r, r["role"]) for r in others.values()]
    per_file = pd.DataFrame(rows).sort_values(["particle_size_nm", "role", "base_name"])
    per_file.to_csv(table_path / f"{prefix}_per_file.csv", index=False, encoding="utf-8-sig")
    avg_rows = []
    for size_nm, pr in sorted(avg_results.items()):
        fit = pr["fit_results_MSD"]
        dyn = sigma2_dynamic_per_size(results, size_nm)
        avg_rows.append({"particle_size_nm": size_nm, "n_files": pr["n_files"],
                         "sum_num_tracks_fit": pr["n_particles_pooled"],
                         "D_um2_per_s": fit["D_um2_per_s"], "D_error": fit["D_error"],
                         "exponent": fit["exponent"], "exponent_error": fit["exponent_error"],
                         "sigma_loc_fixed_nm": fit["sigma_loc_nm"], "weight": WEIGHT_KEY, **dyn})
    pd.DataFrame(avg_rows).to_csv(table_path / f"{prefix}_weighted_average.csv", index=False, encoding="utf-8-sig")
    print(f"Tabellen gespeichert: {table_path} ({prefix}_*.csv)")
    return per_file


def save_pickle(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Cache gespeichert: {path} ({len(data)} Einträge)")


def test_blur_locerror_fit_synthetic() -> None:
    """Recover known K and n with fit_powerlaw_blur_locerror_general(): (1) noise-free model
    eMSDs, (2) simulated Brownian trajectories and (3) simulated fractional Brownian
    trajectories (Cholesky of the fBm covariance), both with continuous exposure (t_E =
    frame time, average over sub-steps) and Gaussian localisation noise. Prints recovered
    vs. true values; nothing is saved."""
    dt = 1.0 / SYNTH_FPS
    sigma_um = SYNTH_SIGMA_NM / 1000.0
    lags = dt * np.arange(1, MSD_FIT_POINTS + 1)
    print("\n── Synthetic test of the MSD fit with localisation error and motion blur (E046) ──")
    for n_true in (0.5, 0.6, 0.9, 1.0):
        for t_e in (dt, dt / 2):
            msd = msd_blur_locerror_general(lags, SYNTH_D_UM2_S, n_true, sigma_um, t_e)
            fit = fit_powerlaw_blur_locerror_general(pd.Series(msd, index=lags), MSD_FIT_POINTS, sigma_um, t_e)
            print(f"  model eMSD, true K = {SYNTH_D_UM2_S}, n = {n_true}, t_E = {t_e * 1e3:.2f} ms: "
                  f"recovered K = {fit['D']:.6f}, n = {fit['n']:.6f}")

    rng = np.random.default_rng(SYNTH_SEED)
    sub_dt = dt / SYNTH_SUBSTEPS
    steps = rng.normal(0.0, np.sqrt(2 * SYNTH_D_UM2_S * sub_dt),
                       size=(SYNTH_N_TRACKS, SYNTH_N_FRAMES * SYNTH_SUBSTEPS, 2))
    path = np.cumsum(steps, axis=1)
    frames = path.reshape(SYNTH_N_TRACKS, SYNTH_N_FRAMES, SYNTH_SUBSTEPS, 2).mean(axis=2)
    frames += rng.normal(0.0, sigma_um, size=frames.shape)
    mpp = 0.3
    df = pd.DataFrame({
        "frame": np.tile(np.arange(SYNTH_N_FRAMES), SYNTH_N_TRACKS),
        "particle": np.repeat(np.arange(SYNTH_N_TRACKS), SYNTH_N_FRAMES),
        "x": frames[..., 0].ravel() / mpp + 100.0,
        "y": frames[..., 1].ravel() / mpp + 100.0,
    })
    emsd = tp.emsd(df, mpp=mpp, fps=SYNTH_FPS)
    fit = fit_powerlaw_blur_locerror_general(emsd, MSD_FIT_POINTS, sigma_um, dt)
    msd1 = float(emsd.iloc[0])
    print(f"  Brownian trajectories ({SYNTH_N_TRACKS} x {SYNTH_N_FRAMES} frames, sigma = {SYNTH_SIGMA_NM} nm, "
          f"t_E = 1/{SYNTH_FPS:.0f} s, seed {SYNTH_SEED}), 4sigma^2/MSD(lag 1) = {4 * sigma_um ** 2 / msd1:.2f}:")
    print(f"    true D = {SYNTH_D_UM2_S}, n = 1; recovered D = {fit['D']:.4f} ± {fit['D_err']:.4f}, "
          f"n = {fit['n']:.4f} ± {fit['n_err']:.4f}")
    dyn = locerror_from_msd_slope(emsd, MSD_FIT_POINTS, dt)
    print(f"    localisation error from the MSD slope (E042): true sigma^2 = {SYNTH_SIGMA_NM ** 2:.1f} nm^2, "
          f"recovered {dyn['sigma2_um2'] * 1e6:.1f} nm^2 (D_lin = {dyn['D_lin']:.4f}); "
          f"offset variance true {SYNTH_SIGMA_NM ** 2 - SYNTH_D_UM2_S * dt / 3 * 1e6:.1f} nm^2, "
          f"recovered {dyn['offset_var_um2'] * 1e6:.1f} nm^2")
    t = dt / SYNTH_FBM_SUBSTEPS * np.arange(1, SYNTH_FBM_FRAMES * SYNTH_FBM_SUBSTEPS + 1)
    n_f = SYNTH_FBM_N
    cov = SYNTH_D_UM2_S * (t[:, None] ** n_f + t[None, :] ** n_f - np.abs(t[:, None] - t[None, :]) ** n_f)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(len(t)))
    pos = (chol @ rng.standard_normal((len(t), 2 * SYNTH_FBM_TRACKS))).T.reshape(
        SYNTH_FBM_TRACKS, 2, SYNTH_FBM_FRAMES * SYNTH_FBM_SUBSTEPS)
    fr = pos.reshape(SYNTH_FBM_TRACKS, 2, SYNTH_FBM_FRAMES, SYNTH_FBM_SUBSTEPS).mean(axis=3)
    fr += rng.normal(0.0, sigma_um, size=fr.shape)
    msd_fbm = np.array([np.mean(np.sum((fr[:, :, k:] - fr[:, :, :-k]) ** 2, axis=1))
                        for k in range(1, MSD_FIT_POINTS + 1)])
    fit_fbm = fit_powerlaw_blur_locerror_general(pd.Series(msd_fbm, index=lags), MSD_FIT_POINTS, sigma_um, dt)
    print(f"  fractional Brownian trajectories ({SYNTH_FBM_TRACKS} x {SYNTH_FBM_FRAMES} frames, "
          f"{SYNTH_FBM_SUBSTEPS} sub-steps per exposure): true K = {SYNTH_D_UM2_S}, n = {n_f}; recovered "
          f"K = {fit_fbm['D']:.4f} ± {fit_fbm['D_err']:.4f}, n = {fit_fbm['n']:.4f} ± {fit_fbm['n_err']:.4f}")

    s0_um = PSF_SIGMA_S0_NM / 1000.0
    t_e = 0.01
    check = calculate_dynamic_localization_error(100.0, s0_um ** 2 / t_e, t_e, PSF_SIGMA_S0_NM)
    print(f"  eq 4.6 (E043): delta_sta = 100 nm, D t_E = s0^2 -> delta_dyn expected {100 * np.sqrt(2):.4f} nm, "
          f"computed {check:.4f} nm")


def run_pipeline(xml_folders: dict[float, list[Path]], rec_overrides: dict[str, Path], cache_file: Path,
                 result_cache_file: Path, table_path: Path, prefix: str) -> None:
    """Synthetic test, MSD analysis of all files, pickles and tables, console summary."""
    test_blur_locerror_fit_synthetic()

    sigma_table = load_sigma_table()
    # Always recompute from the raw XML files — no cache-reuse.
    results, others, exclusions, folders_found = compute_results(xml_folders, rec_overrides, sigma_table)
    if not results:
        raise RuntimeError("No analyzable files found.")
    save_pickle(cache_file, results)

    avg_results = weighted_average_per_size(results, weight_key=WEIGHT_KEY)
    save_pickle(result_cache_file, avg_results)
    per_file = write_tables(table_path, prefix, results, others, exclusions, avg_results)

    print("\n── Folders ──")
    print(pd.DataFrame(folders_found).to_string(index=False))

    print("\n── Weighted average per particle size (weight = tracks >= 10 frames) ──")
    avg_rows = []
    for size_nm, pr in sorted(avg_results.items()):
        fit = pr["fit_results_MSD"]
        avg_rows.append({
            "Größe (nm)":    int(size_nm),
            "D (µm²/s)":     f"{fit['D_um2_per_s']:.4f}",
            "± D":           f"{fit['D_error']:.4f}",
            "n":             f"{fit['exponent']:.3f}",
            "± n":           f"{fit['exponent_error']:.3f}",
            "σ_lok (nm)":    f"{fit['sigma_loc_nm']:.1f}",
            "σ²_stat (nm²)": f"{fit['sigma_loc_nm'] ** 2:.0f}",
            "σ²_dyn gew. (nm²)": f"{sigma2_dynamic_per_size(results, size_nm)['sigma2_loc_dynamic_nm2_weighted']:.0f}",
            "δ²_dyn Gl.4.6 gew. (nm²)": f"{sigma2_dynamic_per_size(results, size_nm)['delta_dyn2_eq46_nm2_weighted']:.0f}",
            "N Tracks Fit":  pr["n_particles_pooled"],
            "N Dateien":     pr["n_files"],
        })
    print(pd.DataFrame(avg_rows).set_index("Größe (nm)").to_string())

    cols = ["base_name", "role", "particle_size_nm", "num_tracks", "num_tracks_fit", "fps", "exposure_ms",
            "sigma_loc_fixed_nm", "D_uncorrected", "D_um2_per_s", "exponent_uncorrected", "exponent",
            "sigma2_loc_static_nm2", "sigma2_loc_dynamic_nm2", "delta_dyn2_eq46_nm2"]
    print("\nPro Datei (MSD):")
    print(per_file[cols].to_string(index=False))
