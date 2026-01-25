"""Analysis functions for diffusion calculations.
Includes MSD analysis and step size diffusion analysis.
Here wont be stored any data loading or visualization methods."""

from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import trackpy as tp

from .physics import calculate_theoretical_diffusion


# Standard analysis settings for canonical keys in result_dict
DEFAULT_MSD_FIT_POINTS = 6
DEFAULT_STEP_INTERVAL = 1


def _get_tracks_df(data: pd.DataFrame | Dict[str, Any]) -> Optional[pd.DataFrame]:
    if isinstance(data, dict):
        return data.get("tracks_df")
    return data


def calculate_step_sizes(
    data: pd.DataFrame | Dict[str, Any],
    step_interval: int = 1,
    sliding: bool = True,
) -> pd.DataFrame:
    """
    Calculate step sizes from either a tracks DataFrame or a standardized result_dict.
    """
    df = _get_tracks_df(data)
    if df is None or df.empty:
        return pd.DataFrame(columns=["particle", "frame", "dx", "dy", "frame_interval"])

    step_rows = []
    for pid in df["particle"].unique():
        g = df.loc[df["particle"] == pid, ["frame", "x", "y"]].sort_values("frame")
        if len(g) < step_interval + 1:
            continue
        frames = g["frame"].to_numpy()
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()

        idx_iter = range(0, len(x) - step_interval) if sliding else range(0, len(x) - step_interval, step_interval)
        for i in idx_iter:
            j = i + step_interval
            if frames[j] - frames[i] != step_interval:
                continue
            step_rows.append(
                {
                    "particle": pid,
                    "frame": int(frames[i]),
                    "dx": float(x[j] - x[i]),
                    "dy": float(y[j] - y[i]),
                    "frame_interval": int(step_interval),
                }
            )
    return pd.DataFrame(step_rows, columns=["particle", "frame", "dx", "dy", "frame_interval"])


def fit_gaussian_diffusion_1d(
    steps_px: np.ndarray | pd.Series,
    mpp_um_per_px: float,
    fps: float,
    frame_interval: int = 1,
    axis: str = "x",
    max_mean_sigma_ratio: float = 0.3,
) -> dict:
    arr_px = np.asarray(steps_px, dtype=float)
    arr_nm = arr_px * mpp_um_per_px * 1000.0

    mu_nm, sigma_nm = stats.norm.fit(arr_nm) if arr_nm.size else (np.nan, np.nan)

    dt_ms = (1000.0 / float(fps)) * int(frame_interval)
    D_nm2_per_ms = (sigma_nm ** 2) / (2.0 * dt_ms) if np.isfinite(sigma_nm) and dt_ms > 0 else np.nan
    D_um2_per_s = D_nm2_per_ms / 1000.0 if np.isfinite(D_nm2_per_ms) else np.nan

    mean_sigma_ratio = (
        (abs(mu_nm) / sigma_nm)
        if np.isfinite(mu_nm) and np.isfinite(sigma_nm) and sigma_nm > 0
        else np.inf
    )
    is_centered = bool(mean_sigma_ratio <= max_mean_sigma_ratio)

    return {
        "axis": axis,
        "n_steps": int(arr_nm.size),
        "frame_interval": int(frame_interval),
        "fps": float(fps),
        "dt_ms": float(dt_ms),
        "mu_nm": float(mu_nm) if np.isfinite(mu_nm) else np.nan,
        "sigma_nm": float(sigma_nm) if np.isfinite(sigma_nm) else np.nan,
        "D_nm2_per_ms": float(D_nm2_per_ms) if np.isfinite(D_nm2_per_ms) else np.nan,
        "D_um2_per_s": float(D_um2_per_s) if np.isfinite(D_um2_per_s) else np.nan,
        "mean_sigma_ratio": float(mean_sigma_ratio),
        "is_centered": is_centered,
        "quality_issues_1d": ([] if is_centered else [f"drift(|mu|/sigma={mean_sigma_ratio:.3g})"]),
    }


def diffusion_2d_from_1d_fits(
    fit_x: dict,
    fit_y: dict,
    max_sigma_ratio: float = 1.5,
    max_mean_sigma_ratio: float = 0.3,
) -> dict:
    dt_x = float(fit_x.get("dt_ms", np.nan))
    dt_y = float(fit_y.get("dt_ms", np.nan))
    if not (np.isfinite(dt_x) and np.isfinite(dt_y) and abs(dt_x - dt_y) < 1e-9):
        return {"quality_flag": "inconsistent_dt", "D_um2_per_s": np.nan}

    sx = float(fit_x.get("sigma_nm", np.nan))
    sy = float(fit_y.get("sigma_nm", np.nan))
    mux = float(fit_x.get("mu_nm", np.nan))
    muy = float(fit_y.get("mu_nm", np.nan))

    D_x_nm2_per_ms = (sx ** 2) / (2.0 * dt_x) if np.isfinite(sx) and sx > 0 else np.nan
    D_y_nm2_per_ms = (sy ** 2) / (2.0 * dt_x) if np.isfinite(sy) and sy > 0 else np.nan

    D_2d_nm2_per_ms = np.nanmean([D_x_nm2_per_ms, D_y_nm2_per_ms])
    D_2d_um2_per_s = D_2d_nm2_per_ms / 1000.0 if np.isfinite(D_2d_nm2_per_ms) else np.nan
    D_dir_disagreement_um2_per_s = (
        (abs(D_x_nm2_per_ms - D_y_nm2_per_ms) / 2.0) / 1000.0
        if np.isfinite(D_x_nm2_per_ms) and np.isfinite(D_y_nm2_per_ms)
        else np.nan
    )

    sigma_ratio = (
        (max(sx, sy) / min(sx, sy))
        if np.isfinite(sx) and np.isfinite(sy) and min(sx, sy) > 0
        else np.inf
    )
    mean_x_ratio = (abs(mux) / sx) if np.isfinite(mux) and np.isfinite(sx) and sx > 0 else np.inf
    mean_y_ratio = (abs(muy) / sy) if np.isfinite(muy) and np.isfinite(sy) and sy > 0 else np.inf

    is_isotropic = bool(sigma_ratio <= max_sigma_ratio)
    is_centered = bool(mean_x_ratio <= max_mean_sigma_ratio and mean_y_ratio <= max_mean_sigma_ratio)

    if is_isotropic and is_centered:
        quality_flag = "good"
    elif (not is_isotropic) and (not is_centered):
        quality_flag = "both"
    elif not is_isotropic:
        quality_flag = "anisotropic"
    else:
        quality_flag = "drift"

    return {
        "D_um2_per_s": float(D_2d_um2_per_s) if np.isfinite(D_2d_um2_per_s) else np.nan,
        "D_x_um2_per_s": float(D_x_nm2_per_ms / 1000.0) if np.isfinite(D_x_nm2_per_ms) else np.nan,
        "D_y_um2_per_s": float(D_y_nm2_per_ms / 1000.0) if np.isfinite(D_y_nm2_per_ms) else np.nan,
        "D_dir_disagreement_um2_per_s": float(D_dir_disagreement_um2_per_s) if np.isfinite(D_dir_disagreement_um2_per_s) else np.nan,
        "sigma_x_nm": float(sx) if np.isfinite(sx) else np.nan,
        "sigma_y_nm": float(sy) if np.isfinite(sy) else np.nan,
        "mu_x_nm": float(mux) if np.isfinite(mux) else np.nan,
        "mu_y_nm": float(muy) if np.isfinite(muy) else np.nan,
        "sigma_ratio": float(sigma_ratio),
        "mean_x_ratio": float(mean_x_ratio),
        "mean_y_ratio": float(mean_y_ratio),
        "is_isotropic": is_isotropic,
        "is_centered": is_centered,
        "quality_flag": quality_flag,
        "dt_ms": float(dt_x),
        "frame_interval": int(fit_x.get("frame_interval", 1)),
        "fps": float(fit_x.get("fps", np.nan)),
        "n_steps_x": int(fit_x.get("n_steps", 0)),
        "n_steps_y": int(fit_y.get("n_steps", 0)),
    }


def fit_powerlaw_with_errors(
    em_series: pd.Series,
    points: int = 10,
    ax=None,
    plot: bool = False,
) -> Dict[str, Any]:
    """Fit power-law model y = A * x^n to ensemble MSD data."""
    xs = em_series.iloc[0:points].index.values.astype(float)
    ys = em_series.iloc[0:points].values.astype(float)

    lx = np.log(xs)
    ly = np.log(ys)
    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)

    n_fit = float(coeffs[0])
    logA_fit = float(coeffs[1])

    se = np.sqrt(np.diag(cov))
    se_n = float(se[0])
    se_logA = float(se[1])

    A_fit = float(np.exp(logA_fit))
    se_A = A_fit * se_logA

    return {
        "A": np.array([A_fit]),
        "n": np.array([n_fit]),
        "A_err": np.array([se_A]),
        "n_err": np.array([se_n]),
        "logA": np.array([logA_fit]),
        "logA_err": np.array([se_logA]),
        "cov": cov,
    }


def perform_msd_analysis(result_dict: Dict[str, Any], fit_points: int = DEFAULT_MSD_FIT_POINTS) -> Dict[str, Any]:
    """
    Perform MSD analysis on tracks from result_dict.
    Updates result_dict in-place. Default fit_points are stored under
    D_MSD / fit_results_MSD; non-default fit_points are stored under
    fit_MSD_fp_{fit_points}.
    """
    tracks = result_dict["tracks_df"]
    mpp = result_dict["mpp"]
    fps = result_dict["fps"]

    if tracks is None or tracks.empty:
        result_dict["D_MSD"] = None
        result_dict["fit_results_MSD"] = None
        return result_dict
    if len(tracks["particle"].unique())/len(tracks["frame"].unique()) > 50:
        tracks_driftless = tp.subtract_drift(tracks)
    else:
        tracks_driftless = tracks.copy()

    imsd = tp.imsd(tracks_driftless, mpp=mpp, fps=fps)
    emsd = tp.emsd(tracks_driftless, mpp=mpp, fps=fps)

    fit_result = fit_powerlaw_with_errors(emsd, points=fit_points)

    D = fit_result["A"][0] / 4.0
    D_err = fit_result["A_err"][0] / 4.0
    n = fit_result["n"][0]
    n_err = fit_result["n_err"][0]

    fit_payload = {
        "D_um2_per_s": D,
        "D_error": D_err,
        "exponent": n,
        "exponent_error": n_err,
        "A": fit_result["A"][0],
        "A_err": fit_result["A_err"][0],
        "logA": fit_result["logA"][0],
        "logA_err": fit_result["logA_err"][0],
        "cov": fit_result["cov"],
        "imsd": imsd,
        "emsd": emsd,
        "fit_points": fit_points,
    }

    if fit_points == DEFAULT_MSD_FIT_POINTS:
        result_dict["D_MSD"] = D
        result_dict["fit_results_MSD"] = fit_payload
    else:
        result_dict[f"fit_MSD_fp_{fit_points}"] = fit_payload

    return result_dict


def perform_stepsize_analysis(
    result_dict: Dict[str, Any],
    step_interval: int = DEFAULT_STEP_INTERVAL,
    sliding: bool = True,
) -> Dict[str, Any]:
    """
    Perform step size diffusion analysis on tracks from result_dict.
    Updates result_dict in-place. Default step_interval is stored under
    D_step / fit_results_step; non-default step_interval is stored under
    fit_step_si_{step_interval}.
    """
    tracks = result_dict["tracks_df"]
    mpp = result_dict["mpp"]
    fps = result_dict["fps"]

    if tracks is None or tracks.empty:
        result_dict["D_step"] = None
        result_dict["fit_results_step"] = None
        return result_dict

    step_df = calculate_step_sizes(result_dict, step_interval=step_interval, sliding=sliding)

    if step_df.empty:
        result_dict["D_step"] = None
        result_dict["fit_results_step"] = None
        return result_dict

    fit_x = fit_gaussian_diffusion_1d(step_df["dx"], mpp, fps, frame_interval=step_interval, axis="x")
    fit_y = fit_gaussian_diffusion_1d(step_df["dy"], mpp, fps, frame_interval=step_interval, axis="y")
    fit_2d = diffusion_2d_from_1d_fits(fit_x, fit_y)

    D = fit_2d["D_um2_per_s"]

    fit_payload = {
        "D_um2_per_s": D,
        "D_x_um2_per_s": fit_2d["D_x_um2_per_s"],
        "D_y_um2_per_s": fit_2d["D_y_um2_per_s"],
        "D_dir_disagreement_um2_per_s": fit_2d["D_dir_disagreement_um2_per_s"],
        "sigma_x_nm": fit_2d["sigma_x_nm"],
        "sigma_y_nm": fit_2d["sigma_y_nm"],
        "mu_x_nm": fit_2d["mu_x_nm"],
        "mu_y_nm": fit_2d["mu_y_nm"],
        "sigma_ratio": fit_2d["sigma_ratio"],
        "mean_x_ratio": fit_2d["mean_x_ratio"],
        "mean_y_ratio": fit_2d["mean_y_ratio"],
        "is_isotropic": fit_2d["is_isotropic"],
        "is_centered": fit_2d["is_centered"],
        "quality_flag": fit_2d["quality_flag"],
        "dt_ms": fit_2d["dt_ms"],
        "frame_interval": fit_2d["frame_interval"],
        "fps": fit_2d["fps"],
        "n_steps_x": fit_2d["n_steps_x"],
        "n_steps_y": fit_2d["n_steps_y"],
        "fit_x": fit_x,
        "fit_y": fit_y,
        "step_df": step_df,
    }

    if step_interval == DEFAULT_STEP_INTERVAL:
        result_dict["D_step"] = D
        result_dict["fit_results_step"] = fit_payload
    else:
        result_dict[f"fit_step_si_{step_interval}"] = fit_payload

    return result_dict


def analyze_multiple_files(
    results_dict: Dict[str, Dict[str, Any]],
    analysis_type: str = "both",
    fit_points: int = DEFAULT_MSD_FIT_POINTS,
    step_interval: int = DEFAULT_STEP_INTERVAL,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple files using MSD and/or step size methods.
    Updates all result_dicts in results_dict in-place.
    """
    total = len(results_dict)
    print(f"\nAnalyzing {total} files...")
    print("=" * 70)

    for idx, (xml_path, result_dict) in enumerate(results_dict.items(), 1):
        base_name = result_dict["base_name"]
        print(f"\n[{idx}/{total}] Processing: {base_name}")

        if analysis_type in ["msd", "both"]:
            try:
                perform_msd_analysis(result_dict, fit_points=fit_points)
                if result_dict["D_MSD"] is not None:
                    print(f"  MSD: D = {result_dict['D_MSD']:.4e} um^2/s")
            except Exception as e:
                print(f"  MSD analysis failed: {e}")
                result_dict["D_MSD"] = None
                result_dict["fit_results_MSD"] = None

        if analysis_type in ["stepsize", "both"]:
            try:
                perform_stepsize_analysis(result_dict, step_interval=step_interval)
                if result_dict["D_step"] is not None:
                    quality = result_dict["fit_results_step"]["quality_flag"]
                    print(f"  StepSize: D = {result_dict['D_step']:.4e} um^2/s (quality: {quality})")
            except Exception as e:
                print(f"  StepSize analysis failed: {e}")
                result_dict["D_step"] = None
                result_dict["fit_results_step"] = None

    print("\n" + "=" * 70)
    print(f"Analysis complete for {total} files")
    print("=" * 70)

    return results_dict


# ============================================================================
# DEPRECATED FUNCTIONS (kept for backwards compatibility)
# ============================================================================

def analyze_all_files(files_df: pd.DataFrame) -> pd.DataFrame:
    print("WARNING: analyze_all_files() is deprecated. Use analyze_multiple_files() instead.")
    return pd.DataFrame()


def analyze_single_file(xml_path: str, mpp: float, fps: float) -> Optional[Dict[str, Any]]:
    print("WARNING: analyze_single_file() is deprecated. Use standardized workflow instead.")
    return None
