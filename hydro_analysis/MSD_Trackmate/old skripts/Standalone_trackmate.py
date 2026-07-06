"""
Standalone TrackMate XML Analysis
==================================
Komplettes eigenständiges Skript – benötigt keine anderen Projektdateien.

Externe Abhängigkeiten (pip-installierbar):
    numpy, pandas, matplotlib, scipy, trackpy, chardet

Analyseschritte:
    1) MSD-Analyse (individual + ensemble)
    2) Schrittgrößen-Diffusionsanalyse (Gauss-Fit)
    3) Plots: MSD, Schrittgrößen, Overlay, dx/dy-Verteilungen, Theorienvergleich

Konfiguration: Abschnitt „CONFIGURATION" weiter unten anpassen.
"""

from __future__ import annotations

# ── Standardbibliothek ───────────────────────────────────────────────────────
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ── Drittanbieter ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
from scipy import stats
import trackpy as tp

try:
    import chardet
    _CHARDET_AVAILABLE = True
except ImportError:
    _CHARDET_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  –  hier anpassen
# ════════════════════════════════════════════════════════════════════════════

DATA_DIR   = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.02.17\Tracks")
OUTPUT_DIR = DATA_DIR.parent / "analysis_results"

MSD_FIT_POINTS = 6   # Anzahl Punkte für den Ensemble-MSD-Fit
STEP_INTERVAL  = 3   # Frame-Intervall für Schrittgrößenanalyse


# ════════════════════════════════════════════════════════════════════════════
# PHYSIKALISCHE KONSTANTEN  (aus core/physics.py)
# ════════════════════════════════════════════════════════════════════════════

_TEMPERATURE_K   = 293.15
_VISCOSITY_PA_S  = 0.001002
_BOLTZMANN       = 1.380649e-23  # J/K


def calculate_theoretical_diffusion(
    particle_size_nm: float,
    temperature: float = _TEMPERATURE_K,
    viscosity: float = _VISCOSITY_PA_S,
) -> float:
    """Stokes-Einstein: D = k_B T / (6π η r) → µm²/s."""
    r_m = (particle_size_nm * 1e-9) / 2.0
    D_m2s = _BOLTZMANN * temperature / (6.0 * np.pi * viscosity * r_m)
    return D_m2s * 1e12


# ════════════════════════════════════════════════════════════════════════════
# IO-FUNKTIONEN  (aus core/io.py)
# ════════════════════════════════════════════════════════════════════════════

def _check_encoding(path: Path) -> str:
    """Datei-Encoding bestimmen (chardet wenn verfügbar, sonst utf-8)."""
    if not _CHARDET_AVAILABLE:
        return "utf-8"
    with open(path, "rb") as f:
        result = chardet.detect(f.read())
    return result.get("encoding") or "utf-8"


def _parse_rec_file(rec_path: Path) -> Dict[str, Any]:
    """PCO CamWare .rec-Datei parsen → fps, mpp."""
    result: Dict[str, Any] = {
        "exposure_ms": None, "delay_ms": None,
        "fps": None, "size_x": None, "size_y": None, "mpp": None,
    }
    if not rec_path.exists():
        return result
    encoding = _check_encoding(rec_path)
    try:
        content = rec_path.read_text(encoding=encoding, errors="replace")

        m = re.search(
            r"Exposure\s*/\s*Delay\s*:\s*([\d.]+)\s*ms\s*/\s*([\d.]+)\s*ms",
            content, re.IGNORECASE,
        )
        if m:
            exposure = float(m.group(1))
            delay    = float(m.group(2))
            result["exposure_ms"] = exposure
            result["delay_ms"]    = delay
            total = exposure + delay
            if total > 0:
                result["fps"] = 1000.0 / total

        sm = re.search(
            r"Picture\s+Size\s+horz\.?/vert\.?\s*:\s*(\d+)\s*/\s*(\d+)",
            content, re.IGNORECASE,
        )
        if sm:
            x = int(sm.group(1))
            y = int(sm.group(2))
            result["size_x"] = x
            result["size_y"] = y
            # µm/px kalibriert via Thorlab-Grid 10 µm
            if x == 200:
                result["mpp"] = 0.30
            elif x == 400:
                result["mpp"] = 0.15
            elif x == 696:
                result["mpp"] = 0.149
            elif x == 696 * 2:
                result["mpp"] = 0.149 / 2.0
    except Exception as e:
        print(f"    Warning: konnte {rec_path.name} nicht lesen: {e}")
    return result


def _find_rec_tif_files(xml_path: Path) -> Dict[str, Any]:
    """Sucht zugehörige .rec- und .tif-Dateien zur XML und liest Kalibrierung."""
    xml_dir  = xml_path.parent.parent
    xml_stem = xml_path.stem
    for suffix in ("_Tracks", "_processed", "_var", "filtered_", "Resultof"):
        xml_stem = xml_stem.replace(suffix, "")

    rec_path = xml_dir / f"{xml_stem}.tif.rec"
    if not rec_path.exists():
        rec_path = xml_dir / f"{xml_stem}.rec"
    if not rec_path.exists():
        print(f"     ERROR: Kein .rec für {xml_path.name} (gesucht: {xml_stem})")

    tif_path = xml_dir / f"{xml_stem}.tif"
    rec_info = _parse_rec_file(rec_path)

    return {
        "fps":      rec_info.get("fps"),
        "mpp":      rec_info.get("mpp"),
        "tiff_file": tif_path if tif_path.exists() else None,
        "rec_file":  rec_path if rec_path.exists() else None,
    }


def _read_trackmate_xml(xml_file_path: Path) -> Optional[pd.DataFrame]:
    """TrackMate-XML parsen → DataFrame [frame, particle, x, y]."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        rows = []
        for pid, particle in enumerate(root.findall("particle"), 1):
            for det in particle.findall("detection"):
                rows.append({
                    "frame":    int(float(det.get("t"))),
                    "particle": pid,
                    "x":        float(det.get("x")),
                    "y":        float(det.get("y")),
                })
        if not rows:
            return pd.DataFrame(columns=["frame", "particle", "x", "y"])
        df = pd.DataFrame(rows)[["frame", "particle", "x", "y"]]
        return df.sort_values(["particle", "frame"]).reset_index(drop=True)
    except Exception as e:
        print(f"Error beim Parsen von {xml_file_path}: {e}")
        return None


def _extract_particle_size_from_path(folder_path: Path) -> Optional[float]:
    """Partikelgröße (nm) aus Pfad/Dateiname extrahieren."""
    candidates: List[str] = []
    if folder_path.is_file():
        candidates.append(folder_path.stem)
    else:
        candidates += [
            folder_path.name,
            folder_path.parent.name,
            folder_path.parent.parent.name,
        ]

    standard_sizes = [1000, 500, 200, 100, 50, 20]
    for name in candidates:
        for size in standard_sizes:
            if re.search(rf"{size}", name, re.IGNORECASE):
                return float(size)
        m = re.search(r"(\d+)", name)
        if m:
            return float(m.group(1))
    return None


def _build_result_dict(
    xml_path: Path,
    tracks_df: Optional[pd.DataFrame],
    mpp: float,
    fps: float,
    particle_size_nm: Optional[float],
    tif_path: Optional[Path],
    rec_path: Optional[Path],
) -> Dict[str, Any]:
    num_tracks = tracks_df["particle"].nunique() if tracks_df is not None else 0
    num_frames = (
        int(tracks_df["frame"].max()) + 1
        if tracks_df is not None and not tracks_df.empty
        else 0
    )
    return {
        "xml_path":        str(xml_path),
        "tif_path":        str(tif_path) if tif_path else None,
        "rec_path":        str(rec_path) if rec_path else None,
        "base_name":       xml_path.name,
        "tracks_df":       tracks_df,
        "mpp":             mpp,
        "fps":             fps,
        "particle_size_nm": particle_size_nm,
        "num_tracks":      num_tracks,
        "num_frames":      num_frames,
        "D_MSD":           None,
        "D_step":          None,
        "fit_results_MSD": None,
        "fit_results_step": None,
    }


def single_file_data(xml_path: Path) -> Optional[Dict[str, Any]]:
    """Lädt eine XML-Datei + zugehöriges .rec → standardisiertes result_dict."""
    particle_size = _extract_particle_size_from_path(xml_path)
    calib         = _find_rec_tif_files(xml_path)
    tracks_df     = _read_trackmate_xml(xml_path)
    fps, mpp      = calib["fps"], calib["mpp"]

    if fps is None or mpp is None:
        print(f"     ERROR: Kalibrierung fehlt für {xml_path.name} (fps={fps}, mpp={mpp})")
        print(f"     Überspringe {xml_path.name}")
        return None

    return _build_result_dict(
        xml_path=xml_path,
        tracks_df=tracks_df,
        mpp=mpp,
        fps=fps,
        particle_size_nm=particle_size,
        tif_path=calib["tiff_file"],
        rec_path=calib["rec_file"],
    )


# ════════════════════════════════════════════════════════════════════════════
# ANALYSE-FUNKTIONEN  (aus core/analysis.py)
# ════════════════════════════════════════════════════════════════════════════

def _calculate_step_sizes(
    data: Dict[str, Any],
    step_interval: int = 1,
    sliding: bool = True,
) -> pd.DataFrame:
    df = data.get("tracks_df")
    if df is None or df.empty:
        return pd.DataFrame(columns=["particle", "frame", "dx", "dy", "frame_interval"])

    rows = []
    for pid in df["particle"].unique():
        g = df.loc[df["particle"] == pid, ["frame", "x", "y"]].sort_values("frame")
        if len(g) < step_interval + 1:
            continue
        frames = g["frame"].to_numpy()
        x      = g["x"].to_numpy()
        y      = g["y"].to_numpy()
        idx_iter = (
            range(0, len(x) - step_interval)
            if sliding
            else range(0, len(x) - step_interval, step_interval)
        )
        for i in idx_iter:
            j = i + step_interval
            if frames[j] - frames[i] != step_interval:
                continue
            rows.append({
                "particle":       pid,
                "frame":          int(frames[i]),
                "dx":             float(x[j] - x[i]),
                "dy":             float(y[j] - y[i]),
                "frame_interval": int(step_interval),
            })
    return pd.DataFrame(rows, columns=["particle", "frame", "dx", "dy", "frame_interval"])


def _fit_gaussian_1d(
    steps_px,
    mpp_um_per_px: float,
    fps: float,
    frame_interval: int = 1,
    axis: str = "x",
    max_mean_sigma_ratio: float = 0.3,
) -> dict:
    arr_px  = np.asarray(steps_px, dtype=float)
    arr_nm  = arr_px * mpp_um_per_px * 1000.0
    mu_nm, sigma_nm = (
        stats.norm.fit(arr_nm) if arr_nm.size else (np.nan, np.nan)
    )
    dt_ms       = (1000.0 / float(fps)) * int(frame_interval)
    D_nm2_per_ms = (sigma_nm**2) / (2.0 * dt_ms) if np.isfinite(sigma_nm) and dt_ms > 0 else np.nan
    D_um2_per_s  = D_nm2_per_ms / 1000.0 if np.isfinite(D_nm2_per_ms) else np.nan
    msr = (
        abs(mu_nm) / sigma_nm
        if np.isfinite(mu_nm) and np.isfinite(sigma_nm) and sigma_nm > 0
        else np.inf
    )
    is_centered = bool(msr <= max_mean_sigma_ratio)
    return {
        "axis":            axis,
        "n_steps":         int(arr_nm.size),
        "frame_interval":  int(frame_interval),
        "fps":             float(fps),
        "dt_ms":           float(dt_ms),
        "mu_nm":           float(mu_nm) if np.isfinite(mu_nm) else np.nan,
        "sigma_nm":        float(sigma_nm) if np.isfinite(sigma_nm) else np.nan,
        "D_nm2_per_ms":    float(D_nm2_per_ms) if np.isfinite(D_nm2_per_ms) else np.nan,
        "D_um2_per_s":     float(D_um2_per_s) if np.isfinite(D_um2_per_s) else np.nan,
        "mean_sigma_ratio": float(msr),
        "is_centered":     is_centered,
        "quality_issues_1d": [] if is_centered else [f"drift(|mu|/sigma={msr:.3g})"],
    }


def _diffusion_2d_from_1d(
    fit_x: dict,
    fit_y: dict,
    max_sigma_ratio: float = 1.5,
    max_mean_sigma_ratio: float = 0.3,
) -> dict:
    dt_x = float(fit_x.get("dt_ms", np.nan))
    dt_y = float(fit_y.get("dt_ms", np.nan))
    if not (np.isfinite(dt_x) and np.isfinite(dt_y) and abs(dt_x - dt_y) < 1e-9):
        return {"quality_flag": "inconsistent_dt", "D_um2_per_s": np.nan}

    sx  = float(fit_x.get("sigma_nm", np.nan))
    sy  = float(fit_y.get("sigma_nm", np.nan))
    mux = float(fit_x.get("mu_nm", np.nan))
    muy = float(fit_y.get("mu_nm", np.nan))

    Dx_nm2ms = (sx**2) / (2.0 * dt_x) if np.isfinite(sx) and sx > 0 else np.nan
    Dy_nm2ms = (sy**2) / (2.0 * dt_x) if np.isfinite(sy) and sy > 0 else np.nan
    D2d_nm2ms = np.nanmean([Dx_nm2ms, Dy_nm2ms])
    D2d       = D2d_nm2ms / 1000.0 if np.isfinite(D2d_nm2ms) else np.nan

    sigma_ratio  = (max(sx, sy) / min(sx, sy)) if np.isfinite(sx) and np.isfinite(sy) and min(sx, sy) > 0 else np.inf
    mx_ratio     = abs(mux) / sx if np.isfinite(mux) and np.isfinite(sx) and sx > 0 else np.inf
    my_ratio     = abs(muy) / sy if np.isfinite(muy) and np.isfinite(sy) and sy > 0 else np.inf
    is_isotropic = bool(sigma_ratio <= max_sigma_ratio)
    is_centered  = bool(mx_ratio <= max_mean_sigma_ratio and my_ratio <= max_mean_sigma_ratio)

    if is_isotropic and is_centered:
        quality_flag = "good"
    elif not is_isotropic and not is_centered:
        quality_flag = "both"
    elif not is_isotropic:
        quality_flag = "anisotropic"
    else:
        quality_flag = "drift"

    return {
        "D_um2_per_s":                float(D2d) if np.isfinite(D2d) else np.nan,
        "D_x_um2_per_s":             float(Dx_nm2ms / 1000.0) if np.isfinite(Dx_nm2ms) else np.nan,
        "D_y_um2_per_s":             float(Dy_nm2ms / 1000.0) if np.isfinite(Dy_nm2ms) else np.nan,
        "D_dir_disagreement_um2_per_s": float(abs(Dx_nm2ms - Dy_nm2ms) / 2000.0) if np.isfinite(Dx_nm2ms) and np.isfinite(Dy_nm2ms) else np.nan,
        "sigma_x_nm":  float(sx) if np.isfinite(sx) else np.nan,
        "sigma_y_nm":  float(sy) if np.isfinite(sy) else np.nan,
        "mu_x_nm":     float(mux) if np.isfinite(mux) else np.nan,
        "mu_y_nm":     float(muy) if np.isfinite(muy) else np.nan,
        "sigma_ratio": float(sigma_ratio),
        "is_isotropic": is_isotropic,
        "is_centered":  is_centered,
        "quality_flag": quality_flag,
        "dt_ms":        float(dt_x),
        "frame_interval": int(fit_x.get("frame_interval", 1)),
        "fps":          float(fit_x.get("fps", np.nan)),
        "n_steps_x":    int(fit_x.get("n_steps", 0)),
        "n_steps_y":    int(fit_y.get("n_steps", 0)),
    }


def _fit_powerlaw(em_series: pd.Series, points: int = 10) -> Dict[str, Any]:
    """Potenzgesetz y = A·x^n auf Ensemble-MSD fitten."""
    xs  = em_series.iloc[:points].index.values.astype(float)
    ys  = em_series.iloc[:points].values.astype(float)
    lx  = np.log(xs)
    ly  = np.log(ys)
    coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
    n_fit   = float(coeffs[0])
    logA    = float(coeffs[1])
    se      = np.sqrt(np.diag(cov))
    A_fit   = float(np.exp(logA))
    se_A    = A_fit * float(se[1])
    return {
        "A":       np.array([A_fit]),
        "n":       np.array([n_fit]),
        "A_err":   np.array([se_A]),
        "n_err":   np.array([float(se[0])]),
        "logA":    np.array([logA]),
        "logA_err": np.array([float(se[1])]),
        "cov":     cov,
    }


def perform_msd_analysis(result_dict: Dict[str, Any], fit_points: int = 6) -> Dict[str, Any]:
    """MSD-Analyse; schreibt D_MSD und fit_results_MSD in result_dict."""
    tracks = result_dict["tracks_df"]
    mpp    = result_dict["mpp"]
    fps    = result_dict["fps"]

    if tracks is None or tracks.empty:
        result_dict["D_MSD"] = None
        result_dict["fit_results_MSD"] = None
        return result_dict

    if len(tracks["particle"].unique()) / len(tracks["frame"].unique()) > 50:
        tracks_use = tp.subtract_drift(tracks)
    else:
        tracks_use = tracks.copy()

    imsd = tp.imsd(tracks_use, mpp=mpp, fps=fps)
    emsd = tp.emsd(tracks_use, mpp=mpp, fps=fps)
    fit  = _fit_powerlaw(emsd, points=fit_points)

    D     = float(fit["A"][0]) / 4.0
    D_err = float(fit["A_err"][0]) / 4.0
    n     = float(fit["n"][0])
    n_err = float(fit["n_err"][0])

    payload = {
        "D_um2_per_s":     D,
        "D_error":         D_err,
        "exponent":        n,
        "exponent_error":  n_err,
        "A":               float(fit["A"][0]),
        "A_err":           float(fit["A_err"][0]),
        "logA":            float(fit["logA"][0]),
        "logA_err":        float(fit["logA_err"][0]),
        "cov":             fit["cov"],
        "imsd":            imsd,
        "emsd":            emsd,
        "fit_points":      fit_points,
    }

    result_dict["D_MSD"] = D
    result_dict["fit_results_MSD"] = payload
    return result_dict


def perform_stepsize_analysis(
    result_dict: Dict[str, Any],
    step_interval: int = 3,
    sliding: bool = True,
) -> Dict[str, Any]:
    """Schrittgrößen-Diffusionsanalyse; schreibt D_step und fit_results_step."""
    tracks = result_dict["tracks_df"]
    mpp    = result_dict["mpp"]
    fps    = result_dict["fps"]

    if tracks is None or tracks.empty:
        result_dict["D_step"] = None
        result_dict["fit_results_step"] = None
        return result_dict

    step_df = _calculate_step_sizes(result_dict, step_interval=step_interval, sliding=sliding)
    if step_df.empty:
        result_dict["D_step"] = None
        result_dict["fit_results_step"] = None
        return result_dict

    fit_x  = _fit_gaussian_1d(step_df["dx"], mpp, fps, frame_interval=step_interval, axis="x")
    fit_y  = _fit_gaussian_1d(step_df["dy"], mpp, fps, frame_interval=step_interval, axis="y")
    fit_2d = _diffusion_2d_from_1d(fit_x, fit_y)
    D      = fit_2d["D_um2_per_s"]

    payload = {
        "D_um2_per_s":                fit_2d["D_um2_per_s"],
        "D_x_um2_per_s":             fit_2d["D_x_um2_per_s"],
        "D_y_um2_per_s":             fit_2d["D_y_um2_per_s"],
        "D_dir_disagreement_um2_per_s": fit_2d["D_dir_disagreement_um2_per_s"],
        "sigma_x_nm":  fit_2d["sigma_x_nm"],
        "sigma_y_nm":  fit_2d["sigma_y_nm"],
        "mu_x_nm":     fit_2d["mu_x_nm"],
        "mu_y_nm":     fit_2d["mu_y_nm"],
        "sigma_ratio": fit_2d["sigma_ratio"],
        "is_isotropic": fit_2d["is_isotropic"],
        "is_centered":  fit_2d["is_centered"],
        "quality_flag": fit_2d["quality_flag"],
        "dt_ms":        fit_2d["dt_ms"],
        "frame_interval": fit_2d["frame_interval"],
        "fps":          fit_2d["fps"],
        "n_steps_x":    fit_2d["n_steps_x"],
        "n_steps_y":    fit_2d["n_steps_y"],
        "fit_x":        fit_x,
        "fit_y":        fit_y,
        "step_df":      step_df,
    }

    result_dict["D_step"] = D
    result_dict["fit_results_step"] = payload
    return result_dict


# ════════════════════════════════════════════════════════════════════════════
# VISUALISIERUNGS-FUNKTIONEN  (aus core/visualization.py)
# ════════════════════════════════════════════════════════════════════════════

def _normalize_results(
    results: Dict[str, Dict[str, Any]] | Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if isinstance(results, dict):
        return [r for r in results.values() if r is not None]
    return [r for r in results if r is not None]


def plot_msd_results(result_dict: Dict[str, Any], save_path: Path = None) -> None:
    """MSD-Plot mit individual- und Ensemble-Kurven."""
    if result_dict["fit_results_MSD"] is None:
        print("Keine MSD-Ergebnisse vorhanden.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    res     = result_dict["fit_results_MSD"]
    imsd    = res["imsd"]
    emsd    = res["emsd"]
    A       = res["A"]
    n       = res["exponent"]
    size    = result_dict.get("particle_size_nm")

    for col in imsd.columns[:50]:
        ax.loglog(imsd.index, imsd[col], alpha=0.1, color="gray")

    sem = imsd.std(axis=1) / np.sqrt(imsd.shape[1])
    ax.errorbar(
        emsd.index, emsd.values, yerr=sem.values,
        fmt="o-", label=f'Ensemble MSD (N={result_dict["num_tracks"]})',
        linewidth=2, markersize=4, color="blue",
        capsize=3, capthick=1, elinewidth=1, alpha=0.8,
    )

    x_fit = np.logspace(np.log10(emsd.index[0]), np.log10(emsd.index[5]), 100)
    ax.loglog(x_fit, A * x_fit**n, "--",
              label=f"Fit: A={A:.3f}±{res['A_err']:.3f}, n={n:.3f}±{res['exponent_error']:.3f}",
              linewidth=2, color="red")

    if size is not None:
        D_thr = calculate_theoretical_diffusion(size)
        ax.loglog(x_fit, 4.0 * D_thr * x_fit, "g:",
                  label=f"Theorie (d={size:.0f} nm)", linewidth=2)

    D     = res["D_um2_per_s"]
    D_err = res["D_error"]
    ax.set_xlabel("Lag Time (s)", fontsize=12)
    ax.set_ylabel("MSD (µm²)", fontsize=12)
    ax.set_title(
        f"MSD Analysis\nD = {D:.4f} ± {D_err:.4f} µm²/s\n"
        f"Exponent n = {n:.3f} ± {res['exponent_error']:.3f}",
        fontsize=14,
    )
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  MSD plot: {save_path}")


def plot_stepsize_results(result_dict: Dict[str, Any], save_path: Path = None) -> None:
    """Schrittgrößen-Histogramm mit Gauss-Fit."""
    if result_dict["fit_results_step"] is None:
        print("Keine Schrittgrößen-Ergebnisse vorhanden.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    step_res = result_dict["fit_results_step"]
    fit_x    = step_res["fit_x"]
    fit_y    = step_res["fit_y"]
    step_df  = step_res["step_df"]
    mpp      = result_dict["mpp"]

    dx_um = step_df["dx"].values * mpp
    dy_um = step_df["dy"].values * mpp

    ax.hist(dx_um, bins=50, density=True, alpha=0.6, color="blue",  edgecolor="black", label="X")
    ax.hist(dy_um, bins=50, density=True, alpha=0.6, color="green", edgecolor="black", label="Y")

    mu_x  = fit_x["mu_nm"]  / 1000.0
    sig_x = fit_x["sigma_nm"] / 1000.0
    mu_y  = fit_y["mu_nm"]  / 1000.0
    sig_y = fit_y["sigma_nm"] / 1000.0
    xvals = np.linspace(
        min(dx_um.min(), dy_um.min()),
        max(dx_um.max(), dy_um.max()),
        200,
    )
    if np.isfinite(sig_x) and sig_x > 0:
        ax.plot(xvals, stats.norm.pdf(xvals, mu_x, sig_x), "b-", linewidth=2.5,
                label=f"Gauss X: µ={mu_x:.4f}, σ={sig_x:.4f} µm")
    if np.isfinite(sig_y) and sig_y > 0:
        ax.plot(xvals, stats.norm.pdf(xvals, mu_y, sig_y), "g-", linewidth=2.5,
                label=f"Gauss Y: µ={mu_y:.4f}, σ={sig_y:.4f} µm")

    dt_s = step_res["dt_ms"] / 1000.0
    Dx   = step_res["D_x_um2_per_s"]
    Dy   = step_res["D_y_um2_per_s"]
    ax.set_xlabel("Displacement (µm)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(
        f"Step Size (Δt={dt_s:.3f} s)\n"
        f"D_x={Dx:.4f}, D_y={Dy:.4f} µm²/s  |  N={step_res['n_steps_x']}",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Step size plot: {save_path}")


def plot_step_size_overlay(
    results: Dict[str, Dict[str, Any]] | Iterable[Dict[str, Any]],
    save_path: Path,
    step_interval: int = 1,
) -> None:
    """Overlay-Histogramm der Schrittgrößen aller Dateien."""
    print("\nErstelle Schrittgrößen-Overlay …")
    results_list = _normalize_results(results)
    fig, ax = plt.subplots(figsize=(14, 10))
    colors  = plt.cm.tab20(np.linspace(0, 1, min(len(results_list), 20)))

    for idx, result in enumerate(results_list):
        mpp       = result.get("mpp")
        size      = result.get("particle_size_nm")
        base_name = result.get("base_name", "unknown")
        step_res  = result.get("fit_results_step")
        step_df   = step_res.get("step_df") if step_res else None
        if step_df is None or step_df.empty or mpp is None:
            continue

        dx_nm = step_df["dx"].values * mpp * 1000.0
        dy_nm = step_df["dy"].values * mpp * 1000.0
        r_nm  = np.sqrt(dx_nm**2 + dy_nm**2)

        label = (
            f"{size:.0f} nm - {base_name[:30]} (N={len(r_nm)}, mean={np.mean(r_nm):.1f} nm)"
            if size is not None
            else f"? - {base_name[:30]}"
        )
        ax.hist(r_nm, bins=60, alpha=0.4, color=colors[idx % len(colors)],
                edgecolor="black", linewidth=0.3, label=label)

    ax.set_xlabel("Step size (nm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency",      fontsize=14, fontweight="bold")
    ax.set_title("Step Size Distribution – alle Dateien", fontsize=16, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "step_size_overlay.png", dpi=300)
    plt.close(fig)
    print(f"[OK] Overlay gespeichert: {save_path / 'step_size_overlay.png'}")


def plot_dx_dy_distributions(
    results: Dict[str, Dict[str, Any]] | Iterable[Dict[str, Any]],
    save_path: Path = None,
    step_interval: int = 1,
) -> None:
    """dx/dy-Verteilungen (2×2 Subplot) für jede Datei."""
    print("\nErstelle dx/dy-Verteilungsplots …")
    for result in _normalize_results(results):
        mpp      = result.get("mpp")
        size     = result.get("particle_size_nm")
        xml_name = result.get("base_name", "unknown")
        step_res = result.get("fit_results_step")
        step_df  = step_res.get("step_df") if step_res else None
        if step_df is None or step_df.empty or mpp is None:
            continue

        dx_nm = step_df["dx"].values * mpp * 1000.0
        dy_nm = step_df["dy"].values * mpp * 1000.0

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax1 = axes[0, 0]
        ax1.hist(dx_nm, bins=50, alpha=0.7, color="cornflowerblue", edgecolor="black")
        ax1.axvline(np.mean(dx_nm), color="red", linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(dx_nm):.1f} nm")
        ax1.set_xlabel("dx (nm)"); ax1.set_ylabel("Frequency")
        ax1.set_title("X-Displacement"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.hist(dy_nm, bins=50, alpha=0.7, color="lightcoral", edgecolor="black")
        ax2.axvline(np.mean(dy_nm), color="red", linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(dy_nm):.1f} nm")
        ax2.set_xlabel("dy (nm)"); ax2.set_ylabel("Frequency")
        ax2.set_title("Y-Displacement"); ax2.legend(); ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        h = ax3.hist2d(dx_nm, dy_nm, bins=50, cmap="Blues", cmin=1)
        plt.colorbar(h[3], ax=ax3, label="Counts")
        ax3.axhline(0, color="red", linewidth=1, alpha=0.5)
        ax3.axvline(0, color="red", linewidth=1, alpha=0.5)
        ax3.set_xlabel("dx (nm)"); ax3.set_ylabel("dy (nm)")
        ax3.set_title("2D-Verteilung"); ax3.set_aspect("equal"); ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        n_pts = len(dx_nm)
        if n_pts > 5000:
            idx = np.random.choice(n_pts, 5000, replace=False)
            dx_p, dy_p, alp = dx_nm[idx], dy_nm[idx], 0.1
        else:
            dx_p, dy_p, alp = dx_nm, dy_nm, 0.3
        ax4.scatter(dx_p, dy_p, s=10, alpha=alp, c="navy")
        ax4.axhline(0, color="red", linewidth=1, alpha=0.5)
        ax4.axvline(0, color="red", linewidth=1, alpha=0.5)
        mean_r = np.mean(np.sqrt(dx_nm**2 + dy_nm**2))
        circle = plt.Circle((0, 0), mean_r, color="red", fill=False,
                             linestyle="--", linewidth=2, label=f"Mean r={mean_r:.1f} nm")
        ax4.add_patch(circle)
        ax4.set_xlabel("dx (nm)"); ax4.set_ylabel("dy (nm)")
        ax4.set_title("Scatter"); ax4.set_aspect("equal")
        ax4.legend(); ax4.grid(True, alpha=0.3)

        size_label = f"{size:.0f} nm" if size is not None else "unknown"
        fig.suptitle(
            f"Displacement – {size_label}\n{xml_name}\n"
            f"N={len(dx_nm)}, σ(dx)={np.std(dx_nm):.1f} nm, σ(dy)={np.std(dy_nm):.1f} nm",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        if save_path is not None:
            safe = xml_name.replace(".xml", "").replace(" ", "_").replace("/", "_")
            slug = f"{size:.0f}nm" if size is not None else "unknown"
            plt.savefig(save_path / f"dx_dy_{slug}_{safe}.png", dpi=300)
            plt.close(fig)

    print(f"[OK] dx/dy-Plots gespeichert: {save_path}")


def plot_theory_comparison(
    summary_df: pd.DataFrame,
    save_path: Path,
    size_override: dict[float, float] | None = None,
    size_err:      dict[float, float] | None = None,
    dls_override:  dict[float, float] | None = None,
    dls_err:       dict[float, float] | None = None,
) -> None:
    """Gemessene vs. theoretische Diffusionskoeffizienten (log/log)."""
    COLOR_MSD_BASE  = "#0000da"
    COLOR_MSD_DARK  = "#000099"
    COLOR_STEP_BASE = "#da0000"
    COLOR_STEP_DARK = "#990000"
    COLOR_DLS       = "#da00bd"
    COLOR_DLS_DARK  = "#9b5191"
    COLOR_THEORY    = "black"

    DLS_FALLBACK = {
        20: 12.388, 50: 8.202, 100: 4.139,
        200: 1.745, 500: 0.622, 1000: 0.357,
    }
    dls_values = dls_override if dls_override else DLS_FALLBACK

    def _map(s):
        return float(size_override[s]) if size_override and s in size_override else float(s)
    def _serr(s):
        return float(size_err[s]) if size_err and s in size_err else 0.0

    df = summary_df.dropna(subset=["particle_size_nm"]).copy()
    if df.empty:
        print("Kein Datensatz für Theorienvergleich.")
        return

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    ax.tick_params(axis="both", which="major", direction="in", length=5.0,
                   width=1.2, top=True, right=True, labelsize=10)
    ax.tick_params(axis="both", which="minor", direction="in", length=2.5,
                   width=1.0, top=True, right=True)

    x_min, x_max = 15, 1500
    x_range = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    ax.plot(x_range, [calculate_theoretical_diffusion(s) for s in x_range],
            color=COLOR_THEORY, linewidth=2.2, linestyle="--", zorder=5)

    off_l, off_r = 0.90, 1.10
    msd_stats, step_stats = [], []

    if "D_MSD_um2_per_s" in df.columns:
        for size in sorted(df["particle_size_nm"].unique()):
            sub  = df[df["particle_size_nm"] == size].dropna(subset=["D_MSD_um2_per_s"])
            vals = sub["D_MSD_um2_per_s"].values
            if vals.size:
                x_off = _map(float(size)) * off_l
                ax.scatter([x_off] * len(vals), vals, s=50, alpha=0.6,
                           facecolors=COLOR_MSD_BASE, edgecolors="none", zorder=2)
                n_p = int(sub["weight"].fillna(0).sum()) if "weight" in sub.columns else len(vals)
                msd_stats.append((float(size), np.mean(vals), np.std(vals), n_p))

    if "D_stepsize_um2_per_s" in df.columns:
        for size in sorted(df["particle_size_nm"].unique()):
            sub  = df[df["particle_size_nm"] == size].dropna(subset=["D_stepsize_um2_per_s"])
            vals = sub["D_stepsize_um2_per_s"].values
            if vals.size:
                x_off = _map(float(size)) * off_r
                ax.scatter([x_off] * len(vals), vals, s=50, alpha=0.6, marker="s",
                           facecolors=COLOR_STEP_BASE, edgecolors="none", zorder=2)
                n_p = int(sub["weight"].fillna(0).sum()) if "weight" in sub.columns else len(vals)
                step_stats.append((float(size), np.mean(vals), np.std(vals), n_p))

    if msd_stats:
        ax.errorbar(
            [_map(s[0]) * off_l for s in msd_stats],
            [s[1] for s in msd_stats],
            yerr=[s[2] for s in msd_stats],
            xerr=[_serr(s[0]) if _serr(s[0]) > 0.1 else 0.0 for s in msd_stats],
            fmt="o", markersize=12, markerfacecolor=COLOR_MSD_BASE,
            markeredgecolor=COLOR_MSD_DARK, markeredgewidth=1.2,
            ecolor=COLOR_MSD_DARK, elinewidth=1.8, capsize=5.0, capthick=1.8, zorder=10,
        )

    if step_stats:
        ax.errorbar(
            [_map(s[0]) * off_r for s in step_stats],
            [s[1] for s in step_stats],
            yerr=[s[2] for s in step_stats],
            xerr=[_serr(s[0]) if _serr(s[0]) > 0.1 else 0.0 for s in step_stats],
            fmt="s", markersize=12, markerfacecolor=COLOR_STEP_BASE,
            markeredgecolor=COLOR_STEP_DARK, markeredgewidth=1.2,
            ecolor=COLOR_STEP_DARK, elinewidth=1.8, capsize=5.0, capthick=1.8, zorder=10,
        )

    # DLS-Referenzwerte
    std_sizes = [20, 50, 100, 200, 500, 1000]
    dls_x  = [_map(float(s)) for s in std_sizes if s in dls_values]
    dls_D  = [dls_values[s] for s in std_sizes if s in dls_values]
    dls_xe = [_serr(float(s)) if _serr(float(s)) > 0.1 else 0.0 for s in std_sizes if s in dls_values]
    dls_ye = [dls_err.get(float(s), 0.0) for s in std_sizes if s in dls_values] if dls_err else None
    if dls_x:
        ax.errorbar(dls_x, dls_D, xerr=dls_xe, yerr=dls_ye,
                    fmt="*", markersize=14,
                    markerfacecolor=COLOR_DLS, markeredgecolor=COLOR_DLS_DARK,
                    markeredgewidth=1.0, ecolor=COLOR_DLS_DARK,
                    elinewidth=1.2, capsize=3.0, capthick=1.2, zorder=8)

    # Theoretische Punkte (×)
    th_x  = [_map(float(s)) for s in std_sizes]
    th_D  = [calculate_theoretical_diffusion(x) for x in th_x]
    th_xe = [_serr(float(s)) if _serr(float(s)) > 0.1 else 0.0 for s in std_sizes]
    ax.errorbar(th_x, th_D, xerr=th_xe, fmt="x", markersize=10,
                markeredgewidth=2.0, color=COLOR_THEORY, ecolor=COLOR_THEORY,
                elinewidth=1.2, capsize=3.0, capthick=1.2, linestyle="None", zorder=6)

    legend_elements = [
        Line2D([0],[0], color=COLOR_THEORY, linewidth=2.2, linestyle="--",
               label="Stokes-Einstein Theorie"),
        Line2D([0],[0], marker="x", color=COLOR_THEORY, markersize=10,
               markeredgewidth=2.0, label="Theor. D", linestyle="None"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=COLOR_DLS,
               markeredgecolor=COLOR_DLS_DARK, markersize=14, markeredgewidth=1.0,
               label="DLS Referenz", linestyle="None"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=COLOR_MSD_BASE,
               markeredgecolor=COLOR_MSD_DARK, markersize=12, markeredgewidth=1.2,
               label="MSD (Mean ± SD)", linestyle="None"),
        Line2D([0],[0], marker="o", color=COLOR_MSD_BASE, markersize=8, alpha=0.6,
               label="MSD Einzelmessung", linestyle="None"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_STEP_BASE,
               markeredgecolor=COLOR_STEP_DARK, markersize=12, markeredgewidth=1.2,
               label="Schrittgröße (Mean ± SD)", linestyle="None"),
        Line2D([0],[0], marker="s", color=COLOR_STEP_BASE, markersize=8, alpha=0.6,
               label="Schrittgröße Einzelmessung", linestyle="None"),
    ]

    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
    ax.xaxis.set_major_formatter(FixedFormatter(["20", "50", "100", "200", "500", "1000"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("Particle Size (nm)", fontsize=12)
    ax.set_ylabel("Diffusion Coefficient (µm²/s)", fontsize=12)
    ax.set_title("Measured vs Theoretical Diffusion", fontsize=13, fontweight="semibold")
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, frameon=False)

    for size, mean, std, n in msd_stats:
        y_pos = mean / (1 + std / mean + 0.4) if mean > 0 and std > 0 else mean * 0.6
        ax.annotate(f"n={n}", xy=(_map(size) * off_l, y_pos),
                    fontsize=9, ha="center", va="top", color=COLOR_MSD_DARK)
    for size, mean, std, n in step_stats:
        y_pos = mean / (1 + std / mean + 0.4) if mean > 0 and std > 0 else mean * 0.6
        ax.annotate(f"n={n}", xy=(_map(size) * off_r, y_pos),
                    fontsize=9, ha="center", va="top", color=COLOR_STEP_DARK)

    plt.show()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Theorienvergleich gespeichert: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# HAUPTPROGRAMM
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not DATA_DIR.exists():
        print(f"Ordner nicht gefunden: {DATA_DIR}")
        return

    xml_files = sorted(DATA_DIR.glob("*.xml"))
    if not xml_files:
        print(f"Keine XML-Dateien in {DATA_DIR}")
        return

    print(f"Gefundene XML-Dateien: {len(xml_files)}")

    # ── Laden ───────────────────────────────────────────────────────────────
    results_dict: dict[str, dict] = {}
    for xml_file in xml_files:
        print("  Verarbeite:", xml_file.name)
        result = single_file_data(xml_file)
        if result is None:
            continue
        results_dict[result["xml_path"]] = result

    if not results_dict:
        print("Keine validen Datensätze gefunden (Kalibrierung fehlt?).")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Analyse ─────────────────────────────────────────────────────────────
    for result in results_dict.values():
        perform_msd_analysis(result,       fit_points=MSD_FIT_POINTS)
        perform_stepsize_analysis(result,  step_interval=STEP_INTERVAL)
        print(f"  • {result['base_name']}  |  mpp={result['mpp']} µm/px, fps={result['fps']:.1f} Hz")

    # ── Pro-Datei-Plots ─────────────────────────────────────────────────────
    for result in results_dict.values():
        base = Path(result["base_name"]).stem
        plot_msd_results(result,      save_path=OUTPUT_DIR / f"{base}_msd.png")
        plot_stepsize_results(result, save_path=OUTPUT_DIR / f"{base}_stepsize.png")

    # ── Gesamt-Plots ────────────────────────────────────────────────────────
    plot_step_size_overlay(results_dict,    OUTPUT_DIR, step_interval=STEP_INTERVAL)
    plot_dx_dy_distributions(results_dict, OUTPUT_DIR, step_interval=STEP_INTERVAL)

    # ── Zusammenfassung + Theorienvergleich ─────────────────────────────────
    summary_rows = []
    for result in results_dict.values():
        tracks_df  = result.get("tracks_df")
        n_particles = (
            len(tracks_df["particle"].unique())
            if tracks_df is not None and "particle" in tracks_df.columns
            else result.get("num_tracks", 0)
        )
        summary_rows.append({
            "xml_path":           result["xml_path"],
            "base_name":          result["base_name"],
            "particle_size_nm":   result.get("particle_size_nm"),
            "D_MSD_um2_per_s":   result.get("D_MSD"),
            "D_stepsize_um2_per_s": result.get("D_step"),
            "weight":             n_particles,
        })

    summary_df  = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR / "summary_results.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary CSV: {summary_csv}")

    theory_plot = OUTPUT_DIR / "diffusion_vs_theory.png"
    plot_theory_comparison(summary_df, theory_plot)

    print(f"\n[OK] Analyse abgeschlossen. Ergebnisse in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
