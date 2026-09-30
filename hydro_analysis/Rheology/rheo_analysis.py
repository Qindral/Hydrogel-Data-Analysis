"""
Numerical analysis of oscillatory shear rheology (parallel plate): moduli
from raw instrument channels, point-level artifact flags, detection of the
weakly frequency-dependent G' region of frequency sweeps, evaluation at a
target frequency, linear-viscoelastic-region (LVR) limits and G'/G''
crossover of amplitude sweeps and replicate statistics.
Here wont be stored any data loading or visualization methods.

Moduli follow the TRIOS definition for a parallel plate of radius R and gap h:
stress constant K_tau = 2 / (pi R^3), strain constant K_gamma = R / h, and the
measured complex torque is corrected for the rotational inertia I of
instrument + geometry, M_sample = M e^{i delta_raw} + I omega^2 theta. Then
G* = K_tau M_sample / (K_gamma theta). This reproduces the TRIOS-exported
G' and G'' of the analysed files to better than 0.2 %.

All acceptance criteria are explicit dataclass fields and are set by the
calling script; none of the defaults is meant as a universal physical limit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ── Derived oscillation quantities ───────────────────────────────────────────

def oscillation_quantities(raw: pd.DataFrame, plate_diameter_m: float, inertia_Nms2: float) -> pd.DataFrame:
    """Return G', G'', tan(delta), strain, stress and inertia share for every point.

    raw needs angular_frequency_rad_s, gap_m, raw_phase_rad, osc_torque_Nm and
    osc_displacement_rad (SI). The stress amplitude is |G*| * gamma_0, i.e. the
    inertia-corrected sample stress.
    """
    radius = plate_diameter_m / 2.0
    omega = raw["angular_frequency_rad_s"].to_numpy(float)
    theta = raw["osc_displacement_rad"].to_numpy(float)
    torque = raw["osc_torque_Nm"].to_numpy(float)
    phase = raw["raw_phase_rad"].to_numpy(float)
    gap = raw["gap_m"].to_numpy(float)

    k_tau = 2.0 / (np.pi * radius ** 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_gamma = radius / gap
        inertia_torque = inertia_Nms2 * omega ** 2 * theta
        m_sample = torque * np.exp(1j * phase) + inertia_torque
        g_star = k_tau * m_sample / (k_gamma * theta)
        strain = theta * k_gamma
        out = pd.DataFrame({
            "frequency_Hz": omega / (2.0 * np.pi),
            "angular_frequency_rad_s": omega,
            "strain_pct": strain * 100.0,
            "stress_Pa": np.abs(g_star) * strain,
            "Gp_Pa": g_star.real,
            "Gpp_Pa": g_star.imag,
            "complex_modulus_Pa": np.abs(g_star),
            "tan_delta": g_star.imag / g_star.real,
            "phase_angle_deg": np.degrees(np.angle(g_star)),
            "inertia_torque_ratio": inertia_torque / np.abs(torque),
        })
    return out


# ── Point-level artifact flags ───────────────────────────────────────────────

@dataclass
class PointCriteria:
    """Conditions under which a single sweep point is excluded from fitting (not deleted)."""

    min_torque_Nm: float = 5e-8          # below: torque resolution-limited
    max_inertia_ratio: float = 1.0       # I w^2 theta / |M| above: inertia-dominated signal
    max_raw_phase_deg: Optional[float] = None
    duplicate_rel_tol: float = 1e-3      # relative tolerance for duplicated x values


def point_artifacts(df: pd.DataFrame, raw: pd.DataFrame, x_column: str, crit: PointCriteria) -> pd.Series:
    """Return a ';'-joined reason string per point; empty string for valid points."""
    reasons: List[List[str]] = [[] for _ in range(len(df))]
    gp, gpp = df["Gp_Pa"].to_numpy(), df["Gpp_Pa"].to_numpy()
    torque = np.abs(raw["osc_torque_Nm"].to_numpy())
    phase_deg = np.degrees(raw["raw_phase_rad"].to_numpy())
    ratio = df["inertia_torque_ratio"].to_numpy()
    x = df[x_column].to_numpy()
    for i in range(len(df)):
        if not (np.isfinite(gp[i]) and np.isfinite(gpp[i])) or gp[i] <= 0 or gpp[i] <= 0:
            reasons[i].append("non-positive or non-finite modulus")
        if torque[i] < crit.min_torque_Nm:
            reasons[i].append("torque below minimum")
        if np.isfinite(ratio[i]) and ratio[i] > crit.max_inertia_ratio:
            reasons[i].append("inertia-dominated")
        if crit.max_raw_phase_deg is not None and phase_deg[i] > crit.max_raw_phase_deg:
            reasons[i].append("raw phase above limit")
        earlier = x[:i]
        if len(earlier) and np.any(np.isclose(earlier, x[i], rtol=crit.duplicate_rel_tol)):
            reasons[i].append("duplicated x value")
    return pd.Series([";".join(r) for r in reasons], index=df.index, dtype=object)


# ── Frequency sweep: weakly frequency-dependent region ──────────────────────

@dataclass
class PlateauCriteria:
    """Acceptance criteria for a contiguous window of the log G' - log f relation."""

    max_abs_slope: float = 0.10              # |m| in log10 G' = a + m log10 f
    min_points: int = 5
    max_rmse_log10: float = 0.02             # residual scatter (decades); 0.02 = about 4.7 %
    max_cv: float = 0.20                     # SD / mean of G' inside the window
    min_fraction_gp_gt_gpp: float = 1.0      # fraction of window points with G' > G''
    min_r2: Optional[float] = None           # not used by default: R^2 -> 0 for an ideal plateau
    exclude_low_points: int = 0              # manual trimming at the low-frequency end
    exclude_high_points: int = 0             # manual trimming at the high-frequency end


def loglog_linear_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """OLS fit of log10 y = a + m log10 x with standard errors, R^2 and residual RMSE."""
    lx, ly = np.log10(x), np.log10(y)
    n = len(lx)
    slope, intercept = np.polyfit(lx, ly, 1)
    residuals = ly - (intercept + slope * lx)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    dof = n - 2
    s2 = ss_res / dof if dof > 0 else np.nan
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_se": float(np.sqrt(s2 / sxx)) if dof > 0 and sxx > 0 else np.nan,
        "intercept_se": float(np.sqrt(s2 * (1.0 / n + lx.mean() ** 2 / sxx))) if dof > 0 and sxx > 0 else np.nan,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rmse_log10": float(np.sqrt(s2)) if dof > 0 else np.nan,
        "n": n,
    }


def _window_statistics(f: np.ndarray, gp: np.ndarray, gpp: np.ndarray) -> Dict[str, float]:
    fit = loglog_linear_fit(f, gp)
    sd = float(np.std(gp, ddof=1))
    mean = float(np.mean(gp))
    return {
        **fit,
        "f_min_Hz": float(f.min()),
        "f_max_Hz": float(f.max()),
        "span_decades": float(np.log10(f.max() / f.min())),
        "Gp_mean_Pa": mean,
        "Gp_median_Pa": float(np.median(gp)),
        "Gp_sd_Pa": sd,
        "Gp_cv": sd / mean,
        "Gpp_mean_Pa": float(np.mean(gpp)),
        "tan_delta_mean": float(np.mean(gpp / gp)),
        "fraction_gp_gt_gpp": float(np.mean(gp > gpp)),
    }


def _failed_criteria(s: Dict[str, float], crit: PlateauCriteria) -> List[str]:
    failed = []
    if abs(s["slope"]) > crit.max_abs_slope:
        failed.append(f"|m|={abs(s['slope']):.3f}>{crit.max_abs_slope}")
    if not s["rmse_log10"] <= crit.max_rmse_log10:
        failed.append(f"rmse={s['rmse_log10']:.3f}>{crit.max_rmse_log10}")
    if s["Gp_cv"] > crit.max_cv:
        failed.append(f"CV={s['Gp_cv']:.3f}>{crit.max_cv}")
    if s["fraction_gp_gt_gpp"] < crit.min_fraction_gp_gt_gpp:
        failed.append(f"G'>G'' fraction={s['fraction_gp_gt_gpp']:.2f}<{crit.min_fraction_gp_gt_gpp}")
    if crit.min_r2 is not None and not s["r2"] >= crit.min_r2:
        failed.append(f"R2={s['r2']:.3f}<{crit.min_r2}")
    return failed


def find_plateau_window(
    f: np.ndarray, gp: np.ndarray, gpp: np.ndarray, valid: np.ndarray, crit: PlateauCriteria
) -> Dict[str, object]:
    """Largest contiguous window of valid points that fulfils all PlateauCriteria.

    Inputs must be sorted by frequency. Invalid points break contiguity. Among
    accepted windows the one with most points wins, then the widest span in
    decades, then the smallest |m|, then the smallest residual scatter. If no
    window is accepted, the least-sloped window with min_points points is
    returned with found=False and the failed criteria listed.
    Indices refer to positions in the sorted input arrays (end inclusive).
    """
    n = len(f)
    usable = valid.copy()
    if crit.exclude_low_points:
        usable[np.flatnonzero(valid)[:crit.exclude_low_points]] = False
    if crit.exclude_high_points:
        idx_valid = np.flatnonzero(usable)
        usable[idx_valid[len(idx_valid) - crit.exclude_high_points:]] = False

    segments, start = [], None
    for i in range(n + 1):
        if i < n and usable[i]:
            start = i if start is None else start
        elif start is not None:
            segments.append((start, i - 1))
            start = None

    best, best_key, fallback, fallback_key = None, None, None, None
    for seg_start, seg_end in segments:
        for i in range(seg_start, seg_end + 1):
            for j in range(i + crit.min_points - 1, seg_end + 1):
                s = _window_statistics(f[i:j + 1], gp[i:j + 1], gpp[i:j + 1])
                s.update(i_start=i, i_end=j)
                failed = _failed_criteria(s, crit)
                if not failed:
                    key = (s["n"], s["span_decades"], -abs(s["slope"]), -s["rmse_log10"])
                    if best_key is None or key > best_key:
                        best, best_key = s, key
                elif j - i + 1 == crit.min_points:
                    key = (-abs(s["slope"]), -s["rmse_log10"])
                    if fallback_key is None or key > fallback_key:
                        fallback, fallback_key = {**s, "failed_criteria": "; ".join(failed)}, key
    if best is not None:
        return {**best, "found": True, "failed_criteria": ""}
    if fallback is not None:
        return {**fallback, "found": False}
    return {"found": False, "failed_criteria": f"fewer than {crit.min_points} contiguous valid points", "n": 0}


def evaluate_at_frequency(
    f: np.ndarray, gp: np.ndarray, valid: np.ndarray, window: Dict[str, object],
    target_Hz: float, tol_decades: float,
) -> Dict[str, object]:
    """Measured G' closest to target (within tol_decades) and window-fit G' at target.

    The fitted value is only returned when target lies inside the selected
    window [f_min, f_max]; there is no extrapolation.
    """
    out: Dict[str, object] = {
        "Gp_target_measured_Pa": np.nan, "f_target_measured_Hz": np.nan,
        "Gp_target_fit_Pa": np.nan, "target_inside_window": False,
    }
    idx = np.flatnonzero(valid)
    if len(idx):
        dist = np.abs(np.log10(f[idx] / target_Hz))
        k = idx[np.argmin(dist)]
        if dist.min() <= tol_decades:
            out["Gp_target_measured_Pa"] = float(gp[k])
            out["f_target_measured_Hz"] = float(f[k])
    if window.get("found") and window["f_min_Hz"] <= target_Hz <= window["f_max_Hz"]:
        out["target_inside_window"] = True
        out["Gp_target_fit_Pa"] = float(10 ** (window["intercept"] + window["slope"] * np.log10(target_Hz)))
    return out


def interp_loglog(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Linear interpolation in (log10 x, log10 y); NaN outside [min x, max x] (no extrapolation).

    x must be sorted ascending and x, y positive.
    """
    x_new = np.atleast_1d(np.asarray(x_new, float))
    out = np.full(x_new.shape, np.nan)
    if len(x) < 2:
        return out
    inside = (x_new >= x.min()) & (x_new <= x.max())
    out[inside] = 10 ** np.interp(np.log10(x_new[inside]), np.log10(x), np.log10(y))
    return out


def max_log_jump(y: np.ndarray, valid: np.ndarray) -> float:
    """Largest |delta log10 y| between consecutive valid points (discontinuity indicator)."""
    v = y[valid]
    if len(v) < 2:
        return np.nan
    return float(np.max(np.abs(np.diff(np.log10(v)))))


# ── Amplitude sweep: LVR and crossover ───────────────────────────────────────

@dataclass
class LVRCriteria:
    n_ref_points: int = 4                          # lowest-strain valid points forming the G' reference
    deviation_levels: Sequence[float] = (0.05, 0.10)
    n_consecutive: int = 2                         # points that must exceed the level in a row
    max_ref_cv: float = 0.05                       # reference plateau CV above this is flagged


def _interp_log(x0: float, x1: float, y0: float, y1: float, y: float) -> float:
    """x at which the linear interpolation in (log10 x, y) reaches y."""
    if y1 == y0:
        return float(x1)
    frac = (y - y0) / (y1 - y0)
    return float(10 ** (np.log10(x0) + frac * (np.log10(x1) - np.log10(x0))))


def _loglog_at(x: np.ndarray, y: np.ndarray, x_query: float) -> float:
    return float(10 ** np.interp(np.log10(x_query), np.log10(x), np.log10(y)))


def amplitude_sweep_analysis(
    strain_pct: np.ndarray, gp: np.ndarray, gpp: np.ndarray, stress_Pa: np.ndarray,
    valid: np.ndarray, crit: LVRCriteria,
) -> Dict[str, object]:
    """LVR limits (deviation of G' from the low-strain reference) and G'=G'' crossover.

    Inputs must be sorted by strain. For each deviation level L the reported
    strain is the log-interpolated strain at which |G'/G'_ref - 1| first
    reaches L, provided that n_consecutive points from there on exceed L; the
    last strain still inside the tolerance is reported as well. The crossover
    is the first change from G' > G'' to G'' >= G'; strain, modulus and stress
    are interpolated in log-log coordinates. No yield criterion is applied.
    """
    x, g1, g2, tau = strain_pct[valid], gp[valid], gpp[valid], stress_Pa[valid]
    out: Dict[str, object] = {"n_valid_points": int(len(x))}
    if len(x) < crit.n_ref_points + 1:
        out["error"] = f"fewer than {crit.n_ref_points + 1} valid points"
        return out

    ref = g1[:crit.n_ref_points]
    g_ref = float(np.mean(ref))
    out.update({
        "Gp_ref_Pa": g_ref,
        "Gp_ref_cv": float(np.std(ref, ddof=1) / g_ref),
        "Gpp_ref_Pa": float(np.mean(g2[:crit.n_ref_points])),
        "tan_delta_ref": float(np.mean(g2[:crit.n_ref_points] / ref)),
        "ref_strain_max_pct": float(x[crit.n_ref_points - 1]),
        "strain_min_pct": float(x.min()),
        "strain_max_pct": float(x.max()),
    })

    dev = np.abs(g1 / g_ref - 1.0)
    for level in crit.deviation_levels:
        tag = f"{int(round(level * 100))}pct"
        found = None
        for k in range(crit.n_ref_points, len(x)):
            if np.all(dev[k:k + crit.n_consecutive] > level):
                found = k
                break
        if found is None:
            out.update({f"strain_{tag}_pct": np.nan, f"stress_{tag}_Pa": np.nan,
                        f"last_strain_within_{tag}_pct": float(x[-1]), f"direction_{tag}": "not reached"})
            continue
        gamma = _interp_log(x[found - 1], x[found], dev[found - 1], dev[found], level)
        out.update({
            f"strain_{tag}_pct": gamma,
            f"stress_{tag}_Pa": _loglog_at(x, tau, gamma),
            f"last_strain_within_{tag}_pct": float(x[found - 1]),
            f"direction_{tag}": "stiffening" if g1[found] > g_ref else "softening",
        })

    diff = np.log10(g1) - np.log10(g2)
    out.update({"crossover_strain_pct": np.nan, "crossover_stress_Pa": np.nan,
                "crossover_modulus_Pa": np.nan, "crossover_status": "not reached in measured range"})
    if diff[0] <= 0:
        out["crossover_status"] = "G'' >= G' already at the lowest strain"
    else:
        for k in range(1, len(x)):
            if diff[k] <= 0 < diff[k - 1]:
                gamma = _interp_log(x[k - 1], x[k], diff[k - 1], diff[k], 0.0)
                out.update({
                    "crossover_strain_pct": gamma,
                    "crossover_stress_Pa": _loglog_at(x, tau, gamma),
                    "crossover_modulus_Pa": _loglog_at(x, g1, gamma),
                    "crossover_status": "found",
                })
                break
    return out


# ── Replicate statistics ─────────────────────────────────────────────────────

def describe(values: Sequence[float]) -> Dict[str, float]:
    """N, mean, SD (ddof=1), CV, min and max of the finite values."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    n = len(v)
    mean = float(v.mean()) if n else np.nan
    sd = float(v.std(ddof=1)) if n > 1 else np.nan
    return {"N": n, "mean": mean, "sd": sd, "cv": sd / mean if n > 1 and mean else np.nan,
            "min": float(v.min()) if n else np.nan, "max": float(v.max()) if n else np.nan}
