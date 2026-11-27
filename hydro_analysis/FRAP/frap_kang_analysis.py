"""
FRAP, D after Kang et al. 2012 (Traffic 13:1589, "Simplified equation to extract diffusion coefficients
from confocal FRAP data"), eq. 5:  D = (r_e^2 + r_n^2) / (8 tau_1/2).
Requested by the user, 2026-11-26: Kang version as an alternative to the propagation D, t_1/2 from the
drift-tracked nominal bleach ROI. No Python package implements the method (searched 2026-11-26; PyFRAP
uses simulations, not eq. 5), so the steps of the paper are implemented here as written in its Methods:

  r_n      nominal bleach radius = half the circle ROI size in the Leica Properties XML (E019, user
           2026-10-08), as read by frap_pyfrap_inputs.bleach_roi_radius_um.
  F(t)     mean intensity in the disk of radius r_n, centred per frame on the Gaussian fit of the bleach
           (E020, user 2026-10-08; cache/frap_pyfrap_inputs.pkl). Kang: F_corr = (F_raw - F_bk) /
           (F_fading - F_bk), F = F_corr / F_corr,i. F_bk = BACKGROUND_COUNTS = 0 (E048). F_fading:
           Kang record a separate unbleached series; none exists here, so the reference region of
           frap_analysis_v2 (pixels with r >= clip(3 r_e, 25, 65) um, same frames) is used as the fading
           curve (open decision E061). Hence F(t) = [F_roi(t) / F_roi,pre] / [F_ref(t) / F_ref,pre].
  t_1/2    Kang, "tau_1/2 measurements": F_0 = F(first post-bleach frame), F_inf = F(last frame),
           F_1/2 = (F_0 + F_inf) / 2; t_1/2 = first frame time with F = F_1/2, else linear interpolation
           between the frames t_k, t_k+1 with F(t_k) < F_1/2 < F(t_k+1). No recovery model is fitted.
  r_e      Kang eq. 2: the first post-bleach profile divided by the pre-bleach profile is fitted with
           1 - K exp(-2 r^2 / r_e^2) (unweighted nonlinear least squares; r_e = half width at 14 % of the
           bleach depth from the top, which is what eq. 2 and the 0.86 K rule of their Fig. 8 define; the
           Methods formula exp(-x^2/r_e^2) contradicts both and is not used). Deviations from the paper:
           azimuthal average around the Gaussian-fit centre of frame 0 instead of one diagonal line, out
           to PROFILE_R_MAX_UM; the profile is fading-corrected with F_ref(0) / F_ref,pre so that its far
           field is 1 as in eq. 2 (diagnostic far_field_ratio is written).
  Unit     one D per measurement (Kang average curves over cells first); group values are mean, SD and
           95 % t confidence interval over the QC-passed measurements (as E049 C for D_eff/D0).

Compute stage on caches (no raw data are read): reads cache/frap_pyfrap_inputs.pkl (frap_pyfrap_inputs.py)
and cache/frap_v2_phase1.pkl (frap_analysis_v2.py), writes cache/frap_kang_result.pkl, read by
frap_kang_figures.py. Measurements: those in frap_pyfrap_inputs.pkl (water, gel, gel_interface; water_2
already left out there, E017). QC gate qc_kang = qc_global of frap_analysis_v2 and t_1/2 found and r_e fit
converged; failing measurements stay in the tables and are shown as open symbols. D0 = mean of the
QC-passed water measurements (user, 2026-10-06); shallow and deep hydrogel pooled (user, 2026-10-14).

Before the real data, synthetic_check() runs the same functions on simulated frames with known D (exact
2D diffusion of the Gaussian bleach, Poisson noise, acquisition fading, 8 bit, real time grids) and writes
the recovered against the true D; it is a test and is never written to the cache.
Writes (always recomputed): CACHE_PATH, and in OUTPUT_DIR frap_kang_values.csv (per measurement),
frap_kang_summary.csv (D0, D_eff, D_eff/D0), frap_kang_synthetic.csv, frap_kang_exclusions.csv.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from hydro_analysis.FRAP.frap_analysis_v2 import (BACKGROUND_COUNTS, PROFILE_R_MAX_UM, RadialBins,
                                                  double_normalised_curve)
from hydro_analysis.FRAP.frap_pyfrap_inputs import disk_mean

# ── Configuration ──────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PYFRAP_INPUTS = HERE / "cache" / "frap_pyfrap_inputs.pkl"
V2_CACHE = HERE / "cache" / "frap_v2_phase1.pkl"
CACHE_PATH = HERE / "cache" / "frap_kang_result.pkl"
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\FRAP\FRAP_v2\kang")

SAMPLE_TYPE = {"water": "Buffer", "gel": "Hydrogel", "gel_interface": "Hydrogel"}   # pooling: user, 2026-10-14
CI_LEVEL = 0.95                                     # group uncertainty as E049 C (user, 2026-10-27)

# Synthetic check (test only). Parameters taken from the real measurements (r_e, K, r_n, time grid of
# the named measurement, fading at the first and last frame); the field of view is 145 um as measured.
SYNTHETIC_SIZE_PX, SYNTHETIC_MPP_UM = 256, 0.5653
SYNTHETIC_CASES = [
    {"case": "buffer", "time_grid_of": "water_3", "D_um2_s": 30.0, "r_e_um": 39.4, "K": 0.72, "r_n_um": 9.65},
    {"case": "gel_short_roi", "time_grid_of": "gel_B1_4", "D_um2_s": 27.0, "r_e_um": 34.5, "K": 0.68, "r_n_um": 9.65},
    {"case": "gel_large_roi", "time_grid_of": "gel_A1_3", "D_um2_s": 27.0, "r_e_um": 46.4, "K": 0.73, "r_n_um": 16.29},
    {"case": "gel_long", "time_grid_of": "gel_B1_2", "D_um2_s": 27.0, "r_e_um": 64.8, "K": 0.80, "r_n_um": 16.29},
    {"case": "interface_short", "time_grid_of": "interface_1", "D_um2_s": 27.0, "r_e_um": 47.1, "K": 0.90, "r_n_um": 16.29},
]
SYNTHETIC_INTENSITY, SYNTHETIC_TEXTURE, SYNTHETIC_FADE = 60.0, 0.1, (0.96, 0.93)   # counts; rel. SD; F_ref(0), F_ref(end)
SYNTHETIC_PRE_FRAMES, SYNTHETIC_SEED = 7, 20261126


# ── Kang steps ─────────────────────────────────────────────────────────────────
def kang_curve(f_roi, f_roi_pre, f_ref, f_ref_pre) -> np.ndarray:
    """Background- and fading-corrected ROI intensity normalised to the pre-bleach value (F_i = 1)."""
    bg = BACKGROUND_COUNTS
    return ((np.asarray(f_roi) - bg) / (f_roi_pre - bg)) / ((np.asarray(f_ref) - bg) / (f_ref_pre - bg))


def kang_half_time(t_s, f) -> dict:
    """Kang's linear interpolation: F_1/2 = (F_0 + F_inf)/2 with F_0 = F(t_0), F_inf = F(t_n)."""
    t_s, f = np.asarray(t_s, float), np.asarray(f, float)
    f0, finf = f[0], f[-1]
    f_half = 0.5 * (f0 + finf)
    out = {"F0": f0, "Finf": finf, "F_half": f_half, "t_half_s": np.nan, "k_half": -1}
    for k in range(len(f) - 1):
        if f[k] == f_half:
            out.update(t_half_s=t_s[k], k_half=k)
            return out
        if f[k] < f_half < f[k + 1]:
            out.update(t_half_s=t_s[k] + (f_half - f[k]) / (f[k + 1] - f[k]) * (t_s[k + 1] - t_s[k]), k_half=k)
            return out
    return out


def _kang_profile(r, K, re):
    return 1.0 - K * np.exp(-2.0 * r ** 2 / re ** 2)


def kang_effective_radius(pre_mean, post_first, centre, mpp_um, fading_0, re_guess_um) -> dict:
    """Fit 1 - K exp(-2 r^2/r_e^2) to the fading-corrected azimuthal post/pre profile of frame 0."""
    bins = RadialBins(pre_mean.shape, centre, mpp_um)
    bg = BACKGROUND_COUNTS
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = bins.profile(post_first - bg) / bins.profile(pre_mean - bg) / fading_0
    ok = np.isfinite(prof) & (bins.counts > 0)
    r, y = bins.r_um[ok], prof[ok]
    far = r > 0.8 * r.max()
    out = {"r_um": r, "profile": y, "far_field_ratio": float(np.mean(y[far])), "fit_ok": False,
           "r_e_um": np.nan, "r_e_se_um": np.nan, "K": np.nan, "profile_r2": np.nan}
    try:
        popt, pcov = curve_fit(_kang_profile, r, y, p0=[float(np.clip(1 - y[:3].mean(), 0.05, 0.95)), re_guess_um],
                               bounds=([0.0, 0.3], [1.0, 5 * r.max()]), maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        out["fail_reason"] = str(exc)
        return out
    resid = y - _kang_profile(r, *popt)
    out.update(fit_ok=True, K=float(popt[0]), r_e_um=float(popt[1]), r_e_se_um=float(np.sqrt(pcov[1, 1])),
               profile_r2=float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)))
    return out


def kang_d(r_e_um, r_n_um, t_half_s) -> float:
    """Kang et al. 2012, eq. 5."""
    return (r_e_um ** 2 + r_n_um ** 2) / (8.0 * t_half_s)


# ── Synthetic check ────────────────────────────────────────────────────────────
def _synthetic_case(case, t_s, rng):
    n, mpp = SYNTHETIC_SIZE_PX, SYNTHETIC_MPP_UM
    rows, cols = np.indices((n, n))
    centre = ((n - 1) / 2 + 3.3, (n - 1) / 2 - 2.1)
    r2 = ((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2) * mpp ** 2
    base = SYNTHETIC_INTENSITY * (1 + SYNTHETIC_TEXTURE * rng.standard_normal((n, n))).clip(0.2)
    pre = np.stack([np.clip(np.round(rng.poisson(base)), 0, 255) for _ in range(SYNTHETIC_PRE_FRAMES)]).astype(float)
    f_a, f_b = SYNTHETIC_FADE
    fade = f_a * (f_b / f_a) ** (t_s / t_s[-1])
    post = []
    for t, fd in zip(t_s, fade):
        w2 = case["r_e_um"] ** 2 + 8 * case["D_um2_s"] * t          # exact 2D diffusion of the Gaussian bleach
        img = base * fd * (1 - case["K"] * case["r_e_um"] ** 2 / w2 * np.exp(-2 * r2 / w2))
        post.append(np.clip(np.round(rng.poisson(img)), 0, 255))
    post = np.stack(post).astype(float)
    radius_px = case["r_n_um"] / mpp
    f_roi = np.array([disk_mean(fr, centre, radius_px)[0] for fr in post])
    f_roi_pre = float(np.mean([disk_mean(fr, centre, radius_px)[0] for fr in pre]))
    ref = double_normalised_curve(pre, post, centre, case["r_e_um"], mpp)
    f = kang_curve(f_roi, f_roi_pre, ref["F_ref"], ref["F_ref_pre"])
    half = kang_half_time(t_s, f)
    prof = kang_effective_radius(pre.mean(axis=0), post[0], centre, mpp, ref["F_ref"][0] / ref["F_ref_pre"],
                                 case["r_e_um"])
    d = kang_d(prof["r_e_um"], case["r_n_um"], half["t_half_s"])
    # the same ROI curve without noise and with a 50x longer series: bias of eq. 5 itself for this geometry
    w2 = lambda t: case["r_e_um"] ** 2 + 8 * case["D_um2_s"] * t
    a = case["r_n_um"] ** 2
    ideal = lambda t: 1 - case["K"] * case["r_e_um"] ** 2 / (2 * a) * (1 - np.exp(-2 * a / w2(t)))
    t_long = np.linspace(0, 50 * t_s[-1], 200000)
    d_ideal = kang_d(case["r_e_um"], case["r_n_um"], kang_half_time(t_long, ideal(t_long))["t_half_s"])
    t_same = np.linspace(0, t_s[-1], 20000)
    d_ideal_same = kang_d(case["r_e_um"], case["r_n_um"], kang_half_time(t_same, ideal(t_same))["t_half_s"])
    return {**case, "n_frames": len(t_s), "t_last_s": float(t_s[-1]), "r_e_fit_um": prof["r_e_um"],
            "r_e_rel_error": prof["r_e_um"] / case["r_e_um"] - 1, "t_half_s": half["t_half_s"],
            "D_kang_um2_s": d, "D_rel_error": d / case["D_um2_s"] - 1,
            "D_noise_free_same_duration_rel_error": d_ideal_same / case["D_um2_s"] - 1,
            "D_noise_free_50x_duration_rel_error": d_ideal / case["D_um2_s"] - 1}


def synthetic_check(time_grids: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SYNTHETIC_SEED)
    rows = [_synthetic_case(c, time_grids[c["time_grid_of"]], rng) for c in SYNTHETIC_CASES]
    return pd.DataFrame(rows)


# ── Real data ──────────────────────────────────────────────────────────────────
def analyse(inputs: dict, v2: dict) -> tuple[pd.DataFrame, dict, list]:
    res = v2["results"].set_index("measurement_id")
    rows, curves, log = [], {}, []
    for mid, d in inputs["data"].items():
        r, det = res.loc[mid], v2["details"][mid]
        t = np.asarray(d["t_s"], float)
        if not np.allclose(t, det["t_s"]):
            raise ValueError(f"{mid}: time axes of the two caches differ.")
        cv = det["curve"]
        f = kang_curve(d["F_roi"], float(np.mean(d["F_roi_pre"])), cv["F_ref"], cv["F_ref_pre"])
        half = kang_half_time(t, f)
        prof = kang_effective_radius(d["pre_mean_image"], d["first_post_image"], (d["row_fit"][0], d["col_fit"][0]),
                                     d["mpp_um"], cv["F_ref"][0] / cv["F_ref_pre"], r.r_e_um)
        D = kang_d(prof["r_e_um"], d["r_n_um"], half["t_half_s"])
        reasons = [txt for ok, txt in [(bool(r.qc_global), "qc_global of frap_analysis_v2 failed: " + str(r.get("qc_reasons", ""))),
                                       (np.isfinite(half["t_half_s"]), "F never crosses F_1/2"),
                                       (prof["fit_ok"], "r_e fit failed")] if not ok]
        rows.append({"measurement_id": mid, "file": d["file"], "condition": d["condition"],
                     "sample_type": SAMPLE_TYPE[d["condition"]], "mpp_um": d["mpp_um"], "n_post_frames": len(t),
                     "t_last_s": t[-1], "r_n_um": d["r_n_um"], "r_e_um": prof["r_e_um"], "r_e_se_um": prof["r_e_se_um"],
                     "K_profile": prof["K"], "profile_r2": prof["profile_r2"],
                     "far_field_ratio": prof["far_field_ratio"], "r_e_v2_um": r.r_e_um,
                     "F0": half["F0"], "Finf": half["Finf"], "F_half": half["F_half"], "t_half_s": half["t_half_s"],
                     "D_kang_um2_s": D, "roi_frames_held": int((~d["fit_ok"]).sum()),
                     "qc_kang": not reasons, "qc_reasons": "; ".join(reasons)})
        curves[mid] = {"t_s": t, "F": f, "profile_r_um": prof["r_um"], "profile": prof["profile"]}
        if (~d["fit_ok"]).any():
            log.append(dict(file=d["file"], step="ROI position held (E020)", parameter="bleach fit failed",
                            n_before=len(t), n_after=int(d["fit_ok"].sum()),
                            reason=f"{mid}: ROI centre held at the last fitted position in {(~d['fit_ok']).sum()} frames"))
        if reasons:
            log.append(dict(file=d["file"], step="qc_kang", parameter="qc_global, t_1/2, r_e fit", n_before=1,
                            n_after=0, reason=f"{mid}: " + "; ".join(reasons) + " (shown open, not in the mean)"))
    for item in inputs.get("exclusions", []) or []:
        log.append(item)
    for mid, why in inputs["config"]["EXCLUDE_IDS"].items():
        log.append(dict(file=mid, step="excluded upstream", parameter="EXCLUDE_IDS (frap_pyfrap_inputs.py)",
                        n_before=1, n_after=0, reason=f"{mid}: {why} (user, 2026-10-06)"))
    return pd.DataFrame(rows), curves, log


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stype in ("Buffer", "Hydrogel"):
        v = df.loc[(df.sample_type == stype) & df.qc_kang, "D_kang_um2_s"].to_numpy()
        n = len(v)
        sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
        t_q = float(stats.t.ppf(0.5 + CI_LEVEL / 2, n - 1)) if n > 1 else np.nan
        rows.append({"sample_type": stype, "quantity": "D0" if stype == "Buffer" else "D_eff", "n": n,
                     "D_mean_um2_s": float(np.mean(v)) if n else np.nan, "D_SD_um2_s": sd,
                     "D_SEM_um2_s": sd / np.sqrt(n) if n > 1 else np.nan, "t_quantile": t_q,
                     "D_CI95_half_width_um2_s": t_q * sd / np.sqrt(n) if n > 1 else np.nan,
                     "measurements": ", ".join(df.loc[(df.sample_type == stype) & df.qc_kang, "measurement_id"]),
                     "qc_failed_not_in_mean": ", ".join(df.loc[(df.sample_type == stype) & ~df.qc_kang, "measurement_id"])})
    out = pd.DataFrame(rows)
    d0 = out.loc[out.quantity == "D0", "D_mean_um2_s"].iloc[0]
    ratio = df.loc[(df.sample_type == "Hydrogel") & df.qc_kang, "D_kang_um2_s"].to_numpy() / d0
    n = len(ratio)
    sd = float(np.std(ratio, ddof=1)) if n > 1 else np.nan
    t_q = float(stats.t.ppf(0.5 + CI_LEVEL / 2, n - 1)) if n > 1 else np.nan
    out["Deff_D0_mean"] = [np.nan, float(np.mean(ratio)) if n else np.nan]
    out["Deff_D0_CI95_half_width"] = [np.nan, t_q * sd / np.sqrt(n) if n > 1 else np.nan]
    out["note"] = "D0 = mean of the QC-passed buffer measurements, entering D_eff/D0 as a constant (E049 C)"
    return out


def main():
    for p in (PYFRAP_INPUTS, V2_CACHE):
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing: run frap_pyfrap_inputs.py and frap_analysis_v2.py first.")
    inputs = pickle.loads(PYFRAP_INPUTS.read_bytes())
    v2 = pickle.loads(V2_CACHE.read_bytes())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    synth = synthetic_check({m: np.asarray(d["t_s"], float) for m, d in inputs["data"].items()})
    synth.to_csv(OUTPUT_DIR / "frap_kang_synthetic.csv", index=False)
    print("Synthetic check (recovered vs true):")
    print(synth[["case", "D_um2_s", "D_kang_um2_s", "D_rel_error", "r_e_rel_error",
                 "D_noise_free_same_duration_rel_error", "D_noise_free_50x_duration_rel_error"]].round(3).to_string(index=False))

    df, curves, log = analyse(inputs, v2)
    summary = group_summary(df)
    df.to_csv(OUTPUT_DIR / "frap_kang_values.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "frap_kang_summary.csv", index=False)
    pd.DataFrame(log, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        OUTPUT_DIR / "frap_kang_exclusions.csv", index=False)
    CACHE_PATH.write_bytes(pickle.dumps({"created": f"{datetime.now():%Y-%m-%d %H:%M}", "results": df,
                                         "summary": summary, "curves": curves,
                                         "sources": [str(PYFRAP_INPUTS), str(V2_CACHE)]}))
    print(df[["measurement_id", "sample_type", "r_n_um", "r_e_um", "r_e_v2_um", "far_field_ratio", "profile_r2",
              "F0", "Finf", "t_half_s", "D_kang_um2_s", "qc_kang"]].round(3).to_string(index=False))
    print(summary[["quantity", "n", "D_mean_um2_s", "D_SD_um2_s", "D_CI95_half_width_um2_s", "Deff_D0_mean",
                   "Deff_D0_CI95_half_width"]].round(3).to_string(index=False))
    print(f"Written: {CACHE_PATH}, {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
