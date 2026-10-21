"""
FRAP with PyFRAP, stage 1 (inputs): drift-tracked bleach ROI curves of the FITC-dextran 150 kDa
FRAP measurements (Leica TCS SP8, LAS X FRAP wizard, circular bleach ROI with Zoom In).
Compute stage: reads the raw TIF series and Leica MetaData XML below DATA_ROOT (read-only, with
frap_analysis_v2.load_measurement) and MEASUREMENTS_CSV, and writes cache/frap_pyfrap_inputs.pkl.
The PyFRAP simulation and fit of D (stage 2) is not written yet; it will read this cache.

Bleach ROI (E019, user 2026-10-08): radius r_n = half the circle ROI size stored in the
Properties XML of the post-bleach series (ROIDATA, REALWORLD_SCALING Size; mean of x and y);
all post-bleach series of a measurement must hold the same single circle. The XML ROI position is
not used: for the 19.2 um ROIs it lies 51 um off the measured bleach (E020).
ROI position (E020, user 2026-10-08): in every post-bleach frame the bleach is fitted with
    F_post(x, y) = c * F_pre(x, y) * (1 - K exp(-2 ((x - x0)^2 + (y - y0)^2) / w^2)),
F_pre = mean pre-bleach image, on all pixels of the raw frame; the ROI is centred on (x0, y0).
If the fit fails (no convergence, K < BLEACH_K_MIN, i.e. the bleach is too flat (E023, user 2026-10-08),
w <= 0, non-finite values or centre outside the image), the ROI stays at the last position. The first post-bleach frame starts from the deficit centroid
(frap_analysis_v2.bleach_centroid) and must succeed. Pre-bleach ROI values use the centre of the
first post-bleach frame. Images are never shifted or resampled.

Measurements: conditions water, gel and gel_interface; water_2 is left out (97 % saturated pixels,
user 2026-10-06, E017); noFITC controls have no bleach to track. Both are logged in the exclusion CSV.
Before the real data, the tracking is checked on synthetic frames with known drift (synthetic_check).

Outputs: CACHE_PATH, and in OUTPUT_DIR frap_pyfrap_inputs_frames.csv (per frame), _summary.csv,
_synthetic.csv and _exclusions.csv. No other script reads them yet.
"""
from __future__ import annotations

import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from hydro_analysis.FRAP.frap_analysis import find_series_folders
from hydro_analysis.FRAP.frap_analysis_v2 import bleach_centroid, load_measurement

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"E:\Daten Promotion Sicherung\Confocal_Measure\2025_09_08_14_57_46--Project_TIF")
MEASUREMENTS_CSV = Path(__file__).resolve().parent / "measurements.csv"
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "frap_pyfrap_inputs.pkl"
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\FRAP\FRAP_v2\pyfrap_inputs")

CONDITIONS = ("water", "gel", "gel_interface")          # FITC measurements; noFITC has no bleach
EXCLUDE_IDS = {"water_2": "97 % saturated pixels"}      # decided by user, 2026-10-06 (E017)
BLEACH_K_MIN = 0.05                # decided by user, 2026-10-08 (E023): flatter bleach fits count as failed

# Synthetic check of the tracking (test only, never written to the cache)
SYNTHETIC_ROI_TOL = 0.01           # PROPOSED, not approved: max relative error of the ROI mean (the PyFRAP input)
SYNTHETIC_K_MIN = 0.1              # PROPOSED, not approved: frames below this depth are reported, not tested
# The centre error is reported, not tested: a first gate of 0.5 px failed at low K (noise limit, 2026-10-08).
SYNTHETIC_PARAMS = {"size_px": 512, "mpp_um": 0.283, "intensity": 60.0, "texture_rel": 0.1, "K0": 0.7,
                    "w0_um": 30.0, "D_um2_s": 25.0, "dt_s": 0.373, "n_frames": 80, "drift_px_per_frame": (0.08, -0.05),
                    "r_n_um": 16.3, "seed": 11}


# ── Loading ────────────────────────────────────────────────────────────────────
def bleach_roi_radius_um(folder: Path) -> tuple[float, float, float]:
    """(r_n, size_x, size_y) in um from the single circle ROI shared by all post-bleach series."""
    _, pb_folders = find_series_folders(folder)
    sizes = set()
    for pb in pb_folders:
        xml = pb / "MetaData" / f"{pb.name}_Properties.xml"
        found = re.findall(r'<ROIDATA Type="(\w+)"[^>]*>\s*<REALWORLD_SCALING>\s*<Size X="([^"]+)" Y="([^"]+)"',
                           xml.read_text("utf-8", "replace"))
        if len(found) != 1 or found[0][0] != "CIRCLE":
            raise ValueError(f"{xml}: expected one CIRCLE bleach ROI, found {found}")
        sizes.add((float(found[0][1]) * 1e6, float(found[0][2]) * 1e6))
    if len(sizes) != 1:
        raise ValueError(f"{folder}: post-bleach series hold different bleach ROIs: {sorted(sizes)}")
    sx, sy = sizes.pop()
    return (sx + sy) / 4.0, sx, sy


# ── Tracking ───────────────────────────────────────────────────────────────────
def _model_and_jac(p, pre, rows, cols):
    c, k, r0, c0, w = p
    dr, dc = rows - r0, cols - c0
    r2 = dr ** 2 + dc ** 2
    g = np.exp(-2.0 * r2 / w ** 2)
    model = c * pre * (1.0 - k * g)
    ckpg = c * k * pre * g
    jac = np.stack([pre * (1.0 - k * g), -c * pre * g, -ckpg * 4.0 * dr / w ** 2,
                    -ckpg * 4.0 * dc / w ** 2, -ckpg * 4.0 * r2 / w ** 3], axis=1)
    return model, jac


def fit_bleach(frame: np.ndarray, pre: np.ndarray, p0) -> tuple[np.ndarray, bool, str]:
    """Least-squares fit of the bleach model to one raw frame; returns (params, ok, reason)."""
    rows, cols = np.indices(frame.shape)
    rows, cols, pre_f, y = rows.ravel().astype(float), cols.ravel().astype(float), pre.ravel(), frame.ravel()
    res = least_squares(lambda p: _model_and_jac(p, pre_f, rows, cols)[0] - y, np.asarray(p0, float),
                        jac=lambda p: _model_and_jac(p, pre_f, rows, cols)[1], method="lm")
    p = res.x
    if not res.success:
        return p, False, "no convergence"
    if not np.all(np.isfinite(p)):
        return p, False, "non-finite parameters"
    if p[1] < BLEACH_K_MIN:
        return p, False, f"K < {BLEACH_K_MIN:g}"
    if p[4] <= 0:
        return p, False, "w <= 0"
    if not (0 <= p[2] < frame.shape[0] and 0 <= p[3] < frame.shape[1]):
        return p, False, "centre outside the image"
    return p, True, ""


def disk_mean(image: np.ndarray, centre: tuple, radius_px: float) -> tuple[float, int]:
    rows, cols = np.indices(image.shape)
    mask = np.hypot(rows - centre[0], cols - centre[1]) <= radius_px
    return float(image[mask].mean()), int(mask.sum())


def track(pre_frames: np.ndarray, post: np.ndarray, mpp_um: float, r_n_um: float) -> dict:
    """Per-frame bleach fit, ROI centre (held on failure) and ROI / field-of-view means."""
    pre = pre_frames.mean(axis=0)
    start = bleach_centroid(pre, post[0])
    if start is None:
        raise ValueError("no bleach deficit in the first post-bleach frame")
    p = np.array([1.0, 0.5, start[0], start[1], 2.0 * r_n_um / mpp_um])
    radius_px = r_n_um / mpp_um
    out = {k: [] for k in ("c", "K", "row_fit", "col_fit", "w_um", "fit_ok", "fail_reason",
                           "row", "col", "F_roi", "n_roi_px", "F_fov")}
    centre = None
    for i, frame in enumerate(post):
        p_fit, ok, reason = fit_bleach(frame, pre, p)
        if i == 0 and not ok:
            raise ValueError(f"bleach fit failed in the first post-bleach frame: {reason}")
        if ok:
            p, centre = p_fit, (p_fit[2], p_fit[3])
        f_roi, n_px = disk_mean(frame, centre, radius_px)
        for key, val in zip(("c", "K", "row_fit", "col_fit"), p_fit[:4]):
            out[key].append(float(val))
        out["w_um"].append(float(p_fit[4] * mpp_um))
        out["fit_ok"].append(ok)
        out["fail_reason"].append(reason)
        out["row"].append(float(centre[0]))
        out["col"].append(float(centre[1]))
        out["F_roi"].append(f_roi)
        out["n_roi_px"].append(n_px)
        out["F_fov"].append(float(frame.mean()))
    out = {k: np.asarray(v) for k, v in out.items()}
    first = (out["row"][0], out["col"][0])
    out["F_roi_pre"] = np.array([disk_mean(f, first, radius_px)[0] for f in pre_frames])
    out["F_fov_pre"] = pre_frames.mean(axis=(1, 2))
    out["pre_mean_image"] = pre
    out["first_post_image"] = post[0]
    return out


# ── Synthetic check ────────────────────────────────────────────────────────────
def synthetic_frames(params: dict) -> dict:
    """Moving Gaussian bleach with known centre on a textured background, Poisson noise, 8 bit."""
    rng = np.random.default_rng(params["seed"])
    n, mpp = params["size_px"], params["mpp_um"]
    rows, cols = np.indices((n, n))
    texture = 1.0 + params["texture_rel"] * rng.standard_normal((n, n))
    truth_pre = params["intensity"] * texture
    pre = np.clip(rng.poisson(np.tile(truth_pre, (7, 1, 1))), 0, 255).astype(float)
    centres, ks, frames, roi_true = [], [], [], []
    radius_px = params["r_n_um"] / mpp
    for i in range(params["n_frames"]):
        t = i * params["dt_s"]
        w2 = params["w0_um"] ** 2 + 8 * params["D_um2_s"] * t
        k = params["K0"] * params["w0_um"] ** 2 / w2
        r0 = n / 2 + params["drift_px_per_frame"][0] * i
        c0 = n / 2 + params["drift_px_per_frame"][1] * i
        truth = truth_pre * (1 - k * np.exp(-2 * ((rows - r0) ** 2 + (cols - c0) ** 2) * mpp ** 2 / w2))
        frames.append(np.clip(rng.poisson(truth), 0, 255).astype(float))
        centres.append((r0, c0))
        ks.append(k)
        roi_true.append(disk_mean(truth, (r0, c0), radius_px)[0])
    return {"pre": pre, "post": np.array(frames), "centres": np.array(centres), "K": np.array(ks),
            "roi_true": np.array(roi_true)}


def synthetic_check() -> pd.DataFrame:
    s = synthetic_frames(SYNTHETIC_PARAMS)
    tr = track(s["pre"], s["post"], SYNTHETIC_PARAMS["mpp_um"], SYNTHETIC_PARAMS["r_n_um"])
    err = np.hypot(tr["row"] - s["centres"][:, 0], tr["col"] - s["centres"][:, 1])
    rel = np.abs(tr["F_roi"] - s["roi_true"]) / s["roi_true"]
    df = pd.DataFrame({"frame": np.arange(len(err)), "K_true": s["K"], "row_true": s["centres"][:, 0],
                       "col_true": s["centres"][:, 1], "row": tr["row"], "col": tr["col"], "fit_ok": tr["fit_ok"],
                       "centre_error_px": err, "roi_rel_error": rel})
    tested = df[df.K_true >= SYNTHETIC_K_MIN]
    print(f"Synthetic check: {len(df)} frames, K_true {df.K_true.min():.3f}-{df.K_true.max():.3f}, "
          f"{int((~df.fit_ok).sum())} failed fits")
    print(f"  K_true >= {SYNTHETIC_K_MIN}: {len(tested)} frames, max centre error {tested.centre_error_px.max():.3f} px, "
          f"max ROI error {tested.roi_rel_error.max():.4f}")
    late = df[df.K_true < SYNTHETIC_K_MIN]
    if len(late):
        print(f"  K_true < {SYNTHETIC_K_MIN} (not tested): {len(late)} frames, max centre error "
              f"{late.centre_error_px.max():.3f} px")
    if tested.roi_rel_error.max() > SYNTHETIC_ROI_TOL:
        raise RuntimeError("Synthetic check of the tracking failed; real data are not analysed.")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_check().to_csv(OUTPUT_DIR / "frap_pyfrap_inputs_synthetic.csv", index=False)

    table = pd.read_csv(MEASUREMENTS_CSV, dtype={"batch": str, "note": str}, keep_default_na=False)
    table["include"] = table["include"].astype(str).str.lower().isin(["true", "1", "yes"])
    log, data, frames, summary = [], {}, [], []
    for _, row in table.iterrows():
        mid, folder = row["measurement_id"], DATA_ROOT / row["file"]
        if row["condition"] not in CONDITIONS:
            log.append(dict(file=row["file"], step="condition", parameter="CONDITIONS", n_before=1, n_after=0,
                            reason=f"{mid}: condition {row['condition']} has no bleach to track"))
            continue
        if mid in EXCLUDE_IDS or not row["include"]:
            log.append(dict(file=row["file"], step="excluded by user", parameter="EXCLUDE_IDS / include",
                            n_before=1, n_after=0, reason=f"{mid}: {EXCLUDE_IDS.get(mid, 'include = False')}"))
            continue
        print(f"Tracking {mid} ({row['file']})")
        r_n, sx, sy = bleach_roi_radius_um(folder)
        meas = load_measurement(folder)
        tr = track(meas["pre"], meas["post"], meas["mpp_um"], r_n)
        n_fail = int((~tr["fit_ok"]).sum())
        if n_fail:
            log.append(dict(file=row["file"], step="ROI position held", parameter="bleach fit failed",
                            n_before=len(tr["fit_ok"]), n_after=len(tr["fit_ok"]) - n_fail,
                            reason=f"{mid}: ROI kept at the last fitted position in {n_fail} frames "
                                   f"({', '.join(sorted(set(tr['fail_reason'][~tr['fit_ok']])))})"))
        data[mid] = {"measurement_id": mid, "file": row["file"], "condition": row["condition"],
                     "folder": str(folder), "mpp_um": meas["mpp_um"], "r_n_um": r_n, "roi_size_um": (sx, sy),
                     "t_s": meas["post_t_s"], "pre_t_s": meas["pre_t_s"], **tr}
        disp = np.hypot(tr["row"] - tr["row"][0], tr["col"] - tr["col"][0]) * meas["mpp_um"]
        frames.append(pd.DataFrame({"measurement_id": mid, "t_s": meas["post_t_s"],
                                    **{k: tr[k] for k in ("fit_ok", "fail_reason", "K", "w_um", "row", "col",
                                                          "F_roi", "n_roi_px", "F_fov")},
                                    "displacement_um": disp}))
        summary.append({"measurement_id": mid, "condition": row["condition"], "r_n_um": r_n,
                        "n_frames": len(tr["fit_ok"]), "n_fit_failed": n_fail,
                        "first_failed_t_s": float(meas["post_t_s"][~tr["fit_ok"]][0]) if n_fail else np.nan,
                        "max_displacement_um": float(disp.max()), "K_first": float(tr["K"][0]),
                        "w_first_um": float(tr["w_um"][0])})

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps({"created": f"{datetime.now():%Y-%m-%d %H:%M}", "data": data,
                                         "config": {"DATA_ROOT": str(DATA_ROOT), "EXCLUDE_IDS": EXCLUDE_IDS}}))
    pd.concat(frames, ignore_index=True).to_csv(OUTPUT_DIR / "frap_pyfrap_inputs_frames.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUTPUT_DIR / "frap_pyfrap_inputs_summary.csv", index=False)
    pd.DataFrame(log, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        OUTPUT_DIR / "frap_pyfrap_inputs_exclusions.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(f"Cache: {CACHE_PATH}\nTables: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
