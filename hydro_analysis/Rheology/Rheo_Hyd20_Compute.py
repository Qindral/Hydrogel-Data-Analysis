"""
Oscillatory rheology of the 20 mg/mL hydrogel (TA Discovery HR-2, 25 mm
parallel plate, 25 degC): mechanical stability over gel age. Compute stage.

Reads every TRIOS run file RAW_DIR/Hyd*.tri (binary, parsed by
trios_tri_parser.py) and recomputes G', G'', tan(delta), strain and stress
from the raw torque, displacement and phase channels including the TRIOS
inertia correction (rheo_analysis.oscillation_quantities). The TRIOS Excel
exports in the same folder (.xls, only for some runs) contain the same runs
as the .tri files of the same name and are used only to validate this
reconstruction (tables/tri_vs_xls_validation.csv); they are never used as
data source.

Sample metadata come from the TRIOS comment field of each run, parsed with
the editable regular expressions in COMMENT_PATTERNS:
  - gel age in days ("9 tagw alt", "4Tagealt", "20 Tage alt"),
  - synthesis date of the hydrogel batch ("21.hergestellt", "von21.10.25"),
  - casting date ("gekastet 22.10.", "gecastet27.10.2025"),
  - specimen position on the casting plate (A2 ... B4),
  - notes on visible cracks, 20x dilution and changed gap.
All runs belong to one hydrogel batch (synthesised 21.10.2025), cast on two
dates (22.10. and 27.10.). Because the casts come from the same batch, gel
age is compared across casts: stability statistics are grouped by batch and
age, the cast is kept as a covariate (stability_summary_by_cast.csv lists
the per-cast values). Runs whose comment gives the age but no casting date
are assigned to the known casting date closest to (run date - age), within
CAST_INFERENCE_TOLERANCE_D (cast_source = "inferred"). The gel age is taken
from the comment. Corrections that cannot be derived from the comments are
entered in SAMPLE_OVERRIDES with a reason (e.g. from the lab note
RAW_DIR/10.Nov.txt); every override appears in the metadata and QC tables.
All runs are 20 mg/mL, so no concentration dependence is evaluated.

Replicate levels: specimen = cast + position (one cast gel piece);
technical repeats = several sweeps of the same specimen on the same day;
longitudinal = the same specimen at different ages. Technical repeats are
averaged per specimen before any group statistic, so N always counts
specimens, never sweep points or repeats.

Analysis per run (criteria in the Configuration block):
  frequency sweeps - largest contiguous window with weak frequency
    dependence of G' (PlateauCriteria), plateau statistics, G' at 0.5 Hz
    (measured point and window fit, never extrapolated) and the
    representative G' chosen by REPRESENTATIVE_GP_ORDER;
  amplitude sweeps - low-strain reference G', strains of 5 % and 10 %
    deviation from it, G'/G'' crossover (strain, stress, modulus).
Points that are inertia-dominated or torque-limited are flagged and left out
of the fits, not deleted. Quality-control warnings never remove a run; group
statistics are reported twice, for all analysable runs ("all_analysable")
and without runs carrying a code of STATS_EXCLUDE_QC_CODES ("qc_filtered").

Writes (unconditionally overwritten, always recomputed from the raw files):
  cache/rheo_hyd20_results.pkl                       (this folder)
  OUTPUT_ROOT/processed_data/<run>.csv, all_measurements.csv
  OUTPUT_ROOT/tables/frequency_sweep_summary.csv, amplitude_sweep_summary.csv,
    stability_summary.csv, stability_summary_by_cast.csv,
    stability_specimen_level.csv, mean_frequency_sweeps.csv (mean +- SD of G', G'',
    tan(delta) per age on a common frequency grid, log-log interpolated within
    each sweep's valid range), quality_control.csv, measurement_metadata.csv,
    fit_parameters.csv, tri_vs_xls_validation.csv
Consumer: Rheo_Hyd20_Figures.py reads the pickle and draws all figures.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hydro_analysis.Rheology.rheo_analysis import (
    LVRCriteria,
    PlateauCriteria,
    PointCriteria,
    amplitude_sweep_analysis,
    describe,
    evaluate_at_frequency,
    find_plateau_window,
    interp_loglog,
    loglog_linear_fit,
    max_log_jump,
    oscillation_quantities,
    point_artifacts,
)
from hydro_analysis.Rheology.trios_tri_parser import TriRun, read_trios_xls, read_tri

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR = Path(r"H:\Daten Promotion Sicherung\Rheo")
TRI_GLOB = "Hyd*.tri"
XLS_GLOB = "Hyd*.xls"
OUTPUT_ROOT = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Rheology_Hyd20"
)
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "rheo_hyd20_results.pkl"

# Metadata rules (editable)
COMMENT_PATTERNS = {
    "age_d": re.compile(r"(\d+)\s*tag\w*\s*alt", re.I),
    "cast_date": re.compile(r"ge[ck]astet\s*(\d{1,2})\.(\d{1,2})\.?(\d{2,4})?", re.I),
    "prep": re.compile(r"(?:von\s*(\d{1,2}\.\d{1,2}\.\d{2,4}))|(?:(\d{1,2})\.\s*hergestellt)", re.I),
    "position": re.compile(r"(?<![A-Za-z0-9])([A-H][1-9])(?![0-9])"),
    "cracks": re.compile(r"riss", re.I),
    "dilution": re.compile(r"(\d+)\s*x\s*verd", re.I),
    "gap_note_mm": re.compile(r"Abstand\s*(\d*[.,]\d+|\d+)\s*mm", re.I),
    "repeat": re.compile(r"(\d+)\.\s*Messung", re.I),
}
CAST_INFERENCE_TOLERANCE_D = 2

# Manual corrections, keyed by file stem. Allowed fields: position, cast_date (datetime.date),
# age_d, exclude_from_stats (bool); "reason" is required. An entry with only "reason" adds an
# INFO note to the QC table without changing anything.
_LAB_NOTE_20_24 = "lab note 10.Nov.txt: '20 -24 eine Probe' (comment positions B3/A2/A2/A2/B2 disregarded)"
SAMPLE_OVERRIDES: Dict[str, dict] = {
    "Hyd20": {"position": "unlabelled", "reason": "no position in comment; first run on the specimen re-measured in Hyd20 (1)"},
    "Hyd20 (1)": {"position": "unlabelled", "reason": "comment '2.Messung!': second run on the specimen of Hyd20.tri"},
    **{f"Hyd20 ({n})": {"position": "runs 20-24", "reason": _LAB_NOTE_20_24} for n in range(20, 25)},
    "Hyd20 (18)": {"reason": "lab note 10.Nov.txt lists runs 11-18 as old gels, the comment of this run gives 13 d "
                             "(new gel); comment kept"},
}

# Strain amplitude of the frequency sweeps (lab note 10.Nov.txt: "0,5% Strain wichtig!")
FS_EXPECTED_STRAIN_PCT = 0.5
FS_STRAIN_REL_TOL = 0.2

# Analysis criteria (editable, see rheo_analysis.py for definitions)
POINT_CRITERIA = PointCriteria(min_torque_Nm=5e-8, max_inertia_ratio=1.0, max_raw_phase_deg=None)
PLATEAU_CRITERIA = PlateauCriteria(
    max_abs_slope=0.10, min_points=5, max_rmse_log10=0.02, max_cv=0.20,
    min_fraction_gp_gt_gpp=1.0, min_r2=None, exclude_low_points=0, exclude_high_points=0,
)
TARGET_FREQUENCY_HZ = 0.5
TARGET_TOLERANCE_DECADES = 0.05          # measured point counts as "at 0.5 Hz" within +-12 %
REPRESENTATIVE_GP_ORDER = ("fit_at_target", "measured_at_target", "plateau_mean")
LVR_CRITERIA = LVRCriteria(n_ref_points=4, deviation_levels=(0.05, 0.10), n_consecutive=2, max_ref_cv=0.05)
# Common grid for the mean frequency sweeps per age: 0.1-100 rad/s, 5 points per decade (TRIOS default grid)
MEAN_CURVE_OMEGA_RAD_S = 10.0 ** np.arange(-1.0, 2.0 + 1e-9, 1 / 5)

# Quality-control thresholds
QC_STRONG_SLOPE = 0.20            # |m| over all valid points of a frequency sweep
QC_RESIDUAL_SCATTER = 0.03        # relative scatter around the window fit, 10**rmse - 1
QC_MAX_LOG_JUMP = 0.10            # decades between neighbouring valid points (about 26 %)
QC_MAX_AXIAL_FORCE_N = 0.5        # |F_N| above: specimen under strong normal load
QC_TEMPERATURE_TOL_C = 0.5
QC_GROUP_OUTLIER_FACTOR = 2.0     # representative G' vs. median of the unflagged runs of the same age
QC_GROUP_MIN_REFERENCE = 3        # unflagged runs needed for that median
QC_GPP_DOMINANT_FRACTION = 0.5    # fraction of valid points with G'' > G'
QC_FREQUENCY_RANGE_MIN_DECADES = 1.0

# Runs carrying one of these QC codes are left out of the "qc_filtered" statistics
# (FAIL runs are left out of every statistic). They stay in all data tables and figures.
STATS_EXCLUDE_QC_CODES = ("axial_force", "presheared", "cracks", "group_outlier")


# ── Metadata ───────────────────────────────────────────────────────────────────

def _flat(text: str) -> str:
    return " | ".join(line.strip() for line in text.splitlines() if line.strip())


def parse_metadata(run: TriRun) -> dict:
    comment = run.comments
    run_date = run.run_datetime.date() if run.run_datetime else None
    meta = {
        "run": run.stem,
        "file": run.path.name,
        "run_datetime": run.run_datetime,
        "run_date": run_date,
        "procedure": run.procedure,
        "comment": _flat(comment),
        "geometry": run.geometry_name,
        "plate_diameter_mm": run.plate_diameter_m * 1e3,
        "instrument_inertia_uNms2": run.instrument_inertia_Nms2 * 1e6,
        "geometry_inertia_uNms2": run.geometry_inertia_Nms2 * 1e6,
        "n_points": len(run.raw),
    }
    m = COMMENT_PATTERNS["age_d"].search(comment)
    meta["age_d_comment"] = int(m.group(1)) if m else np.nan

    meta["cast_date"] = None
    m = COMMENT_PATTERNS["cast_date"].search(comment)
    if m:
        year = int(m.group(3)) if m.group(3) else (run_date.year if run_date else 2000)
        year = year + 2000 if year < 100 else year
        meta["cast_date"] = date(year, int(m.group(2)), int(m.group(1)))

    m = COMMENT_PATTERNS["prep"].search(comment)
    meta["synthesis_note"] = (m.group(1) or f"day {m.group(2)}") if m else ""
    meta["synthesis_date"] = None
    if m and m.group(1):
        day, month, year = (int(v) for v in m.group(1).split("."))
        meta["synthesis_date"] = date(year + 2000 if year < 100 else year, month, day)
    elif m and meta["cast_date"]:
        # Only the day is given ("21.hergestellt"): month and year of the casting date, synthesis before casting.
        cast = meta["cast_date"]
        day = int(m.group(2))
        meta["synthesis_date"] = date(cast.year, cast.month, day) if day <= cast.day else \
            date(cast.year - (cast.month == 1), (cast.month - 2) % 12 + 1, day)

    positions = list(dict.fromkeys(COMMENT_PATTERNS["position"].findall(comment)))
    meta["positions_in_comment"] = "/".join(positions)
    meta["position"] = "/".join(positions) if positions else ""
    meta["position_ambiguous"] = len(positions) > 1

    meta["cracks_noted"] = bool(COMMENT_PATTERNS["cracks"].search(comment))
    m = COMMENT_PATTERNS["dilution"].search(comment)
    meta["dilution_note"] = f"{m.group(1)}x" if m else ""
    m = COMMENT_PATTERNS["gap_note_mm"].search(comment)
    meta["gap_note_mm"] = float(m.group(1).replace(",", ".")) if m else np.nan
    m = COMMENT_PATTERNS["repeat"].search(comment)
    meta["repeat_note"] = f"measurement {m.group(1)}" if m else ""
    meta["gap_um"] = float(np.nanmedian(run.raw["gap_m"])) * 1e6 if len(run.raw) else np.nan
    return meta


def assign_casts(metas: List[dict]) -> None:
    """Cast = casting date (inferred from run date - age where the comment lacks it); batch = synthesis date."""
    for meta in metas:
        override = SAMPLE_OVERRIDES.get(meta["run"], {})
        meta["override_reason"] = override.get("reason", "")
        meta["override_fields"] = ", ".join(k for k in override if k != "reason")
        for key in ("position", "cast_date"):
            if key in override:
                meta[key] = override[key]
        if "age_d" in override:
            meta["age_d_comment"] = override["age_d"]
        meta["exclude_from_stats"] = bool(override.get("exclude_from_stats", False))
        meta["cast_source"] = "comment" if meta["cast_date"] else ""

    known_casts = sorted({m["cast_date"] for m in metas if m["cast_date"]})
    known_synthesis = sorted({m["synthesis_date"] for m in metas if m["synthesis_date"]})
    for meta in metas:
        if meta["cast_date"] is None and known_casts and meta["run_date"] and np.isfinite(meta["age_d_comment"]):
            estimate = meta["run_date"].toordinal() - int(meta["age_d_comment"])
            nearest = min(known_casts, key=lambda d: abs(d.toordinal() - estimate))
            if abs(nearest.toordinal() - estimate) <= CAST_INFERENCE_TOLERANCE_D:
                meta["cast_date"] = nearest
                meta["cast_source"] = "inferred from age and run date"
        if meta["synthesis_date"] is None and len(known_synthesis) == 1:
            meta["synthesis_date"] = known_synthesis[0]
        meta["cast"] = f"cast {meta['cast_date']:%Y-%m-%d}" if meta["cast_date"] else "unknown"
        meta["batch"] = f"batch {meta['synthesis_date']:%Y-%m-%d}" if meta["synthesis_date"] else "unknown"
        meta["age_d_from_dates"] = (
            (meta["run_date"] - meta["cast_date"]).days if meta["cast_date"] and meta["run_date"] else np.nan
        )
        meta["age_d"] = meta["age_d_comment"] if np.isfinite(meta["age_d_comment"]) else meta["age_d_from_dates"]
        position = meta["position"] or f"unlabelled:{meta['run']}"
        meta["specimen_id"] = f"{meta['cast']} | {position}"


# ── Per-run analysis ───────────────────────────────────────────────────────────

def classify(run: TriRun, derived: pd.DataFrame) -> str:
    proc = run.procedure.lower()
    if "frequency" in proc:
        return "frequency_sweep"
    if "amplitude" in proc:
        return "amplitude_sweep"
    if len(derived) > 1:
        f_span = np.nanmax(derived["frequency_Hz"]) / np.nanmin(derived["frequency_Hz"])
        s_span = np.nanmax(derived["strain_pct"]) / np.nanmin(derived["strain_pct"])
        if f_span > 1.5 and s_span < 1.5:
            return "frequency_sweep"
        if s_span > 1.5 and f_span < 1.5:
            return "amplitude_sweep"
    return "unknown"


def analyse_frequency_sweep(data: pd.DataFrame) -> dict:
    f = data["frequency_Hz"].to_numpy()
    gp, gpp = data["Gp_Pa"].to_numpy(), data["Gpp_Pa"].to_numpy()
    valid = (data["point_flag"] == "").to_numpy()
    window = find_plateau_window(f, gp, gpp, valid, PLATEAU_CRITERIA)
    target = evaluate_at_frequency(f, gp, valid, window, TARGET_FREQUENCY_HZ, TARGET_TOLERANCE_DECADES)

    candidates = {
        "fit_at_target": target["Gp_target_fit_Pa"],
        "measured_at_target": target["Gp_target_measured_Pa"],
        "plateau_mean": window.get("Gp_mean_Pa", np.nan) if window.get("found") else np.nan,
    }
    rep_source, rep_value = "none", np.nan
    for key in REPRESENTATIVE_GP_ORDER:
        if np.isfinite(candidates[key]):
            rep_source, rep_value = key, candidates[key]
            break

    overall = loglog_linear_fit(f[valid], gp[valid]) if valid.sum() >= 3 else {}
    gp_interp = interp_loglog(f[valid], gp[valid], TARGET_FREQUENCY_HZ)[0]
    gpp_interp = interp_loglog(f[valid], gpp[valid], TARGET_FREQUENCY_HZ)[0]
    return {
        "window": window,
        "target": target,
        "Gp_rep_Pa": rep_value,
        "Gp_rep_source": rep_source,
        "Gp_0p5Hz_interp_Pa": gp_interp,
        "Gpp_0p5Hz_interp_Pa": gpp_interp,
        "tan_delta_0p5Hz_interp": gpp_interp / gp_interp,
        "overall_slope": overall.get("slope", np.nan),
        "overall_rmse_log10": overall.get("rmse_log10", np.nan),
        "max_log_jump_Gp": max_log_jump(gp, valid),
        "n_valid": int(valid.sum()),
        "fraction_gpp_gt_gp": float(np.mean(gpp[valid] > gp[valid])) if valid.any() else np.nan,
        "f_valid_min_Hz": float(f[valid].min()) if valid.any() else np.nan,
        "f_valid_max_Hz": float(f[valid].max()) if valid.any() else np.nan,
        "strain_median_pct": float(np.nanmedian(data["strain_pct"])),
    }


def analyse_amplitude_sweep(data: pd.DataFrame) -> dict:
    valid = (data["point_flag"] == "").to_numpy()
    return amplitude_sweep_analysis(
        data["strain_pct"].to_numpy(), data["Gp_Pa"].to_numpy(), data["Gpp_Pa"].to_numpy(),
        data["stress_Pa"].to_numpy(), valid, LVR_CRITERIA,
    )


# ── Quality control ────────────────────────────────────────────────────────────

class QC:
    """Collects one row per finding: Sample | Status | Code | Warning | Relevant parameter."""

    LEVELS = {"OK": 0, "INFO": 0, "WARNING": 1, "FAIL": 2}

    def __init__(self) -> None:
        self.rows: List[dict] = []

    def add(self, run: str, status: str, code: str, warning: str, parameter: str) -> None:
        self.rows.append({"Sample": run, "Status": status, "Code": code, "Warning": warning,
                          "Relevant parameter": parameter})

    def status_of(self, run: str) -> str:
        worst = max((self.LEVELS[r["Status"]] for r in self.rows if r["Sample"] == run), default=0)
        return {0: "OK", 1: "WARNING", 2: "FAIL"}[worst]

    def codes_of(self, run: str) -> set:
        return {r["Code"] for r in self.rows if r["Sample"] == run}

    def excluded_from_filtered(self, run: str) -> bool:
        return self.status_of(run) == "FAIL" or bool(self.codes_of(run) & set(STATS_EXCLUDE_QC_CODES))


def qc_common(qc: QC, meta: dict, data: pd.DataFrame, raw: pd.DataFrame) -> None:
    run = meta["run"]
    if meta["n_points"] == 0:
        qc.add(run, "FAIL", "no_data", "no data points stored (run aborted)", "n_points=0")
        return
    n_bad = int((data["point_flag"] != "").sum())
    if n_bad:
        reasons = sorted({r for flag in data["point_flag"] if flag for r in flag.split(";")})
        qc.add(run, "INFO", "points_excluded", "points excluded from fits (kept in data)",
               f"{n_bad}/{len(data)}: {', '.join(reasons)}")
    nonpos = int(((data["Gp_Pa"] <= 0) | (data["Gpp_Pa"] <= 0)).sum())
    if nonpos:
        qc.add(run, "WARNING", "non_positive", "non-positive modulus values (not representable on log axes)",
               f"{nonpos} points")
    force = np.nanmax(np.abs(raw["axial_force_N"])) if raw["axial_force_N"].notna().any() else np.nan
    if np.isfinite(force) and force > QC_MAX_AXIAL_FORCE_N:
        qc.add(run, "WARNING", "axial_force", "large axial force: specimen under strong normal load "
               "(compression or tension), moduli possibly biased",
               f"max |F_N|={force:.2f} N > {QC_MAX_AXIAL_FORCE_N} N (median {np.nanmedian(raw['axial_force_N']):+.2f} N)")
    temp = raw["temperature_C"]
    if temp.notna().any() and (temp.max() - temp.min()) > QC_TEMPERATURE_TOL_C:
        qc.add(run, "WARNING", "temperature", "temperature drift", f"{temp.min():.2f}-{temp.max():.2f} degC")
    if meta["position_ambiguous"]:
        qc.add(run, "WARNING", "position_ambiguous", "several specimen positions in comment; specimen identity ambiguous",
               f"positions={meta['positions_in_comment']}")
    if not meta["position"]:
        qc.add(run, "INFO", "position_missing", "no specimen position in comment; treated as its own specimen", "position=''")
    if meta["cast"] == "unknown":
        qc.add(run, "WARNING", "cast_unknown", "casting date could not be determined", f"comment='{meta['comment']}'")
    elif meta["cast_source"].startswith("inferred"):
        qc.add(run, "INFO", "cast_inferred", "casting date inferred from age and run date", meta["cast"])
    if meta["batch"] == "unknown":
        qc.add(run, "WARNING", "batch_unknown", "synthesis batch could not be determined", f"comment='{meta['comment']}'")
    if np.isfinite(meta["age_d_comment"]) and np.isfinite(meta["age_d_from_dates"]) and \
            abs(meta["age_d_comment"] - meta["age_d_from_dates"]) > 1:
        qc.add(run, "INFO", "age_mismatch", "age in comment differs from run date - casting date",
               f"comment {meta['age_d_comment']:.0f} d vs. dates {meta['age_d_from_dates']:.0f} d")
    if meta["cracks_noted"]:
        qc.add(run, "WARNING", "cracks", "visible cracks noted in comment", meta["comment"])
    if meta["override_reason"]:
        if meta["override_fields"]:
            qc.add(run, "INFO", "override", f"manual metadata override ({meta['override_fields']})", meta["override_reason"])
        else:
            qc.add(run, "INFO", "note", "note on metadata", meta["override_reason"])
    if meta["exclude_from_stats"]:
        qc.add(run, "INFO", "manual_exclusion", "excluded from statistics by SAMPLE_OVERRIDES", meta["override_reason"])


def qc_frequency_sweep(qc: QC, run: str, res: dict) -> None:
    w = res["window"]
    if abs(res["strain_median_pct"] / FS_EXPECTED_STRAIN_PCT - 1) > FS_STRAIN_REL_TOL:
        qc.add(run, "WARNING", "strain_setting", f"strain amplitude differs from {FS_EXPECTED_STRAIN_PCT} %",
               f"median strain {res['strain_median_pct']:.2f} %")
    if res["n_valid"] < PLATEAU_CRITERIA.min_points:
        qc.add(run, "FAIL", "too_few_points", "fewer valid points than required for the plateau fit",
               f"n_valid={res['n_valid']} < {PLATEAU_CRITERIA.min_points}")
        return
    if not w.get("found"):
        qc.add(run, "FAIL", "no_plateau", "no window fulfils the plateau criteria", w.get("failed_criteria", ""))
    elif 10 ** w["rmse_log10"] - 1 > QC_RESIDUAL_SCATTER:
        qc.add(run, "WARNING", "window_scatter", "high scatter of G' around the window fit",
               f"relative scatter {10 ** w['rmse_log10'] - 1:.3f} (CV incl. slope {w['Gp_cv']:.3f})")
    if np.isfinite(res["overall_slope"]) and abs(res["overall_slope"]) > QC_STRONG_SLOPE:
        qc.add(run, "WARNING", "strong_slope", "unusually strong frequency dependence of G'",
               f"m_all={res['overall_slope']:.3f}")
    if res["fraction_gpp_gt_gp"] > QC_GPP_DOMINANT_FRACTION:
        qc.add(run, "WARNING", "gpp_dominant", "G'' > G' over most of the sweep", f"fraction={res['fraction_gpp_gt_gp']:.2f}")
    if np.isfinite(res["max_log_jump_Gp"]) and res["max_log_jump_Gp"] > QC_MAX_LOG_JUMP:
        qc.add(run, "WARNING", "discontinuity", "discontinuity in G' between neighbouring points",
               f"max |dlog10 G'|={res['max_log_jump_Gp']:.3f}")
    if np.log10(res["f_valid_max_Hz"] / res["f_valid_min_Hz"]) < QC_FREQUENCY_RANGE_MIN_DECADES:
        qc.add(run, "WARNING", "narrow_range", "valid frequency range narrower than required",
               f"{res['f_valid_min_Hz']:.3g}-{res['f_valid_max_Hz']:.3g} Hz")
    t = res["target"]
    if not t["target_inside_window"]:
        qc.add(run, "WARNING", "target_outside_window",
               f"{TARGET_FREQUENCY_HZ} Hz outside the selected window; no fitted value",
               f"window {w.get('f_min_Hz', np.nan):.3g}-{w.get('f_max_Hz', np.nan):.3g} Hz")
    if not np.isfinite(t["Gp_target_measured_Pa"]):
        qc.add(run, "INFO" if t["target_inside_window"] else "WARNING", "target_not_measured",
               f"no measured point within {TARGET_TOLERANCE_DECADES} decades of {TARGET_FREQUENCY_HZ} Hz",
               "fitted value used" if t["target_inside_window"] else "no value at target frequency")


def qc_amplitude_sweep(qc: QC, run: str, res: dict) -> None:
    if "error" in res:
        qc.add(run, "FAIL", "as_not_analysable", "amplitude sweep not analysable", res["error"])
        return
    if res["Gp_ref_cv"] > LVR_CRITERIA.max_ref_cv:
        qc.add(run, "WARNING", "ref_scatter", "low-strain reference points scatter", f"CV={res['Gp_ref_cv']:.3f}")
    if res["crossover_status"].startswith("G''"):
        qc.add(run, "WARNING", "no_solid_response", "G'' >= G' at the lowest strain", res["crossover_status"])
    for level in LVR_CRITERIA.deviation_levels:
        tag = f"{int(round(level * 100))}pct"
        if not np.isfinite(res[f"strain_{tag}_pct"]):
            qc.add(run, "INFO", "lvr_not_left", f"G' stays within {tag} of the reference over the measured range",
                   f"max strain {res['strain_max_pct']:.3g} %")
    if res["crossover_status"] == "not reached in measured range":
        qc.add(run, "INFO", "no_crossover", "no G'/G'' crossover in measured range",
               f"max strain {res['strain_max_pct']:.3g} %")


def qc_sequence(qc: QC, metas: Dict[str, dict], results: Dict[str, dict]) -> None:
    """Frequency sweeps that follow a large-strain amplitude sweep on the same specimen and day."""
    for run, meta in metas.items():
        if results[run]["type"] != "frequency_sweep" or meta["run_datetime"] is None:
            continue
        for other, om in metas.items():
            res = results[other]
            if res["type"] != "amplitude_sweep" or res["as"] is None or om["specimen_id"] != meta["specimen_id"]:
                continue
            if om["run_date"] != meta["run_date"] or om["run_datetime"] >= meta["run_datetime"]:
                continue
            gamma10 = res["as"].get("strain_10pct_pct", np.nan)
            if np.isfinite(gamma10) and res["as"].get("strain_max_pct", 0) > gamma10:
                qc.add(run, "WARNING", "presheared", "measured after a large-strain amplitude sweep on the same "
                       "specimen (possibly pre-sheared, unless the specimen was reloaded)",
                       f"{other}: max strain {res['as']['strain_max_pct']:.3g} % > gamma_10% {gamma10:.3g} %")


def qc_group_outliers(qc: QC, fs_table: pd.DataFrame) -> None:
    """Compare every run with the median of the runs of its group that carry no excluding code."""
    for _, group in fs_table.dropna(subset=["Gp_rep_Pa"]).groupby(["batch", "age_d"]):
        reference = group[~group["run"].map(qc.excluded_from_filtered)]
        if len(reference) < QC_GROUP_MIN_REFERENCE:
            continue
        median = reference["Gp_rep_Pa"].median()
        for _, row in group.iterrows():
            ratio = row["Gp_rep_Pa"] / median
            if ratio > QC_GROUP_OUTLIER_FACTOR or ratio < 1 / QC_GROUP_OUTLIER_FACTOR:
                qc.add(row["run"], "WARNING", "group_outlier", "representative G' deviates strongly from the runs of the same age",
                       f"G'={row['Gp_rep_Pa']:.0f} Pa vs. median {median:.0f} Pa of {len(reference)} unflagged runs (x{ratio:.2f})")


# ── Tables ─────────────────────────────────────────────────────────────────────

SELECTIONS = ("all_analysable", "qc_filtered")
META_COLUMNS = ["run", "batch", "cast", "age_d", "specimen_id", "position", "gap_um",
                "run_datetime", "cracks_noted", "dilution_note"]
FS_VALUES = ["Gp_rep_Pa", "Gp_mean_Pa", "Gpp_mean_Pa", "tan_delta_mean", "slope_m",
             "Gpp_0p5Hz_interp_Pa", "tan_delta_0p5Hz_interp"]
AS_VALUES = ["Gp_ref_Pa", "strain_5pct_pct", "strain_10pct_pct", "crossover_strain_pct", "crossover_stress_Pa"]


def frequency_sweep_row(meta: dict, res: dict, qc_status: str) -> dict:
    w, t = res["window"], res["target"]
    row = {k: meta[k] for k in META_COLUMNS}
    row.update({
        "qc_status": qc_status,
        "plateau_found": w.get("found", False),
        "window_f_min_Hz": w.get("f_min_Hz", np.nan),
        "window_f_max_Hz": w.get("f_max_Hz", np.nan),
        "window_n_points": w.get("n", 0),
        "Gp_mean_Pa": w.get("Gp_mean_Pa", np.nan),
        "Gp_median_Pa": w.get("Gp_median_Pa", np.nan),
        "Gp_sd_Pa": w.get("Gp_sd_Pa", np.nan),
        "Gp_cv": w.get("Gp_cv", np.nan),
        "slope_m": w.get("slope", np.nan),
        "slope_m_se": w.get("slope_se", np.nan),
        "intercept_log10_Pa": w.get("intercept", np.nan),
        "r2": w.get("r2", np.nan),
        "rmse_log10": w.get("rmse_log10", np.nan),
        "Gpp_mean_Pa": w.get("Gpp_mean_Pa", np.nan),
        "tan_delta_mean": w.get("tan_delta_mean", np.nan),
        "window_failed_criteria": w.get("failed_criteria", ""),
        "Gp_0p5Hz_measured_Pa": t["Gp_target_measured_Pa"],
        "f_measured_near_0p5Hz_Hz": t["f_target_measured_Hz"],
        "Gp_0p5Hz_fit_Pa": t["Gp_target_fit_Pa"],
        "Gp_rep_Pa": res["Gp_rep_Pa"],
        "Gp_rep_source": res["Gp_rep_source"],
        "Gp_0p5Hz_interp_Pa": res["Gp_0p5Hz_interp_Pa"],
        "Gpp_0p5Hz_interp_Pa": res["Gpp_0p5Hz_interp_Pa"],
        "tan_delta_0p5Hz_interp": res["tan_delta_0p5Hz_interp"],
        "overall_slope_m": res["overall_slope"],
        "n_points": meta["n_points"],
        "n_valid_points": res["n_valid"],
    })
    if not w.get("found"):
        for key in ("Gp_mean_Pa", "Gp_median_Pa", "Gp_sd_Pa", "Gp_cv", "slope_m", "slope_m_se",
                    "intercept_log10_Pa", "r2", "rmse_log10", "Gpp_mean_Pa", "tan_delta_mean"):
            row[key] = np.nan
    return row


def amplitude_sweep_row(meta: dict, res: dict, qc_status: str) -> dict:
    row = {k: meta[k] for k in META_COLUMNS}
    row["qc_status"] = qc_status
    row["n_points"] = meta["n_points"]
    row.update({k: v for k, v in res.items()})
    return row


def specimen_level(table: pd.DataFrame, value_columns: List[str]) -> pd.DataFrame:
    """Mean of technical repeats per specimen and age (N_repeats kept)."""
    keys = ["batch", "cast", "age_d", "specimen_id"]
    agg = table.groupby(keys, dropna=False)[value_columns].mean().reset_index()
    agg["n_repeats"] = table.groupby(keys, dropna=False).size().to_numpy()
    agg["runs"] = table.groupby(keys, dropna=False)["run"].apply(lambda s: ", ".join(s)).to_numpy()
    return agg


def _group_summary(specimens: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    """Specimen-level values -> N, mean, SD, CV per group, per selection; G' normalised to the youngest age."""
    rows = []
    fs_spec = specimens[specimens["sweep_type"] == "frequency_sweep"]
    as_spec = specimens[specimens["sweep_type"] == "amplitude_sweep"]
    for selection in SELECTIONS:
        groups = specimens.loc[specimens["selection"] == selection, group_keys].drop_duplicates()
        for _, g in groups.sort_values(group_keys).iterrows():
            row = {"selection": selection, **g.to_dict()}
            for frame, values, kind in ((fs_spec, FS_VALUES, "fs"), (as_spec, AS_VALUES, "as")):
                mask = frame["selection"] == selection
                for key in group_keys:
                    mask &= frame[key] == g[key]
                sub = frame[mask]
                row[f"{kind}_N_specimens"] = len(sub)
                row[f"{kind}_N_runs"] = int(sub["n_repeats"].sum())
                row[f"{kind}_casts"] = ", ".join(f"{c} (N={n})" for c, n in sub["cast"].value_counts().sort_index().items())
                for col in values:
                    d = describe(sub[col].tolist())
                    row.update({f"{col}_mean": d["mean"], f"{col}_sd": d["sd"], f"{col}_cv": d["cv"]})
            rows.append(row)
    summary = pd.DataFrame(rows)
    if len(summary):
        summary["Gp_rep_norm_to_first_age"] = np.nan
        summary["normalisation_age_d"] = np.nan
        norm_keys = ["selection"] + [k for k in group_keys if k != "age_d"]
        for _, group in summary.groupby(norm_keys):
            first = group.dropna(subset=["Gp_rep_Pa_mean"]).sort_values("age_d")
            if len(first):
                summary.loc[group.index, "Gp_rep_norm_to_first_age"] = group["Gp_rep_Pa_mean"] / first["Gp_rep_Pa_mean"].iloc[0]
                summary.loc[group.index, "normalisation_age_d"] = first["age_d"].iloc[0]
    return summary


def _select(table: pd.DataFrame, selection: str, metas: Dict[str, dict]) -> pd.DataFrame:
    """Runs entering the statistics of a selection: never FAIL or manually excluded runs."""
    t = table[table["qc_status"] != "FAIL"]
    t = t[~t["run"].map(lambda r: metas[r]["exclude_from_stats"])]
    return t[~t["excluded_by_qc_codes"].astype(bool)] if selection == "qc_filtered" else t


def mean_frequency_sweeps(fs: pd.DataFrame, results: Dict[str, dict], metas: Dict[str, dict]) -> pd.DataFrame:
    """Mean +- SD of G'(f), G''(f), tan(delta)(f) over specimens of the same batch and age.

    Every sweep is interpolated in log-log onto the common grid MEAN_CURVE_OMEGA_RAD_S within the
    range of its valid points (no extrapolation); technical repeats are averaged per specimen
    first, so N counts specimens at every grid frequency. tan(delta) is G''/G' per specimen.
    """
    grid_f = MEAN_CURVE_OMEGA_RAD_S / (2 * np.pi)
    frames = []
    for selection in SELECTIONS:
        for _, row in _select(fs, selection, metas).iterrows():
            if not np.isfinite(row.get("Gp_rep_Pa", np.nan)):
                continue
            data = results[row["run"]]["data"]
            valid = (data["point_flag"] == "").to_numpy()
            f = data["frequency_Hz"].to_numpy()[valid]
            frames.append(pd.DataFrame({
                "selection": selection, "batch": row["batch"], "cast": row["cast"], "age_d": row["age_d"],
                "specimen_id": row["specimen_id"], "frequency_Hz": grid_f,
                "Gp_Pa": interp_loglog(f, data["Gp_Pa"].to_numpy()[valid], grid_f),
                "Gpp_Pa": interp_loglog(f, data["Gpp_Pa"].to_numpy()[valid], grid_f),
            }))
    runs = pd.concat(frames, ignore_index=True).dropna(subset=["Gp_Pa", "Gpp_Pa"])
    keys = ["selection", "batch", "age_d", "frequency_Hz"]
    spec = runs.groupby(keys + ["cast", "specimen_id"])[["Gp_Pa", "Gpp_Pa"]].mean().reset_index()
    spec["tan_delta"] = spec["Gpp_Pa"] / spec["Gp_Pa"]
    grouped = spec.groupby(keys)
    out = grouped[["Gp_Pa", "Gpp_Pa", "tan_delta"]].agg(["mean", "std"])
    out.columns = [f"{col}_{stat.replace('std', 'sd')}" for col, stat in out.columns]
    out["N_specimens"] = grouped.size()
    out["casts"] = grouped["cast"].apply(lambda s: ", ".join(sorted(set(s))))
    return out.reset_index()


def stability_tables(fs: pd.DataFrame, amp: pd.DataFrame, metas: Dict[str, dict]
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Specimen-level table, summary per batch and age (casts pooled) and per batch, cast and age."""
    spec_frames = []
    for selection in SELECTIONS:
        for table, values, kind in ((fs, FS_VALUES, "frequency_sweep"), (amp, AS_VALUES, "amplitude_sweep")):
            chosen = _select(table, selection, metas)
            if len(chosen):
                spec_frames.append(specimen_level(chosen, values).assign(selection=selection, sweep_type=kind))
    specimens = pd.concat(spec_frames, ignore_index=True)
    pooled = _group_summary(specimens, ["batch", "age_d"])
    by_cast = _group_summary(specimens, ["batch", "cast", "age_d"])
    return pooled, by_cast, specimens


def fit_parameter_table(fs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in fs.iterrows():
        for p, se in (("slope_m", "slope_m_se"), ("intercept_log10_Pa", None), ("r2", None), ("rmse_log10", None)):
            rows.append({"fit": "frequency sweep window: log10 G' = a + m log10 f", "run": r["run"],
                         "parameter": p, "value": r[p], "stderr": r[se] if se else np.nan,
                         "x_min": r["window_f_min_Hz"], "x_max": r["window_f_max_Hz"], "n_points": r["window_n_points"]})
    return pd.DataFrame(rows)


# ── TRIOS export validation ────────────────────────────────────────────────────

def validate_against_xls(results: Dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import xlrd  # noqa: F401
    except ImportError:
        print("xlrd not installed; TRIOS .xls validation skipped.")
        return pd.DataFrame(), pd.DataFrame()
    rows, points = [], []
    for xls in sorted(RAW_DIR.glob(XLS_GLOB)):
        details, table = read_trios_xls(xls)
        run = str(details.get("Name", xls.stem))
        if run not in results or len(results[run]["data"]) != len(table):
            rows.append({"xls_file": xls.name, "run": run, "column": "-", "status": "no matching .tri run"})
            continue
        data = results[run]["data"].sort_values("raw_step_time_s").reset_index(drop=True)
        table = table.sort_values("step_time_s").reset_index(drop=True)
        for col in table.columns:
            ours = data[col].to_numpy() if col in data else data[f"raw_{col}"].to_numpy()
            ref = table[col].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(ours / ref - 1.0)
            rows.append({"xls_file": xls.name, "run": run, "column": col, "status": "compared",
                         "n_points": len(ref), "max_rel_deviation": float(np.nanmax(rel)),
                         "median_rel_deviation": float(np.nanmedian(rel))})
        points.append(pd.DataFrame({"run": run, "Gp_xls_Pa": table["Gp_Pa"], "Gp_tri_Pa": data["Gp_Pa"],
                                    "Gpp_xls_Pa": table["Gpp_Pa"], "Gpp_tri_Pa": data["Gpp_Pa"]}))
    return pd.DataFrame(rows), (pd.concat(points, ignore_index=True) if points else pd.DataFrame())


# ── Main ───────────────────────────────────────────────────────────────────────

def _safe_name(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def main() -> None:
    tri_files = sorted(RAW_DIR.glob(TRI_GLOB), key=lambda p: (len(p.stem), p.stem))
    if not tri_files:
        raise FileNotFoundError(f"No {TRI_GLOB} files in {RAW_DIR}")

    runs = {p.stem: read_tri(p) for p in tri_files}
    meta_list = [parse_metadata(r) for r in runs.values()]
    assign_casts(meta_list)
    metas = {m["run"]: m for m in meta_list}

    qc = QC()
    results: Dict[str, dict] = {}
    for stem, run in runs.items():
        meta = metas[stem]
        for note in run.parse_notes:
            qc.add(stem, "WARNING", "parser_note", "parser note", note)
        inertia = run.instrument_inertia_Nms2 + run.geometry_inertia_Nms2
        derived = oscillation_quantities(run.raw, run.plate_diameter_m, inertia)
        kind = classify(run, derived)
        x_col = "frequency_Hz" if kind == "frequency_sweep" else "strain_pct"
        order = np.argsort(derived[x_col].to_numpy(), kind="stable")
        derived = derived.iloc[order].reset_index(drop=True)
        raw = run.raw.iloc[order].reset_index(drop=True)
        derived["point_flag"] = point_artifacts(derived, raw, x_col, POINT_CRITERIA)
        data = pd.concat([derived, raw.add_prefix("raw_")], axis=1)
        results[stem] = {"type": kind, "data": data, "raw": raw, "fs": None, "as": None}

        qc_common(qc, meta, derived, raw)
        if meta["n_points"] == 0:
            continue
        if kind == "frequency_sweep":
            results[stem]["fs"] = analyse_frequency_sweep(derived)
            qc_frequency_sweep(qc, stem, results[stem]["fs"])
        elif kind == "amplitude_sweep":
            results[stem]["as"] = analyse_amplitude_sweep(derived)
            qc_amplitude_sweep(qc, stem, results[stem]["as"])
        else:
            qc.add(stem, "FAIL", "type_unknown", "measurement type not identified", f"procedure='{run.procedure}'")

    qc_sequence(qc, metas, results)
    fs_rows = [frequency_sweep_row(metas[s], r["fs"], "") for s, r in results.items() if r["fs"] is not None]
    fs_table = pd.DataFrame(fs_rows)
    qc_group_outliers(qc, fs_table)

    status = {s: qc.status_of(s) for s in results}
    excluded = {s: qc.excluded_from_filtered(s) for s in results}
    fs_table["qc_status"] = fs_table["run"].map(status)
    as_table = pd.DataFrame([amplitude_sweep_row(metas[s], r["as"], status[s])
                             for s, r in results.items() if r["as"] is not None])
    # Runs without any analysis (empty / unknown type) still appear in both summaries.
    for s, r in results.items():
        if r["fs"] is None and r["as"] is None:
            row = {k: metas[s][k] for k in META_COLUMNS}
            row.update(qc_status=status[s], n_points=metas[s]["n_points"])
            if r["type"] == "amplitude_sweep":
                as_table = pd.concat([as_table, pd.DataFrame([row])], ignore_index=True)
            else:
                fs_table = pd.concat([fs_table, pd.DataFrame([row])], ignore_index=True)

    for table in (fs_table, as_table):
        table["excluded_by_qc_codes"] = table["run"].map(excluded)
        table["qc_codes"] = table["run"].map(lambda s: ", ".join(sorted(qc.codes_of(s))))
    stability, stability_by_cast, specimens = stability_tables(fs_table, as_table, metas)
    mean_sweeps = mean_frequency_sweeps(fs_table, results, metas)
    fit_params = fit_parameter_table(fs_table.dropna(subset=["window_n_points"]))
    validation, validation_points = validate_against_xls(results)

    flagged = {row["Sample"] for row in qc.rows}
    for s in results:
        if s not in flagged:
            qc.add(s, "OK", "", "", "")
    qc_table = pd.DataFrame(qc.rows)
    qc_table["run_status"] = qc_table["Sample"].map(status)
    order = {s: i for i, s in enumerate(results)}
    qc_table = qc_table.sort_values("Sample", key=lambda c: c.map(order), kind="stable")

    meta_table = pd.DataFrame(meta_list)
    meta_table["measurement_type"] = meta_table["run"].map(lambda s: results[s]["type"])
    meta_table["qc_status"] = meta_table["run"].map(status)
    meta_table["excluded_by_qc_codes"] = meta_table["run"].map(excluded)

    # ── Write outputs ──
    dirs = {name: OUTPUT_ROOT / name for name in ("processed_data", "tables")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    long_frames = []
    for stem, r in results.items():
        info = {"run": stem, "measurement_type": r["type"],
                **{k: metas[stem][k] for k in ("batch", "cast", "age_d", "specimen_id", "position")}}
        frame = r["data"].assign(**info)[list(info) + list(r["data"].columns)]
        frame.to_csv(dirs["processed_data"] / f"{_safe_name(stem)}.csv", index=False)
        long_frames.append(frame)
    pd.concat(long_frames, ignore_index=True).to_csv(dirs["processed_data"] / "all_measurements.csv", index=False)

    tables = {
        "frequency_sweep_summary": fs_table,
        "amplitude_sweep_summary": as_table,
        "stability_summary": stability,
        "stability_summary_by_cast": stability_by_cast,
        "stability_specimen_level": specimens,
        "mean_frequency_sweeps": mean_sweeps,
        "quality_control": qc_table,
        "measurement_metadata": meta_table,
        "fit_parameters": fit_params,
        "tri_vs_xls_validation": validation,
    }
    for name, table in tables.items():
        table.to_csv(dirs["tables"] / f"{name}.csv", index=False)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump({
            "created": datetime.now(),
            "raw_dir": str(RAW_DIR),
            "output_root": str(OUTPUT_ROOT),
            "config": {
                "point_criteria": asdict(POINT_CRITERIA), "plateau_criteria": asdict(PLATEAU_CRITERIA),
                "lvr_criteria": asdict(LVR_CRITERIA), "target_frequency_Hz": TARGET_FREQUENCY_HZ,
                "representative_gp_order": REPRESENTATIVE_GP_ORDER,
                "stats_exclude_qc_codes": STATS_EXCLUDE_QC_CODES,
            },
            "metadata": metas,
            "results": results,
            "tables": tables,
            "validation_points": validation_points,
        }, fh)

    # ── Console summary ──
    print(f"{len(results)} runs parsed from {RAW_DIR}")
    print(meta_table[["run", "measurement_type", "batch", "cast", "age_d", "position", "gap_um", "qc_status"]]
          .to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    if len(validation):
        compared = validation[validation["status"] == "compared"]
        print("\nTRIOS export check (max relative deviation per column):")
        print(compared.groupby("column")["max_rel_deviation"].max().to_string())
    cols = ["selection", "age_d", "fs_N_specimens", "fs_N_runs", "fs_casts", "Gp_rep_Pa_mean", "Gp_rep_Pa_sd",
            "Gp_rep_norm_to_first_age", "tan_delta_mean_mean", "as_N_specimens", "strain_10pct_pct_mean"]
    print("\nStability summary (casts pooled):")
    print(stability[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nTables: {dirs['tables']}\nCache:  {CACHE_PATH}")


if __name__ == "__main__":
    main()
