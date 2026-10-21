"""
Irradiance overview of all SPT measurements: laser power from the .rec files,
irradiance E (W/m^2, W/cm^2) and incident power per particle for the particle size
of each measurement.

File set: the same as REC_files scraping.py (Parameter_Check), i.e. every TrackMate
XML under SPT_ROOT with its paired .rec file found by core.io.find_rec_tif_files.
XMLs of the same recording (Analysis/, Tracks/, ...) share one .rec file and are
collapsed to one row per recording; copies of a recording in several folders (same
name and record time) count once. No tracks are loaded and no MSD is fitted.
Particle size: core.io.extract_particle_size_from_path on the XML file name, as in
REC_files scraping.py; if that gives no nominal size, the parent folders and a
"<n>k" token (e.g. "1k" = 1000 nm) are used. The diameter is the DLS label of the
nominal size (core.io.get_dls_labels).
Wavelength: 488 or 514 nm if stated in the .rec comment, else DEFAULT_WAVELENGTH_NM.
Laser power and depth: core.io.load_inherited_rec_metadata(), i.e. the stated value or,
if none is stated, the value inherited from the preceding recording in the same folder
(Parameter_Check/rec_metadata_inheritance.py; power_status and power_inherited_from
document the source). The .rec files are not modified.

Normalization: for a fixed beam profile E scales linearly with P, so
    E = P * (E / P)_calibration,
with (E / P) averaged over the power series of that wavelength in
irradiance_results.csv (the S170C power is distributed over the 60 x 45 µm SPT image;
E is an upper bound, see laser_beam_characterization.py). E_focus is the maximum of
the fitted beam inside the image. Power per particle: E_focus * pi d^2 / 4, the power
incident on the geometric cross-section, not the absorbed power.

Stage: consumer stage for the beam calibration. Reads OUTPUT_DIR/irradiance_results.csv
(run laser_beam_characterization.py first) and the .rec metadata of the SPT
measurements (read-only) and Parameter_Check/cache/rec_metadata_inherited.pkl (run
rec_metadata_inheritance.py first); writes OUTPUT_DIR/spt_irradiance_overview.xlsx with the
sheets "measurements" (one row per recording) and "summary" (per particle size,
wavelength and power). No consumers.
"""
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_analysis.core.io import (
    check_text_encoding, extract_particle_size_from_path, find_rec_tif_files, get_dls_labels,
    load_inherited_rec_metadata, parse_rec_comment_metadata, scan_xml_folder,
)

# ── Configuration ──────────────────────────────────────────────────────────────
SPT_ROOT = Path(r"E:\PhD Data Analysis\SPT 2025 II")    # root of REC_files scraping.py
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
                  r"\Experiments and Results - Data\Laser_Characterization\Laser_Beam_Characterization")
CALIBRATION_CSV = OUTPUT_DIR / "irradiance_results.csv"
OUTPUT_XLSX = OUTPUT_DIR / "spt_irradiance_overview.xlsx"
DEFAULT_WAVELENGTH_NM = 488
WAVELENGTH_RE = re.compile(r"\b(488|514)\s*nm", re.IGNORECASE)
KILO_RE = re.compile(r"(?<!\d)(\d+)\s*k(?![a-z])", re.IGNORECASE)


def load_calibration() -> pd.DataFrame:
    if not CALIBRATION_CSV.exists():
        raise FileNotFoundError(f"{CALIBRATION_CSV} missing; run laser_beam_characterization.py first.")
    cal = pd.read_csv(CALIBRATION_CSV)
    cal["k_mean_per_m2"] = cal["E_mean_W_m2"] / cal["P_total_W"]
    cal["k_focus_per_m2"] = cal["E_focus_W_m2"] / cal["P_total_W"]
    return cal.groupby("wavelength_nm").agg(k_mean_per_m2=("k_mean_per_m2", "mean"),
                                            k_focus_per_m2=("k_focus_per_m2", "mean"),
                                            P_cal_min_W=("P_total_W", "min"), P_cal_max_W=("P_total_W", "max"))


def particle_size(xml_path: Path, nominal_sizes: set[float]) -> float:
    size = extract_particle_size_from_path(xml_path)
    if size in nominal_sizes:
        return size
    kilo = KILO_RE.search(xml_path.stem)
    if kilo and float(kilo.group(1)) * 1000.0 in nominal_sizes:
        return float(kilo.group(1)) * 1000.0
    for folder in xml_path.parents[:3]:
        size = extract_particle_size_from_path(folder)
        if size in nominal_sizes:
            return size
    return np.nan


def collect_recordings(dls_labels: dict[float, int]) -> tuple[pd.DataFrame, int]:
    """One row per .rec file; second value: number of XMLs without a .rec file."""
    recordings, n_without_rec = {}, 0
    inherited = load_inherited_rec_metadata()
    for xml_path in sorted(scan_xml_folder(SPT_ROOT)):
        # find_rec_tif_files prints an error for every XML without .rec; counted here instead.
        with contextlib.redirect_stdout(io.StringIO()):
            rec_path = find_rec_tif_files(xml_path)["rec_file"]
        if rec_path is None:
            n_without_rec += 1
            continue
        if rec_path in recordings:
            recordings[rec_path]["n_xml"] += 1
            continue
        text = rec_path.read_text(encoding=check_text_encoding(rec_path) or "utf-16", errors="replace")
        comment = text.partition("Comment:")[2]
        wl = WAVELENGTH_RE.search(comment)
        meta = parse_rec_comment_metadata(rec_path)
        mapped = inherited.loc[str(rec_path.resolve())]
        nominal = particle_size(xml_path, set(dls_labels))
        recordings[rec_path] = {
            "recording": rec_path.name.removesuffix(".rec").removesuffix(".tif"),
            "folder": str(rec_path.parent.relative_to(SPT_ROOT)),
            "particle_nominal_nm": nominal, "particle_dls_nm": dls_labels.get(nominal, np.nan),
            "wavelength_nm": int(wl.group(1)) if wl else DEFAULT_WAVELENGTH_NM,
            "wavelength_source": ".rec comment" if wl else "assumed",
            "laser_power_mW": mapped["laser_power_mW"], "power_status": mapped["power_status"],
            "power_inherited_from": mapped["power_inherited_from"], "power_origin": mapped["power_origin"],
            "depth_um": mapped["depth_um"], "depth_status": mapped["depth_status"],
            "recorded_at": meta["recorded_at"],
            "comment": " | ".join(line.strip() for line in comment.splitlines() if line.strip()),
            "n_xml": 1, "rec_path": str(rec_path)}
    df = pd.DataFrame(recordings.values())
    # Copies of one recording in several folders (date folder and sorted folder) count once.
    key = np.where(df["recorded_at"].notna(), df["recorded_at"].astype(str) + df["recording"], df["rec_path"])
    agg = {c: "first" for c in df.columns} | {"folder": " ; ".join, "rec_path": " ; ".join, "n_xml": "sum"}
    return df.groupby(key, sort=False).agg(agg).reset_index(drop=True), n_without_rec


def normalize(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    c = cal.reindex(df["wavelength_nm"]).reset_index(drop=True)
    df = df.reset_index(drop=True)
    df["P_W"] = df["laser_power_mW"] * 1e-3
    df["E_mean_W_m2"] = df["P_W"] * c["k_mean_per_m2"]
    df["E_focus_W_m2"] = df["P_W"] * c["k_focus_per_m2"]
    df["E_focus_W_cm2"] = df["E_focus_W_m2"] * 1e-4
    df["P_particle_W"] = df["E_focus_W_m2"] * np.pi * (df["particle_dls_nm"] * 1e-9) ** 2 / 4.0
    df["P_particle_nW"] = df["P_particle_W"] * 1e9
    df["P_in_calibrated_range"] = df["P_W"].between(c["P_cal_min_W"], c["P_cal_max_W"])
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["particle_nominal_nm", "particle_dls_nm", "wavelength_nm", "laser_power_mW"]
    return (df.dropna(subset=["laser_power_mW", "particle_dls_nm"])
              .groupby(keys)
              .agg(n_recordings=("recording", "size"), P_W=("P_W", "first"),
                   E_mean_W_m2=("E_mean_W_m2", "first"), E_focus_W_m2=("E_focus_W_m2", "first"),
                   E_focus_W_cm2=("E_focus_W_cm2", "first"), P_particle_W=("P_particle_W", "first"),
                   P_particle_nW=("P_particle_nW", "first"))
              .reset_index())


def main() -> None:
    cal = load_calibration()
    dls_labels = get_dls_labels()
    df, n_without_rec = collect_recordings(dls_labels)
    df = normalize(df, cal)
    summary = summarize(df)

    with pd.ExcelWriter(OUTPUT_XLSX) as writer:
        df.sort_values(["particle_nominal_nm", "laser_power_mW", "recorded_at"]).to_excel(
            writer, sheet_name="measurements", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print("Calibration (E / P, m^-2):")
    print(cal.to_string(float_format=lambda v: f"{v:.3e}"))
    print(f"\n{len(df)} recordings with .rec ({int(df['n_xml'].sum())} XMLs), {n_without_rec} XMLs without .rec; "
          f"{int(df['laser_power_mW'].isna().sum())} recordings without laser power, "
          f"{int(df['particle_dls_nm'].isna().sum())} without particle size.")
    print(f"\n{'d_DLS (nm)':>10} {'nm':>4} {'P (W)':>10} {'E_mean (W/m^2)':>15} {'E_focus (W/m^2)':>16} "
          f"{'E_focus (W/cm^2)':>17} {'P/particle (nW)':>16} {'n':>4}")
    for r in summary.itertuples(index=False):
        print(f"{r.particle_dls_nm:>10.0f} {r.wavelength_nm:>4} {r.P_W:>10.2e} {r.E_mean_W_m2:>15.3e} "
              f"{r.E_focus_W_m2:>16.3e} {r.E_focus_W_cm2:>17.3g} {r.P_particle_nW:>16.3g} {r.n_recordings:>4}")
    print(f"\nOutput: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
