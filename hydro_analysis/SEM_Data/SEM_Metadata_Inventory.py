"""
SEM metadata inventory.

Recursively finds all TIFFs under ROOT_DIR, reads each one's embedded FEI/
Thermo Fisher SEM metadata block (TIFF tag 34682 -- an INI-style text block
written by the microscope's acquisition software: [Section] / key=value),
and collects the acquisition parameters relevant to interpreting the image
(beam voltage, spot size, working distance, pixel size, field of view,
magnification, dwell/frame time, detector, vacuum condition, stage, ...)
into one row per file.

Files without a tag-34682 block (e.g. cropped/re-exported derivatives) are
skipped with a warning, not silently dropped -- they're listed in the
"skipped" summary at the end.

A second table then groups files by their shared acquisition "recipe"
(detector, HV, spot size, scan speed/dwell time, line integration, vacuum
mode, magnification) and reports how many files used each combination --
e.g. "ETD, SE, 2.0 kV, dwell 5 us -> 12 files".

Metadata parsing (parse_fei_text_block / safe_float) follows the same
approach already used for FEI TIFFs in SEM_Particles/Sem_Peakfinding.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

# ── Configuration ───────────────────────────────────────────────────────────
ROOT_DIR = Path(r"H:\Daten Promotion Sicherung\SEM")
ROOT_DIR = Path(r"H:\Daten Promotion Sicherung\Diffusion in Hydrogel Data\SEM Particles")
OUT_CSV = Path(__file__).parent / "sem_metadata_inventory.csv"
SUMMARY_CSV = Path(__file__).parent / "sem_metadata_settings_summary.csv"

FEI_TAG = 34682

# Parameters that define an acquisition "recipe" -- files sharing all of
# these were recorded with the same instrument settings, regardless of
# which sample/region or how far the operator zoomed in (magnification is
# reported as a range per recipe instead of grouped on, since it varies
# essentially continuously shot-to-shot).
RECIPE_COLUMNS = [
    "detector_name", "detector_signal", "HV_kV", "spot_size",
    "dwell_time_us", "line_integration", "vacuum_mode",
]


# ── FEI text-block parsing ──────────────────────────────────────────────────

def parse_fei_text_block(text: str) -> dict[str, dict[str, str]]:
    """Parse the [Section]/key=value FEI metadata block into a nested dict."""
    sections: dict[str, dict[str, str]] = {}
    current_section = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            sections[current_section] = {}
            continue
        if "=" in line and current_section is not None:
            key, value = line.split("=", 1)
            sections[current_section][key.strip()] = value.strip()
    return sections


def safe_float(d: dict, key: str) -> float | None:
    val = d.get(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ── Per-file extraction ──────────────────────────────────────────────────────

def read_sem_metadata(tif_path: Path) -> dict | None:
    """Open one TIFF and return a flat dict of relevant acquisition
    parameters, or None if it has no FEI metadata block."""
    with Image.open(tif_path) as img:
        width, height = img.size
        raw = img.tag_v2.get(FEI_TAG)

    if raw is None:
        return None
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    sections = parse_fei_text_block(text)

    user = sections.get("User", {})
    system = sections.get("System", {})
    ebeam = sections.get("EBeam", {})
    scan = sections.get("Scan", {})
    escan = sections.get("EScan", {})
    stage = sections.get("Stage", {})
    detectors = sections.get("Detectors", {})
    vacuum = sections.get("Vacuum", {})

    detector_name = detectors.get("Name")
    detector_sec = sections.get(detector_name, {}) if detector_name else {}

    px_w_m = safe_float(scan, "PixelWidth")
    px_h_m = safe_float(scan, "PixelHeight")
    hfw_m = safe_float(scan, "HorFieldsize") or safe_float(ebeam, "HFW")
    vfw_m = safe_float(scan, "VerFieldsize") or safe_float(ebeam, "VFW")
    display_width_m = safe_float(system, "DisplayWidth")
    dwell_s = safe_float(scan, "Dwelltime") or safe_float(escan, "Dwell")
    hv_V = safe_float(ebeam, "HV")
    wd_m = safe_float(ebeam, "WD") or safe_float(stage, "WorkingDistance")

    magnification = (display_width_m / hfw_m) if (display_width_m and hfw_m) else None

    date = user.get("Date")
    time = user.get("Time")
    acquired = f"{date} {time}" if date and time else (date or time)

    return {
        "file": str(tif_path),
        "folder": str(tif_path.parent.relative_to(ROOT_DIR)),
        "filename": tif_path.name,
        "acquired": acquired,
        "microscope_type": system.get("Type"),
        "software_version": system.get("Software"),
        "image_width_px": width,
        "image_height_px": height,
        "HV_kV": (hv_V / 1000.0) if hv_V is not None else None,
        "spot_size": safe_float(sections.get("Beam", {}), "Spot"),
        "WD_mm": (wd_m * 1e3) if wd_m is not None else None,
        "pixel_width_nm": (px_w_m * 1e9) if px_w_m is not None else None,
        "pixel_height_nm": (px_h_m * 1e9) if px_h_m is not None else None,
        "HFW_um": (hfw_m * 1e6) if hfw_m is not None else None,
        "VFW_um": (vfw_m * 1e6) if vfw_m is not None else None,
        "magnification_x": magnification,
        "dwell_time_us": (dwell_s * 1e6) if dwell_s is not None else None,
        "frame_time_s": safe_float(scan, "FrameTime"),
        "line_integration": safe_float(escan, "LineIntegration"),
        "detector_name": detector_name,
        "detector_signal": detector_sec.get("Signal"),
        "vacuum_mode": vacuum.get("UserMode"),
        "chamber_pressure_mbar": safe_float(vacuum, "ChPressure"),
        "active_stage": stage.get("ActiveStage"),
        "scan_rotation_deg": safe_float(ebeam, "ScanRotation"),
    }


# ── Acquisition-recipe summary ──────────────────────────────────────────────

def summarize_acquisition_settings(df: pd.DataFrame,
                                   group_cols: list[str] = RECIPE_COLUMNS) -> pd.DataFrame:
    """Group files by shared acquisition settings and count how many files
    used each combination -- e.g. 'ETD, SE, 2.0 kV, dwell 5 us -> 12 files'.
    """
    cols = [c for c in group_cols if c in df.columns]
    g = df.groupby(cols, dropna=False)
    summary = g.agg(
        n_files=("filename", "count"),
        magnification_min_x=("magnification_x", "min"),
        magnification_max_x=("magnification_x", "max"),
        folders=("folder", lambda s: ", ".join(sorted(s.unique()))),
        example_files=("filename", lambda s: ", ".join(sorted(s)[:3]) + (", ..." if len(s) > 3 else "")),
    ).reset_index()
    return summary.sort_values("n_files", ascending=False).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ROOT_DIR.exists():
        print(f"ERROR: folder not found: {ROOT_DIR}")
        return

    tif_paths = sorted(set(ROOT_DIR.rglob("*.tif")) | set(ROOT_DIR.rglob("*.tiff")))
    print(f"Found {len(tif_paths)} TIFF file(s) under {ROOT_DIR}")

    rows = []
    skipped = []
    for p in tif_paths:
        try:
            meta = read_sem_metadata(p)
        except Exception as e:
            print(f"  [ERROR] {p.name}: {e}")
            skipped.append((p, f"error: {e}"))
            continue
        if meta is None:
            print(f"  [SKIP no metadata] {p.relative_to(ROOT_DIR)}")
            skipped.append((p, "no FEI metadata tag"))
            continue
        rows.append(meta)

    if not rows:
        print("No files with readable SEM metadata found.")
        return

    df = pd.DataFrame(rows).sort_values(["folder", "filename"]).reset_index(drop=True)

    print(f"\nRead metadata from {len(df)} file(s); skipped {len(skipped)}.")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    settings_summary = summarize_acquisition_settings(df)
    print(f"\n── Acquisition recipes ({len(settings_summary)} distinct combinations) ──")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(settings_summary.to_string(index=False))
    settings_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved: {SUMMARY_CSV}")

    if skipped:
        print("\nSkipped files:")
        for p, reason in skipped:
            print(f"  {p.relative_to(ROOT_DIR)}: {reason}")


if __name__ == "__main__":
    main()
