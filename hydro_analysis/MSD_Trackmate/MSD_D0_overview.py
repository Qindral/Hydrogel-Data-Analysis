"""
D₀ overview — individual per-file power-law fits + 20 mg/mL C16 eMSD overlay.

D₀ (water) XML files: all fps allowed, per-file fit lines (thin, colour).
20 mg/mL C16 XML files: all fps allowed, per-file eMSD (thicker, darker).
Stokes-Einstein theory line per size (dashed).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import trackpy as tp

from hydro_analysis.core.io import single_file_data, get_dls_sizes, get_dls_labels
from hydro_analysis.core.analysis import (
    fit_powerlaw_with_errors,
    DEFAULT_MSD_FIT_POINTS,
    MIN_TRACK_LENGTH,
)
from hydro_analysis.core.physics import calculate_theoretical_diffusion
from hydro_analysis.MSD_Trackmate.plot_emsd_publication import plot_fit_lines_overview

tp.quiet()

# ── D₀ water folders ───────────────────────────────────────────────────────────
XML_FOLDERS_D0: dict[float, list[Path]] = {
    20.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm\Tracks"),
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.16\Tracks_20"),
    ],
    50.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_50"),
    ],
    100.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\100 nm\Tracks"),
    ],
    200.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\200 nm\Tracks"),
    ],
    500.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\500 nm\Tracks"),
    ],
    1000.0: [
        Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_1000"),
    ],
}

# ── 20 mg/mL C16 hydrogel folders ─────────────────────────────────────────────
_ROOT_20MG = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")
XML_FOLDERS_20MG: dict[float, list[Path]] = {
    20.0:   [_ROOT_20MG / "20 nm" / "20 nm 20 mg" / "Tracks"],
    50.0:   [_ROOT_20MG / "50 nm" / "50 nm 20 mg" / "Tracks_new"],
    100.0:  [_ROOT_20MG / "100 nm" / "Tracks"],
    200.0:  [_ROOT_20MG / "200 nm" / "Tracks"],
    500.0:  [_ROOT_20MG / "500 nm" / "Tracks"],
    1000.0: [_ROOT_20MG / "1000 nm" / "Tracks"],
}

SAVE_PATH = Path(
    rf"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung"
    rf"\Plots_{pd.Timestamp.now().strftime('%Y%m%d')}"
)
MSD_FIT_POINTS = DEFAULT_MSD_FIT_POINTS


def _process_file(xml_path: Path) -> dict | None:
    """Load one XML → eMSD Series + power-law fit dict, or None on failure."""
    rd = single_file_data(xml_path)
    if rd is None:
        return None

    tracks = rd.get("tracks_df")
    if tracks is None or tracks.empty:
        return None

    mpp = rd["mpp"]
    fps = rd["fps"]

    tracks = tp.filter_stubs(tracks, threshold=MIN_TRACK_LENGTH)
    if tracks.empty:
        return None

    try:
        emsd = tp.emsd(tracks, mpp=mpp, fps=fps)
    except Exception as e:
        print(f"  [SKIP emsd error] {xml_path.name}: {e}")
        return None

    emsd = emsd[emsd > 0]
    if len(emsd) < MSD_FIT_POINTS:
        return None

    try:
        fr = fit_powerlaw_with_errors(emsd, points=MSD_FIT_POINTS)
    except Exception:
        return None

    lag = emsd.index.values.astype(float)
    return {
        "A":       float(fr["A"][0]),
        "n":       float(fr["n"][0]),
        "tau_min": float(lag.min()),
        "tau_max": float(lag.max()),
        "fps":     fps,
        "file":    xml_path.name,
        "emsd":    emsd,
    }


def _collect(folders: dict[float, list[Path]],
             ) -> tuple[dict[float, list[dict]], dict[float, list[pd.Series]]]:
    """Walk all folders, return (fits_by_size, emsd_by_size)."""
    fits:  dict[float, list[dict]]       = {}
    emsds: dict[float, list[pd.Series]] = {}

    for size_nm, folder_list in folders.items():
        fits[size_nm]  = []
        emsds[size_nm] = []

        for folder in folder_list:
            if not folder.exists():
                print(f"WARNING: folder not found: {folder}")
                continue
            for xml_path in sorted(folder.glob("*.xml")):
                result = _process_file(xml_path)
                if result is None:
                    print(f"  [SKIP] {xml_path.name}")
                    continue
                fit_dict = {k: v for k, v in result.items() if k != "emsd"}
                fits[size_nm].append(fit_dict)
                emsds[size_nm].append(result["emsd"])
                print(f"  [{int(size_nm)} nm  {result['fps']:.0f} fps] "
                      f"A={result['A']:.3e}  n={result['n']:.3f}  "
                      f"{xml_path.name}")

        print(f"  → {int(size_nm)} nm: {len(fits[size_nm])} files")

    return fits, emsds


def main() -> None:
    dls_sizes  = get_dls_sizes()
    dls_labels = get_dls_labels()

    theory_by_size: dict[float, float] = {
        size_nm: calculate_theoretical_diffusion(
            particle_size_nm=dls_sizes.get(size_nm, size_nm)
        )
        for size_nm in XML_FOLDERS_D0
    }

    print("=== D₀ water ===")
    fits_by_size, _ = _collect(XML_FOLDERS_D0)

    print("\n=== 20 mg/mL C16 ===")
    fits_20mg, _ = _collect(XML_FOLDERS_20MG)

    plot_fit_lines_overview(
        fits_by_size=fits_by_size,
        theory_by_size=theory_by_size,
        dls_labels=dls_labels,
        fits_hydrogel_by_size=fits_20mg,
        hydrogel_label="20 mg/mL C16",
        save_path=SAVE_PATH,
        filename="emsd_D0_vs_20mg_overview",
        sample_label=r"$D_0$ water vs. 20 mg/mL C16",
    )


if __name__ == "__main__":
    main()
