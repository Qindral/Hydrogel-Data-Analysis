"""
Drives the REAL TrackMate detector+tracker from Python by launching Fiji
headlessly (fiji_scripts/run_trackmate_headless.py, a Jython script run
inside Fiji's own bundled interpreter -- not a Python package import).
Replaces the earlier from-scratch Python reimplementation of TrackMate's
DoG detector and LAP tracker (Reimplement_DoG_LAP_Compute.py /
Reimplement_DoG_LAP_Compare.py, both removed): running the actual algorithm
headlessly gives byte-identical results to the interactive GUI (verified:
for 50nmwater_10.xml, a headless run with the exact recorded parameters
reproduced n_spots=13584 exactly, and n_tracks_filtered=267, which matches
the real recorded FilteredTracks count once the 43 edge-artifact splits
that Compare_Tracks_vs_RawAnalysis.py's remove_edge_artifacts() adds on top
are accounted for: 310 - 43 = 267) -- with none of a reimplementation's
approximation caveats (subpixel localization, disk-crop intensity features,
etc.), for less code than the reimplementation needed.

run_trackmate_headless() is the general-purpose function other scripts
should import: given a movie's raw parameters (typically read via
_trackmate_settings_parser.parse_trackmate_full_settings() from a real
Analysis_X session XML, then optionally perturbed for a parameter sweep),
it runs TrackMate on the actual TIFF and returns {"ok", "error", "n_spots",
"n_tracks_total", "n_tracks_filtered"}; the written output XML is a
complete TrackMate session, parseable by the existing, unmodified
Validation_CPT/trackmate_session.py::read_trackmate_session() and
Compare_Tracks_vs_RawAnalysis.py::extract_tracks_and_parameters() exactly
like any other Analysis_X file already in this project.

Known load-bearing gotcha (see fiji_scripts/run_trackmate_headless.py for
the fix): IJ.openImage() interprets these multi-page TIFFs as Z-stacks by
default, not time series -- silently producing near-zero spots/tracks with
no error if not corrected.

Requires: Fiji installed at FIJI_EXE (adjust if this changes) with the
TrackMate update site enabled -- no Python-side dependency (no pyimagej/
scyjava; Fiji is invoked as an external subprocess, not embedded in this
process).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

# ── Configuration ──────────────────────────────────────────────────────────────
FIJI_EXE = Path(r"E:\PhD Data Analysis\Fiji\fiji-windows-x64.exe")
JYTHON_SCRIPT = Path(__file__).parent / "fiji_scripts" / "run_trackmate_headless.py"
DEFAULT_TIMEOUT_S = 600


def params_from_settings(settings: dict[str, Any], tif_path: str, output_xml_path: str) -> dict[str, Any]:
    """Build a run_trackmate_headless() params dict from
    _trackmate_settings_parser.parse_trackmate_full_settings()'s output,
    for the given (possibly different) tif_path/output_xml_path -- e.g. to
    re-run the exact recorded parameters on the same movie, or as the
    starting point for a parameter sweep (see module docstring)."""
    return {
        "tif_path": tif_path,
        "output_xml_path": output_xml_path,
        "detector_name": settings["detector_name"],
        "detector": {
            "RADIUS": settings["radius"],
            "THRESHOLD": settings["threshold"],
            "TARGET_CHANNEL": settings["target_channel"],
            "DO_SUBPIXEL_LOCALIZATION": settings["do_subpixel_localization"],
            "DO_MEDIAN_FILTERING": settings["do_median_filtering"],
        },
        "initial_quality": settings["initial_quality"],
        "spot_filters": [list(f) for f in settings["spot_filters"]],
        "tracker": {
            "LINKING_MAX_DISTANCE": settings["linking_max_distance"],
            "ALLOW_GAP_CLOSING": settings["allow_gap_closing"],
            "GAP_CLOSING_MAX_DISTANCE": settings["gap_closing_max_distance"],
            "MAX_FRAME_GAP": settings["max_frame_gap"],
            "ALLOW_TRACK_SPLITTING": settings["allow_track_splitting"],
            "ALLOW_TRACK_MERGING": settings["allow_track_merging"],
            "CUTOFF_PERCENTILE": settings["cutoff_percentile"],
            "ALTERNATIVE_LINKING_COST_FACTOR": settings["alternative_linking_cost_factor"],
            "LINKING_FEATURE_PENALTIES": dict(settings["linking_feature_penalties"]),
            "GAP_CLOSING_FEATURE_PENALTIES": dict(settings["gap_closing_feature_penalties"]),
        },
        "track_filters": [list(f) for f in settings["track_filters"]],
    }


def run_trackmate_headless(
    params: dict[str, Any], params_json_path: Path, timeout: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Runs real TrackMate headlessly with `params` (see
    fiji_scripts/run_trackmate_headless.py for the exact schema; typically
    built via params_from_settings()). Writes params to params_json_path
    first (left on disk after the run, including on failure, for
    debugging) and points the Jython script at it via the
    TRACKMATE_PARAMS_JSON environment variable -- confirmed empirically
    that ImageJ2's --run does not forward arguments to sys.argv, so an env
    var is the reliable channel here.

    Returns the Jython script's own result dict: {"ok", "error", "n_spots",
    "n_tracks_total", "n_tracks_filtered"}. Raises RuntimeError if Fiji
    fails to produce a result file at all (a JVM-level crash before the
    Jython script's own try/except, or a timeout) or if the run completed
    but reported ok=False.
    """
    params_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_json_path, "w") as f:
        json.dump(params, f)

    env = os.environ.copy()
    env["TRACKMATE_PARAMS_JSON"] = str(params_json_path)

    try:
        proc = subprocess.run(
            [str(FIJI_EXE), "--ij2", "--headless", "--console", "--run", str(JYTHON_SCRIPT)],
            env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Fiji headless run timed out after {timeout}s for {params['tif_path']}") from exc

    result_path = Path(params["output_xml_path"] + "_result.json")
    if not result_path.exists():
        raise RuntimeError(
            f"Fiji produced no result file (exit code {proc.returncode}) for {params['tif_path']}.\n"
            f"stdout(tail): {proc.stdout[-2000:]}\nstderr(tail): {proc.stderr[-2000:]}"
        )
    with open(result_path) as f:
        result = json.load(f)
    if not result["ok"]:
        raise RuntimeError(f"TrackMate run failed for {params['tif_path']}: {result['error']}")
    return result


def main() -> None:
    """Smoke test: re-run the real, recorded parameters for one known movie
    and confirm the spot/track counts match what was already established
    for it this session (see module docstring)."""
    from hydro_analysis.MSD_Trackmate.Validation_Claude._trackmate_settings_parser import parse_trackmate_full_settings

    analysis_xml = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Analysis_50\50nmwater_10.xml")
    tif_path = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\50 nm water_10.tif")
    out_dir = Path(__file__).parent.parent / "cache" / "trackmate_headless_smoketest"
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = parse_trackmate_full_settings(analysis_xml)
    params = params_from_settings(settings, str(tif_path), str(out_dir / "smoketest_output.xml"))

    print(f"Running real headless TrackMate on {tif_path.name} with the exact recorded parameters...")
    result = run_trackmate_headless(params, out_dir / "smoketest_params.json")
    print(f"n_spots={result['n_spots']} (expected 13584), "
          f"n_tracks_filtered={result['n_tracks_filtered']} (expected 267, i.e. the recorded 310 minus "
          f"the 43 edge-artifact splits remove_edge_artifacts() adds on top)")


if __name__ == "__main__":
    main()
