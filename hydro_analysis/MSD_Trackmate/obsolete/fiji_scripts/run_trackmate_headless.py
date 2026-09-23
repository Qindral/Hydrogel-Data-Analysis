# Jython script, run inside Fiji/ImageJ2's bundled Jython interpreter via
# headless batch mode -- NOT plain CPython (no pip packages, Jython 2.7
# stdlib only). Not part of the hydro_analysis package; invoked externally
# by hydro_analysis.MSD_Trackmate.Validation_Claude.trackmate_headless_runner via:
#
#   fiji-windows-x64.exe --ij2 --headless --console --run run_trackmate_headless.py
#
# with the parameter file's path passed through the TRACKMATE_PARAMS_JSON
# environment variable (ImageJ2's --run does not forward extra arguments to
# sys.argv, confirmed empirically against this project's Fiji install --
# env vars are the reliable channel).
#
# Runs the REAL TrackMate detector+tracker (this is not a reimplementation
# of the algorithm -- see the now-removed Reimplement_DoG_LAP_* scripts for
# that abandoned approach and why it was replaced: running actual TrackMate
# headlessly gives byte-identical results to the interactive GUI, with none
# of a reimplementation's approximation caveats). DOG_DETECTOR and
# LOG_DETECTOR are both supported (selected via "detector_name"); tracking
# is always SPARSE_LAP_TRACKER. Every parameter is supplied explicitly from
# the params JSON, none left at TrackMate's own defaults, so a sweep driver
# can vary any of them without hidden defaults changing underneath it.
#
# Params JSON schema (written by the Python driver):
#   {
#     "tif_path": str, "output_xml_path": str,
#     "detector_name": "DOG_DETECTOR" | "LOG_DETECTOR",
#     "detector": {"RADIUS": float, "THRESHOLD": float, "TARGET_CHANNEL": int,
#                  "DO_SUBPIXEL_LOCALIZATION": bool, "DO_MEDIAN_FILTERING": bool},
#     "initial_quality": float,
#     "spot_filters": [[feature, value, isabove], ...],
#     "tracker": {"LINKING_MAX_DISTANCE": float, "ALLOW_GAP_CLOSING": bool,
#                 "GAP_CLOSING_MAX_DISTANCE": float, "MAX_FRAME_GAP": int,
#                 "ALLOW_TRACK_SPLITTING": bool, "ALLOW_TRACK_MERGING": bool,
#                 "CUTOFF_PERCENTILE": float, "ALTERNATIVE_LINKING_COST_FACTOR": float,
#                 "LINKING_FEATURE_PENALTIES": {feature: factor, ...},
#                 "GAP_CLOSING_FEATURE_PENALTIES": {feature: factor, ...}},
#     "track_filters": [[feature, value, isabove], ...]
#   }
#
# Writes a result JSON next to output_xml_path (same path with
# "_result.json" appended) with {"ok": bool, "error": str_or_null,
# "n_spots": int, "n_tracks_total": int, "n_tracks_filtered": int}, so the
# Python driver can distinguish "ran but found 0 tracks" from "crashed"
# without depending solely on the process exit code.
#
# Known, load-bearing gotcha (cost real debugging time to find): IJ.openImage()
# on one of this project's multi-page TIFFs is interpreted by ImageJ as a
# Z-STACK (nSlices=N, nFrames=1), not a time series -- TrackMate then
# detects/links across the wrong axis and silently returns near-zero
# spots/tracks with no error. imp.setDimensions(1, 1, imp.getStackSize())
# below re-interprets the same pixel data as (nChannels=1, nSlices=1,
# nFrames=N) before Settings(imp) is built. Do not remove this line.

import os
import json
import traceback

from ij import IJ
from fiji.plugin.trackmate import Model, Settings, TrackMate, Logger
from fiji.plugin.trackmate.detection import DogDetectorFactory, LogDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory, LAPUtils
from fiji.plugin.trackmate.features import FeatureFilter
from fiji.plugin.trackmate.io import TmXmlWriter
import java.io.File as JFile
import java.util.HashMap as HashMap


def _to_java_map(d):
    m = HashMap()
    for k, v in d.items():
        m.put(k, v)
    return m


def main():
    params_path = os.environ.get("TRACKMATE_PARAMS_JSON")
    if not params_path:
        raise RuntimeError("TRACKMATE_PARAMS_JSON environment variable not set.")
    with open(params_path, "r") as f:
        params = json.load(f)

    output_xml_path = params["output_xml_path"]
    result_path = output_xml_path + "_result.json"
    result = {"ok": False, "error": None, "n_spots": None, "n_tracks_total": None, "n_tracks_filtered": None}

    try:
        imp = IJ.openImage(params["tif_path"])
        if imp is None:
            raise RuntimeError("IJ.openImage returned None for " + params["tif_path"])
        imp.setDimensions(1, 1, imp.getStackSize())

        model = Model()
        model.setLogger(Logger.VOID_LOGGER)

        detector_name = params.get("detector_name", "DOG_DETECTOR")
        if detector_name == "LOG_DETECTOR":
            detector_factory = LogDetectorFactory()
        elif detector_name == "DOG_DETECTOR":
            detector_factory = DogDetectorFactory()
        else:
            raise RuntimeError("Unsupported detector_name: " + str(detector_name))

        settings = Settings(imp)
        settings.detectorFactory = detector_factory
        settings.detectorSettings = _to_java_map(params["detector"])

        settings.trackerFactory = SparseLAPTrackerFactory()
        # Start from TrackMate's own defaults (this supplies mandatory keys
        # this project never varies/parses, e.g. SPLITTING_MAX_DISTANCE,
        # MERGING_MAX_DISTANCE, BLOCKING_VALUE -- splitting/merging are
        # always disabled here, see module docstring), then overlay every
        # key the params JSON explicitly provides.
        tracker_settings = LAPUtils.getDefaultSegmentSettingsMap()
        for key, value in params["tracker"].items():
            tracker_settings.put(key, value)
        settings.trackerSettings = tracker_settings

        settings.initialSpotFilterValue = params["initial_quality"]

        settings.addAllAnalyzers()

        for feature, value, isabove in params.get("spot_filters", []):
            settings.addSpotFilter(FeatureFilter(feature, value, isabove))
        for feature, value, isabove in params.get("track_filters", []):
            settings.addTrackFilter(FeatureFilter(feature, value, isabove))

        trackmate = TrackMate(model, settings)
        if not trackmate.checkInput():
            raise RuntimeError("checkInput failed: " + str(trackmate.getErrorMessage()))
        if not trackmate.process():
            raise RuntimeError("process failed: " + str(trackmate.getErrorMessage()))

        result["n_spots"] = model.getSpots().getNSpots(False)
        result["n_tracks_total"] = model.getTrackModel().nTracks(False)
        result["n_tracks_filtered"] = model.getTrackModel().nTracks(True)

        out_file = JFile(output_xml_path)
        writer = TmXmlWriter(out_file, Logger.VOID_LOGGER)
        writer.appendModel(model)
        writer.appendSettings(settings)
        writer.writeToFile()

        result["ok"] = True

    except Exception as e:
        result["error"] = str(e) + "\n" + traceback.format_exc()

    with open(result_path, "w") as f:
        json.dump(result, f)


main()
