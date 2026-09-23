# Jython inside Fiji. Runs actual DoG detection, not tracking, on sampled frames.
import os
import json
import traceback
from ij import IJ
from fiji.plugin.trackmate import Model, Settings, TrackMate, Logger
from fiji.plugin.trackmate.detection import DogDetectorFactory
from java.util import HashMap
from java.lang import Double, Integer, Boolean, System

params = json.load(open(os.environ["SNR_DOG_PARAMS"]))
result = {"ok": False, "cases": [], "error": None}
try:
    imp = IJ.openImage(params["tiff_path"])
    if imp is None:
        raise RuntimeError("Cannot open sampled TIFF")
    imp.setDimensions(1, 1, imp.getStackSize())
    # XML radius has already been converted to pixels in the Python caller.
    cal = imp.getCalibration()
    cal.pixelWidth = cal.pixelHeight = cal.pixelDepth = 1.0
    cal.setUnit("pixel")
    for rf in params["radius_factors"]:
        for tf in params["threshold_factors"]:
            model = Model()
            model.setLogger(Logger.VOID_LOGGER)
            settings = Settings(imp)
            settings.detectorFactory = DogDetectorFactory()
            values = HashMap()
            values.put("RADIUS", Double(params["radius"] * rf))
            values.put("THRESHOLD", Double(params["threshold"] * tf))
            values.put("TARGET_CHANNEL", Integer(1))
            values.put("DO_MEDIAN_FILTERING", Boolean(params["median"]))
            values.put("DO_SUBPIXEL_LOCALIZATION", Boolean(params["subpixel"]))
            settings.detectorSettings = values
            trackmate = TrackMate(model, settings)
            if not trackmate.execDetection():
                raise RuntimeError(str(trackmate.getErrorMessage()))
            frames = {}
            spots = model.getSpots()
            for f in spots.keySet():
                frame_spots = []
                for spot in spots.iterable(f, False):
                    frame_spots.append([float(spot.getFeature("POSITION_X")), float(spot.getFeature("POSITION_Y"))])
                frames[str(int(f))] = frame_spots
            result["cases"].append({"radius_factor": rf, "threshold_factor": tf, "frames": frames})
    imp.close()
    result["ok"] = True
except Exception:
    result["error"] = traceback.format_exc()
with open(params["result_path"], "w") as stream:
    json.dump(result, stream)
System.exit(0 if result["ok"] else 1)
