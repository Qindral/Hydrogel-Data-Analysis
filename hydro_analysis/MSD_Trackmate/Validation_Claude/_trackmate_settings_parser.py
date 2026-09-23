"""
Shared helper, not run standalone (same convention as
_detectability_selection.py in this folder). Tail-reads a raw TrackMate
session XML (the full Fiji project export -- Model/AllSpots/AllTracks/
FilteredTracks + Settings -- as opposed to the simplified
"<Tracks><particle><detection>" schema read by core.io.read_trackmate_xml)
and extracts the exact detector/tracker/filter parameters that produced it,
in a structure directly usable both to drive a fresh headless TrackMate run
with the same parameters (see run_trackmate_headless.py /
TrackMate_Headless_Sweep.py) and to build a parameter sweep around a known
real-world baseline.

Only DOG_DETECTOR/LOG_DETECTOR and QUALITY-based InitialSpotFilter are
supported (real headless TrackMate computes every other spot/track feature
itself via Settings.addAllAnalyzers(), so SpotFilterCollection/
TrackFilterCollection entries on any feature name are passed through as-is
-- no Python-side feature computation needed, unlike the earlier from-
scratch reimplementation this replaced). Splitting/merging enabled, or a
detector/InitialSpotFilter this parser doesn't recognize, raise
NotImplementedError rather than being silently ignored, since a caller
relying on this parser to reproduce a real run needs to know when
something wasn't captured.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SETTINGS_TAIL_BYTES = 500_000


def _attr(tag_text: str, name: str) -> str | None:
    m = re.search(rf'{name}="([^"]*)"', tag_text)
    return m.group(1) if m else None


def _find_tag(text: str, tag: str) -> tuple[str | None, str | None]:
    """Returns (full_tag_text, inner_text_or_None). Handles both
    self-closing (<Tag .../>) and block (<Tag ...>...</Tag>) forms."""
    m = re.search(rf"<{tag}\b[^>]*?/>", text)
    if m:
        return m.group(0), None
    m = re.search(rf"<{tag}\b[^>]*?>(.*?)</{tag}>", text, re.DOTALL)
    if m:
        return m.group(0), m.group(1)
    return None, None


_FILTER_ITEM_RE = re.compile(r'<Filter feature="([^"]+)" value="([\d.eE+-]+)" isabove="(true|false)"')


def _parse_filters(inner_text: str) -> list[tuple[str, float, bool]]:
    return [(feat, float(val), above == "true") for feat, val, above in _FILTER_ITEM_RE.findall(inner_text)]


_PENALTY_ATTR_RE = re.compile(r'(\w+)="([\d.eE+-]+)"')


def _parse_feature_penalties(inner_text: str | None) -> dict[str, float]:
    m = re.search(r"<FeaturePenalties\b([^/>]*)/>", inner_text or "")
    if not m:
        return {}
    return {name: float(val) for name, val in _PENALTY_ATTR_RE.findall(m.group(1))}


def parse_trackmate_full_settings(xml_path: Path, tail_bytes: int = SETTINGS_TAIL_BYTES) -> dict[str, Any]:
    """Tail-read a raw TrackMate session XML and extract every parameter
    needed to drive an equivalent headless TrackMate run. Raises
    NotImplementedError for detector/filter/tracker features this project's
    headless runner does not handle (unsupported detector, non-QUALITY
    InitialSpotFilter feature, or enabled track splitting/merging)."""
    size = xml_path.stat().st_size
    with open(xml_path, "rb") as f:
        if size > tail_bytes:
            f.seek(-tail_bytes, 2)
        text = f.read().decode("utf-8", errors="ignore")

    image_tag, _ = _find_tag(text, "ImageData")
    width_px = int(_attr(image_tag, "width"))
    height_px = int(_attr(image_tag, "height"))

    det_tag, _ = _find_tag(text, "DetectorSettings")
    detector_name = _attr(det_tag, "DETECTOR_NAME")
    if detector_name not in ("DOG_DETECTOR", "LOG_DETECTOR"):
        raise NotImplementedError(
            f"{xml_path.name}: only DOG_DETECTOR/LOG_DETECTOR are implemented, found {detector_name!r}."
        )
    radius = float(_attr(det_tag, "RADIUS"))
    threshold = float(_attr(det_tag, "THRESHOLD"))
    do_median_filtering = _attr(det_tag, "DO_MEDIAN_FILTERING") == "true"
    do_subpixel_localization = _attr(det_tag, "DO_SUBPIXEL_LOCALIZATION") == "true"
    target_channel = int(_attr(det_tag, "TARGET_CHANNEL"))

    init_tag, _ = _find_tag(text, "InitialSpotFilter")
    if _attr(init_tag, "feature") != "QUALITY":
        raise NotImplementedError(f"{xml_path.name}: InitialSpotFilter on a non-QUALITY feature is not implemented.")
    initial_quality = float(_attr(init_tag, "value"))

    _, spot_filter_inner = _find_tag(text, "SpotFilterCollection")
    spot_filters = _parse_filters(spot_filter_inner or "")

    tracker_tag, tracker_inner = _find_tag(text, "TrackerSettings")
    cutoff_percentile = float(_attr(tracker_tag, "CUTOFF_PERCENTILE"))
    alternative_linking_cost_factor = float(_attr(tracker_tag, "ALTERNATIVE_LINKING_COST_FACTOR"))

    linking_tag, linking_inner = _find_tag(tracker_inner, "Linking")
    linking_max_distance = float(_attr(linking_tag, "LINKING_MAX_DISTANCE"))
    linking_feature_penalties = _parse_feature_penalties(linking_inner)

    gap_tag, gap_inner = _find_tag(tracker_inner, "GapClosing")
    allow_gap_closing = _attr(gap_tag, "ALLOW_GAP_CLOSING") == "true"
    gap_closing_max_distance = float(_attr(gap_tag, "GAP_CLOSING_MAX_DISTANCE"))
    max_frame_gap = int(_attr(gap_tag, "MAX_FRAME_GAP"))
    gap_closing_feature_penalties = _parse_feature_penalties(gap_inner)

    split_tag, _ = _find_tag(tracker_inner, "TrackSplitting")
    merge_tag, _ = _find_tag(tracker_inner, "TrackMerging")
    allow_track_splitting = _attr(split_tag, "ALLOW_TRACK_SPLITTING") == "true"
    allow_track_merging = _attr(merge_tag, "ALLOW_TRACK_MERGING") == "true"
    if allow_track_splitting or allow_track_merging:
        raise NotImplementedError(f"{xml_path.name}: track splitting/merging is enabled and not implemented.")

    _, track_filter_inner = _find_tag(text, "TrackFilterCollection")
    track_filters = _parse_filters(track_filter_inner or "")

    return {
        "width_px": width_px, "height_px": height_px,
        "detector_name": detector_name,
        "radius": radius, "threshold": threshold, "target_channel": target_channel,
        "do_median_filtering": do_median_filtering, "do_subpixel_localization": do_subpixel_localization,
        "initial_quality": initial_quality,
        "spot_filters": spot_filters,
        "cutoff_percentile": cutoff_percentile, "alternative_linking_cost_factor": alternative_linking_cost_factor,
        "linking_max_distance": linking_max_distance, "linking_feature_penalties": linking_feature_penalties,
        "allow_gap_closing": allow_gap_closing, "gap_closing_max_distance": gap_closing_max_distance,
        "max_frame_gap": max_frame_gap, "gap_closing_feature_penalties": gap_closing_feature_penalties,
        "allow_track_splitting": allow_track_splitting, "allow_track_merging": allow_track_merging,
        "track_filters": track_filters,
    }
