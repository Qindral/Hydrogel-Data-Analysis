"""Reader for complete Fiji TrackMate sessions (not legacy particle XML)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackMateSession:
    path: Path
    spots: pd.DataFrame
    trajectories: pd.DataFrame
    pixelwidth_um: float
    pixelheight_um: float
    timeinterval_s: float
    spatialunits: str
    timeunits: str
    spot_features: tuple[str, ...]


def _as_float(attrs: dict[str, str], key: str, path: Path) -> float:
    value = attrs.get(key)
    if value is None:
        raise ValueError(f"{path.name}: required XML attribute {key!r} is missing.")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {key}={value!r} is not numeric.") from exc


def _ordered_paths(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Return paths through a TrackMate track; retain all branches explicitly."""
    children: dict[int, list[int]] = {}
    parents: dict[int, list[int]] = {}
    for source, target in edges:
        children.setdefault(source, []).append(target)
        parents.setdefault(target, []).append(source)
    nodes = set(children) | set(parents)
    starts = sorted(node for node in nodes if not parents.get(node))
    paths: list[list[int]] = []
    # TrackMate trajectories can exceed Python's recursion limit, so this is
    # deliberately iterative rather than a recursive depth-first traversal.
    pending: list[list[int]] = [[start] for start in starts]
    while pending:
        path = pending.pop()
        next_nodes = children.get(path[-1], [])
        if not next_nodes:
            paths.append(path)
            continue
        for nxt in next_nodes:
            if nxt not in path:  # defensive protection against malformed cycles
                pending.append(path + [nxt])
    return paths


def read_trackmate_session(path: str | Path) -> TrackMateSession:
    """Read filtered trajectories and all per-spot attributes from one session.

    `POSITION_X/Y` are retained in the XML's spatial unit. The caller must
    explicitly convert to pixels only when indexing the TIFF.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    model = root.find("Model")
    image = root.find("Settings/ImageData")
    if model is None or image is None:
        raise ValueError(f"{path.name}: not a complete TrackMate session (Model/ImageData missing).")
    spatialunits = model.attrib.get("spatialunits", "")
    timeunits = model.attrib.get("timeunits", "")
    pixelwidth = _as_float(image.attrib, "pixelwidth", path)
    pixelheight = _as_float(image.attrib, "pixelheight", path)
    dt = _as_float(image.attrib, "timeinterval", path)
    if pixelwidth <= 0 or pixelheight <= 0 or dt <= 0:
        raise ValueError(f"{path.name}: XML calibration must be positive.")

    rows: list[dict[str, object]] = []
    for spot in root.findall("Model/AllSpots/SpotsInFrame/Spot"):
        attrs = dict(spot.attrib)
        if "ID" not in attrs or "FRAME" not in attrs:
            continue
        rows.append({
            "spot_id": int(float(attrs.pop("ID"))),
            "frame": int(float(attrs.pop("FRAME"))),
            "x": _as_float(attrs, "POSITION_X", path),
            "y": _as_float(attrs, "POSITION_Y", path),
            **{key: value for key, value in attrs.items()},
        })
    spots = pd.DataFrame(rows)
    if spots.empty:
        raise ValueError(f"{path.name}: AllSpots contains no usable spots.")
    features = tuple(sorted(c for c in spots.columns if c not in {"spot_id", "frame", "x", "y"}))

    filtered_node = root.find("Model/FilteredTracks")
    if filtered_node is None:
        raise ValueError(f"{path.name}: FilteredTracks missing; the validation requires the analysed track set.")
    filtered_ids = {int(node.attrib["TRACK_ID"]) for node in root.findall("Model/FilteredTracks/TrackID")
                    if "TRACK_ID" in node.attrib}
    tracks: list[tuple[int, list[tuple[int, int]]]] = []
    for node in root.findall("Model/AllTracks/Track"):
        track_id = int(node.attrib.get("TRACK_ID", "-1"))
        if track_id not in filtered_ids:
            continue
        edges = [(int(edge.attrib["SPOT_SOURCE_ID"]), int(edge.attrib["SPOT_TARGET_ID"]))
                 for edge in node.findall("Edge")
                 if "SPOT_SOURCE_ID" in edge.attrib and "SPOT_TARGET_ID" in edge.attrib]
        if edges:
            tracks.append((track_id, edges))
    if not tracks:
        raise ValueError(f"{path.name}: no filtered tracks with edges found.")

    indexed = spots.set_index("spot_id", drop=False)
    trajectory_rows: list[pd.DataFrame] = []
    for track_id, edges in tracks:
        for branch, ids in enumerate(_ordered_paths(edges)):
            available = [sid for sid in ids if sid in indexed.index]
            if len(available) < 2:
                continue
            frame = indexed.loc[available].copy().sort_values("frame")
            frame["track_id"] = track_id
            frame["branch_id"] = branch
            frame["trajectory_id"] = f"{track_id}:{branch}"
            trajectory_rows.append(frame.reset_index(drop=True))
    trajectories = pd.concat(trajectory_rows, ignore_index=True) if trajectory_rows else pd.DataFrame()
    if trajectories.empty:
        raise ValueError(f"{path.name}: no reconstructable filtered trajectories.")
    return TrackMateSession(path, spots, trajectories, pixelwidth, pixelheight, dt,
                            spatialunits, timeunits, features)
