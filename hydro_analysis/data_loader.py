"""Image loading for Hydro Analysis."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

from .metadata import DatasetMetadata, collect_notes

OME_NS = {
    "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
    "ome2015": "http://www.openmicroscopy.org/Schemas/OME/2015-01",
}


@dataclass
class LoadedDataset:
    data: np.ndarray
    metadata: DatasetMetadata


class DatasetLoader:
    """Load TIFF datasets and extract metadata."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def load(self) -> LoadedDataset:
        with tifffile.TiffFile(self.path) as tif:
            series = tif.series[0]
            data = series.asarray()
            axes = series.axes
            raw_meta = self._extract_basic_tags(tif)
            metadata = DatasetMetadata(
                path=self.path,
                axes=axes,
                shape=data.shape,
                dtype=str(data.dtype),
                raw_metadata=raw_meta,
            )
            self._populate_metadata_from_ome(tif, metadata)
            self._populate_metadata_from_leica(tif, metadata)
            self._populate_metadata_from_standard_tags(tif, metadata)
            self._populate_metadata_from_sidecars(metadata)
            if not metadata.notes:
                metadata.notes = collect_notes(raw_meta.get("description"))
            return LoadedDataset(data=data, metadata=metadata)

    def _extract_basic_tags(self, tif: tifffile.TiffFile) -> Dict[str, object]:
        tags = {}
        first_page = tif.pages[0]
        for key in ("Artist", "Software", "DateTime", "ImageDescription"):
            if key in first_page.tags:
                value = first_page.tags[key].value
                if isinstance(value, bytes):
                    value = value.decode(errors="ignore")
                tags[key.lower()] = value
        return tags

    def _populate_metadata_from_ome(self, tif: tifffile.TiffFile, metadata: DatasetMetadata) -> None:
        if not tif.ome_metadata:
            return
        xml = tif.ome_metadata
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return
        ns = None
        for candidate in ("ome", "ome2015"):
            if root.tag.startswith("{" + OME_NS[candidate]):
                ns = {"ome": OME_NS[candidate]}
                break
        if ns is None:
            ns = {"ome": root.tag[root.tag.find("{") + 1 : root.tag.find("}")]} if "{" in root.tag else {}
        pixels = root.find(".//ome:Pixels", ns) if ns else root.find(".//Pixels")
        if pixels is not None:
            metadata.px_size_xy_um = _safe_float(pixels.get("PhysicalSizeX"))
            if not metadata.px_size_xy_um:
                metadata.px_size_xy_um = _safe_float(pixels.get("PhysicalSizeY"))
            metadata.z_step_um = _safe_float(pixels.get("PhysicalSizeZ"))
            time_increment = _safe_float(pixels.get("TimeIncrement"))
        else:
            time_increment = None
        plane_elems = root.findall(".//ome:Plane", ns) if ns else root.findall(".//Plane")
        timestamps: Dict[int, float] = {}
        positions: List[Tuple[float, float]] = []
        for plane in plane_elems:
            the_t = plane.get("TheT")
            delta_t = plane.get("DeltaT")
            if the_t is not None and delta_t is not None:
                timestamps[int(the_t)] = float(delta_t)
            pos_x = plane.get("PositionX")
            pos_y = plane.get("PositionY")
            if pos_x is not None and pos_y is not None:
                positions.append((float(pos_x) / 1000.0, float(pos_y) / 1000.0))
        if not timestamps and time_increment:
            size_t = metadata.counts.get("T", len(metadata.shape))
            timestamps = {i: i * time_increment for i in range(size_t)}
        if timestamps:
            metadata.timestamps = [timestamps[i] for i in sorted(timestamps)]
        if positions:
            metadata.stage_position_mm = positions[0]
        channels = root.findall(".//ome:Channel", ns) if ns else root.findall(".//Channel")
        if channels:
            metadata.channel_names = [ch.get("Name") or ch.get("ID") or f"C{i}" for i, ch in enumerate(channels)]
        notes = []
        for node_name in ("Experiment", "Description", "Instrument"):
            elems = root.findall(f".//ome:{node_name}", ns) if ns else root.findall(f".//{node_name}")
            for elem in elems:
                text = (elem.text or "").strip()
                if text:
                    notes.append(text)
        metadata.notes = collect_notes(metadata.notes, notes)

    def _populate_metadata_from_leica(self, tif: tifffile.TiffFile, metadata: DatasetMetadata) -> None:
        description = metadata.raw_metadata.get("imagedescription") or metadata.raw_metadata.get("description")
        if description and "Leica" not in str(description):
            return
        leica_xml = None
        if description and "<?xml" in str(description):
            leica_xml = str(description)
        if not leica_xml:
            sidecar = self.path.with_suffix(".xml")
            if sidecar.exists():
                leica_xml = sidecar.read_text()
        if not leica_xml:
            return
        try:
            root = ET.fromstring(leica_xml)
        except ET.ParseError:
            return
        px_size = root.findtext(".//PixelSize")
        if px_size:
            metadata.px_size_xy_um = _safe_float(px_size)
        z_step = root.findtext(".//ZStep")
        if z_step:
            metadata.z_step_um = _safe_float(z_step)
        timestamps = []
        for elem in root.findall(".//TimeStamp"):
            val = _safe_float(elem.text)
            if val is not None:
                timestamps.append(val)
        if timestamps:
            metadata.timestamps = sorted(set(timestamps))
        stage_x = root.findtext(".//StageX")
        stage_y = root.findtext(".//StageY")
        if stage_x and stage_y:
            metadata.stage_position_mm = (_safe_float(stage_x), _safe_float(stage_y))
        metadata.notes = collect_notes(metadata.notes, root.findtext(".//Comment"))

    def _populate_metadata_from_standard_tags(self, tif: tifffile.TiffFile, metadata: DatasetMetadata) -> None:
        first_page = tif.pages[0]
        x_res = first_page.tags.get("XResolution")
        y_res = first_page.tags.get("YResolution")
        unit_tag = first_page.tags.get("ResolutionUnit")
        unit = unit_tag.value if unit_tag is not None else None
        px_size = None
        if x_res and isinstance(x_res.value, tuple) and x_res.value[0] and x_res.value[1]:
            px_size = _resolution_to_microns(x_res.value, unit)
        elif y_res and isinstance(y_res.value, tuple) and y_res.value[0] and y_res.value[1]:
            px_size = _resolution_to_microns(y_res.value, unit)
        if px_size:
            metadata.px_size_xy_um = metadata.px_size_xy_um or px_size
        datetime_tag = metadata.raw_metadata.get("datetime")
        if datetime_tag:
            metadata.datetime_acquired = _parse_datetime(datetime_tag)
        metadata.notes = collect_notes(metadata.notes, metadata.raw_metadata.get("description"))
        metadata.raw_metadata.setdefault("artist", metadata.raw_metadata.get("artist"))
        metadata.raw_metadata.setdefault("software", metadata.raw_metadata.get("software"))

    def _populate_metadata_from_sidecars(self, metadata: DatasetMetadata) -> None:
        base = self.path.with_suffix("")
        for ext in (".xml", ".txt"):
            sidecar = base.with_suffix(ext)
            if sidecar.exists():
                text = sidecar.read_text(errors="ignore")
                metadata.notes = collect_notes(metadata.notes, text)
                if "frap" in text.lower():
                    break


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolution_to_microns(resolution: Tuple[int, int], unit: Optional[int]) -> Optional[float]:
    # resolution given as (numerator, denominator) in units per inch/centimeter
    num, den = resolution
    if den == 0:
        return None
    value = num / den
    # Resolution unit: 2=inch, 3=centimeter
    if unit == 2:
        # pixels per inch
        return 25400.0 / value if value else None
    if unit == 3:
        # pixels per centimeter
        return 10000.0 / value if value else None
    return None


def _parse_datetime(text: str) -> Optional[datetime]:
    text = text.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    match = re.search(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})", text)
    if match:
        return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}")
    return None
