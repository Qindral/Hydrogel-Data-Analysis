"""Metadata utilities for Hydro Analysis."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import re
import xml.etree.ElementTree as ET

FILETIME_EPOCH = datetime(1601, 1, 1)
NUMERIC_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass
class TimeSummary:
    """Summary statistics for timestamps."""

    timestamps: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        if not self.timestamps:
            return {}
        if len(self.timestamps) == 1:
            return {"start": float(self.timestamps[0])}
        diffs = [b - a for a, b in zip(self.timestamps[:-1], self.timestamps[1:]) if b is not None and a is not None]
        diffs = [d for d in diffs if d > 0]
        if not diffs:
            return {"start": float(self.timestamps[0])}
        diffs_sorted = sorted(diffs)
        mid = len(diffs_sorted) // 2
        if len(diffs_sorted) % 2:
            median = diffs_sorted[mid]
        else:
            median = 0.5 * (diffs_sorted[mid - 1] + diffs_sorted[mid])
        q1_idx = max(0, len(diffs_sorted) // 4)
        q3_idx = min(len(diffs_sorted) - 1, 3 * len(diffs_sorted) // 4)
        iqr = diffs_sorted[q3_idx] - diffs_sorted[q1_idx]
        fps = 1.0 / median if median else None
        return {
            "start": float(self.timestamps[0]),
            "end": float(self.timestamps[-1]),
            "median_delta": median,
            "iqr_delta": iqr,
            "fps": fps,
        }


@dataclass
class DatasetMetadata:
    """Container for image metadata."""

    path: Path
    axes: str
    shape: Tuple[int, ...]
    dtype: str
    px_size_xy_um: Optional[float] = None
    z_step_um: Optional[float] = None
    timestamps: List[float] = field(default_factory=list)
    datetime_acquired: Optional[datetime] = None
    stage_position_mm: Optional[Tuple[float, float]] = None
    channel_names: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        if self.datetime_acquired:
            data["datetime_acquired"] = self.datetime_acquired.isoformat()
        data["time_summary"] = TimeSummary(self.timestamps).to_dict()
        return data

    @property
    def width_height(self) -> Tuple[int, int]:
        axes_to_dim = dict(zip(self.axes, self.shape))
        width = axes_to_dim.get("X", 0)
        height = axes_to_dim.get("Y", 0)
        return width, height

    @property
    def counts(self) -> Dict[str, int]:
        axes_to_dim = dict(zip(self.axes, self.shape))
        return {
            axis: axes_to_dim.get(axis, 1)
            for axis in ("C", "Z", "T")
        }

    def infer_kind(self) -> str:
        """Heuristic dataset kind classification."""
        text_blob = " ".join(self.notes + [str(self.raw_metadata.get("artist", "")), str(self.raw_metadata.get("software", ""))]).lower()
        counts = self.counts
        if "frap" in text_blob or "bleach" in text_blob:
            return "FRAP"
        if "spt" in text_blob or counts.get("C", 1) == 1 and counts.get("T", 1) > 100:
            return "SPT"
        if counts.get("Z", 1) > 1 and counts.get("T", 1) > 1:
            return "FULL"
        return "FULL" if counts.get("T", 1) > 1 else "SPT"

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True))


def format_timestamp_summary(timestamps: Sequence[float]) -> str:
    if not timestamps:
        return "–"
    summary = TimeSummary(list(timestamps)).to_dict()
    if not summary:
        return "–"
    fps = summary.get("fps")
    parts = []
    if fps is not None and fps > 0:
        parts.append(f"FPS≈{fps:.2f}")
    median = summary.get("median_delta")
    if median:
        parts.append(f"Δt₅₀={median:.3f}s")
    iqr = summary.get("iqr_delta")
    if iqr:
        parts.append(f"IQR={iqr:.3f}s")
    return ", ".join(parts) if parts else "–"


def format_stage_position(stage: Optional[Tuple[float, float]]) -> str:
    if not stage:
        return "–"
    x, y = stage
    return f"x={x:.3f} mm, y={y:.3f} mm"


def ensure_dataset_root(tiff_path: Path) -> Path:
    root = tiff_path.parent / tiff_path.stem
    root.mkdir(parents=True, exist_ok=True)
    qc_dir = root / "qc"
    qc_dir.mkdir(exist_ok=True)
    return root


def collect_notes(*sources: Iterable[str]) -> List[str]:
    notes: List[str] = []
    for source in sources:
        if not source:
            continue
        if isinstance(source, str):
            text = source.strip()
            if text:
                notes.extend([line.strip() for line in text.splitlines() if line.strip()])
        else:
            for item in source:
                if not item:
                    continue
                text = str(item).strip()
                if text:
                    notes.append(text)
    deduped: List[str] = []
    seen = set()
    for entry in notes:
        low = entry.lower()
        if low in seen:
            continue
        seen.add(low)
        deduped.append(entry)
    return deduped


def infer_dataset_base_name(path: Path) -> str:
    """Infer the logical dataset base name from a TIFF path."""

    stem = Path(path).stem
    # Strip conventional Leica style suffixes for time, channel, location, z.
    for pattern in (
        r"_[tT]\d+$",
        r"_[zZ]\d+$",
        r"_[cC]\d+$",
        r"_[lL]\d+$",
    ):
        stem = re.sub(pattern, "", stem)
    return stem


def _parse_numeric_token(value: Optional[str]) -> Optional[float]:
    """Parse Leica numeric strings that may include units or minutes."""

    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    minutes_match = re.search(r"(\d+)\s*m", value)
    seconds_match = re.search(r"(-?\d+(?:[.,]\d+)?)\s*s", value)
    if minutes_match and seconds_match:
        minutes = int(minutes_match.group(1))
        seconds = float(seconds_match.group(1).replace(",", "."))
        return minutes * 60 + seconds
    cleaned = value.replace(",", "")
    match = NUMERIC_RE.search(cleaned)
    if not match:
        return None
    token = match.group(0)
    token = token.replace(",", "").replace(" ", "")
    return float(token)


def _decode_leica_timestamp_list(text: str) -> Tuple[List[float], Optional[datetime]]:
    """Decode Leica FILETIME encoded timestamps into seconds and acquisition time."""

    timestamps: List[datetime] = []
    for token in text.split():
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token, 16)
        except ValueError:
            continue
        timestamps.append(FILETIME_EPOCH + timedelta(microseconds=value / 10))
    if not timestamps:
        return [], None
    first = timestamps[0]
    rel: List[float] = []
    seen = set()
    for stamp in timestamps:
        delta = (stamp - first).total_seconds()
        key = round(delta, 9)
        if key in seen:
            continue
        seen.add(key)
        rel.append(delta)
    return rel, first


def parse_leica_properties_xml(path: Path) -> Dict[str, object]:
    """Parse a Leica *_Properties.xml file into structured metadata."""

    if not path.exists():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    dimensions: Dict[str, Dict[str, str]] = {}
    for elem in root.iter("DimensionDescription"):
        dim_id = elem.attrib.get("DimID")
        if not dim_id:
            continue
        dimensions[dim_id] = elem.attrib
    px_size_xy_um = _parse_numeric_token(dimensions.get("X", {}).get("Voxel"))
    if px_size_xy_um is None:
        px_size_xy_um = _parse_numeric_token(dimensions.get("Y", {}).get("Voxel"))
    z_step_um = _parse_numeric_token(dimensions.get("Z", {}).get("Voxel"))
    size_x = int(dimensions.get("X", {}).get("NumberOfElements", "0") or 0)
    size_y = int(dimensions.get("Y", {}).get("NumberOfElements", "0") or 0)
    size_t = int(dimensions.get("T", {}).get("NumberOfElements", "0") or 0)
    size_c = int(dimensions.get("C", {}).get("NumberOfElements", "0") or 0)
    size_z = int(dimensions.get("Z", {}).get("NumberOfElements", "0") or 0)
    time_voxel_seconds = _parse_numeric_token(dimensions.get("T", {}).get("Voxel"))
    timestamp_list = root.findtext(".//TimeStampList")
    timestamps: List[float] = []
    datetime_acquired: Optional[datetime] = None
    if timestamp_list:
        timestamps, datetime_acquired = _decode_leica_timestamp_list(timestamp_list)
    if not timestamps and time_voxel_seconds and size_t:
        timestamps = [i * time_voxel_seconds for i in range(size_t)]
    stage_node = root.find(".//ATLConfocalSettingDefinition")
    stage_position_mm: Optional[Tuple[float, float]] = None
    if stage_node is not None:
        stage_x = _parse_numeric_token(stage_node.attrib.get("StagePosX"))
        stage_y = _parse_numeric_token(stage_node.attrib.get("StagePosY"))
        if stage_x is not None and stage_y is not None:
            stage_position_mm = (stage_x / 1000.0, stage_y / 1000.0)
    channel_names: List[str] = []
    for ch in root.iter("ChannelDescription"):
        for key in ("Name", "NameOfMeasuredQuantity", "LUTName", "ChannelTag"):
            name = ch.attrib.get(key)
            if name:
                channel_names.append(name.strip())
                break
    comment = root.findtext(".//Comment")
    return {
        "px_size_xy_um": px_size_xy_um,
        "z_step_um": z_step_um,
        "time_voxel_seconds": time_voxel_seconds,
        "timestamps": timestamps,
        "datetime_acquired": datetime_acquired,
        "stage_position_mm": stage_position_mm,
        "channel_names": channel_names,
        "size_x": size_x,
        "size_y": size_y,
        "size_t": size_t,
        "size_c": size_c,
        "size_z": size_z,
        "comment": comment or "",
    }
