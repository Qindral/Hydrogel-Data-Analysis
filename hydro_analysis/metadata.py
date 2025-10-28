"""Metadata utilities for Hydro Analysis."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json


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
