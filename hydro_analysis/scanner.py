"""Filesystem scanner for dataset metadata extraction."""
from __future__ import annotations

import argparse
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .data_loader import DatasetLoader
from .metadata import (
    DatasetMetadata,
    collect_notes,
    infer_dataset_base_name,
    parse_leica_properties_xml,
)


class DatasetScanner:
    """Scan a directory tree for TIFF datasets and persist metadata."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def scan(self) -> List[DatasetMetadata]:
        results: List[DatasetMetadata] = []
        processed: set[Tuple[Path, str]] = set()
        for tif_path in sorted(self.root.rglob("*.tif")):
            dataset_dir = tif_path.parent
            base_name = infer_dataset_base_name(tif_path)
            key = (dataset_dir, base_name)
            if key in processed:
                continue
            processed.add(key)
            if self._is_leica_dataset(dataset_dir, base_name):
                metadata = self._scan_leica_dataset(dataset_dir, base_name)
            else:
                metadata = self._scan_custom_dataset(dataset_dir, base_name)
            if metadata is None:
                continue
            dataset_root = self._ensure_dataset_root(dataset_dir, base_name)
            metadata.raw_metadata.setdefault("dataset_root", str(dataset_root))
            metadata.to_json(dataset_root / "meta.json")
            results.append(metadata)
        return results

    def _is_leica_dataset(self, dataset_dir: Path, base_name: str) -> bool:
        metadata_dir = dataset_dir / "MetaData"
        if not metadata_dir.exists():
            return False
        if (metadata_dir / f"{base_name}_Properties.xml").exists():
            return True
        return any(metadata_dir.glob(f"{base_name}*_Properties.xml"))

    def _scan_leica_dataset(self, dataset_dir: Path, base_name: str) -> Optional[DatasetMetadata]:
        metadata_dir = dataset_dir / "MetaData"
        properties_path = metadata_dir / f"{base_name}_Properties.xml"
        if not properties_path.exists():
            matches = sorted(metadata_dir.glob(f"{base_name}*_Properties.xml"))
            if matches:
                properties_path = matches[0]
        if not properties_path.exists():
            return None
        info = parse_leica_properties_xml(properties_path)
        tif_files = self._collect_dataset_frames(dataset_dir, base_name)
        if not tif_files:
            return None
        from tifffile import TiffFile

        with TiffFile(tif_files[0]) as tif:
            series = tif.series[0]
            dtype = str(series.dtype)
            height = series.shape[-2]
            width = series.shape[-1]
        size_y = info.get("size_y") or height
        size_x = info.get("size_x") or width
        size_t = info.get("size_t") or len(tif_files)
        size_c = info.get("size_c") or (len(info.get("channel_names", [])) or 1)
        size_z = info.get("size_z") or 1
        axes, shape = _compose_axes_and_shape(size_t, size_c, size_z, size_y, size_x)
        timestamps = [float(t) for t in info.get("timestamps", []) if t is not None]
        if not timestamps:
            step = info.get("time_voxel_seconds")
            if step and (size_t and size_t > 1):
                timestamps = [i * float(step) for i in range(size_t)]
        datetime_acquired = info.get("datetime_acquired")
        notes = collect_notes(info.get("comment", ""))
        metadata = DatasetMetadata(
            path=tif_files[0],
            axes=axes,
            shape=shape,
            dtype=dtype,
            px_size_xy_um=info.get("px_size_xy_um"),
            z_step_um=info.get("z_step_um"),
            timestamps=timestamps,
            datetime_acquired=datetime_acquired if isinstance(datetime_acquired, datetime) else None,
            stage_position_mm=info.get("stage_position_mm"),
            channel_names=info.get("channel_names", []),
            notes=notes,
            raw_metadata={
                "dataset_type": "leica",
                "dataset_base": base_name,
                "source_properties": str(properties_path),
                "frame_count": size_t or len(tif_files),
                "time_voxel_seconds": info.get("time_voxel_seconds"),
                "files": [path.name for path in tif_files],
            },
        )
        return metadata

    def _scan_custom_dataset(self, dataset_dir: Path, base_name: str) -> Optional[DatasetMetadata]:
        tif_files = self._collect_dataset_frames(dataset_dir, base_name)
        if not tif_files:
            return None
        loader = DatasetLoader(tif_files[0])
        loaded = loader.load()
        metadata = replace(loaded.metadata)
        frame_count = len(tif_files)
        if frame_count > 1:
            metadata.axes = "T" + metadata.axes
            metadata.shape = (frame_count,) + tuple(metadata.shape)
        timestamps = list(metadata.timestamps)
        datetime_acquired = metadata.datetime_acquired
        if not timestamps:
            timestamps, datetime_acquired = _derive_custom_timestamps(tif_files)
        metadata.timestamps = timestamps
        if datetime_acquired and not metadata.datetime_acquired:
            metadata.datetime_acquired = datetime_acquired
        metadata.raw_metadata.setdefault("dataset_type", "custom")
        metadata.raw_metadata.update(
            {
                "dataset_base": base_name,
                "frame_count": frame_count,
                "files": [path.name for path in tif_files],
            }
        )
        return metadata

    def _collect_dataset_frames(self, dataset_dir: Path, base_name: str) -> List[Path]:
        return sorted(
            path
            for path in dataset_dir.glob("*.tif")
            if infer_dataset_base_name(path) == base_name
        )

    def _ensure_dataset_root(self, dataset_dir: Path, base_name: str) -> Path:
        root = dataset_dir / base_name
        root.mkdir(parents=True, exist_ok=True)
        qc_dir = root / "qc"
        qc_dir.mkdir(exist_ok=True)
        return root


def _compose_axes_and_shape(
    size_t: int,
    size_c: int,
    size_z: int,
    size_y: int,
    size_x: int,
) -> Tuple[str, Tuple[int, ...]]:
    axes: List[str] = []
    shape: List[int] = []
    if size_t and size_t > 1:
        axes.append("T")
        shape.append(int(size_t))
    if size_c and size_c > 1:
        axes.append("C")
        shape.append(int(size_c))
    if size_z and size_z > 1:
        axes.append("Z")
        shape.append(int(size_z))
    axes.extend(["Y", "X"])
    shape.extend([int(size_y), int(size_x)])
    return "".join(axes), tuple(shape)


def _derive_custom_timestamps(paths: Sequence[Path]) -> Tuple[List[float], Optional[datetime]]:
    tokens: List[Tuple[float, Path]] = []
    for path in sorted(paths):
        match = re.search(r"_[tT](\d+)", path.stem)
        if match:
            tokens.append((float(match.group(1)), path))
    if tokens:
        tokens.sort(key=lambda item: item[0])
        base = tokens[0][0]
        return [value - base for value, _ in tokens], None
    mtimes = [path.stat().st_mtime for path in sorted(paths)]
    if not mtimes:
        return [], None
    base_time = mtimes[0]
    return [t - base_time for t in mtimes], datetime.fromtimestamp(base_time)


def scan_datasets(root: Path) -> List[DatasetMetadata]:
    """Convenience wrapper for scanning datasets under *root*."""

    return DatasetScanner(root).scan()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scan datasets and extract metadata")
    parser.add_argument("root", type=Path, help="Root directory to scan")
    args = parser.parse_args(argv)
    scanner = DatasetScanner(args.root)
    results = scanner.scan()
    print(f"Processed {len(results)} datasets under {args.root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
