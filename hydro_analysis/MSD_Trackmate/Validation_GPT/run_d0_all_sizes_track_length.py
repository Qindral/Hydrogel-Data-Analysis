"""D0 all-size track-length-bias analysis from complete TrackMate sessions.

Compute stage: discovers the unprocessed complete sessions in the six water
particle-size folders, matches raw TIFF/.rec pairs, and writes a manifest plus
the four-lag, minimum-10-localization D0 bias analysis to an output folder.
It is independent of existing caches and legacy validation scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd

try:
    from .complete_spt_validation import RC, load_manifest, task_b, track_table
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from complete_spt_validation import RC, load_manifest, task_b, track_table

import matplotlib.pyplot as plt


# DLS labels used throughout the dissertation figures; folder names remain in
# the manifest so the mapping is auditable.
NOMINAL_TO_LABEL_NM = {20: 35, 50: 50, 100: 100, 200: 240, 500: 560, 1000: 1370}
SESSION_DIR_NAMES = {"analysis", "analyses", "trackmate_analyses"}
REJECT_TOKENS = ("_tracks", "processed", "resultof", "_var")


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def image_filename(xml_path: Path) -> str | None:
    try:
        node = ET.parse(xml_path).getroot().find("Settings/ImageData")
        return node.attrib.get("filename") if node is not None else None
    except ET.ParseError:
        return None


def discover(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected, skipped = [], []
    for nominal, label in NOMINAL_TO_LABEL_NM.items():
        size_dir = root / f"{nominal} nm"
        if not size_dir.is_dir():
            skipped.append({"nominal_nm": nominal, "xml_path": "", "reason": "size folder missing"})
            continue
        tiffs = [p for suffix in ("*.tif", "*.tiff") for p in size_dir.rglob(suffix)]
        by_name: dict[str, list[Path]] = {}
        for tif in tiffs:
            by_name.setdefault(normalise(tif.name), []).append(tif)
        seen_tiffs: set[Path] = set()
        for xml in size_dir.rglob("*.xml"):
            if xml.parent.name.lower() not in SESSION_DIR_NAMES or any(token in xml.stem.lower() for token in REJECT_TOKENS):
                continue
            image_name = image_filename(xml)
            # Exact filename is authoritative. Normalisation is only a
            # fallback for historical exports that differ solely in spacing.
            exact = [tif for tif in tiffs if image_name and tif.name.lower() == image_name.lower()]
            matches = exact if len(exact) == 1 else by_name.get(normalise(image_name or ""), [])
            if len(matches) != 1:
                skipped.append({"nominal_nm": nominal, "xml_path": str(xml),
                                "reason": f"TIFF match count for {image_name!r}: {len(matches)}"})
                continue
            tif = matches[0].resolve()
            if tif in seen_tiffs:
                skipped.append({"nominal_nm": nominal, "xml_path": str(xml), "reason": "duplicate TIFF/session"})
                continue
            seen_tiffs.add(tif)
            selected.append({"xml_path": str(xml), "tiff_path": str(tif), "condition": "water",
                             "particle_size_nm": label, "label": f"D0 {label} nm | {xml.stem}",
                             "nominal_folder_nm": nominal})
    return pd.DataFrame(selected), pd.DataFrame(skipped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, skipped = discover(args.d0_root)
    if manifest.empty:
        raise RuntimeError("No unambiguous complete-session XML/TIFF pairs found.")
    manifest_path = args.output / "d0_all_sizes_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    skipped.to_csv(args.output / "d0_all_sizes_skipped.csv", index=False)
    records = load_manifest(manifest_path)
    tracks = track_table(records)
    with plt.rc_context(RC):
        task_b(tracks, args.output)
    print(f"Selected {len(manifest)} XML/TIFF pairs; wrote {args.output}")


if __name__ == "__main__":
    main()
