"""
Audit track-length filtering in the existing D0 and Deff MSD caches.

Reads msd_d0_results.pkl, msd_20mg_files.pkl and their source TrackMate
track exports. Compares cached tracks_df IDs with saved fit_results_MSD
['imsd'] columns and the current MIN_TRACK_LENGTH. Optional settings inventory
provides candidate full-project XML paths; their NUMBER_SPOTS filters are read
fresh, but filename matching does not prove that a project produced an export.
No raw-data refitting, cache changes, or analysis-function changes.
Writes per-file and per-size audit CSVs into Auswertungsbilder/TrackLength_Audit.
Run this file directly using the project's Python environment.
"""
from pathlib import Path
import pickle
import re
import sys
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from hydro_analysis.core.io import read_trackmate_xml, get_dls_labels
from hydro_analysis.core.analysis import MIN_TRACK_LENGTH

# Configuration
CACHE_DIR = PROJECT_ROOT / "hydro_analysis" / "MSD_Trackmate" / "cache"
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
    r"\Experiments and Results - Data\Auswertungsbilder\TrackLength_Audit"
)


def main():
    inventory_path = CACHE_DIR / "trackmate_settings_inventory.pkl"
    inventory = {}
    if inventory_path.exists():
        with inventory_path.open("rb") as handle:
            table = pickle.load(handle)["table"]
        inventory = {r["xml_path"]: r for r in table.to_dict("records")}
    labels = get_dls_labels()
    rows = []
    for condition, name in [("D0", "msd_d0_results.pkl"), ("Deff", "msd_20mg_files.pkl")]:
        with (CACHE_DIR / name).open("rb") as handle:
            results = pickle.load(handle)
        for record in results.values():
            lengths = record["tracks_df"].groupby("particle").size()
            imsd = (record.get("fit_results_MSD") or {}).get("imsd")
            ids = set(imsd.columns) if imsd is not None else set()
            expected = set(lengths[lengths >= MIN_TRACK_LENGTH].index)
            source = Path(record["xml_path"])
            raw = read_trackmate_xml(source) if source.exists() else None
            raw_lengths = raw.groupby("particle").size() if raw is not None and not raw.empty else None
            settings_path = inventory.get(str(source), {}).get("settings_xml")
            number_spots_filters = []
            if isinstance(settings_path, str) and Path(settings_path).exists():
                with open(settings_path, "rb") as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell() - 4_000_000))
                    tail = handle.read().decode("utf-8", errors="replace")
                match = re.search(r"<TrackFilterCollection>(.*?)</TrackFilterCollection>", tail, re.S)
                if match:
                    collection = ET.fromstring(match.group(0))
                    number_spots_filters = [element.attrib for element in collection if element.get("feature") == "NUMBER_SPOTS"]
            rows.append(dict(
                condition=condition, dls_size_nm=labels.get(record["particle_size_nm"], record["particle_size_nm"]),
                xml_path=str(source), source_available=raw_lengths is not None,
                source_tracks=len(raw_lengths) if raw_lengths is not None else None,
                source_below_10=int((raw_lengths < 10).sum()) if raw_lengths is not None else None,
                source_min_frames=int(raw_lengths.min()) if raw_lengths is not None else None,
                cached_tracks=len(lengths), cached_below_10=int((lengths < 10).sum()),
                msd_tracks=len(ids), msd_below_10=len(ids & set(lengths[lengths < 10].index)),
                msd_min_frames=lengths.reindex(list(ids)).min() if ids else None,
                ids_match_length_filter=ids == expected, min_track_length_setting=MIN_TRACK_LENGTH,
                stored_num_tracks=record["num_tracks"],
                edge_removed=record.get("edge_filter_removed"), edge_splits=record.get("edge_filter_splits"),
                candidate_settings_xml=settings_path, candidate_number_spots_filters=str(number_spots_filters),
            ))
    per_file = pd.DataFrame(rows)
    summary = per_file.groupby(["condition", "dls_size_nm"]).agg(
        files=("xml_path", "size"), available_sources=("source_available", "sum"),
        source_tracks=("source_tracks", "sum"), source_below_10=("source_below_10", "sum"),
        source_min_frames=("source_min_frames", "min"),
        cached_tracks=("cached_tracks", "sum"), cached_below_10=("cached_below_10", "sum"),
        msd_tracks=("msd_tracks", "sum"), msd_below_10=("msd_below_10", "sum"),
        msd_min_frames=("msd_min_frames", "min"), all_ids_match=("ids_match_length_filter", "all"),
    )
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    per_file.to_csv(SAVE_PATH / "track_length_filter_audit_files.csv", index=False)
    summary.to_csv(SAVE_PATH / "track_length_filter_audit_summary.csv")
    print(summary.to_string())
    print(f"Saved audit: {SAVE_PATH}")


if __name__ == "__main__":
    main()
