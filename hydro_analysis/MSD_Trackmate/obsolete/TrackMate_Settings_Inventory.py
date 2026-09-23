"""
Inventory of TrackMate detector/tracker/filter parameters actually used for
every D0 (water) and D_eff (20 mg/mL hydrogel) raw movie, matched to its
extracted tracks.

Compute stage (reads raw XML, no TIFF pixels touched): reuses
WATER_HYDROGEL_FOLDERS from Loc_Error_Analyse_immob_particle.py (as
MOVIE_FOLDERS; the same per-size D0/D_eff folder configuration already
established in MSD_per_file_publication.py / MSD_FromTrackmate_20mg.py,
formerly duplicated in the now-removed TaskC_Detectability_Compute.py) to
enumerate every *_Tracks.xml file via
core.io.build_datasets() (unmodified), and -- separately, for each one --
looks for that movie's own full, un-suffixed TrackMate project XML (the
actual Fiji export that recorded which detector/tracker/filter parameters
were used, as opposed to guessing/re-deriving them), extracting:

  segmentation : DetectorSettings (DETECTOR_NAME, RADIUS, THRESHOLD,
                 DO_MEDIAN_FILTERING, DO_SUBPIXEL_LOCALIZATION, TARGET_CHANNEL)
  filtering    : InitialSpotFilter (QUALITY threshold), SpotFilterCollection
                 (any additional spot filters), TrackFilterCollection (any
                 track-level filters, e.g. NUMBER_SPOTS)
  linking      : TrackerSettings (TRACKER_NAME, CUTOFF_PERCENTILE),
                 Linking (LINKING_MAX_DISTANCE), GapClosing
                 (ALLOW_GAP_CLOSING, GAP_CLOSING_MAX_DISTANCE, MAX_FRAME_GAP),
                 TrackSplitting, TrackMerging

This project's raw settings XML lives in one of several places relative to
a given *_Tracks.xml, tried in this order (find_settings_xml()):
  1. a loose {stem}.xml file next to the .tif (e.g. 2026.01.16\\20nm_water.xml)
  2. {condition folder}\\analysis\\{stem}.xml       (hydrogel convention)
  3. {condition folder}\\Analysis\\{stem}.xml       (capitalized variant)
  4. {condition folder}\\Analysis_new\\{stem}.xml   (50 nm hydrogel, matches its own Tracks_new)
  5. {condition folder}\\Trackmate_Analyses\\{stem}.xml (D_0 Wassermessung\\20 nm convention)
  6. {condition folder}\\Analyses\\{stem}.xml / \\analyses\\{stem}.xml (plural, no "Trackmate_"
     prefix -- confirmed present e.g. D_0 Wassermessung\\500 nm\\Analyses, distinct from
     its sibling \\Analysis\\ folder in the same directory)
Confirmed by direct inspection that D_0 Wassermessung\\20 nm\\Trackmate_Analyses\\
exists but its filenames (20nm.xml, 20nm_2.xml, ...) do NOT correspond 1:1
by stem to that folder's own Tracks\\*_Tracks.xml files (20 nm water_4_Tracks.xml,
...) -- a different recording/naming scheme, not just a missing file. When a
candidate subfolder exists but has no exact-stem match, that is recorded
distinctly (missing_note) from "no candidate folder/file found at all" --
both are real, different findings worth knowing apart.

Also confirmed by direct inspection (D_0 Wassermessung\\100 nm\\analysis\\
contains "100nmTriSpuffer_2.xml" for Tracks\\'s "100 nm TriSpuffer_2_Tracks.xml")
that the raw analysis exports frequently drop the spaces/underscores present
in the manually-renamed *_Tracks.xml files -- same movie, different manual
spacing, not a missing file. If no EXACT stem match is found, a second pass
compares a whitespace/underscore-stripped, lowercased key
(match_type="normalized" in the output, vs. "exact"); if that normalized
key matches more than one candidate file it is reported as ambiguous rather
than guessed -- never silently picks between multiple candidates.

Outputs (Auswertungsbilder\\TrackMate_Settings_Inventory\\):
  trackmate_settings_inventory.csv   one row per *_Tracks.xml file: every
                                      parameter above, plus track-count/
                                      calibration summary and whether
                                      settings were found at all
  trackmate_settings_missing.csv     only the rows with no matched settings
                                      XML -- exactly "where those files are
                                      missing"
Also pickles the full per-file result_dicts (tracks_df included, from
core.io.build_datasets()) alongside the parsed settings to
cache/trackmate_settings_inventory.pkl, so this lookup -- and the tracks
themselves -- are the single cached source of truth: nothing downstream
needs to re-parse the raw Fiji XML or reload the raw TrackMate tracks XML
to find out what parameters were actually used.

Run whenever the raw XML data changes -- unconditionally overwrites both
outputs every run (pure audit/compute script, no plotting).
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

from hydro_analysis.core.io import build_datasets
from hydro_analysis.MSD_Trackmate.Validation_Claude.Loc_Error_Analyse_immob_particle import (
    WATER_HYDROGEL_FOLDERS as MOVIE_FOLDERS,
)

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_FILE = Path(__file__).parent.parent / "cache" / "trackmate_settings_inventory.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH = SAVE_PATH_BASE / "TrackMate_Settings_Inventory"

SETTINGS_TAIL_BYTES = 4_000_000   # <Settings> sits near the end of the file; generous margin
SETTINGS_SUBFOLDER_CANDIDATES = (
    "analysis", "Analysis", "Analysis_new", "Trackmate_Analyses", "Analyses", "analyses",
)
_PIPELINE_SUFFIXES = ("_Tracks", "filtered_", "Resultof", "_processed", "_var")

_RE = {
    "detector_name": re.compile(r'<DetectorSettings[^>]*DETECTOR_NAME="([A-Za-z_]+)"'),
    "radius": re.compile(r'<DetectorSettings[^>]*\bRADIUS="([\d.eE+-]+)"'),
    "threshold": re.compile(r'<DetectorSettings[^>]*\bTHRESHOLD="([\d.eE+-]+)"'),
    "target_channel": re.compile(r'<DetectorSettings[^>]*TARGET_CHANNEL="(\d+)"'),
    "do_median_filtering": re.compile(r'<DetectorSettings[^>]*DO_MEDIAN_FILTERING="(true|false)"'),
    "do_subpixel_localization": re.compile(r'<DetectorSettings[^>]*DO_SUBPIXEL_LOCALIZATION="(true|false)"'),
    "initial_quality": re.compile(r'<InitialSpotFilter[^>]*feature="QUALITY"[^>]*value="([\d.eE+-]+)"'),
    "tracker_name": re.compile(r'<TrackerSettings[^>]*TRACKER_NAME="([A-Za-z_]+)"'),
    "cutoff_percentile": re.compile(r'<TrackerSettings[^>]*CUTOFF_PERCENTILE="([\d.eE+-]+)"'),
    "linking_max_distance": re.compile(r'<Linking\s[^>]*LINKING_MAX_DISTANCE="([\d.eE+-]+)"'),
    "allow_gap_closing": re.compile(r'<GapClosing[^>]*ALLOW_GAP_CLOSING="(true|false)"'),
    "gap_closing_max_distance": re.compile(r'<GapClosing[^>]*GAP_CLOSING_MAX_DISTANCE="([\d.eE+-]+)"'),
    "max_frame_gap": re.compile(r'<GapClosing[^>]*MAX_FRAME_GAP="(\d+)"'),
    "allow_track_splitting": re.compile(r'<TrackSplitting[^>]*ALLOW_TRACK_SPLITTING="(true|false)"'),
    "splitting_max_distance": re.compile(r'<TrackSplitting[^>]*SPLITTING_MAX_DISTANCE="([\d.eE+-]+)"'),
    "allow_track_merging": re.compile(r'<TrackMerging[^>]*ALLOW_TRACK_MERGING="(true|false)"'),
    "merging_max_distance": re.compile(r'<TrackMerging[^>]*MERGING_MAX_DISTANCE="([\d.eE+-]+)"'),
}
_BOOL_FIELDS = {"do_median_filtering", "do_subpixel_localization", "allow_gap_closing",
                "allow_track_splitting", "allow_track_merging"}
_INT_FIELDS = {"target_channel", "max_frame_gap"}
_NUMERIC_FIELDS = {"radius", "threshold", "initial_quality", "cutoff_percentile",
                    "linking_max_distance", "gap_closing_max_distance",
                    "splitting_max_distance", "merging_max_distance"}
_EMPTY_SETTINGS = {k: None for k in _RE} | {"extra_spot_filters": None, "track_filters": None}

_SPOT_FILTER_BLOCK_RE = re.compile(r"<SpotFilterCollection>(.*?)</SpotFilterCollection>", re.DOTALL)
_TRACK_FILTER_BLOCK_RE = re.compile(r"<TrackFilterCollection>(.*?)</TrackFilterCollection>", re.DOTALL)
_FILTER_ITEM_RE = re.compile(r'<Filter feature="([^"]+)" value="([\d.eE+-]+)" isabove="(true|false)"')


def _clean_stem(xml_path: Path) -> str:
    stem = xml_path.stem
    for suffix in _PIPELINE_SUFFIXES:
        stem = stem.replace(suffix, "")
    return stem


def _normalize(name: str) -> str:
    """Loose comparison key: strips all whitespace/underscores and
    lowercases. Bridges naming differences like "100 nm TriSpuffer_2"
    (Tracks\\, manually renamed) vs "100nmTriSpuffer_2" (raw Fiji export,
    named from the .tif) -- same movie, same characters, just different
    separators -- NOT a fuzzy/approximate match: every letter and digit
    must still agree, only whitespace/underscore/case is ignored."""
    return re.sub(r"[\s_]+", "", name).lower()


def _find_in_folder(folder: Path, stem: str, exclude: Path) -> tuple[Path | None, str | None, int, int]:
    """Search one folder for stem (exact, then normalized). Returns
    (match_or_None, match_type_or_None, n_other_files, n_ambiguous_normalized_matches)."""
    if not folder.is_dir():
        return None, None, 0, 0
    exact = folder / f"{stem}.xml"
    if exact.exists() and exact != exclude:
        return exact, "exact", 0, 0

    candidates = [p for p in folder.glob("*.xml") if p != exclude]
    if not candidates:
        return None, None, 0, 0

    target_norm = _normalize(stem)
    norm_matches = [p for p in candidates if _normalize(_clean_stem(p)) == target_norm]
    if len(norm_matches) == 1:
        return norm_matches[0], "normalized", len(candidates), 0
    return None, None, len(candidates), len(norm_matches)


def find_settings_xml(xml_path: Path) -> tuple[Path | None, str | None, dict[str, str]]:
    """Locate this movie's own full TrackMate project XML. Returns
    (path_or_None, match_type, diagnostics), where diagnostics maps a
    candidate location to a short human-readable note when that location
    exists but yielded no usable match -- distinguishing "no such
    folder/file", "folder exists, names don't correspond even normalized",
    and "folder exists, but the normalized name is AMBIGUOUS (matches >1
    file)" as three different, honestly reported outcomes."""
    stem = _clean_stem(xml_path)
    movie_dir = xml_path.parent.parent
    diagnostics: dict[str, str] = {}

    match, match_type, n_other, n_ambig = _find_in_folder(movie_dir, stem, xml_path)
    if match is not None:
        return match, match_type, diagnostics
    if n_ambig > 1:
        diagnostics["(loose file next to .tif)"] = f"{n_ambig} ambiguous normalized matches among {n_other} .xml files"
    elif n_other:
        diagnostics["(loose file next to .tif)"] = f"{n_other} other .xml present, none match"

    for sub in SETTINGS_SUBFOLDER_CANDIDATES:
        folder = movie_dir / sub
        match, match_type, n_other, n_ambig = _find_in_folder(folder, stem, xml_path)
        if match is not None:
            return match, match_type, diagnostics
        if n_ambig > 1:
            diagnostics[sub] = f"{n_ambig} ambiguous normalized matches among {n_other} .xml files"
        elif n_other:
            diagnostics[sub] = f"{n_other} other .xml present, none match"

    return None, None, diagnostics


def _read_tail(path: Path, max_bytes: int) -> str:
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, 2)
        return f.read().decode("utf-8", errors="ignore")


def _cast(key: str, raw: str | None):
    if raw is None:
        return None
    if key in _BOOL_FIELDS:
        return raw == "true"
    if key in _INT_FIELDS:
        return int(raw)
    if key in _NUMERIC_FIELDS:
        return float(raw)
    return raw


def _format_filters(block_match: re.Match | None) -> str | None:
    if block_match is None:
        return None
    items = _FILTER_ITEM_RE.findall(block_match.group(1))
    if not items:
        return None
    return "; ".join(f"{feat}{'>' if above == 'true' else '<'}{val}" for feat, val, above in items)


def parse_settings(settings_xml: Path) -> dict:
    tail = _read_tail(settings_xml, SETTINGS_TAIL_BYTES)
    out: dict = {}
    for key, pattern in _RE.items():
        m = pattern.search(tail)
        out[key] = _cast(key, m.group(1) if m else None)
    out["extra_spot_filters"] = _format_filters(_SPOT_FILTER_BLOCK_RE.search(tail))
    out["track_filters"] = _format_filters(_TRACK_FILTER_BLOCK_RE.search(tail))
    return out


def main() -> None:
    all_rows: list[dict] = []
    result_dicts_by_path: dict[str, dict] = {}

    for condition, size_folders in MOVIE_FOLDERS.items():
        for size_nm, folders in size_folders.items():
            for folder in folders:
                if not folder.exists():
                    print(f"  [SKIP] {condition} {size_nm:.0f} nm: Ordner nicht gefunden: {folder}")
                    continue
                datasets = build_datasets(folder)
                print(f"{condition} {size_nm:.0f} nm: {len(datasets)} Dateien aus {folder}")
                for _, result in datasets.items():
                    if result is None:
                        continue
                    xml_path = Path(result["xml_path"])
                    settings_xml, match_type, diagnostics = find_settings_xml(xml_path)

                    row = {
                        "condition": condition,
                        "particle_size_nm": size_nm,
                        "base_name": result.get("base_name"),
                        "xml_path": str(xml_path),
                        "tif_path": str(result.get("tif_path")) if result.get("tif_path") else None,
                        "num_tracks": result.get("num_tracks"),
                        "num_frames": result.get("num_frames"),
                        "mpp": result.get("mpp"),
                        "fps": result.get("fps"),
                        "settings_xml": str(settings_xml) if settings_xml else None,
                        "settings_found": settings_xml is not None,
                        "match_type": match_type,
                        "missing_note": None,
                    }

                    if settings_xml is not None:
                        row.update(parse_settings(settings_xml))
                        threshold_str = f"{row['threshold']:.4g}" if row.get("threshold") is not None else "n/a"
                        print(f"  [OK, {match_type}] {row['base_name']}: {row.get('detector_name')} "
                              f"radius={row.get('radius')} threshold={threshold_str}")
                    else:
                        row.update(_EMPTY_SETTINGS)
                        note = "; ".join(f"{k}: {v}" for k, v in diagnostics.items())
                        row["missing_note"] = note if note else "no candidate settings file/folder found"
                        print(f"  [MISSING] {row['base_name']}: {row['missing_note']}")

                    all_rows.append(row)
                    result_dicts_by_path[str(xml_path)] = result

    df = pd.DataFrame(all_rows)
    n_missing = int((~df["settings_found"]).sum())
    print(f"\nGesamt: {len(df)} Track-Dateien, {n_missing} ohne zugehörige Settings-XML.")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = SAVE_PATH / "trackmate_settings_inventory.csv"
    df.to_csv(csv_path, index=False)
    print(f"Tabelle gespeichert: {csv_path}")

    missing_cols = ["condition", "particle_size_nm", "base_name", "xml_path", "missing_note"]
    missing_df = df.loc[~df["settings_found"], missing_cols]
    missing_csv = SAVE_PATH / "trackmate_settings_missing.csv"
    missing_df.to_csv(missing_csv, index=False)
    print(f"Fehlende Settings gespeichert: {missing_csv} ({len(missing_df)} Dateien)")
    if not missing_df.empty:
        print(missing_df.to_string(index=False))

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"table": df, "result_dicts": result_dicts_by_path}, f)
    print(f"Cache gespeichert: {CACHE_FILE}")


if __name__ == "__main__":
    main()
