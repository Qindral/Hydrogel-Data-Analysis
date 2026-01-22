'''This Script provides experiment data extraction and parsing. It is solely for the SPT experiment conducted in the keylab microscopy.'''
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Tuple, Any
import xml.etree.ElementTree as ET

import pandas as pd
# -----------------------------
# Dataclass requested by you
# -----------------------------

@dataclass
class DatasetFiles:
    """Container for a dataset with base TIF, REC, processed TIFs, and XMLs.

    XMLs are grouped by suffix/variant:
    - 'base': XMLs for the original TIF
    - 'processed': XMLs for processed versions
    - other suffixes as found
    """
    base_tif: Path
    rec_path: Optional[Path]
    processed_tifs: List[Path]
    xml_paths: List[Path]  # All XMLs (backward compatibility)
    xml_groups: Dict[str, List[Path]]  # {'base': [...], 'processed': [...], ...}
    base_name: str

    # Optional: store parsed rec metadata cache here (lazy)
    _rec_meta_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def all_tifs(self) -> List[Path]:
        """All TIF files (base + processed)."""
        return [self.base_tif] + list(self.processed_tifs)

    @property
    def base_xmls(self) -> List[Path]:
        """XMLs for the base (non-processed) TIF."""
        return self.xml_groups.get("base", [])

    @property
    def processed_xmls(self) -> List[Path]:
        """XMLs for processed versions."""
        return self.xml_groups.get("processed", [])

    @property
    def rec_metadata(self) -> Optional[Dict[str, Any]]:
        """Parsed metadata from .rec (PCO CamWare comment file), if available."""
        if self.rec_path is None:
            return None
        if self._rec_meta_cache is None:
            try:
                text = self.rec_path.read_text(encoding="utf-8", errors="replace")
                self._rec_meta_cache = parse_pco_camware_rec_text(text)
                self._rec_meta_cache["rec_file"] = str(self.rec_path)
            except Exception as e:
                self._rec_meta_cache = {"rec_file": str(self.rec_path), "error": str(e)}
        return self._rec_meta_cache


# -----------------------------
# REC parsing 
# -----------------------------

def _to_float(s: str) -> float:
    return float(s.strip().replace(",", "."))

def parse_rec_file(rec_path: Path) -> Dict[str, any]:
    """
    Parse PCO CamWare .rec file to extract metadata.    
    Args:
        rec_path: Path to .rec file
        
    Returns:
        Dictionary with keys: exposure_ms, delay_ms, fps, size_x, size_y, mpp
    """
    result = {
        'exposure_ms': None,
        'delay_ms': None,
        'fps': None,
        'size_x': None,
        'size_y': None,
        'mpp': None
    }
    
    try:
        
        content = rec_path.read_text(encoding='utf-16', errors='replace')
        
        # Extract exposure/delay
        match = re.search(r'Exposure\s*/\s*Delay\s*:\s*([\d.]+)\s*ms\s*/\s*([\d.]+)\s*ms', 
                         content, re.IGNORECASE)
        
        if match:
            exposure = float(match.group(1))
            delay = float(match.group(2))
            
            result['exposure_ms'] = exposure
            result['delay_ms'] = delay
            
            total_time_ms = exposure + delay
            if total_time_ms > 0:
                result['fps'] = 1000.0 / total_time_ms
        
        # Extract image size
        size_match = re.search(r'Picture\s+Size\s+horz\.?/vert\.?\s*:\s*(\d+)\s*/\s*(\d+)', 
                              content, re.IGNORECASE)
        
        if size_match:
            result['size_x'] = int(size_match.group(1))
            result['size_y'] = int(size_match.group(2))
            
            
            x, y = result['size_x'], result['size_y']
            # mpp measured via Thorlab Grid 10µm
            if x==200:
                result['mpp'] = 0.30 
            elif x == 400:
                result['mpp'] = 0.15 
            elif x==696:
                result['mpp'] = 0.149 
            elif x==696*2:
                result['mpp'] = 0.149 / 2
    
    except Exception as e:
        print(f"    Warning: Could not parse {rec_path.name}: {e}")
    
    return result

# -----------------------------
# XML parsing 
# -----------------------------

def read_trackmate_xml(xml_file_path: Path) -> Optional[pd.DataFrame]:
    """
    Parse TrackMate XML file and convert to pandas DataFrame.
    
    The TrackMate XML format contains 'particle' elements with nested
    'detection' elements for each time point.
    
    Args:
        xml_file_path: Path to TrackMate XML file
        
    Returns:
        DataFrame with columns ['frame', 'particle', 'x', 'y'], or None on error
    """
    try:
        # Parse XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        data_rows = []
        
        # Iterate through all particle tracks
        for particle_id, particle in enumerate(root.findall('particle')):
            # Extract detections (position at each time point)
            for detection in particle.findall('detection'):
                # Get attributes
                t_raw = detection.get('t')
                x_raw = detection.get('x')
                y_raw = detection.get('y')
                
                # Convert to appropriate types and store
                row = {
                    'frame': int(float(t_raw)),  # Handle "40.0" format
                    'particle': particle_id + 1,  # 1-indexed particle IDs
                    'x': float(x_raw),
                    'y': float(y_raw)
                }
                data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        if not df.empty:
            # Ensure correct column order
            df = df[['frame', 'particle', 'x', 'y']]
            
            # Sort for better readability
            df = df.sort_values(by=['frame', 'particle']).reset_index(drop=True)
            
        return df

    except Exception as e:
        print(f"Error parsing {xml_file_path}: {e}")
        return None


# -----------------------------
# Canonicalization / grouping
# -----------------------------

TIFF_EXT = {".tif", ".tiff"}
REC_EXT = {".rec"}
XML_EXT = {".xml"}

# repeated suffixes at end: processed, Tracks, track
END_TOKEN_RE = re.compile(r"([ _\-.]+)(processed|tracks|track)\Z", re.IGNORECASE)

def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def canonical_base(stem: str) -> str:
    """
    Make a stable base name from any derived filename:
      A4_xxx_processed_processed -> A4_xxx
      A4_xxx_Tracks -> A4_xxx
    """
    s = _norm_spaces(stem)
    while True:
        m = END_TOKEN_RE.search(s)
        if not m:
            break
        s = s[:m.start()].rstrip(" _-.\t")
        s = _norm_spaces(s)
    return s

def processed_level(stem: str) -> int:
    """Count how often 'processed' appears at the end (processed_processed => 2)."""
    s = _norm_spaces(stem)
    level = 0
    while True:
        m = re.search(r"([ _\-.]+)processed\Z", s, flags=re.IGNORECASE)
        if not m:
            break
        level += 1
        s = s[:m.start()].rstrip(" _-.\t")
        s = _norm_spaces(s)
    return level

def xml_variant_from_path(xml_path: Path) -> str:
    """
    Classify XML variant mainly by name:
      - contains processed -> 'processed'
      - else -> 'base'
    You can extend this later with more variants.
    """
    name = xml_path.stem.lower()
    if "processed" in name:
        return "processed"
    return "base"


# -----------------------------
# Index class: universal usage
# -----------------------------

class DatasetIndex:
    """
    Build once per experiment folder and use everywhere:
      idx = DatasetIndex.from_root(root)
      ds  = idx.from_any_path(tif_or_rec_or_xml)
    """

    def __init__(self, datasets: Dict[str, DatasetFiles]):
        self._datasets = datasets

        # Lookup tables for fast resolution from arbitrary path
        self._by_tif: Dict[Path, str] = {}
        self._by_rec: Dict[Path, str] = {}
        self._by_xml: Dict[Path, str] = {}

        for base, ds in datasets.items():
            self._by_tif[ds.base_tif] = base
            for p in ds.processed_tifs:
                self._by_tif[p] = base
            if ds.rec_path:
                self._by_rec[ds.rec_path] = base
            for x in ds.xml_paths:
                self._by_xml[x] = base

    @property
    def datasets(self) -> Dict[str, DatasetFiles]:
        return self._datasets

    def list_bases(self) -> List[str]:
        return sorted(self._datasets.keys())

    def get(self, base_name: str) -> DatasetFiles:
        return self._datasets[base_name]

    def from_any_path(self, path: Path) -> DatasetFiles:
        """
        Given a .tif/.rec/.xml path (raw or processed), return the connected DatasetFiles.
        If it's a derived file we haven't stored by exact Path (e.g., relative vs absolute),
        we try canonical resolution.
        """
        p = Path(path).resolve()

        # Direct hit
        if p in self._by_tif:
            return self._datasets[self._by_tif[p]]
        if p in self._by_rec:
            return self._datasets[self._by_rec[p]]
        if p in self._by_xml:
            return self._datasets[self._by_xml[p]]

        # Fallback: infer base from name
        ext = p.suffix.lower()
        if ext in TIFF_EXT or ext in REC_EXT:
            base = canonical_base(p.stem)
            if base in self._datasets:
                return self._datasets[base]
        if ext in XML_EXT:
            # remove trailing _Tracks if present and canonicalize
            stem = re.sub(r"([ _\-.]+)tracks\Z", "", p.stem, flags=re.IGNORECASE)
            base = canonical_base(stem)
            if base in self._datasets:
                return self._datasets[base]

        raise KeyError(f"Could not resolve dataset for path: {p}")

    @classmethod
    def from_root(cls, root: Path) -> "DatasetIndex":
        datasets = build_datasets(root)
        return cls(datasets)


# -----------------------------
# Builder: scan folder and bundle
# -----------------------------

def _iter_files(root: Path) -> Iterable[Path]:
    for p in Path(root).rglob("*"):
        if p.is_file():
            yield p.resolve()

def build_datasets(root: Path) -> Dict[str, DatasetFiles]:
    """
    Scan root and build DatasetFiles bundles keyed by canonical base_name.
    Requires at least one base tif per dataset (will pick a best candidate).
    """
    root = Path(root).resolve()

    # temp collections keyed by base
    tifs_raw: Dict[str, List[Path]] = {}
    tifs_processed: Dict[str, List[Tuple[int, Path]]] = {}
    recs: Dict[str, List[Path]] = {}
    xmls: Dict[str, List[Path]] = {}

    for p in _iter_files(root):
        ext = p.suffix.lower()

        if ext in TIFF_EXT:
            base = canonical_base(p.stem)
            lvl = processed_level(p.stem)
            # also treat anything inside preprocess/ as processed candidate
            is_proc_folder = p.parent.name.lower() in {"preprocess", "processed", "proc"}
            if lvl > 0 or is_proc_folder:
                tifs_processed.setdefault(base, []).append((lvl, p))
            else:
                tifs_raw.setdefault(base, []).append(p)

        elif ext in REC_EXT:
            base = canonical_base(p.stem)
            recs.setdefault(base, []).append(p)

        elif ext in XML_EXT:
            # accept XML from Analysis/ or Tracks/ or containing "tracks"
            name_low = p.name.lower()
            if ("tracks" in name_low) or (p.parent.name.lower() in {"analysis", "tracks"}):
                stem = re.sub(r"([ _\-.]+)tracks\Z", "", p.stem, flags=re.IGNORECASE)
                base = canonical_base(stem)
                xmls.setdefault(base, []).append(p)

    # union of all seen bases
    all_bases = set(tifs_raw) | set(tifs_processed) | set(recs) | set(xmls)

    datasets: Dict[str, DatasetFiles] = {}
    for base in sorted(all_bases):
        raw_list = sorted(set(tifs_raw.get(base, [])))
        proc_list = sorted(set(tifs_processed.get(base, [])), key=lambda x: (x[0], str(x[1])))
        rec_list = sorted(set(recs.get(base, [])))
        xml_list = sorted(set(xmls.get(base, [])))

        # choose base tif:
        # prefer a true raw tif; else fall back to first processed (still better than nothing)
        if raw_list:
            base_tif = raw_list[0]
        elif proc_list:
            base_tif = proc_list[0][1]
        else:
            # No tif at all: skip dataset (or raise). Here we skip to avoid broken bundles.
            continue

        rec_path = rec_list[0] if rec_list else None
        processed_tifs = [p for _, p in proc_list]

        # group xmls
        xml_groups: Dict[str, List[Path]] = {}
        for x in xml_list:
            v = xml_variant_from_path(x)
            xml_groups.setdefault(v, []).append(x)

        datasets[base] = DatasetFiles(
            base_tif=base_tif,
            rec_path=rec_path,
            processed_tifs=processed_tifs,
            xml_paths=xml_list,
            xml_groups={k: sorted(v) for k, v in xml_groups.items()},
            base_name=base,
        )

    return datasets


def _to_float(s: str) -> float:
    # akzeptiert auch Komma als Dezimaltrenner
    return float(s.strip().replace(",", "."))


# -----------------------------
# Datenmodell
# -----------------------------

@dataclass
class TrackFile:
    path: Path
    source: str               # "TracksFolder" | "AnalysisFolder" | "Unknown"
    variant: str              # "raw" | "processed" | "unknown"

@dataclass
class ImageFile:
    path: Path
    kind: str                 # "raw" | "processed"
    processing_level: int = 0

@dataclass
class Dataset:
    base: str
    tiffs: List[ImageFile] = field(default_factory=list)
    recs: List[Path] = field(default_factory=list)
    tracks: List[TrackFile] = field(default_factory=list)
    logs: List[Path] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def best_raw_tiff(self) -> Optional[Path]:
        raws = [x.path for x in self.tiffs if x.kind == "raw"]
        return sorted(raws)[0] if raws else None

    def best_rec(self) -> Optional[Path]:
        return sorted(self.recs)[0] if self.recs else None

    def processed_candidates(self) -> List[Path]:
        procs = [x for x in self.tiffs if x.kind == "processed"]
        # sortiert: level 1 vor level 2, usw.
        procs_sorted = sorted(procs, key=lambda x: (x.processing_level, str(x.path)))
        return [x.path for x in procs_sorted]


# -----------------------------
# Normalisierung / Canonical Base
# -----------------------------

# Tokens, die wir am ENDE wiederholt abstrippen
STRIP_END_TOKENS = {
    "processed", "tracks", "track", "xml", "tif", "tiff"
}

# matcht _processed, " processed", -processed, .processed am Ende (wiederholt)
END_TOKEN_RE = re.compile(r"([ _\-.]+)(processed|tracks|track)\Z", re.IGNORECASE)

def normalize_spaces(s: str) -> str:
    # mehrere Spaces/Tabs -> ein Space
    s = re.sub(r"\s+", " ", s.strip())
    return s

def canonical_base_from_stem(stem: str) -> str:
    """
    Macht aus z.B.:
      "B3_inside_1d_1mmfrom injection_50_20mg_processed_processed"
    -> "B3_inside_1d_1mmfrom injection_50_20mg"
    """
    s = normalize_spaces(stem)
    while True:
        m = END_TOKEN_RE.search(s)
        if not m:
            break
        # entferne den token-Teil am Ende
        s = s[:m.start()].rstrip(" _-.\t")
        s = normalize_spaces(s)
    return s

def processed_level_from_stem(stem: str) -> int:
    """
    Zählt wie oft 'processed' am Ende (in Kaskade) vorkommt.
    """
    s = normalize_spaces(stem)
    level = 0
    while True:
        m = re.search(r"([ _\-.]+)processed\Z", s, flags=re.IGNORECASE)
        if not m:
            break
        level += 1
        s = s[:m.start()].rstrip(" _-.\t")
        s = normalize_spaces(s)
    return level


# -----------------------------
# Erkennung nach Datei/Ordner
# -----------------------------

TIFF_EXT = {".tif", ".tiff"}

def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p

def classify_track_source(path: Path) -> str:
    parent = path.parent.name.lower()
    if parent == "tracks":
        return "TracksFolder"
    if parent == "analysis":
        return "AnalysisFolder"
    return "Unknown"

def infer_track_variant_from_name(path: Path) -> str:
    name = path.stem.lower()
    # erkennt ..._processed_Tracks.xml oder ..._processed.xml
    if re.search(r"processed(\Z|[_ \-.])", name):
        return "processed"
    # erkennt ..._Tracks.xml oder ... .xml
    return "raw"


# -----------------------------
# Index builder
# -----------------------------

def build_index(root: Path) -> Dict[str, Dataset]:
    root = root.resolve()
    datasets: Dict[str, Dataset] = {}

    def ds(base: str) -> Dataset:
        if base not in datasets:
            datasets[base] = Dataset(base=base)
        return datasets[base]

    for p in iter_files(root):
        ext = p.suffix.lower()

        # TIFF
        if ext in TIFF_EXT:
            stem = p.stem
            base = canonical_base_from_stem(stem)
            level = processed_level_from_stem(stem)
            kind = "processed" if level > 0 or p.parent.name.lower() in {"preprocess", "processed", "proc"} else "raw"
            ds(base).tiffs.append(ImageFile(path=p, kind=kind, processing_level=level))

        # REC
        elif ext == ".rec":
            stem = p.stem
            base = canonical_base_from_stem(stem)
            ds(base).recs.append(p)

        # Logs aus preprocess
        elif ext == ".txt" and p.name.lower().endswith("_log.txt"):
            stem = p.stem
            base = canonical_base_from_stem(stem.replace("_log", ""))
            ds(base).logs.append(p)

        # XML (Tracks oder Analysis)
        elif ext == ".xml":
            # nur als Track-XML zählen, wenn:
            # - in Analysis/ liegt ODER in Tracks/ liegt ODER Name enthält "tracks"
            name_low = p.name.lower()
            if ("tracks" in name_low) or (p.parent.name.lower() in {"analysis", "tracks"}):
                stem = p.stem
                # falls Dateiname "..._Tracks" -> entfernen
                stem = re.sub(r"([ _\-.]+)tracks\Z", "", stem, flags=re.IGNORECASE)
                base = canonical_base_from_stem(stem)
                ds(base).tracks.append(
                    TrackFile(
                        path=p,
                        source=classify_track_source(p),
                        variant=infer_track_variant_from_name(p),
                    )
                )

    # Post-checks
    for base, d in datasets.items():
        if d.tiffs and not d.recs:
            d.notes.append("TIFF vorhanden, aber keine .rec Datei gefunden.")
        if d.recs and not d.tiffs:
            d.notes.append("REC vorhanden, aber keine TIFF Datei gefunden.")
        if d.tiffs and not d.tracks:
            d.notes.append("TIFF vorhanden, aber keine Track-XML gefunden.")
        # “processed_processed” Hinweis
        if any(x.processing_level >= 2 for x in d.tiffs):
            d.notes.append("Mehrfach preprocess erkannt (processed_processed).")

        # sortieren für stabile Ausgabe
        d.tiffs.sort(key=lambda x: (x.kind != "raw", x.processing_level, str(x.path)))
        d.recs.sort()
        d.logs.sort()
        d.tracks.sort(key=lambda x: (x.source, x.variant, str(x.path)))

    return datasets


def export_index_json(datasets: Dict[str, Dataset], out_path: Path) -> None:
    payload: Dict[str, Any] = {}
    for base, d in datasets.items():
        payload[base] = {
            "base": base,
            "tiffs": [
                {"path": str(x.path), "kind": x.kind, "processing_level": x.processing_level}
                for x in d.tiffs
            ],
            "recs": [str(p) for p in d.recs],
            "logs": [str(p) for p in d.logs],
            "tracks": [
                {"path": str(t.path), "source": t.source, "variant": t.variant}
                for t in d.tracks
            ],
            "best_raw_tiff": str(d.best_raw_tiff()) if d.best_raw_tiff() else None,
            "processed_candidates": [str(p) for p in d.processed_candidates()],
            "best_rec": str(d.best_rec()) if d.best_rec() else None,
            "notes": d.notes,
        }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str)
    ap.add_argument("--export", type=str, default="")
    ap.add_argument("--show", type=str, default="")
    args = ap.parse_args()

    root = Path(args.root)
    datasets = build_index(root)
    print(f"Datasets gefunden: {len(datasets)}")

    if args.export:
        out = Path(args.export)
        export_index_json(datasets, out)
        print(f"Index geschrieben: {out}")

    if args.show:
        base = args.show
        if base not in datasets:
            # hilfreiches Fallback: show arbeitet auch mit "ungefährem" base-string
            candidates = [b for b in datasets.keys() if base.lower() in b.lower()]
            raise KeyError(f"Base '{base}' nicht gefunden. Kandidaten: {candidates[:20]}")
        d = datasets[base]
        print(json.dumps({
            "base": d.base,
            "best_raw_tiff": str(d.best_raw_tiff()) if d.best_raw_tiff() else None,
            "processed_candidates": [str(p) for p in d.processed_candidates()],
            "best_rec": str(d.best_rec()) if d.best_rec() else None,
            "tracks": [{"path": str(t.path), "source": t.source, "variant": t.variant} for t in d.tracks],
            "notes": d.notes,
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# -----------------------------
# Convenience helper functions
# -----------------------------

def load_index(root: Path) -> DatasetIndex:
    """Small convenience wrapper."""
    return DatasetIndex.from_root(root)
