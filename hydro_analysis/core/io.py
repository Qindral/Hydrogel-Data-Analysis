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
# REC parsing 
# -----------------------------


def find_rec_tif_files(xml_path: Path) -> Dict[str, any]:
    """
    Search for TIFF or .rec files in the same directory as XML.
    .rec files will be always in the parent-parent folder of the XML.
    
    Args:
        xml_path: Path to TrackMate XML file
        
    Returns:
        Dictionary with calibration info: fps, mpp, mode, files_found
    """
    xml_dir = xml_path.parent.parent
    # print('dir', xml_dir)
    # Clean up XML path to find corresponding .rec file
    # Remove common suffixes and patterns
    xml_stem = xml_path.stem
    xml_stem = xml_stem.replace('_Tracks', '') 
    xml_stem = xml_stem.replace('_processed', '')
    xml_stem = xml_stem.replace('Resultsof', '')
    xml_stem = xml_stem.replace('_var', '')
    

    rec_path = xml_dir / f"{xml_stem}.tif.rec"
    if not rec_path.exists():
        rec_path = xml_dir / f"{xml_stem}.rec"


    tif_path = xml_dir / f"{xml_stem}.tif"
    # print("rec: ", rec_path)
    result = {
        'fps': None,
        'mpp': None,
        'tiff_file':    None,
        'rec_file': None,
    }
    

    

    # Try to parse .rec file for FPS
    
    rec_info = parse_rec_file(rec_path)
    fps = rec_info.get('fps')
    mpp = rec_info.get('mpp')
    

    
    # Determine calibration
    
    result['mpp'] = mpp
    result['fps'] = fps
    result['tiff_file'] = tif_path if tif_path.exists() else None
    result['rec_file'] = rec_path if rec_path.exists() else None
    
    return result



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
    encoding = check_text_encoding(rec_path)
    try:
        
        content = rec_path.read_text(encoding=encoding, errors='replace')
        
        
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
                
                row = {
                    'frame': int(float(t_raw)),  
                    'particle': particle_id + 1,  
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
            df = df.sort_values(by=[ 'particle','frame']).reset_index(drop=True)
            
        return df

    except Exception as e:
        print(f"Error parsing {xml_file_path}: {e}")
        return None

def check_text_encoding(path: Path) -> None:
    """Utility to check text encoding of a file."""
    import chardet
    with open(path, 'rb') as f:
        result = chardet.detect(f.read())
    
        # print(result['encoding'])
    return result['encoding']
    #print(path.read_text(encoding=result['encoding'], errors='replace'))
    
# -----------------------------
# Canonicalization / grouping
# -----------------------------




# def _norm_spaces(s: str) -> str:
#     return re.sub(r"\s+", " ", s.strip())

# def canonical_base(stem: str) -> str:
#     """
#     Make a stable base name from any derived filename:
#       A4_xxx_processed_processed -> A4_xxx
#       A4_xxx_Tracks -> A4_xxx
#     """
#     s = _norm_spaces(stem)
#     while True:
#         m = END_TOKEN_RE.search(s)
#         if not m:
#             break
#         s = s[:m.start()].rstrip(" _-.\t")
#         s = _norm_spaces(s)
#     return s

# def processed_level(stem: str) -> int:
#     """Count how often 'processed' appears at the end (processed_processed => 2)."""
#     s = _norm_spaces(stem)
#     level = 0
#     while True:
#         m = re.search(r"([ _\-.]+)processed\Z", s, flags=re.IGNORECASE)
#         if not m:
#             break
#         level += 1
#         s = s[:m.start()].rstrip(" _-.\t")
#         s = _norm_spaces(s)
#     return level

# def xml_variant_from_path(xml_path: Path) -> str:
#     """
#     Classify XML variant mainly by name:
#       - contains processed -> 'processed'
#       - else -> 'base'
#     You can extend this later with more variants.
#     """
#     name = xml_path.stem.lower()
#     if "processed" in name:
#         return "processed"
#     return "base"


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

def scan_xml_folder(root: Path) -> List[Path]:
    """Recursively find all TrackMate XML files in the folder."""
    xml_files = list(root.rglob("*.xml"))
    return xml_files


def build_datasets(root: Path) -> Dict[str, DatasetFiles]:
    """
    Scan the root folder for datasets and build DatasetFiles instances.
    
    Args:
        root: Root folder to scan
        """
    xml_files = scan_xml_folder(root)
    
    datasets = {}
    print(root)
    for xml_file in xml_files:
        print("_find",xml_file)
        rec_tif_info = find_rec_tif_files(xml_file)
        print("names")
        base_tif = rec_tif_info['tiff_file']
        rec_path = rec_tif_info['rec_file']
        tracks = read_trackmate_xml(xml_file)
        
        datasets[xml_file.stem] = {
            "base_tif": base_tif if base_tif else Path(),
            "rec_tif": rec_tif_info,
            "xml_paths": [xml_file],
            "tracks": tracks
        }

    
    return datasets



# -----------------------------
# Compare
# ---
def compare_xml(path1: Path, xml_path2: Path) -> float:
    """
    Compare two TrackMate XML files and return similarity score.
    
    Args:
        path1: Path to first XML file
        xml_path2: Path to second XML file
    """
    tracks_1 = read_trackmate_xml(path1)
    tracks_2 = read_trackmate_xml(xml_path2)
    for row1, row2 in zip(tracks_1.itertuples(), tracks_2.itertuples()):
        if row1.frame != row2.frame:
            similarity = 0
            continue
        else: 
            similarity = 100
    if similarity == 100:
        print("Die XML Dateien sind identisch")
        return similarity
    frame_max = max(tracks_1['frame'].max(), tracks_2['frame'].max())
    matched_particles = {}
    matched = 0
    for frame in range(frame_max + 1):
        particles_in_frame_1 = tracks_1[tracks_1['frame'] == frame]
        particles_in_frame_2 = tracks_2[tracks_2['frame'] == frame]
        for _, particle1 in particles_in_frame_1.iterrows():
            for _, particle2 in particles_in_frame_2.iterrows():
                dist = ((particle1['x'] - particle2['x'])**2 + (particle1['y'] - particle2['y'])**2)**0.5
                if dist < 1e-5:
                    matched_particles[particle1['particle']] = particle2['particle']
                    break
        
        unmatched_1 = set(particles_in_frame_1['particle']) - set(matched_particles.keys())
        unmatched_2 = set(particles_in_frame_2['particle']) - set(matched_particles.values())
        
        if len(matched_particles) == len(particles_in_frame_1) == len(particles_in_frame_2):
            print(f"Frame {frame}: Gleiche Partikel")
        elif len(matched_particles) == 0:
            matched += 0
        else:
            print(f"Frame {frame}: Partielle Übereinstimmung")
            print(f"  Zugeordnete Partikel: {matched_particles}")
            print(f"  Ungepaarte in Datei 1: {unmatched_1}")
            print(f"  Ungepaarte in Datei 2: {unmatched_2}")
            matched += len(matched_particles)
    print(len(matched_particles)/max(len(tracks_1), len(tracks_2))*100)
    
    


    for match1, match2 in matched_particles.items():
        particle1 = tracks_1[tracks_1['particle'] == match1]
        particle2 = tracks_2[tracks_2['particle'] == match2]
        


# -----------------------------
# CLI
# -----------------------------

def main():
    path = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm\Tracks_old\20 nm_2_Tracks.xml" # add a path to a file Tracks_xml

    rec_tif = find_rec_tif_files(Path(path))
    print(rec_tif)
    meta_data = parse_rec_file(rec_tif['rec_file'])
    print(json.dumps(meta_data, indent=4))
    from matplotlib import pyplot as plt
    import tifffile as tiff
    with tiff.TiffFile(rec_tif['tiff_file']) as tif:
        img = tif.pages[0].asarray()
    # plt.imshow(img, cmap='gray')
    # plt.show()
    tracks = read_trackmate_xml(Path(path))
    print(tracks.head())

    print('Folder Analysis:')
    folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\500 nm\Tracks") # add a path to a folder
    datasets = build_datasets(folder)

    for base_name, ds in datasets.items():
        print(f"Dataset: {base_name}")
        print(f"  Base TIFF: {ds['base_tif']}")
        print(f"  REC file: {ds['rec_tif']}")
        print(f"  tracks: {ds['tracks'].head() if ds['tracks'] is not None else 'None'}")

    print("XML Comparison:")
    xml1 = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\Tracks\1000 nm_2_60_Tracks.xml"
    # xml2 = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\Tracks\1000 nm_2_Tracks.xml"
    # xml1 = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\Tracks\1000 nm_2_Tracks.xml"
    xml2 = r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\1000 nm\Tracks\1000 nm_60_Tracks.xml"
    # similarity = compare_xml(Path(xml1), Path(xml2))
    xml_test = read_trackmate_xml(xml2)
    number = xml_test[(xml_test['particle']==1) & (xml_test['frame']==3)]['x'].values[0]



if __name__ == "__main__":
    main()



