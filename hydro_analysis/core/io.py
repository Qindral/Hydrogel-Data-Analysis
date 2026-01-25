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


def extract_particle_size_from_path(folder_path: Path) -> Optional[float]:
    """
    Extract particle size (in nm) from folder name.
    
    Supports various naming formats:
    - "50nm"
    - "100_nm" 
    - "200 nm"
    
    Args:
        folder_path: Path object of the folder
        
    Returns:
        Particle size in nanometers, or None if not found
    """
    folder_name = folder_path.name
    match = re.search(r'(\d+)\s*nm', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


    """
    Scan directory structure and collect XML track files with associated FPS data.
    
    This function focuses on XML files (which contain the track data) and retrieves
    FPS information from .rec files in the same particle size folder.
    
    Expected structure:
        root_path/
            50nm/
                *.rec (for FPS extraction)
                Tracks/
                    *.xml (primary data)
            100nm/
                *.rec
                Tracks/
                    *.xml
    
    Args:
        root_path: Root directory to scan
        
    Returns:
        DataFrame with columns:
        - particle_size_nm: Particle size in nanometers
        - xml_path: Path to XML track file
        - xml_name: Basename of XML file
        - exposure_ms: Exposure time from matched .rec file
        - delay_ms: Delay time from matched .rec file
        - fps: Calculated frames per second from matched .rec file
        - mpp: Micrometers per pixel calibration
        - mode: Detection mode description
    """
    file_records = []
    
    print(f"\nDEBUG: Scanning folders in {root_path}")
    print("=" * 70)
    
    # Walk through subdirectories
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        # Extract particle size from folder name
        particle_size = extract_particle_size_from_path(subfolder)
        
        print(f"\nFolder: {subfolder.name}")
        print(f"   Particle size: {particle_size} nm" if particle_size else "   Warning: No particle size detected")
        
        if particle_size is None:
            continue
        
        # Find all REC files in main folder and parse them
        rec_files = sorted(list(subfolder.glob("*.rec")))
        print(f"   Found {len(rec_files)} .rec files")
        
        # Parse all REC files and store by basename for individual matching
        rec_info_by_basename = {}
        
        for rec_file in rec_files:
            rec_info = parse_rec_file(rec_file)
            if rec_info['fps'] is not None:
                basename = rec_file.stem
                rec_info_by_basename[basename] = rec_info
                print(f"     • {rec_file.name}: {rec_info['fps']:.2f} fps, size={rec_info['size_x']}x{rec_info['size_y']} px" 
                      if rec_info['size_x'] else f"     • {rec_file.name}: {rec_info['fps']:.2f} fps")
        
        if not rec_info_by_basename:
            print(f"   Warning: No valid FPS data from .rec files")
        
        # Find all XML files in Tracks subfolder
        tracks_folder = subfolder / "Tracks"
        xml_files = []
        
        if tracks_folder.exists() and tracks_folder.is_dir():
            xml_files = sorted(list(tracks_folder.glob("*.xml")))
            print(f"   Found {len(xml_files)} XML track files in Tracks/")
        else:
            print(f"   Warning: No Tracks/ subfolder found")
        
        if len(xml_files) == 0:
            print(f"   Skipping - no XML files found")
            continue
        
        # Create one record per XML file with individual .rec matching
        for xml_file in xml_files:
            # Extract image dimensions from XML
            x_max, y_max = extract_image_dimensions_from_xml(xml_file)
            
            # Match XML to .rec file based on filename
            xml_basename = xml_file.stem.replace('_Tracks', '').replace(' Tracks', '')
            matched_rec_info = None
            
            for rec_basename, rec_info in rec_info_by_basename.items():
                if rec_basename in xml_basename or xml_basename in rec_basename:
                    matched_rec_info = rec_info
                    break
            
            # Use matched .rec parameters if available
            if matched_rec_info:
                rec_fps = matched_rec_info['fps']
                exposure_ms = matched_rec_info['exposure_ms']
                delay_ms = matched_rec_info['delay_ms']
                if matched_rec_info['size_x'] is not None:
                    x_max = matched_rec_info['size_x']
                    y_max = matched_rec_info['size_y']
            else:
                rec_fps = None
                exposure_ms = None
                delay_ms = None
            
            # Determine mpp based on image size (prioritize), then fps
            if x_max is not None and y_max is not None:
                mpp = get_mpp_from_size(x_max, y_max)
                fps = rec_fps if rec_fps else DEFAULT_FPS
                mode = f'{x_max}x{y_max}px'
            else:
                mpp, fps, mode = get_mpp_from_fps_and_size(
                    fps=rec_fps,
                    x_max=x_max,
                    y_max=y_max
                )
            
            file_records.append({
                'particle_size_nm': particle_size,
                'xml_path': str(xml_file),
                'xml_name': xml_file.name,
                'x_max': x_max,
                'y_max': y_max,
                'exposure_ms': exposure_ms,
                'delay_ms': delay_ms,
                'fps': fps,
                'mpp': mpp,
                'mode': mode,
            })
            
            size_info = f"{x_max}x{y_max}" if x_max and y_max else "unknown size"
        
    
    
    df = pd.DataFrame(file_records)
    
    return df

# -----------------------------
# Single Files
# -----------------------------

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

def Collect_from_list (folder_list: Dict[str, List[Path]]) -> pd.DataFrame:
    """
    Collect all XML files from a list of folders and build a DataFrame.
    {'20':[Path1, Path2], '50':[Path3, Path4], '200':[Path32,Path23,Path14,Path3, Path4]}
    Args:
        folder_list: List of folder paths to scan
    """
    all_records = []
    for particle_size_nm, folders in folder_list.items():
        for folder in folders:
            datasets = build_datasets(folder)
            for base_name, ds in datasets.items():
                rec_tif_info = ds['rec_tif']
                tracks = ds['tracks']
                rec_path = rec_tif_info['rec_file']
                meta_data = parse_rec_file(rec_path) if rec_path else {}
                mpp = meta_data.get('mpp')
                fps = meta_data.get('fps')
                particle_size = extract_particle_size_from_path(folder)
                all_records.append({
                    'particle_size_nm': particle_size_nm,
                    'xml_name': base_name,
                    'xml_path': ds['xml_paths'][0],
                    'exposure_ms': meta_data.get('exposure_ms'),
                    'delay_ms': meta_data.get('delay_ms'),
                    'fps': fps,
                    'mpp': mpp,
                    'mode': f"{meta_data.get('size_x')}x{meta_data.get('size_y')}" if meta_data.get('size_x') else None,
                    'tracks': tracks
                })
    
    combined_df = pd.DataFrame(all_records)
    return combined_df


def collect_all_files_by_particle_size(root_path: Path) -> pd.DataFrame:

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



