'''This Script provides experiment data extraction and parsing. It is solely for the SPT experiment conducted in the keylab microscopy.
Here wont be stored any further analysis methods or visualization methods.'''
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET

import pandas as pd

# -----------------------------
# REC parsing 
# -----------------------------


def find_rec_tif_files(xml_path: Path) -> Dict[str, Any]:
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
    
    xml_stem = xml_stem.replace('filtered_', '')
    xml_stem = xml_stem.replace('Resultof', '')
    xml_stem = xml_stem.replace('_var', '')
    

    rec_path = xml_dir / f"{xml_stem}.tif.rec"
    if not rec_path.exists():
        rec_path = xml_dir / f"{xml_stem}.rec"
    if not rec_path.exists():
        print(f"     ERROR: Missing calibration for {xml_path.name},rec {xml_stem}")
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



def parse_rec_file(rec_path: Path) -> Dict[str, Any]:
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
    if not rec_path.exists():
        return result
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

def check_text_encoding(path: Path) -> str:
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
    candidates = [folder_path.name]
    if folder_path.suffix:
        candidates.append(folder_path.parent.name)
        candidates.append(folder_path.parent.parent.name)
    else:
        candidates.append(folder_path.parent.name)

    for name in candidates:
        match = re.search(r'(\d+)\s*nm', name, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None

# -----------------------------
# Single Files
# -----------------------------

def _build_result_dict(
    xml_path: Path,
    tracks_df: Optional[pd.DataFrame],
    mpp: float,
    fps: float,
    particle_size_nm: Optional[float],
    tif_path: Optional[Path],
    rec_path: Optional[Path],
) -> Dict[str, Any]:
    num_tracks = tracks_df['particle'].nunique() if tracks_df is not None else 0
    num_frames = tracks_df['frame'].max() + 1 if tracks_df is not None and not tracks_df.empty else 0

    return {
        'xml_path': str(xml_path),
        'tif_path': str(tif_path) if tif_path else None,
        'rec_path': str(rec_path) if rec_path else None,
        'base_name': xml_path.name,
        'tracks_df': tracks_df,
        'mpp': mpp,
        'fps': fps,
        'particle_size_nm': particle_size_nm,
        'num_tracks': num_tracks,
        'num_frames': num_frames,
        'D_MSD': None,
        'D_step': None,
        'fit_results_MSD': None,
        'fit_results_step': None,
    }

def single_file_data(xml_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract data from a single XML file and its corresponding .rec file.
    
    Args:
        xml_path: Path to TrackMate XML file

    Returns:
        Standardized result_dict or None if calibration is missing
    """
    particle_size = extract_particle_size_from_path(xml_path)
    calib_info = find_rec_tif_files(xml_path)
    tracks_df = read_trackmate_xml(xml_path)
    fps = calib_info['fps']
    mpp = calib_info['mpp']
    
    # Validate required parameters
    if fps is None or mpp is None:
        print(f"     ERROR: Missing calibration for {xml_path.name} (fps={fps}, mpp={mpp})")
        print(f"     Skipping {xml_path.name}")
        return None
    
    return _build_result_dict(
        xml_path=xml_path,
        tracks_df=tracks_df,
        mpp=mpp,
        fps=fps,
        particle_size_nm=particle_size,
        tif_path=calib_info['tiff_file'],
        rec_path=calib_info['rec_file'],
    )

# -----------------------------
# Builder: scan folder and bundle
# -----------------------------

def scan_xml_folder(root: Path) -> List[Path]:
    """Recursively find all TrackMate XML files in the folder."""
    xml_files = list(root.rglob("*.xml"))
    return xml_files


def build_datasets(root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Scan the root folder for datasets and build standardized result_dict entries.
    
    Args:
        root: Root folder to scan
        """
    xml_files = scan_xml_folder(root)
    
    datasets = {}
    print(root)
    for xml_file in xml_files:
        print("_find",xml_file)
        particle_size = extract_particle_size_from_path(xml_file.parent.parent)
        result = single_file_data(xml_file)
        if result is None:
            continue
        if result['particle_size_nm'] is None:
            result['particle_size_nm'] = particle_size

        datasets[xml_file.stem] = result

    
    return datasets

def collect_files_from_list (folder_list: Dict[str, List[Path]]) -> Dict[str, Dict[str, Any]]:
    """
    Collect all XML files from a list of folders and build a results dict.
    {'20':[Path1, Path2], '50':[Path3, Path4], '200':[Path32,Path23,Path14,Path3, Path4]}
    Args:
        folder_list: List of folder paths to scan
    """
    results_dict: Dict[str, Dict[str, Any]] = {}
    for particle_size_nm, folders in folder_list.items():
        for folder in folders:
            datasets = build_datasets(folder)
            for _, result in datasets.items():
                if result is None:
                    continue
                if result['particle_size_nm'] is None:
                    result['particle_size_nm'] = particle_size_nm
                results_dict[result['xml_path']] = result

    return results_dict


def collect_all_files_by_particle_size(root_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Scan the directory structure and collect XML track files with matched .rec file parameters.
    Each XML file yields one standardized result_dict.
    
    Expected structure:
        root_path/
            50nm/
                file.rec (one or more)
                Tracks/
                    file1_Tracks.xml
                    file2_Tracks.xml
    
    Args:
        root_path: Root directory to scan
        
    Returns:
        Dictionary mapping xml_path -> standardized result_dict
    """
    file_records: List[Dict[str, Any]] = []
    
    print(f"\nScanning folders in {root_path}")
    print("=" * 70)
    
    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
        
        particle_size = extract_particle_size_from_path(subfolder)
        
        print(f"\nFolder: {subfolder.name}")
        if particle_size:
            print(f"   Particle size: {particle_size} nm")
        else:
            print(f"   Warning: No particle size detected - skipping")
            continue
        
        # Look for Tracks subfolder
        tracks_folder = subfolder / "Tracks"
        if not tracks_folder.exists():
            print(f"   Warning: No Tracks/ subfolder - skipping")
            continue
        
        xml_files = sorted(list(tracks_folder.glob("*.xml")))
        print(f"   Found {len(xml_files)} XML files in Tracks/")
        
        if not xml_files:
            continue
        
        # Process each XML file using find_calibration_files
        for xml_file in xml_files:
            result = single_file_data(xml_file)
            if result is None:
                continue
            result['particle_size_nm'] = particle_size
            file_records.append(result)

            print(f"     [OK] {xml_file.name} [mpp={result['mpp']} µm/px]")
            
    print(f"\n" + "=" * 70)
    print(f"[OK] Total XML files collected: {len(file_records)}")
    print("=" * 70 + "\n")
    
    results_dict: Dict[str, Dict[str, Any]] = {}
    for record in file_records:
        xml_str = record['xml_path']
        results_dict[xml_str] = record
    
    return results_dict



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

    data = single_file_data(Path(path))
    print(data["tracks_df"].head())

    print('Folder Analysis:')
    folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\500 nm\Tracks") # add a path to a folder
    datasets = build_datasets(folder)

    for base_name, ds in datasets.items():
        print(f"Dataset: {base_name}")
        print(f"  TIFF: {ds['tif_path']}")
        print(f"  REC file: {ds['rec_path']}")
        print(f"  mpp: {ds['mpp']} µm/px, fps: {ds['fps']} Hz")
        print(f"  tracks: {ds['tracks_df'].head() if ds['tracks_df'] is not None else 'None'}")

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



