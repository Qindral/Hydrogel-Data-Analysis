"""I/O functions for loading tracks from XML."""

from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re


@dataclass
class DatasetFiles:
    """Container for a dataset with base TIF, REC, processed TIFs, and XMLs."""
    base_tif: Path
    rec_path: Path
    processed_tifs: List[Path]
    xml_paths: List[Path]
    base_name: str
    
    @property
    def all_tifs(self) -> List[Path]:
        """All TIF files (base + processed)."""
        return [self.base_tif] + self.processed_tifs


class TrackLoader:
    """Unified track loading from TrackMate XML."""
    
    @staticmethod
    def from_trackmate_xml(
        xml_path: Path,
        mpp: Optional[float] = None,
        fps: Optional[float] = None,
        min_length: int = 10,
        rec_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Load tracks from TrackMate XML with automatic metadata extraction.
        
        Args:
            xml_path: Path to XML file
            mpp: Micrometers per pixel (if None, extracts from .rec or XML)
            fps: Frames per second (if None, extracts from .rec or XML)
            min_length: Minimum track length to include
            rec_path: Optional path to .rec file for metadata extraction
        
        Returns:
            DataFrame with columns: ['particle', 'frame', 'x', 'y']
            and attrs: {'mpp': float, 'fps': float, 'mode': str}
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Try to get metadata from .rec file first
        rec_metadata = {}
        if rec_path and rec_path.exists():
            rec_metadata = parse_rec_file(rec_path)
        
        # Extract metadata from XML if not provided
        x_max, y_max = TrackLoader._extract_image_dimensions(root)
        
        # Use .rec file dimensions if available
        if 'width' in rec_metadata and 'height' in rec_metadata:
            x_max = rec_metadata['width']
            y_max = rec_metadata['height']
        
        # FPS: prefer .rec, then manual, then XML
        if fps is None:
            if rec_metadata.get('fps'):
                fps = rec_metadata['fps']
            else:
                frame_interval = float(root.get('frameInterval', '1.0'))
                fps = 1.0 / frame_interval if frame_interval > 0 else 20.0
        
        # MPP: prefer manual, then dimension-based, then mode detection
        if mpp is None:
            if x_max and y_max:
                mpp = get_mpp_from_dimensions(x_max, y_max)
                mode = f'{x_max}x{y_max}'
            else:
                mpp, fps, mode = TrackLoader._detect_mode(fps, x_max, y_max)
        else:
            mode = 'Custom'
        
        # Parse tracks
        data = []
        for pid, particle in enumerate(root.findall('particle')):
            for detection in particle.findall('detection'):
                data.append({
                    'particle': pid,
                    'frame': int(float(detection.get('t'))),
                    'x': float(detection.get('x')),
                    'y': float(detection.get('y')),
                })
        
        if not data:
            df = pd.DataFrame(columns=['particle', 'frame', 'x', 'y'])
        else:
            df = pd.DataFrame(data)
            df = df.sort_values(['particle', 'frame']).reset_index(drop=True)
        
        # Filter by track length
        if not df.empty and min_length > 0:
            counts = df.groupby('particle').size()
            valid = counts[counts >= min_length].index
            df = df[df['particle'].isin(valid)].reset_index(drop=True)
            
            # Renumber particles consecutively
            if not df.empty:
                unique_ids = sorted(df['particle'].unique())
                id_map = {old: new for new, old in enumerate(unique_ids)}
                df['particle'] = df['particle'].map(id_map)
        
        df.attrs['mpp'] = mpp
        df.attrs['fps'] = fps
        df.attrs['mode'] = mode
        
        return df
    
    @staticmethod
    def _extract_image_dimensions(root: ET.Element) -> Tuple[Optional[int], Optional[int]]:
        """Extract max x, y coordinates from XML."""
        x_coords, y_coords = [], []
        
        for particle in root.findall('particle'):
            for detection in particle.findall('detection'):
                x_raw = detection.get('x')
                y_raw = detection.get('y')
                if x_raw and y_raw:
                    x_coords.append(float(x_raw))
                    y_coords.append(float(y_raw))
        
        if x_coords and y_coords:
            return int(np.ceil(max(x_coords))), int(np.ceil(max(y_coords)))
        return None, None
    
    @staticmethod
    def _detect_mode(fps: float, x_max: Optional[int], y_max: Optional[int]) -> Tuple[float, float, str]:
        """Detect acquisition mode and return (mpp, fps, mode_name)."""
        # First try image size
        if x_max is not None and y_max is not None:
            if x_max <= 250 and y_max <= 200:
                return 0.3, 60.0, '60 FPS'
            else:
                return 0.15, 20.0, '20 FPS'
        
        # Fallback to FPS
        if 50 <= fps <= 70:
            return 0.3, fps, '60 FPS'
        elif 15 <= fps <= 30:
            return 0.15, fps, '20 FPS'
        
        return 0.15, 20.0, 'Unknown'


def _extract_base_name(filename: str) -> str:
    """Extract base name by removing only specific prefixes/suffixes.
    
    Preserves: spaces, underscores, capitalization, numbers like _1, _01
    Removes: Resultof, processed_, preprocessed_ (prefix), _Tracks, _processed (suffix)
    """
    name = filename
    
    # Remove prefixes (case-insensitive)
    for prefix in ['Resultof', 'resultof', 'processed_', 'preprocessed_']:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix):]
            break
    
    # Remove suffixes (case-insensitive)
    for suffix in ['_Tracks', '_tracks', '_processed', '_preprocessed']:
        if name.lower().endswith(suffix.lower()):
            name = name[:-len(suffix)]
            break
    
    return name


def find_dataset_files(root_path: Path) -> List[DatasetFiles]:
    """Find and group TIF files with their REC and XML files.
    
    Handles:
    - Single file input: match that file specifically
    - Folder without subfolders: search files in that folder
    - Folder with subfolders: search recursively
    
    Matching rules:
    - TIF filename is the base name (preserved exactly)
    - REC: basename.tif.rec or basename.rec
    - XML: can have prefixes (Resultof) or suffixes (_Tracks, _processed)
    """
    # Case A: Single file input
    if root_path.is_file():
        if root_path.suffix == '.tif':
            return [_match_single_tif(root_path)]
        else:
            return []
    
    # Case B: Folder - check for subfolders
    subfolders = [d for d in root_path.iterdir() if d.is_dir()]
    
    if not subfolders:
        # No subfolders: search files in this folder only
        return _search_folder(root_path, search_subfolders=False)
    else:
        # Has subfolders: search recursively
        datasets = []
        datasets.extend(_search_folder(root_path, search_subfolders=False))  # Root files
        for subfolder in subfolders:
            datasets.extend(_search_folder(subfolder, search_subfolders=False))  # Subfolder files
        return datasets


def _search_folder(folder: Path, search_subfolders: bool) -> List[DatasetFiles]:
    """Search for TIF files in a folder and match with REC/XML."""
    tif_files = list(folder.glob("*.tif"))
    
    # Group by exact base name
    tif_groups = {}
    for tif in tif_files:
        base = _extract_base_name(tif.stem)
        if base not in tif_groups:
            tif_groups[base] = []
        tif_groups[base].append(tif)
    
    # Build datasets
    datasets = []
    for base_name, tifs in sorted(tif_groups.items()):
        # Sort: non-processed first
        tifs_sorted = sorted(tifs, key=lambda t: (
            'processed' in t.stem.lower(),
            'preprocessed' in t.stem.lower(),
            t.stem
        ))
        
        base_tif = tifs_sorted[0]
        processed_tifs = tifs_sorted[1:]
        
        # Find REC
        rec_path = None
        for rec_name in [f"{base_tif.name}.rec", f"{base_tif.stem}.rec"]:
            rec_candidate = base_tif.parent / rec_name
            if rec_candidate.exists():
                rec_path = rec_candidate
                break
        
        if not rec_path:
            continue  # Skip if no REC
        
        # Find XMLs - search in same folder and Tracks subfolder
        xml_paths = _find_xmls_for_base(base_name, folder)
        
        datasets.append(DatasetFiles(
            base_tif=base_tif,
            rec_path=rec_path,
            processed_tifs=processed_tifs,
            xml_paths=xml_paths,
            base_name=base_name
        ))
    
    return datasets


def _match_single_tif(tif_path: Path) -> DatasetFiles:
    """Match a single TIF file with its REC and XMLs."""
    folder = tif_path.parent
    base_name = _extract_base_name(tif_path.stem)
    
    # Find REC
    rec_path = None
    for rec_name in [f"{tif_path.name}.rec", f"{tif_path.stem}.rec"]:
        rec_candidate = folder / rec_name
        if rec_candidate.exists():
            rec_path = rec_candidate
            break
    
    if not rec_path:
        raise FileNotFoundError(f"No REC file found for {tif_path.name}")
    
    # Find processed versions
    processed_tifs = []
    for pattern in [f"processed_{tif_path.name}", f"preprocessed_{tif_path.name}"]:
        candidate = folder / pattern
        if candidate.exists():
            processed_tifs.append(candidate)
    
    # Find XMLs
    xml_paths = _find_xmls_for_base(base_name, folder)
    
    return DatasetFiles(
        base_tif=tif_path,
        rec_path=rec_path,
        processed_tifs=processed_tifs,
        xml_paths=xml_paths,
        base_name=base_name
    )


def _find_xmls_for_base(base_name: str, folder: Path) -> List[Path]:
    """Find XMLs matching the base name (exact match, case-insensitive)."""
    xml_folders = [folder]
    for sub in ["Tracks", "Track"]:
        sub_path = folder / sub
        if sub_path.exists():
            xml_folders.append(sub_path)
    
    matched = []
    for xml_folder in xml_folders:
        for xml in xml_folder.glob("*.xml"):
            xml_base = _extract_base_name(xml.stem)
            # Exact match, case-insensitive
            if xml_base.lower() == base_name.lower():
                matched.append(xml)
    
    return sorted(set(matched))


def find_xml_files(root_path: Path, pattern: str = "**/*Tracks.xml") -> List[Path]:
    """Legacy function - use find_dataset_files() instead."""
    datasets = find_dataset_files(root_path)
    return [ds.xml_path for ds in datasets]


def _normalize_name(name: str) -> str:
    """Normalize filename for fuzzy matching (lowercase, no spaces/underscores)."""
    return name.lower().replace(' ', '').replace('_', '').replace('-', '')


def parse_rec_file(rec_path: Path) -> dict:
    """Parse PCO .rec file for camera metadata.
    
    Returns dict with 'fps', 'width', 'height', 'exposure_ms'
    """
    metadata = {}
    
    try:
        with open(rec_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract picture size: "Picture Size horz./vert.: 200/150"
        size_match = re.search(r'Picture Size horz\./vert\.\s*:\s*(\d+)/(\d+)', content)
        if size_match:
            metadata['width'] = int(size_match.group(1))
            metadata['height'] = int(size_match.group(2))
        
        # Extract exposure time: "Exposure / Delay : 15.000000 ms"
        exposure_match = re.search(r'Exposure / Delay\s*:\s*([\d.]+)\s*ms', content)
        if exposure_match:
            exposure_ms = float(exposure_match.group(1))
            metadata['exposure_ms'] = exposure_ms
            # FPS = 1000 / exposure_ms (assuming no delay between frames)
            metadata['fps'] = 1000.0 / exposure_ms if exposure_ms > 0 else None
    
    except Exception:
        pass
    
    return metadata


def get_mpp_from_dimensions(width: int, height: int) -> float:
    """Get micrometers per pixel based on image dimensions.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        mpp in µm/px
    """
    # Known dimension → mpp mappings
    dimension_map = {
        (200, 150): 0.3,
        (400, 300): 0.15,
        (696, 520): 0.149,
    }
    
    # Check exact match
    if (width, height) in dimension_map:
        return dimension_map[(width, height)]
    
    # Fallback: estimate based on known mappings
    # Assuming inverse relationship: larger image = smaller mpp
    if width <= 200:
        return 0.3
    elif width <= 400:
        return 0.15
    else:
        return 0.149


def extract_particle_size_from_path(path: Path) -> Optional[float]:
    """Extract particle size (nm) from folder or file name."""
    match = re.search(r'(\d+)\s*nm', path.name, re.IGNORECASE)
    return float(match.group(1)) if match else None
