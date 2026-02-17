# spt_folder_scan_and_compare_tracks.py
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import trackpy as tp

# Import analysis functions from core module
from hydro_analysis.core.io import read_trackmate_xml as core_read_trackmate_xml
from hydro_analysis.core.analysis import (
    fit_powerlaw_with_errors,
    calculate_step_sizes,
    fit_gaussian_diffusion_1d,
    diffusion_2d_from_1d_fits,
)


# =========================
# Datamodel
# =========================

@dataclass
class RecMeta:
    exposure_ms: Optional[float] = None
    delay_ms: Optional[float] = None
    fps_nominal: Optional[float] = None
    size_px: Optional[Dict[str, int]] = None
    roi_size_px: Optional[Dict[str, int]] = None
    binning: Optional[Dict[str, int]] = None
    camera_type: Optional[str] = None
    camera_serial: Optional[str] = None
    pixelrate_mhz: Optional[float] = None
    comment_raw: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrackSummary:
    path: Path
    n_tracks: int
    n_edges: int
    n_spots_est: int
    track_lengths: List[int]  # edges+1 pro Track (wenn möglich)
    notes: List[str] = field(default_factory=list)

    @property
    def mean_track_length(self) -> Optional[float]:
        return (sum(self.track_lengths) / len(self.track_lengths)) if self.track_lengths else None

    @property
    def median_track_length(self) -> Optional[float]:
        if not self.track_lengths:
            return None
        s = sorted(self.track_lengths)
        mid = len(s) // 2
        if len(s) % 2 == 1:
            return float(s[mid])
        return 0.5 * (s[mid - 1] + s[mid])

@dataclass
class Dataset:
    series_base: str          # z.B. "20 nm_water"
    series_index: Optional[int]  # None für "base", sonst 2,3,4...
    tif_path: Optional[Path] = None
    rec_path: Optional[Path] = None
    track_xmls: List[Path] = field(default_factory=list)
    other_xmls: List[Path] = field(default_factory=list)
    rec_meta: Optional[RecMeta] = None

    @property
    def key(self) -> str:
        idx = "base" if self.series_index is None else f"{self.series_index:02d}"
        return f"{self.series_base}__{idx}"


# =========================
# REC parser (PCO CamWare Comment File)
# =========================

def _to_float(s: str) -> float:
    return float(s.strip().replace(",", "."))

def parse_pco_camware_rec_text(text: str) -> RecMeta:
    out: Dict[str, Any] = {}
    def grab(pattern: str, flags=0) -> Optional[re.Match]:
        return re.search(pattern, text, flags)

    m = grab(r"Camera Type\s*:\s*(.+)")
    if m:
        out["camera_type"] = m.group(1).strip()

    m = grab(r"Picture Size\s*horz\./vert\.\s*:\s*(\d+)\s*/\s*(\d+)")
    if m:
        out["size_px"] = {"x": int(m.group(1)), "y": int(m.group(2))}

    m = grab(r"ROI\s*horz\./vert\.\s*:\s*(\d+)\s*-\s*(\d+)\s*/\s*(\d+)\s*-\s*(\d+)")
    if m:
        roi_px = {"x0": int(m.group(1)), "x1": int(m.group(2)),
                  "y0": int(m.group(3)), "y1": int(m.group(4))}
        out["roi_px"] = roi_px
        out["roi_size_px"] = {"x": roi_px["x1"] - roi_px["x0"] + 1,
                              "y": roi_px["y1"] - roi_px["y0"] + 1}

    m = grab(r"Binning\s*horz\./vert\.\s*:\s*x(\d+)\s*/\s*x(\d+)", flags=re.IGNORECASE)
    if m:
        out["binning"] = {"x": int(m.group(1)), "y": int(m.group(2))}

    m = grab(r"Exposure\s*/\s*Delay\s*:\s*([0-9\.,]+)\s*ms\s*/\s*([0-9\.,]+)\s*ms", flags=re.IGNORECASE)
    if m:
        exposure_ms = _to_float(m.group(1))
        delay_ms = _to_float(m.group(2))
        out["exposure_ms"] = exposure_ms
        out["delay_ms"] = delay_ms
        period = exposure_ms + delay_ms
        out["fps_nominal"] = (1000.0 / period) if period > 0 else None

    m = grab(r"Pixelrate\s*:\s*([0-9\.,]+)\s*MHz", flags=re.IGNORECASE)
    if m:
        out["pixelrate_mhz"] = _to_float(m.group(1))

    m = grab(r"Camera serial number\s*:\s*(\S+)")
    if m:
        out["camera_serial"] = m.group(1).strip()

    m = grab(r"Comment:\s*(.*)\Z", flags=re.DOTALL)
    if m:
        out["comment_raw"] = m.group(1).strip()

    meta = RecMeta(
        exposure_ms=out.get("exposure_ms"),
        delay_ms=out.get("delay_ms"),
        fps_nominal=out.get("fps_nominal"),
        size_px=out.get("size_px"),
        roi_size_px=out.get("roi_size_px"),
        binning=out.get("binning"),
        camera_type=out.get("camera_type"),
        camera_serial=out.get("camera_serial"),
        pixelrate_mhz=out.get("pixelrate_mhz"),
        comment_raw=out.get("comment_raw"),
        raw=out
    )
    return meta


# =========================
# Name normalization & series parsing
# =========================

def normalize_base_name(name: str) -> str:
    """
    Normalisiert grob:
    - mehrere Whitespaces -> 1 underscore
    - underscores/spaces mix -> underscore
    - entfernt doppelte underscores
    - tolerate missing spaces/underscores: "20nmwater" -> "20_nm_water"
    """
    s = name.strip()
    
    # Handle common patterns like "20nmwater" -> "20 nm water"
    # Insert space before "nm" if preceded by digit
    s = re.sub(r"(\d+)(nm)", r"\1 \2 ", s, flags=re.IGNORECASE)
    
    # Replace multiple whitespaces with single underscore
    s = re.sub(r"[ \t]+", "_", s)
    
    # Remove double underscores
    s = re.sub(r"_+", "_", s)
    
    return s

SERIES_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?P<idx>\d{2})$", re.IGNORECASE)
SINGLE_DIGIT_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_(?P<idx>\d)$", re.IGNORECASE)  # für _4 Fälle

def split_series(stem: str) -> Tuple[str, Optional[int]]:
    """
    Erkennt _02/_03/_04 strikt; zusätzlich _4 (weil dein Beispiel das hat).
    Rückgabe: (series_base_normalized, index or None)
    """
    s = normalize_base_name(stem)

    m = SERIES_SUFFIX_RE.match(s)
    if m:
        return normalize_base_name(m.group("base")), int(m.group("idx"))

    # Fallback für _4 (damit dein Beispielordner funktioniert)
    m = SINGLE_DIGIT_SUFFIX_RE.match(s)
    if m:
        return normalize_base_name(m.group("base")), int(m.group("idx"))

    return s, None


def get_mpp_from_size(width: int, height: int) -> Optional[float]:
    """
    Determine micrometers per pixel based on image dimensions.
    Based on known camera configurations.
    """
    # Known dimension → mpp mappings
    dimension_map = {
        (200, 150): 0.3,
        (400, 300): 0.15,
        (696, 520): 0.149,
    }
    
    if (width, height) in dimension_map:
        return dimension_map[(width, height)]
    
    # Fallback: assume smaller images have higher magnification
    if width <= 250 and height <= 200:
        return 0.3
    elif width <= 450 and height <= 350:
        return 0.15
    else:
        return 0.149


def clean_xml_name_for_parsing(xml_stem: str) -> str:
    """
    Remove common prefixes and suffixes from XML name before parsing series.
    Examples:
    - '20 nm water_4_Tracks' → '20 nm water_4'
    - 'filtered_20 nm water_4_Tracks' → '20 nm water_4'
    - 'Resultof20nmwater_4_Tracks' → '20nmwater_4'
    """
    s = xml_stem
    
    # Remove common prefixes (case-insensitive)
    for prefix in ['filtered_', 'resultof', 'result', 'processed_']:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):]
            break
    
    # Remove common suffixes (case-insensitive)
    for suffix in ['_Tracks', '_tracks', '_Track', '_track']:
        if s.lower().endswith(suffix.lower()):
            s = s[:-len(suffix)]
            break
    
    return s


def is_track_xml_name(xml_name: str) -> bool:
    """
    Entscheidet ob XML ein Track-XML ist:
    - enthält "tracks" (typisch *_Tracks.xml)
    - oder beginnt mit "filtered_" / "resultof" + enthält tracks
    """
    n = xml_name.lower()
    return ("tracks" in n) or n.startswith("filtered_") or n.startswith("resultof")


# =========================
# TrackMate XML summary
# =========================

def _tag_endswith(el: ET.Element, suffix: str) -> bool:
    # XML namespaces robust: {ns}Track -> endswith("Track")
    return el.tag.lower().endswith(suffix.lower())

def summarize_track_xml(xml_path: Path) -> TrackSummary:
    notes: List[str] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        return TrackSummary(
            path=xml_path, n_tracks=0, n_edges=0, n_spots_est=0,
            track_lengths=[], notes=[f"XML parse error: {e}"]
        )

    tracks = [el for el in root.iter() if _tag_endswith(el, "Track")]
    edges = [el for el in root.iter() if _tag_endswith(el, "Edge")]
    spots = [el for el in root.iter() if _tag_endswith(el, "Spot")]

    n_tracks = len(tracks)
    n_edges = len(edges)
    n_spots = len(spots)

    # Track lengths: count edges within each Track if possible
    track_lengths: List[int] = []
    for tr in tracks:
        # Edges that are descendants of this Track
        tr_edges = [e for e in tr.iter() if _tag_endswith(e, "Edge")]
        if tr_edges:
            track_lengths.append(len(tr_edges) + 1)

    # If no <Spot> elements exist (some exports), estimate spots from edge ids
    if n_spots == 0 and n_edges > 0:
        # Common TrackMate attributes: SPOT_SOURCE_ID / SPOT_TARGET_ID
        ids = set()
        for e in edges:
            for k in ("SPOT_SOURCE_ID", "SPOT_TARGET_ID", "source", "target"):
                if k in e.attrib:
                    ids.add(e.attrib[k])
        if ids:
            n_spots = len(ids)
            notes.append("Spots estimated from Edge source/target IDs (no <Spot> nodes).")

    if n_tracks == 0 and n_edges > 0:
        notes.append("Found edges but no <Track> nodes; XML may be non-standard TrackMate export.")
    if n_tracks > 0 and not track_lengths:
        notes.append("Tracks found but could not derive per-track lengths (no nested Edge nodes).")

    return TrackSummary(
        path=xml_path,
        n_tracks=n_tracks,
        n_edges=n_edges,
        n_spots_est=n_spots,
        track_lengths=track_lengths,
        notes=notes
    )


# =========================
# Folder scanning & bundling
# =========================

def detect_rec_for_tif(tif_path: Path) -> Optional[Path]:
    """
    In deinem Beispiel gibt es:
      20 nm_water_02.tif.rec  (Suffix .tif.rec)
    Also suchen wir:
      - <tif>.rec
      - <stem>.rec
    """
    p1 = tif_path.with_suffix(tif_path.suffix + ".rec")  # .tif.rec
    if p1.exists():
        return p1
    p2 = tif_path.with_suffix(".rec")  # .rec
    if p2.exists():
        return p2
    return None


def scan_folder(root: Path) -> Dict[str, Dataset]:
    """
    Start from .rec files, then map to TIFs and XMLs.
    
    Strategy:
    1. Find all .rec files (anchor files with metadata)
    2. For each .rec, find matching TIF (e.g., 20 nm_water_02.tif.rec → 20 nm_water_02.tif)
    3. For each TIF/REC pair, find matching XML files
    4. Group by series_base and index
    """
    root = Path(root).resolve()
    datasets: Dict[str, Dataset] = {}

    # Step 1: Find all .rec files and create datasets
    rec_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".rec"]
    
    for rec_path in rec_files:
        # Derive TIF name from REC name
        # Example: "20 nm_water_02.tif.rec" → "20 nm_water_02.tif"
        stem = rec_path.stem
        
        # Check if stem ends with .tif or .tiff
        tif_path = None
        if stem.lower().endswith('.tif'):
            tif_path = root / stem
        elif stem.lower().endswith('.tiff'):
            tif_path = root / stem
        else:
            # Try adding .tif extension
            tif_path = root / (stem + ".tif")
        
        # Verify TIF exists
        if not tif_path.exists():
            # Try .tiff as fallback
            tif_path = root / (stem + ".tiff")
            if not tif_path.exists():
                print(f"Warning: No TIF found for REC {rec_path.name}, skipping")
                continue
        
        # Extract series base and index from TIF stem
        series_base, idx = split_series(tif_path.stem)
        key = f"{series_base}__{'base' if idx is None else f'{idx:02d}'}"
        
        # Create or get dataset
        ds = Dataset(series_base=series_base, series_index=idx)
        ds.tif_path = tif_path
        ds.rec_path = rec_path
        
        # Parse REC metadata
        try:
            # Try UTF-16 first (PCO CamWare often uses this), then UTF-8
            try:
                rec_text = rec_path.read_text(encoding="utf-16", errors="replace")
            except:
                rec_text = rec_path.read_text(encoding="utf-8", errors="replace")
            
            ds.rec_meta = parse_pco_camware_rec_text(rec_text)
        except Exception as e:
            print(f"Warning: Failed to parse REC {rec_path.name}: {e}")
        
        datasets[key] = ds
    
    # Step 2: Find and attach XML files to matching datasets
    xml_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".xml"]
    
    for xml_path in xml_files:
        # Clean XML name before parsing (remove _Tracks, filtered_, etc.)
        xml_stem_cleaned = clean_xml_name_for_parsing(xml_path.stem)
        xml_series_base, xml_idx = split_series(xml_stem_cleaned)
        
        # Try to find exact match by series_base AND index
        matched = False
        for key, ds in datasets.items():
            # Normalize both series_base for comparison
            ds_base_norm = normalize_base_name(ds.series_base)
            xml_base_norm = normalize_base_name(xml_series_base)
            
            # Series base must match exactly (after normalization)
            series_match = ds_base_norm == xml_base_norm
            
            # Index must match exactly (both None or both same number)
            index_match = ds.series_index == xml_idx
            
            if series_match and index_match:
                ds.track_xmls.append(xml_path)
                matched = True
                break
        
        # If no exact match, add to other_xmls
        if not matched:
            # Find closest dataset by series_base only (for reporting)
            for key, ds in datasets.items():
                ds_base_norm = normalize_base_name(ds.series_base)
                xml_base_norm = normalize_base_name(xml_series_base)
                if ds_base_norm in xml_base_norm or xml_base_norm in ds_base_norm:
                    if xml_path not in ds.other_xmls:
                        ds.other_xmls.append(xml_path)
    
    return datasets


def print_dataset_summary(datasets):
    for key, ds in datasets.items():
        print("=" * 60)
        print(f"Dataset: {key}")

        # --- Core files
        print(f"  TIF : {ds.tif_path.name if ds.tif_path else 'None'}")
        print(f"  REC : {ds.rec_path.name if ds.rec_path else 'None'}")

        # --- REC metadata (if available)
        if ds.rec_meta:
            print("  REC metadata:")
            
            # Image size and MPP
            if ds.rec_meta.size_px:
                width = ds.rec_meta.size_px['x']
                height = ds.rec_meta.size_px['y']
                mpp = get_mpp_from_size(width, height)
                print(f"    Image size: {width} x {height} px")
                print(f"    MPP: {mpp:.3f} µm/px")
            
            # Timing info
            if ds.rec_meta.exposure_ms is not None:
                print(f"    Exposure: {ds.rec_meta.exposure_ms:.2f} ms")
            if ds.rec_meta.delay_ms is not None:
                print(f"    Delay: {ds.rec_meta.delay_ms:.2f} ms")
            if ds.rec_meta.fps_nominal is not None:
                print(f"    FPS: {ds.rec_meta.fps_nominal:.2f}")
            
            # ROI info (if different from full size)
            if ds.rec_meta.roi_size_px:
                rx = ds.rec_meta.roi_size_px
                print(f"    ROI size : {rx['x']} x {rx['y']} px")
        else:
            print("  REC metadata: None")

        # --- Track XMLs
        print(f"  Track XMLs ({len(ds.track_xmls)}):")
        for p in ds.track_xmls:
            print(f"    - {p.name}")

        # --- Other XMLs
        if ds.other_xmls:
            print(f"  Other XMLs ({len(ds.other_xmls)}):")
            for p in ds.other_xmls:
                print(f"    - {p.name}")

        # --- Comparison hint
        if ds.series_index == 4 and len(ds.track_xmls) >= 2:
            print("  -> Candidate for TRACK COMPARISON (_04, multiple XMLs)")

        print()


# =========================
# Track loading and analysis
# =========================

def load_tracks_from_xml(xml_path: Path, mpp: float, fps: float) -> pd.DataFrame:
    """
    Load tracks from TrackMate XML and return as trackpy-compatible DataFrame.
    Uses core.analysis.read_trackmate_xml
    """
    df = core_read_trackmate_xml(xml_path)
    if df is None or df.empty:
        return pd.DataFrame(columns=['particle', 'frame', 'x', 'y'])
    
    df.attrs['mpp'] = mpp
    df.attrs['fps'] = fps
    return df


def compute_msd_analysis(tracks: pd.DataFrame, max_lagtime: Optional[int] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Compute MSD and extract diffusion coefficient.
    Uses core.analysis.fit_powerlaw_with_errors from MSD_FromTrackmate_D0.py
    
    Returns:
        msd: DataFrame with MSD values
        results: Dict with D, D_error, n, n_error
    """
    if len(tracks) == 0:
        return pd.DataFrame(), {}
    
    msd = tp.imsd(tracks, mpp=tracks.attrs['mpp'], fps=tracks.attrs['fps'], max_lagtime=max_lagtime)
    em = msd.mean(axis=1)
    
    fit_points = min(6, len(em))
    if fit_points < 3:
        return msd, {}
    
    # Use core function for power-law fitting
    params = fit_powerlaw_with_errors(em, points=fit_points, plot=False)
    
    # D = A / 4 for 2D
    D = params.A[0] / 4.0
    D_err = params.A_err[0] / 4.0
    
    return msd, {
        'D_um2_per_s': D,
        'D_error': D_err,
        'exponent': params.n[0],
        'exponent_error': params.n_err[0],
        'n_particles': tracks['particle'].nunique(),
        'n_detections': len(tracks)
    }


def compute_step_size_analysis(tracks: pd.DataFrame) -> dict:
    """
    Compute diffusion from displacement distributions.
    Uses fit_gaussian_diffusion_1d + diffusion_2d_from_1d_fits from core.analysis.

    D = sigma^2 / (2 * dt)
    """
    if len(tracks) == 0:
        return {}

    mpp = tracks.attrs['mpp']
    fps = tracks.attrs['fps']

    step_df = calculate_step_sizes(tracks, step_interval=1)

    if step_df.empty:
        return {}

    # Fit Gaussian to dx and dy separately, then combine for 2D diffusion
    fit_x = fit_gaussian_diffusion_1d(step_df["dx"], mpp, fps, frame_interval=1, axis="x")
    fit_y = fit_gaussian_diffusion_1d(step_df["dy"], mpp, fps, frame_interval=1, axis="y")
    results = diffusion_2d_from_1d_fits(fit_x, fit_y)

    return {
        'D_um2_per_s': results['D_um2_per_s'],
        'D_error': results['D_dir_disagreement_um2_per_s'],
        'sigma_x': results['sigma_x_nm'],
        'sigma_y': results['sigma_y_nm'],
        'mean_x': results['mu_x_nm'],
        'mean_y': results['mu_y_nm'],
        'n_steps': results['n_steps_x'],
        'quality_flag': results['quality_flag'],
        'is_isotropic': results['is_isotropic'],
        'is_centered': results['is_centered'],
    }


def analyze_track_statistics(tracks: pd.DataFrame) -> dict:
    """
    Compute basic track statistics.
    """
    if len(tracks) == 0:
        return {'n_particles': 0, 'n_detections': 0, 'track_lengths': []}
    
    track_lengths = tracks.groupby('particle').size().values
    
    return {
        'n_particles': tracks['particle'].nunique(),
        'n_detections': len(tracks),
        'track_lengths': track_lengths.tolist(),
        'mean_length': float(np.mean(track_lengths)),
        'median_length': float(np.median(track_lengths)),
        'min_length': int(np.min(track_lengths)),
        'max_length': int(np.max(track_lengths))
    }


def compare_track_xmls(dataset: Dataset, output_folder: Path) -> Dict[str, Any]:
    """
    Compare multiple track XMLs for a dataset (_04 files).
    
    Performs:
    1. MSD analysis for each XML
    2. Step size analysis for each XML
    3. Track statistics comparison
    4. Creates comparison plots
    
    Returns:
        Dictionary with all results
    """
    if not dataset.track_xmls:
        print(f"No track XMLs for {dataset.key}")
        return {}
    
    if len(dataset.track_xmls) < 2:
        print(f"Only one track XML for {dataset.key}, skipping comparison")
        return {}
    
    # Get MPP and FPS from metadata
    if not dataset.rec_meta or not dataset.rec_meta.size_px:
        print(f"No REC metadata for {dataset.key}, cannot proceed")
        return {}
    
    width = dataset.rec_meta.size_px['x']
    height = dataset.rec_meta.size_px['y']
    mpp = get_mpp_from_size(width, height)
    fps = dataset.rec_meta.fps_nominal or 20.0
    
    print(f"\n{'='*70}")
    print(f"Comparing {len(dataset.track_xmls)} track XMLs for {dataset.key}")
    print(f"MPP: {mpp:.3f} µm/px, FPS: {fps:.2f}")
    print(f"{'='*70}\n")
    
    results = {
        'dataset_key': dataset.key,
        'mpp': mpp,
        'fps': fps,
        'xml_results': {}
    }
    
    # Analyze each XML
    for xml_path in dataset.track_xmls:
        xml_name = xml_path.stem
        print(f"Analyzing {xml_name}...")
        
        # Load tracks
        tracks = load_tracks_from_xml(xml_path, mpp, fps)
        
        # MSD analysis
        msd, msd_results = compute_msd_analysis(tracks, max_lagtime=100)
        
        # Step size analysis
        step_results = compute_step_size_analysis(tracks)
        
        # Track statistics
        stats = analyze_track_statistics(tracks)
        
        results['xml_results'][xml_name] = {
            'msd': msd_results,
            'stepsize': step_results,
            'statistics': stats,
            'msd_dataframe': msd
        }
        
        print(f"  Particles: {stats['n_particles']}, Detections: {stats['n_detections']}")
        print(f"  MSD D: {msd_results.get('D_um2_per_s', np.nan):.4f} ± {msd_results.get('D_error', np.nan):.4f} µm²/s")
        print(f"  Step D: {step_results.get('D_um2_per_s', np.nan):.4f} ± {step_results.get('D_error', np.nan):.4f} µm²/s")
        if 'quality_flag' in step_results:
            print(f"  Step quality: {step_results['quality_flag']}, isotropic={step_results.get('is_isotropic', 'N/A')}, centered={step_results.get('is_centered', 'N/A')}\n")
    
    # Create and show comparison plots
    fig = create_comparison_plots(results, dataset.key)
    plt.show()
    
    return results


def create_comparison_plots(results: dict, dataset_key: str) -> Figure:
    """
    Create comprehensive comparison plots.
    
    Layout:
    - Row 1: MSD curves, Diffusion coefficients (MSD vs Step)
    - Row 2: Track length distributions, Particle counts
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Track Comparison: {dataset_key}", fontsize=16, fontweight='bold')
    
    xml_names = list(results['xml_results'].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(xml_names)))
    
    # Plot 1: MSD curves
    ax = axes[0, 0]
    for i, (xml_name, data) in enumerate(results['xml_results'].items()):
        msd_df = data.get('msd_dataframe')
        if msd_df is not None and not msd_df.empty:
            em = msd_df.mean(axis=1)
            ax.loglog(em.index, em.values, 'o-', color=colors[i], label=xml_name, alpha=0.7)
    
    ax.set_xlabel('Lag time (s)')
    ax.set_ylabel('MSD (µm²)')
    ax.set_title('Ensemble MSD')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Diffusion coefficients comparison
    ax = axes[0, 1]
    x_pos = np.arange(len(xml_names))
    
    msd_D = [results['xml_results'][name]['msd'].get('D_um2_per_s', np.nan) for name in xml_names]
    msd_D_err = [results['xml_results'][name]['msd'].get('D_error', np.nan) for name in xml_names]
    step_D = [results['xml_results'][name]['stepsize'].get('D_um2_per_s', np.nan) for name in xml_names]
    step_D_err = [results['xml_results'][name]['stepsize'].get('D_error', np.nan) for name in xml_names]
    
    width = 0.35
    ax.bar(x_pos - width/2, msd_D, width, yerr=msd_D_err, label='MSD', alpha=0.7, capsize=5)
    ax.bar(x_pos + width/2, step_D, width, yerr=step_D_err, label='Step Size', alpha=0.7, capsize=5)
    
    ax.set_ylabel('D (µm²/s)')
    ax.set_title('Diffusion Coefficient')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name[:15] for name in xml_names], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: MSD Exponent
    ax = axes[0, 2]
    exponents = [results['xml_results'][name]['msd'].get('exponent', np.nan) for name in xml_names]
    exp_errors = [results['xml_results'][name]['msd'].get('exponent_error', np.nan) for name in xml_names]
    
    ax.bar(x_pos, exponents, yerr=exp_errors, alpha=0.7, capsize=5, color=colors)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Normal diffusion (n=1)')
    ax.set_ylabel('MSD Exponent n')
    ax.set_title('Diffusion Mode')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name[:15] for name in xml_names], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Track length distributions
    ax = axes[1, 0]
    for i, (xml_name, data) in enumerate(results['xml_results'].items()):
        lengths = data['statistics'].get('track_lengths', [])
        if lengths:
            ax.hist(lengths, bins=20, alpha=0.5, label=xml_name, color=colors[i])
    
    ax.set_xlabel('Track Length (frames)')
    ax.set_ylabel('Count')
    ax.set_title('Track Length Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Particle counts
    ax = axes[1, 1]
    n_particles = [results['xml_results'][name]['statistics']['n_particles'] for name in xml_names]
    n_detections = [results['xml_results'][name]['statistics']['n_detections'] for name in xml_names]
    
    ax.bar(x_pos - width/2, n_particles, width, label='Particles', alpha=0.7, color='skyblue')
    ax.bar(x_pos + width/2, n_detections, width, label='Detections', alpha=0.7, color='salmon')
    
    ax.set_ylabel('Count')
    ax.set_title('Particle & Detection Counts')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name[:15] for name in xml_names], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Mean track lengths
    ax = axes[1, 2]
    mean_lengths = [results['xml_results'][name]['statistics']['mean_length'] for name in xml_names]
    median_lengths = [results['xml_results'][name]['statistics']['median_length'] for name in xml_names]
    
    ax.bar(x_pos - width/2, mean_lengths, width, label='Mean', alpha=0.7, color='green')
    ax.bar(x_pos + width/2, median_lengths, width, label='Median', alpha=0.7, color='orange')
    
    ax.set_ylabel('Track Length (frames)')
    ax.set_title('Average Track Lengths')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([name[:15] for name in xml_names], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return fig


def analyze_all_tracks(datasets: Dict[str, Dataset]) -> pd.DataFrame:
    """
    Analyze all track XMLs across all datasets.
    
    Returns:
        DataFrame with columns: dataset_key, xml_name, n_particles, D_msd, D_msd_err, D_step, D_step_err, quality
    """
    results = []
    
    for key, ds in datasets.items():
        if not ds.track_xmls:
            continue
        
        # Get calibration
        if not ds.rec_meta or not ds.rec_meta.size_px:
            print(f"Skipping {key}: No REC metadata")
            continue
        
        width = ds.rec_meta.size_px['x']
        height = ds.rec_meta.size_px['y']
        mpp = get_mpp_from_size(width, height)
        fps = ds.rec_meta.fps_nominal or 20.0
        
        print(f"\nAnalyzing {key}...")
        
        # Collect tracks from all XMLs first to determine which to use for step size
        xml_track_counts = {}
        
        for xml_path in ds.track_xmls:
            xml_name = xml_path.name
            print(f"  {xml_name}...", end=" ")
            
            try:
                # Load tracks
                tracks = load_tracks_from_xml(xml_path, mpp, fps)
                
                if len(tracks) == 0:
                    print("no tracks")
                    xml_track_counts[xml_path] = 0
                    continue
                
                # Store track count
                n_particles = tracks['particle'].nunique()
                xml_track_counts[xml_path] = n_particles
                
                # MSD analysis (always for all XMLs)
                _, msd_results = compute_msd_analysis(tracks, max_lagtime=100)
                
                # For step size: use only unfiltered XMLs (no "filtered_" prefix)
                is_unfiltered = not xml_name.lower().startswith('filtered_')
                
                if is_unfiltered:
                    step_results = compute_step_size_analysis(tracks)
                    step_d = step_results.get('D_um2_per_s', np.nan)
                    step_d_err = step_results.get('D_error', np.nan)
                    quality = step_results.get('quality_flag', 'unknown')
                else:
                    # For filtered XMLs, use NaN for step size
                    step_d = np.nan
                    step_d_err = np.nan
                    quality = 'n/a'
                
                # Store results
                results.append({
                    'dataset_key': key,
                    'xml_name': xml_name,
                    'n_particles': msd_results.get('n_particles', 0),
                    'n_detections': msd_results.get('n_detections', 0),
                    'D_msd': msd_results.get('D_um2_per_s', np.nan),
                    'D_msd_err': msd_results.get('D_error', np.nan),
                    'D_step': step_d,
                    'D_step_err': step_d_err,
                    'quality': quality,
                    'mpp': mpp,
                    'fps': fps
                })
                
                print(f"OK {msd_results.get('n_particles', 0)} particles")
                
            except Exception as e:
                print(f"failed: {e}")
    
    return pd.DataFrame(results)


def plot_diffusion_comparison(df: pd.DataFrame):
    """
    Create comparison plot of diffusion coefficients.
    """
    if df.empty:
        print("No results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Diffusion Coefficient Comparison - All Tracks', fontsize=16, fontweight='bold')
    
    # Sort by dataset key for consistent ordering
    df = df.sort_values(['dataset_key', 'xml_name']).reset_index(drop=True)
    
    # Create x-axis labels
    x_labels = [f"{row['dataset_key']}\n{row['xml_name']}" for _, row in df.iterrows()]
    x_pos = np.arange(len(df))
    
    # Plot 1: MSD method
    ax1.errorbar(x_pos, df['D_msd'], yerr=df['D_msd_err'], 
                 fmt='o', markersize=8, capsize=5, capthick=2, 
                 label='MSD Method', color='steelblue', linewidth=2)
    ax1.set_xlabel('Dataset / Track XML', fontsize=12)
    ax1.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax1.set_title('MSD Method', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Step size method
    colors = ['green' if q == 'good' else 'orange' if q == 'anisotropic' else 'red' 
              for q in df['quality']]
    
    ax2.errorbar(x_pos, df['D_step'], yerr=df['D_step_err'],
                 fmt='o', markersize=8, capsize=5, capthick=2,
                 label='Step Size Method', linewidth=2)
    
    # Color code by quality
    for i, (idx, row) in enumerate(df.iterrows()):
        color = 'green' if row['quality'] == 'good' else 'orange' if row['quality'] == 'anisotropic' else 'red'
        ax2.plot(i, row['D_step'], 'o', markersize=10, color=color, alpha=0.7)
    
    ax2.set_xlabel('Dataset / Track XML', fontsize=12)
    ax2.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax2.set_title('Step Size Method (color = quality)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Add quality legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Good'),
        Patch(facecolor='orange', alpha=0.7, label='Anisotropic'),
        Patch(facecolor='red', alpha=0.7, label='Drift/Both')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Dataset':<25} {'XML':<35} {'Particles':>10} {'D_MSD':>12} {'D_Step':>12} {'Quality':>10}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['dataset_key']:<25} {row['xml_name']:<35} "
              f"{row['n_particles']:>10} "
              f"{row['D_msd']:>12.4f} "
              f"{row['D_step']:>12.4f} "
              f"{row['quality']:>10}")
    print("="*100)


def plot_diffusion_vs_fps(df: pd.DataFrame):
    """
    Plot diffusion coefficients vs frame rate (FPS).
    """
    if df.empty:
        print("No results to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Diffusion Coefficient vs Frame Rate', fontsize=16, fontweight='bold')
    
    # Sort by FPS for better visualization
    df_sorted = df.sort_values('fps')
    
    # Plot 1: MSD method
    ax1.errorbar(df_sorted['fps'], df_sorted['D_msd'], yerr=df_sorted['D_msd_err'],
                 fmt='o', markersize=10, capsize=5, capthick=2,
                 color='steelblue', linewidth=2, alpha=0.7)
    
    # Add labels for each point
    for _, row in df_sorted.iterrows():
        ax1.annotate(row['xml_name'].replace('_Tracks.xml', '').replace('20 nm water_', '').replace('20 nm_water_', ''),
                    (row['fps'], row['D_msd']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=8, alpha=0.7)
    
    ax1.set_xlabel('Frame Rate (FPS)', fontsize=12)
    ax1.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax1.set_title('MSD Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Step size method
    colors = ['green' if q == 'good' else 'orange' if q == 'anisotropic' else 'red' 
              for q in df_sorted['quality']]
    
    ax2.errorbar(df_sorted['fps'], df_sorted['D_step'], yerr=df_sorted['D_step_err'],
                 fmt='o', markersize=10, capsize=5, capthick=2,
                 linewidth=2, alpha=0.7)
    
    # Color code by quality
    for _, row in df_sorted.iterrows():
        color = 'green' if row['quality'] == 'good' else 'orange' if row['quality'] == 'anisotropic' else 'red'
        ax2.plot(row['fps'], row['D_step'], 'o', markersize=12, color=color, alpha=0.7)
        
        # Add labels
        ax2.annotate(row['xml_name'].replace('_Tracks.xml', '').replace('20 nm water_', '').replace('20 nm_water_', ''),
                    (row['fps'], row['D_step']),
                    textcoords="offset points", xytext=(0,10), ha='center',
                    fontsize=8, alpha=0.7)
    
    ax2.set_xlabel('Frame Rate (FPS)', fontsize=12)
    ax2.set_ylabel('Diffusion Coefficient (µm²/s)', fontsize=12)
    ax2.set_title('Step Size Method (color = quality)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    # Add quality legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Good'),
        Patch(facecolor='orange', alpha=0.7, label='Anisotropic'),
        Patch(facecolor='red', alpha=0.7, label='Drift/Both')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_folder = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg")
    
    datasets = scan_folder(test_folder)
    
    print(f"Found {len(datasets)} datasets\n")
    print_dataset_summary(datasets)
    
    print("\n\n" + "="*70)
    print("ANALYZING ALL TRACKS")
    print("="*70)
    
    # Analyze all tracks
    results_df = analyze_all_tracks(datasets)
    
    # Plot comparison
    if not results_df.empty:
        plot_diffusion_comparison(results_df)
        plot_diffusion_vs_fps(results_df)
    else:
        print("\nNo tracks found to analyze")
