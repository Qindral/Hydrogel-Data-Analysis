import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import trackpy as tp


def trackmate_xml_to_traj(xml_path: str) -> pd.DataFrame:
    """
    Parse a TrackMate XML file and return a trackpy-style trajectory DataFrame.

    Returns a DataFrame with columns:
        - particle : track ID (int)
        - frame    : frame index (int)
        - x, y     : position (same units as stored in the XML)
        - (optional) z
        - t        : physical time in seconds

    The DataFrame has attributes:
        - traj.attrs['mpp'] : microns per pixel (float)
        - traj.attrs['fps'] : frames per second (float)
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # --- Calibration from <ImageData> ---
    image_data = root.find(".//ImageData")
    if image_data is None:
        raise ValueError("No <ImageData> tag found in XML – is this a TrackMate file?")

    # TrackMate stores calibration in ImageData as attributes like:
    # timeinterval="0.05"  pixelwidth="0.1" ...
    # Example in the official FAQ. :contentReference[oaicite:0]{index=0}
    dt = float(image_data.get("timeinterval", "1.0"))   # seconds per frame
    fps = 1.0 / dt
    mpp = float(image_data.get("pixelwidth", "1.0"))    # microns per pixel (if calibrated)

    # --- Collect all spots ---
    # Spots are typically under <AllSpots><SpotsInFrame frame="..."><Spot ... />
    spots = {}
    for spot in root.iter("Spot"):
        sid = int(spot.get("ID"))
        spots[sid] = {
            "frame": int(spot.get("FRAME")),
            "x": float(spot.get("POSITION_X")),
            "y": float(spot.get("POSITION_Y")),
            # POSITION_Z may be missing in 2D data
            "z": float(spot.get("POSITION_Z", "nan")),
        }

    if not spots:
        raise ValueError("No <Spot> elements found in XML. Export finished tracks in TrackMate first.")

    # --- Map Spot-ID -> Track-ID via <AllTracks><Track TRACK_ID=...><Edge ... /> ---
    spot_to_track = {}
    for track in root.iter("Track"):
        track_id = int(track.get("TRACK_ID"))
        for edge in track.iter("Edge"):
            s = int(edge.get("SPOT_SOURCE_ID"))
            t = int(edge.get("SPOT_TARGET_ID"))
            # Don't overwrite if already assigned (shouldn't collide in normal TrackMate output)
            spot_to_track.setdefault(s, track_id)
            spot_to_track.setdefault(t, track_id)

    # --- Build trajectory DataFrame in trackpy format ---
    rows = []
    for sid, dat in spots.items():
        # Optional: skip unlinked spots (no track)
        if sid not in spot_to_track:
            continue

        row = {
            "particle": spot_to_track[sid],
            "frame": dat["frame"],
            "x": dat["x"],
            "y": dat["y"],
        }
        if not np.isnan(dat["z"]):
            row["z"] = dat["z"]
        rows.append(row)

    if not rows:
        raise ValueError("No spots belonging to tracks. Did you export 'filtered tracks' in TrackMate?")

    traj = (
        pd.DataFrame(rows)
        .sort_values(["particle", "frame"])
        .reset_index(drop=True)
    )

    # Physical time axis in seconds
    traj["t"] = traj["frame"] * dt

    # Attach calibration so you don’t have to remember it later
    traj.attrs["mpp"] = mpp
    traj.attrs["fps"] = fps

    return traj


def diffusion_from_trackmate_xml(
    xml_path: str,
    max_lagtime: int = 50,
    dims: int = 2,
    fit_range: tuple[float, float] | None = None,
):
    """
    High-level helper:

    1. Parse TrackMate-XML -> traj DataFrame (trackpy format)
    2. Compute per-particle MSD (imsd) and ensemble MSD (emsd) with trackpy
    3. Fit a power law MSD(t) = A * t^alpha to get:
          D = A / (2 * dims)   (for alpha ~ 1, normal Diffusion)

    Parameters
    ----------
    xml_path : str
        Path to TrackMate XML file.
    max_lagtime : int
        Maximal lag (in frames) for MSD calculation.
    dims : int
        Dimension for D (2 for x,y; 3 for x,y,z).
    fit_range : (t_min, t_max) in seconds, optional
        Time window used for the linear fit in log-log. If None, use all lags.

    Returns
    -------
    traj : DataFrame
        Trajectories in trackpy format.
    imsd : DataFrame
        Per-particle MSD (index: lag time [s], columns: particle IDs). :contentReference[oaicite:1]{index=1}
    emsd : Series
        Ensemble MSD (index: lag time [s]). :contentReference[oaicite:2]{index=2}
    D_particles : DataFrame
        Columns: ['particle', 'D', 'alpha'] (per-track D and anomalous exponent).
    D_ensemble : (D_ens, alpha_ens)
        Ensemble diffusion coefficient and exponent.
    """
    traj = trackmate_xml_to_traj(xml_path)
    mpp = traj.attrs["mpp"]
    fps = traj.attrs["fps"]

    # --- MSDs mit trackpy ---
    # imsd: pro Track eine Spalte, index = lag time in s
    imsd = tp.imsd(
        traj,
        mpp=mpp,
        fps=fps,
        max_lagtime=max_lagtime,
        pos_columns=["x", "y"],  # für 3D: ["x", "y", "z"]
    )  # :contentReference[oaicite:3]{index=3}

    # emsd: Ensemble-MSD
    emsd = tp.emsd(
        traj,
        mpp=mpp,
        fps=fps,
        max_lagtime=max_lagtime,
    )  # :contentReference[oaicite:4]{index=4}

    # --- Fit pro Partikel: MSD(t) ~ A * t^alpha ---
    D_rows = []
    for col in imsd.columns:
        series = imsd[col].dropna()
        if series.empty:
            continue

        # Optionales Zeitfenster auswählen
        if fit_range is not None:
            t = series.index.values
            mask = (t >= fit_range[0]) & (t <= fit_range[1])
            series = series.loc[mask]

        if series.empty:
            continue

        A, alpha = tp.utils.fit_powerlaw(series)  # log-log-Fit: MSD = A * t^alpha :contentReference[oaicite:5]{index=5}
        D = A / (2 * dims)  # für normales Brown’sches (alpha ~ 1)
        D_rows.append({"particle": col, "D": D, "alpha": alpha})

    D_particles = pd.DataFrame(D_rows)

    # --- Ensemble-Fit ---
    emsd_clean = emsd.dropna()
    if len(emsd_clean) > 0:
        if fit_range is not None:
            t = emsd_clean.index.values
            mask = (t >= fit_range[0]) & (t <= fit_range[1])
            emsd_clean = emsd_clean.loc[mask]

    if len(emsd_clean) > 1:
        A_ens, alpha_ens = tp.utils.fit_powerlaw(emsd_clean)
        D_ens = A_ens / (2 * dims)
        D_ensemble = (D_ens, alpha_ens)
    else:
        D_ensemble = (np.nan, np.nan)

    return traj, imsd, emsd, D_particles, D_ensemble

XML_DATEI = r"Z:\Diffusion in Hydrogel Data\20mg_20nm\Trajektorien\ResultofB1_20nm_20mg_1d_nichtzentral_1_ohne_Tracks.xml"
XML_DATEI = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml"
dx = 0.150  # µm per pixel

if __name__ == "__main__":
    # Beispielaufruf
    xml_file = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml"

    traj, imsd, emsd, D_particles, D_ensemble = diffusion_from_trackmate_xml(
        xml_file,
        max_lagtime=50,
        dims=2,
        fit_range=(0.1, 2.0),  # z.B. lineares Regime zwischen 0.1 s und 2 s
    )

    print("Calibration:")
    print(f"  mpp = {traj.attrs['mpp']} µm/px")
    print(f"  fps = {traj.attrs['fps']} 1/s")
    print()

    print("Per-particle diffusion coefficients:")
    print(D_particles.head())

    print("\nEnsemble D, alpha:")
    print(D_ensemble)
