import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import trackpy as tp


def load_tracks_xml(
    xml_path: str,
    dx: float,
    dt: Optional[float] = None,
) -> pd.DataFrame:
    """
    Parse a 'Tracks' XML of the form:

    <Tracks frameInterval="1.0" timeUnits="frame" spaceUnits="pixel" ...>
      <particle nSpots="...">
        <detection t="0" x="..." y="..." z="0.0" />
        ...
      </particle>
      ...
    </Tracks>

    und erzeuge ein trackpy-kompatibles DataFrame:

        columns = ['particle', 'frame', 'x', 'y']

    Parameter
    ---------
    xml_path : str
        Pfad zur XML-Datei.
    dx : float
        Mikrometer pro Pixel (µm/px).
    dt : float oder None
        Sekunden pro Frame (s/frame). Wenn None → 'frameInterval' aus
        der XML benutzen (falls vorhanden), sonst 1.0.

    Rückgabe
    --------
    traj : DataFrame
        Trajektorien, sortiert nach particle, frame.
        Wichtige Attribute:
            traj.attrs['mpp'] = dx       (µm/px, für trackpy)
            traj.attrs['fps'] = 1/dt     (1/s, für trackpy)
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # dt ggf. aus dem Attribut 'frameInterval' holen
    if dt is None:
        frame_interval = root.get("frameInterval")
        if frame_interval is not None:
            dt = float(frame_interval)
        else:
            dt = 1.0  # Fallback

    rows = []
    # Partikel bekommen eine fortlaufende ID basierend auf der Reihenfolge im XML
    for pid, p in enumerate(root.findall("particle")):
        for det in p.findall("detection"):
            t = int(det.get("t"))
            x = float(det.get("x"))
            y = float(det.get("y"))
            rows.append(
                {
                    "particle": pid,
                    "frame": t,
                    "x": x,  # Pixel
                    "y": y,  # Pixel
                }
            )

    if not rows:
        raise ValueError("Keine <particle>/<detection>-Einträge in der XML-Datei gefunden.")

    traj = (
        pd.DataFrame(rows)
        .sort_values(["particle", "frame"])
        .reset_index(drop=True)
    )

    # trackpy braucht mpp (µm/px) und fps (Frames/s)
    traj.attrs["dx"] = dx            # nur Info
    traj.attrs["dt"] = dt
    traj.attrs["mpp"] = dx           # µm/px
    traj.attrs["fps"] = 1.0 / dt     # Frames/s

    return traj
traj = load_tracks_xml(r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml", dx=0.150, dt=0.050)

print(traj.head())

def _fit_powerlaw_trackpy(series: pd.Series) -> Optional[Tuple[float, float]]:
    """
    Robust wrapper um trackpy.utils.fit_powerlaw.

    Rückgabe: (A, n) mit MSD(t) = A * t**n
    oder None, falls Fit nicht möglich.
    """
    # Alles in echte floats verwandeln
    series = pd.to_numeric(series, errors="coerce").dropna()
    series.index = pd.to_numeric(series.index, errors="coerce")
    series = series.dropna()
    if series.empty:
        return None

    params = tp.utils.fit_powerlaw(series, plot=False)  # trackpy 0.5+ gibt DataFrame zurück

    if isinstance(params, pd.DataFrame):
        try:
            n = float(params.loc[0, "n"])
            A = float(params.loc[0, "A"])
        except (KeyError, ValueError, TypeError, IndexError):
            return None
    else:
        # Fallback für ältere trackpy-Versionen: Series mit Keys 'n', 'A'
        try:
            n = float(params["n"])
            A = float(params["A"])
        except (KeyError, ValueError, TypeError):
            return None

    return A, n

#A,n = _fit_powerlaw_trackpy(pd.Series(traj.index, index=traj['x']**2 + traj['y']**2))


def msd_and_diffusion_from_xml(
    xml_path: str,
    dx: float,
    dt: float,
    max_lagtime: int = 50,
    dims: int = 2,
    min_track_length: int = 5,
    fit_range: Optional[Tuple[float, float]] = None,
):
    """
    Gesamte Pipeline:
      XML → Trajektorien → imsd/emsd → (D_i, alpha_i) & (D_ens, alpha_ens).

    Parameter
    ---------
    xml_path : str
        Pfad zur 'Tracks'-XML-Datei.
    dx : float
        Mikrometer pro Pixel (µm/px).
    dt : float
        Sekunden pro Frame (s/frame).
    max_lagtime : int
        Maximaler Lag in FRAMES für die MSD-Berechnung.
    dims : int
        Raumdimension (2 für x,y; 3 für x,y,z).
    min_track_length : int
        Minimale Anzahl an Punkten pro Track, die für D-Fits berücksichtigt werden.
    fit_range : (t_min, t_max) in Sekunden oder None
        Zeitfenster, in dem der Power-Law-Fit durchgeführt wird.
        None → alle verfügbaren Lags.

    Rückgabe
    --------
    traj : DataFrame
        Trajektorien mit Attributen mpp, fps.
    imsd : DataFrame
        per-particle MSD (Spalten: Partikel-IDs, Index: lag time in s).
    emsd : Series
        Ensemble-MSD (Index: lag time in s).
    D_particles : DataFrame
        Spalten ['particle', 'D', 'alpha', 'n_points'].
    D_ensemble : dict
        {'D': D_ens, 'alpha': alpha_ens}
    """
    # 1) Tracks einlesen
    traj = load_tracks_xml(xml_path, dx=dx, dt=dt)
    mpp = traj.attrs["mpp"]
    fps = traj.attrs["fps"]

    # 2) Optional: kurze Tracks filtern
    if min_track_length is not None and min_track_length > 1:
        lengths = traj.groupby("particle")["frame"].count()
        valid_ids = lengths[lengths >= min_track_length].index
        traj = traj[traj["particle"].isin(valid_ids)].copy()

    if traj.empty:
        raise ValueError("Keine Trajektorien nach Filterung übrig.")

    # 3) MSDs mit trackpy
    # imsd: pro Track eine Spalte, Index = lag time (s)
    imsd = tp.imsd(
        traj,
        mpp=mpp,
        fps=fps,
        max_lagtime=max_lagtime,
        pos_columns=["x", "y"],  # bei Bedarf ['x', 'y', 'z']
    )  # 

    # emsd: Ensemble-MSD
    emsd = tp.emsd(
        traj,
        mpp=mpp,
        fps=fps,
        max_lagtime=max_lagtime,
    )  # 

    # 4) Per-Track Fits
    D_rows = []
    for col in imsd.columns:
        series = imsd[col].dropna()
        if series.empty:
            continue

        # Fitbereich in t beschränken (Sekunden)
        if fit_range is not None:
            t = series.index.values
            mask = (t >= fit_range[0]) & (t <= fit_range[1])
            series = series.loc[mask]
            if series.empty:
                continue

        result = _fit_powerlaw_trackpy(series)
        if result is None:
            continue
        A, n = result

        D = A / (2.0 * dims)  # MSD(t) ~ 2 d D t für n≈1

        D_rows.append(
            {
                "particle": col,
                "D": D,
                "alpha": n,
                "n_points": len(series),
            }
        )

    D_particles = pd.DataFrame(D_rows)

    # 5) Ensemble-Fit
    emsd_clean = emsd.dropna()
    if fit_range is not None and len(emsd_clean) > 0:
        t = emsd_clean.index.values
        mask = (t >= fit_range[0]) & (t <= fit_range[1])
        emsd_clean = emsd_clean.loc[mask]

    D_ensemble = {"D": np.nan, "alpha": np.nan}
    if len(emsd_clean) > 1:
        result = _fit_powerlaw_trackpy(emsd_clean)
        if result is not None:
            A_ens, n_ens = result
            D_ens = A_ens / (2.0 * dims)
            D_ensemble = {"D": D_ens, "alpha": n_ens}

    return traj, imsd, emsd, D_particles, D_ensemble

xml_file = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml"
xml_file = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks.xml"
xml_file = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\preprocess\tracks\20 nm_2_processed_Tracks.xml"
traj, imsd, emsd, D_particles, D_ensemble  = msd_and_diffusion_from_xml(xml_file, dx=0.150, dt=0.050)

print("Kalibration:")
print(f"  dx  = {traj.attrs['dx']} µm/px")
print(f"  dt  = {traj.attrs['dt']} s/frame")
print(f"  mpp = {traj.attrs['mpp']} µm/px")
print(f"  fps = {traj.attrs['fps']} 1/s\n")

print("Per-Track Diffusionskoeffizienten (erste Zeilen):")
print(D_particles.head())

print("\nEnsemble:")
print(f"  D_ensemble  = {D_ensemble['D']:.4g} µm²/s")
print(f"  alpha_ens   = {D_ensemble['alpha']:.3f}")

# Optional: Ensemble-MSD plotten
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.loglog(emsd.index, emsd.values, "o-")
ax.set_xlabel("lag time τ [s]")
ax.set_ylabel("MSD(τ) [µm²]")
ax.grid(True, which="both")
plt.show()