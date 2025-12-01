import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import trackpy as tp


def load_simple_tracks_xml(
    xml_path: str,
    dx: float,
    dt: float | None = None,
) -> pd.DataFrame:
    """
    Parse a 'Tracks' XML of the form:

    <Tracks frameInterval="1.0" ...>
      <particle nSpots="...">
        <detection t="0" x="..." y="..." z="0.0" />
        ...
      </particle>
      ...
    </Tracks>

    und gebe ein trackpy-kompatibles DataFrame zurück:
        columns = ['particle', 'frame', 'x', 'y']

    Parameter
    ---------
    xml_path : str
        Pfad zur XML-Datei.
    dx : float
        Mikrometer pro Pixel (µm/px).
    dt : float | None
        Sekunden pro Frame (s/frame). Wenn None, wird die 'frameInterval'-
        Angabe aus <Tracks> verwendet (falls vorhanden), ansonsten 1.0.

    Rückgabe
    --------
    traj : pd.DataFrame
        Trajektorien. Pixelkoordinaten, aber mit dx, dt als Attribute.
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Falls dt nicht explizit angegeben wurde, versuche frameInterval aus dem XML
    if dt is None:
        dt_xml = root.get("frameInterval")
        dt = float(dt_xml) if dt_xml is not None else 1.0

    rows = []
    # particle-Elemente sequenziell durchgehen → ID = laufender Index
    for pid, p in enumerate(root.findall("particle")):
        for det in p.findall("detection"):
            t = int(det.get("t"))
            x = float(det.get("x"))
            y = float(det.get("y"))
            # z ist hier immer 0.0, kann man ignorieren oder später für 3D ergänzen

            rows.append(
                {
                    "particle": pid,
                    "frame": t,
                    "x": x,
                    "y": y,
                }
            )

    if not rows:
        raise ValueError("Keine <particle>/<detection>-Einträge im XML gefunden.")

    traj = (
        pd.DataFrame(rows)
        .sort_values(["particle", "frame"])
        .reset_index(drop=True)
    )

    # Kalibrierung an das DataFrame hängen – nur als Info für nachgelagerte Funktionen
    traj.attrs["dx"] = dx          # µm / px
    traj.attrs["dt"] = dt          # s / frame
    traj.attrs["mpp"] = dx         # trackpy: microns per pixel
    traj.attrs["fps"] = 1.0 / dt   # trackpy: frames per second

    return traj

def diffusion_from_trackmate_with_trackpy(
    xml_path: str,
    dx: float,
    dt: float,
    max_lagtime: int = 50,
    dims: int = 2,
    fit_range: tuple[float, float] | None = None,
):
    """
    Komplette Pipeline:

      XML (dein 'Tracks'-Format)
        → trackpy-Trajektorien-DF
        → imsd / emsd mit trackpy
        → per-Track & Ensemble-D über Power-Law-Fit.

    Parameter
    ---------
    xml_path : str
        Pfad zur 'Tracks'-XML-Datei.
    dx : float
        Mikrometer pro Pixel (µm/px).
    dt : float
        Sekunden pro Frame (s/frame).
    max_lagtime : int
        Maximaler Lag (in FRAMES) für MSD.
    dims : int
        Raumdimension (2 oder 3).
    fit_range : (t_min, t_max) in Sekunden oder None
        Zeitfenster für MSD(t)=A*t^alpha-Fits. None → alle Lags.

    Rückgabe
    --------
    traj : DataFrame
        Trajektorien (Pixelkoordinaten, aber mit mpp/fps-Attributen).
    imsd : DataFrame
        Per-Particle MSD (Index: lag time [s], Spalten: Partikel-IDs).
    emsd : Series
        Ensemble MSD (Index: lag time [s]).
    D_particles : DataFrame
        Spalten ['particle', 'D', 'alpha'].
    D_ensemble : (D_ens, alpha_ens)
        Ensemble-Diffusionskoeffizient und Exponent.
    """
    # 1) Trajektorien einlesen
    traj = load_simple_tracks_xml(xml_path, dx=dx, dt=dt)

    mpp = traj.attrs["mpp"]
    fps = traj.attrs["fps"]

    # 2) MSD mit trackpy
    imsd = tp.imsd(
        traj,
        mpp=mpp,
        fps=fps,
        max_lagtime=max_lagtime,
        pos_columns=["x", "y"],  # bei 3D: ["x", "y", "z"]
    )

    emsd = tp.emsd(
        traj,
        mpp=mpp,
        fps=fps,
        max_lagtime=max_lagtime,
    )

    # 3) per-Track Fit: MSD(t) = A * t^alpha
    D_rows = []
    for col in imsd.columns:
        series = imsd[col].dropna()

        if series.empty:
            continue

        # --- WICHTIG: robust gegen Strings / dtype=object ---
        series = pd.to_numeric(series, errors="coerce")
        series = series.dropna()
        series.index = pd.to_numeric(series.index, errors="coerce")
        series = series.dropna()

        # optional Zeitfenster auswählen
        if fit_range is not None:
            t = series.index.values  # [s]
            mask = (t >= fit_range[0]) & (t <= fit_range[1])
            series = series.loc[mask]

        if series.empty:
            continue

        # Power-Law Fit (jetzt sicher Float)
        params = tp.utils.fit_powerlaw(series, plot=False)
        if isinstance(params, pd.DataFrame):
            try:
                alpha = float(params.loc[0, "n"])
                A     = float(params.loc[0, "A"])
            except (KeyError, ValueError, TypeError, IndexError):
                continue
        else:
            # Fallback, falls andere Version: Series mit Keys "n", "A"
            try:
                alpha = float(params["n"])
                A     = float(params["A"])
            except (KeyError, ValueError, TypeError):
                continue

        # Falls A oder alpha trotzdem nicht float sind → skip
        try:
            A = float(A)
            alpha = float(alpha)
        except:
            continue

        D = A / (2 * dims)
        D_rows.append({"particle": col, "D": D, "alpha": alpha})

    D_particles = pd.DataFrame(D_rows)

    # 4) Ensemble-Fit
    emsd_clean = emsd.dropna()
    if fit_range is not None and len(emsd_clean) > 0:
        t = emsd_clean.index.values
        mask = (t >= fit_range[0]) & (t <= fit_range[1])
        emsd_clean = emsd_clean.loc[mask]

    if len(emsd_clean) > 1:
        params = tp.utils.fit_powerlaw(emsd_clean, plot=False)

        # trackpy >= 0.5 liefert ein DataFrame mit einer Zeile
        if isinstance(params, pd.DataFrame):
            try:
                alpha_ens = float(params.loc[0, "n"])   # Exponent
                A_ens     = float(params.loc[0, "A"])   # Präfaktor
            except (KeyError, ValueError, TypeError, IndexError):
                D_ensemble = (np.nan, np.nan)
                return traj, imsd, emsd, D_particles, D_ensemble

        # Fallback: ältere trackpy-Version → Series
        else:
            try:
                alpha_ens = float(params["n"])
                A_ens     = float(params["A"])
            except (KeyError, ValueError, TypeError):
                D_ensemble = (np.nan, np.nan)
                return traj, imsd, emsd, D_particles, D_ensemble

        # Diffusionskoeffizient (2D oder 3D)
        D_ens = A_ens / (2 * dims)

        # Rückgabeformat wie vorher
        D_ensemble = (D_ens, alpha_ens)

    else:
        D_ensemble = (np.nan, np.nan)

    return traj, imsd, emsd, D_particles, D_ensemble



XML_DATEI = r"Z:\Diffusion in Hydrogel Data\20mg_20nm\Trajektorien\ResultofB1_20nm_20mg_1d_nichtzentral_1_ohne_Tracks.xml"
XML_DATEI = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml"
dx = 0.150  # µm per pixel

if __name__ == "__main__":
    xml_file = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml"
    xml_file = r"Z:\Diffusion in Hydrogel Data\20mg_20nm\Trajektorien\ResultofB1_20nm_20mg_1d_nichtzentral_1_ohne_Tracks.xml"
    # HIER definierst du deine Kalibration:
    dx = 0.15   # µm/px 
    dt = 0.050   # s/frame 

    traj, imsd, emsd, D_particles, D_ensemble = diffusion_from_trackmate_with_trackpy(
        xml_file,
        dx=dx,
        dt=dt,
        max_lagtime=100,       # z.B. max. 100 Frames Lag
        dims=2,                # xy
        fit_range=(0.1, 2.0),  # lineares Regime in Sekunden
    )

    print("Kalibration, die wir verwenden:")
    print(f"  mpp = {traj.attrs['mpp']} µm/px")
    print(f"  fps = {traj.attrs['fps']} 1/s\n")

    print("Per-particle D (erste Zeilen):")
    print(D_particles.head())

    print("\nEnsemble D, alpha:")
    print(D_ensemble)

    # Optional: Ensemble-MSD plotten
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.loglog(emsd.index, emsd.values, "o-")
    ax.set_xlabel("lag time τ [s]")
    ax.set_ylabel("MSD(τ) [µm²]")
    ax.grid(True, which="both")
    plt.show()

