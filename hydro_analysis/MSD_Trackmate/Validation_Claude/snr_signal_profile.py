"""
snr_signal_profile.py

Paired raw / DoG reanalysis: same particles, scan coordinates and annulus.
Exports absolute profiles, normalized overlays, per-particle metrics and
condition summaries. DoG uses session detector radius with TrackMate /
ImgLib2 scale conventions, computed with SciPy (not exact Fiji replay).
No median prefilter is applied, to isolate DoG alone. Original LoG sessions
are explicitly recorded as such; this is a new DoG analysis of their images.
SNR gain is descriptive local contrast-to-background variability, not photon
SNR or a statistical detection significance. Filtered pixels are correlated;
the annulus may contain particle/filter tails. No guaranteed improvement.


Erzeugt PRO PARTIKELGROESSE (20/50/100/200/500/1000 nm) eine EIGENE
Abbildungsdatei (nicht eine gemeinsame Abbildung fuer alle Groessen). Je
Groesse und je verfuegbarem Fall (immobilized/water/hydrogel, siehe FILES)
werden bis zu N_PARTICLES_PER_CONDITION=5 Partikel ausgewertet -- also die
volle Quote PRO FALL, nicht ueber die Faelle gepoolt/aufgeteilt. Innerhalb
einer Abbildung: Zeilen = Fall, Spalten = die bis zu 5 Partikel dieses
Falls, farbcodiert nach Fall. EIN PANEL PRO PARTIKEL.

Nur ISOLIERTE Partikel werden ausgewaehlt: ein Kandidat wird verworfen,
wenn ein anderer Spot aus Model/AllSpots (unabhaengig von Tracking/
Sichtbarkeit) so liegt, dass er im horizontalen Linienscan (Signalbereich
+ Hintergrund-Anhang) desselben Bildzeilenbereichs sichtbar mit auftauchen
wuerde (siehe _is_isolated: Distanz in x innerhalb der gescannten Offsets,
Distanz in y innerhalb ISOLATION_ROW_TOLERANCE_PX + eigener Radius des
anderen Spots). Das verhindert, dass im langen Hintergrund-Anhang
Nachbarpartikel als zusaetzliche Peaks auftauchen (beobachtet z.B. bei
100 nm immobilized). Die Auswahl nutzt weiterhin den Retry-Pool-Ansatz
(ganzer Kandidaten-Pool des Frames zufaellig durchmischt, naechster
Kandidat bei Ablehnung), damit die Isolationspruefung die Ausbeute nicht
einfach stillschweigend reduziert, ohne Alternativen zu versuchen.

Jede Linie ist EIN durchgehender Zug (kein gestrichelter Abschnitt): der
horizontale Linienscan durchs Spot-Zentrum (+/- SIGNAL_HALF_WIDTH_PX Pixel)
direkt gefolgt vom selben Linienscan weiter fortgesetzt, ueber
BACKGROUND_TAIL_FACTOR x so viele Pixel wie der Signalbereich lang ist --
zeigt, wie der Hintergrund ueber eine deutlich laengere Strecke
tatsaechlich aussieht. Der fuer die SNR-Berechnung verwendete Peak (Maximum
des Signalbereichs) ist als Marker auf der Linie hervorgehoben.

Da die absolute Helligkeit zwischen 20 nm- und 1000 nm-Partikeln oder
zwischen den drei Faellen um Groessenordnungen variiert, zeigt die Y-Achse
NICHT die Rohintensitaet, sondern das auf den lokalen Hintergrund normierte
Signal (Intensitaet - mu_bg) / sigma_bg -- numerisch identisch mit dem
lokalen SNR an dieser Position, dadurch sind alle Kurven vergleichbar.

Hintergrund-Statistik (mu_bg, sigma_bg -- getrennt vom oben gezeigten
Hintergrund-Linien-Anhang) pro Spot: lokaler Annulus (1.5*R bis 3.0*R um
das Spot-Zentrum, R = Spot-eigener RADIUS-Wert, oder das Detektor-RADIUS
als Fallback). Alle Spots aus AllSpots -- auch nicht getrackte/nicht
sichtbare -- werden mit ihrem eigenen Radius aus dem Annulus ausmaskiert.
SNR = (Peak - mu_bg) / sigma_bg, Peak = Maximum des Signalbereichs.

FILES haelt fuer jede (Fall, Partikelgroesse)-Kombination den Pfad zur
*_Tracks.xml -- die zugehoerige rohe Settings-XML (mit Model/AllSpots) und
das TIFF werden von find_session_and_tif() automatisch daneben gesucht
(jeder Unterordner mit "analys" im Namen, da dieses Projekt dafuer viele
verschiedene Namen nebeneinander verwendet: Analysis, analysis,
Analysis_immob_NEW, Analysis_50, Analysis_1000, Trackmate_Analyses).
Settings/ImageData/pixelwidth in diesen rohen Session-XMLs ist projektweit
nur ein Platzhalter ("1.0") -- die echte mpp-Kalibrierung kommt aus einer
Bildbreiten-Zuordnung (dieselbe wie core/io.py::parse_rec_file, per
Kalibriergitter gemessen), hier per bereits geladener TIFF-Breite
nachgeschlagen (MPP_BY_IMAGE_WIDTH_PX). Fehlende Kombinationen (siehe
Kommentare bei FILES) werden explizit uebersprungen, nie stillschweigend
ausgelassen.

Darstellung nach hydro_analysis/Style_guide.txt: PNG, 600 dpi, Seiten-
verhaeltnis 1.43:1 pro Panel, Ticks nach innen auf allen vier Seiten,
kein Grid, Open-Sans-Schriftfamilie, Panel-Raster wie in den uebrigen
Validation_Claude-Skripten (_grid()-Konvention) -- als eigener _RC-Dict
hier dupliziert (identisch mit MSD_per_file_publication._RC), da dieses
Skript bewusst keine projektinternen Imports hat.

Ersetzt/konsolidiert die bisherige mehrteilige SNR-/Detectability-Analyse
(TaskC_Detectability_Compute.py, ..._Distribution.py, ..._Figure.py,
Detectability_Sweep_Demo.py, _detectability_selection.py,
SNR_DoG_Compute.py/_snr_dog.py/SNR_DoG_Figures.py -- alle entfernt).

Nur Standardbibliothek + numpy/pandas/matplotlib/tifffile/scipy/xml.etree/re,
keine projektinternen Imports -- eigenstaendig lauffaehig, auch ausserhalb
dieses Repos.
"""
from __future__ import annotations

import os
import argparse
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ── Eingabe ──────────────────────────────────────────────────────────────────────
# (Fall, Partikelgroesse_nm) -> Pfad zur *_Tracks.xml. Settings-XML/TIFF
# werden automatisch daneben gesucht (siehe find_session_and_tif()).
FILES: dict[tuple[str, float], str] = {
    # 20nm_immob_Tracks.xml (die Basis-Datei) hat leere FilteredTracks
    # (keine rekonstruierbaren Edges) -- 20nm_immob_2 stattdessen verwendet.
    ("immobilized", 20.0): r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks\20nm_immob_2_Tracks.xml",
    ("immobilized", 50.0): r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks\50nm_immob_1_Tracks.xml",
    ("immobilized", 100.0): r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks\100 nm_Tracks.xml",
    ("immobilized", 200.0): r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks\200 nm_immob_Tracks.xml",
    ("immobilized", 500.0): r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks\500 nm_Tracks.xml",
    ("immobilized", 1000.0): r"E:\PhD Data Analysis\SPT 2025 II\2026.01.31_immobilized\Tracks\1k_Tracks.xml",

    # water 20 nm: gefunden in einem bislang nicht referenzierten Wasser-
    # Messordner (2025.11.27_water\tracks\).
    ("water", 20.0): r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27_water\tracks\20 nm_Tracks.xml",
    ("water", 50.0): r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_50\50 nm water 02_Tracks.xml",
    ("water", 100.0): r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\100 nm\Tracks\100 nm TriSpuffer_2_Tracks.xml",
    ("water", 200.0): r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\200 nm\Tracks\200 nm_Tracks.xml",
    ("water", 500.0): r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\500 nm\Tracks\500 nm water_01_Tracks.xml",
    ("water", 1000.0): r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\Tracks_1000\1000_nm_01_Tracks.xml",

    ("hydrogel", 20.0): r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\20 nm\20 nm 20 mg\Tracks\20nm_1d_A2_02_Tracks.xml",
    ("hydrogel", 50.0): r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16\50 nm\50 nm 20 mg\Tracks_new\50 nm A2_2d_01_Tracks.xml",
    # hydrogel 100/200/500/1000 nm: keine Rohdaten vorhanden (100 nm-Ordner
    # ist leer, 200/500/1000 nm-Ordner existieren im Projekt nicht).
}

OUTPUT_DIR = r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder\PSF_Analyse"
SEED = 42

N_PARTICLES_PER_CONDITION = 5   # volle Quote pro Fall (nicht ueber Faelle gepoolt)
INNER_RADIUS_FACTOR = 1.5
OUTER_RADIUS_FACTOR = 3.0
SIGNAL_HALF_WIDTH_PX = 10
BACKGROUND_TAIL_FACTOR = 4  # Hintergrund-Anhang ist 4x so lang wie der Signalbereich
# Isolationspruefung: ein anderer Spot gilt als "sichtbar im Linienscan", wenn
# seine x-Distanz zum Kandidat-Zentrum innerhalb der gescannten Offsets liegt
# UND seine y-Distanz innerhalb dieser Toleranz + seinem eigenen Radius --
# solche Kandidaten werden verworfen (siehe _is_isolated).
ISOLATION_ROW_TOLERANCE_PX = 2.0

CONDITION_COLORS = {"immobilized": "#000000", "water": "#0000da", "hydrogel": "#da0000"}

# Settings/ImageData/pixelwidth in den rohen Session-XMLs ist projektweit nur
# ein Platzhalter ("1.0"), NICHT die echte Kalibrierung -- die kommt aus den
# .rec-Metadaten dieses Projekts als Bildbreite->mpp-Zuordnung (per
# Kalibriergitter gemessen, siehe core/io.py::parse_rec_file). Hier per
# Bildbreite aus dem bereits geladenen TIFF nachgeschlagen, ohne .rec-Datei
# parsen zu muessen.
MPP_BY_IMAGE_WIDTH_PX = {200: 0.30, 400: 0.15, 696: 0.149, 696 * 2: 0.149 / 2, 800: 0.130, 348: 0.299}

# Style_guide.txt Abschnitt 12 (identisch mit MSD_per_file_publication._RC) --
# hier dupliziert, da dieses Skript bewusst keine projektinternen Imports hat.
_RC = {
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "font.family": "sans-serif",
    "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5.0,
    "ytick.major.size": 5.0,
    "xtick.minor.size": 3.0,
    "ytick.minor.size": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 7,
}

# SciPy DoG reanalysis using TrackMate/ImgLib2 scale conventions.
# No median prefilter; not a bit-identical replay of Fiji's detector.
def dog_filter(image, radius_yx):
    radius = np.asarray(radius_yx, dtype=float)
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("DoG requires a finite positive detector radius")
    target = np.maximum(1.0, 0.9 * radius / np.sqrt(2.0))
    small = np.sqrt(target ** 2 - 0.25)
    large = np.sqrt((target * 1.1 / 0.9) ** 2 - 0.25)
    image = np.asarray(image, dtype=np.float64)
    return (gaussian_filter(image, small, mode="mirror") -
            gaussian_filter(image, large, mode="mirror")), small, large


random.seed(SEED)


def _norm(s: str) -> str:
    return re.sub(r"[\s_]", "", s).lower()


def find_session_and_tif(tracks_xml: Path) -> tuple[Path | None, Path | None]:
    """Findet die rohe Settings-XML (mit Model/AllSpots) und das TIFF zu
    einer *_Tracks.xml (Gross-/Kleinschreibung des "_Tracks"/"_tracks"-
    Suffixes wird ignoriert -- einige Ordner verwenden Kleinschreibung).
    Settings-XML: jeder Unterordner von movie_dir, dessen Name "analys"
    enthaelt (Gross-/Kleinschreibung egal) -- diese Projektdaten verwenden
    dafuer viele verschiedene Namen nebeneinander (Analysis, analysis,
    Analysis_immob_NEW, Analysis_50, Analysis_1000, Trackmate_Analyses),
    ein fester Namens-Katalog waere nicht robust genug. Exakter Stem-
    Abgleich zuerst, sonst Leerzeichen/Unterstrich-normalisiert. Das TIFF
    wird direkt im Movie-Ordner gesucht, ebenso normalisiert falls der
    exakte Name nicht existiert."""
    stem = re.sub(r"_tracks$", "", tracks_xml.stem, flags=re.IGNORECASE)
    movie_dir = tracks_xml.parent.parent
    target = _norm(stem)

    settings_xml = None
    analysis_dirs = [p for p in movie_dir.iterdir() if p.is_dir() and "analys" in p.name.lower()]
    for folder in analysis_dirs:
        exact = folder / f"{stem}.xml"
        if exact.exists():
            settings_xml = exact
            break
        matches = [p for p in folder.glob("*.xml") if _norm(p.stem) == target]
        if len(matches) == 1:
            settings_xml = matches[0]
            break

    tif_path = movie_dir / f"{stem}.tif"
    if not tif_path.exists():
        matches = [p for p in movie_dir.glob("*.tif") if _norm(p.stem) == target]
        tif_path = matches[0] if len(matches) == 1 else None

    return settings_xml, tif_path


def _is_isolated(spot_id: int, xi: int, yi: int, frame_all_spots: pd.DataFrame,
                  offsets_px: np.ndarray, row_tolerance_px: float) -> bool:
    """True, wenn kein anderer Spot dieses Frames im horizontalen
    Linienscan (Signal + Hintergrund-Anhang) sichtbar mit auftauchen
    wuerde: x-Distanz zum Zentrum innerhalb der gescannten Offsets
    (asymmetrisch, der Anhang reicht nur nach rechts) UND y-Distanz
    innerhalb row_tolerance_px + dem eigenen Radius des anderen Spots."""
    lo, hi = float(offsets_px.min()), float(offsets_px.max())
    others = frame_all_spots[frame_all_spots["id"] != spot_id]
    for _, other in others.iterrows():
        dx = other["x_px"] - xi
        if not (lo <= dx <= hi):
            continue
        other_r = other["radius_px"] if np.isfinite(other["radius_px"]) and other["radius_px"] > 0 else 2.0
        if abs(other["y_px"] - yi) <= row_tolerance_px + other_r:
            return False
    return True


def analyze_file(tracks_xml: Path, n_particles: int) -> list[dict]:
    """Ein (Fall, Groesse)-Eintrag -> bis zu n_particles Ergebnis-Dicts mit
    Signal- und Hintergrund-Anhang-Offsets (px + nm), Profilwerten, mu_bg,
    sigma_bg, SNR. Leere Liste bei fehlenden Dateien oder fehlenden
    auswertbaren Spots -- vom Aufrufer mit einer Konsolenmeldung zu
    behandeln, nie stillschweigend."""
    settings_xml, tif_path = find_session_and_tif(tracks_xml)
    if settings_xml is None or tif_path is None:
        print(f"  [UEBERSPRUNGEN] {tracks_xml.name}: Settings-XML oder TIFF nicht gefunden "
              f"(settings={settings_xml}, tif={tif_path})")
        return []

    root = ET.parse(settings_xml).getroot()

    image_tag = root.find("Settings/ImageData")
    if image_tag is None:
        print(f"  [UEBERSPRUNGEN] {settings_xml.name}: Settings/ImageData fehlt")
        return []

    model_tag = root.find("Model")
    spatialunits = (model_tag.attrib.get("spatialunits", "pixel") if model_tag is not None else "pixel").lower()
    needs_px_conversion = "pixel" not in spatialunits
    pixelwidth_xml = float(image_tag.attrib["pixelwidth"])   # nur fuer needs_px_conversion, siehe Docstring
    pixelheight_xml = float(image_tag.attrib["pixelheight"])

    detector_tag = root.find("Settings/DetectorSettings")
    default_radius_raw = (
        float(detector_tag.attrib["RADIUS"]) if detector_tag is not None and "RADIUS" in detector_tag.attrib
        else np.nan
    )

    spot_rows = []
    for spot in root.findall("Model/AllSpots/SpotsInFrame/Spot"):
        a = spot.attrib
        if not {"ID", "FRAME", "POSITION_X", "POSITION_Y"} <= a.keys():
            continue
        radius_raw = float(a["RADIUS"]) if a.get("RADIUS") not in (None, "") else default_radius_raw
        spot_rows.append({
            "id": int(float(a["ID"])), "frame": int(float(a["FRAME"])),
            "x_raw": float(a["POSITION_X"]), "y_raw": float(a["POSITION_Y"]),
            "radius_raw": radius_raw, "visibility": int(float(a.get("VISIBILITY", 0))),
        })
    all_spots = pd.DataFrame(spot_rows)
    if all_spots.empty:
        print(f"  [UEBERSPRUNGEN] {settings_xml.name}: keine Spots in Model/AllSpots")
        return []

    if needs_px_conversion:
        all_spots["x_px"] = all_spots["x_raw"] / pixelwidth_xml
        all_spots["y_px"] = all_spots["y_raw"] / pixelheight_xml
        all_spots["radius_px"] = all_spots["radius_raw"] / pixelwidth_xml
    else:
        all_spots["x_px"] = all_spots["x_raw"]
        all_spots["y_px"] = all_spots["y_raw"]
        all_spots["radius_px"] = all_spots["radius_raw"]

    filtered_track_ids = {int(node.attrib["TRACK_ID"]) for node in root.findall("Model/FilteredTracks/TrackID")}
    tracked_spot_ids: set[int] = set()
    for track in root.findall("Model/AllTracks/Track"):
        track_id = int(float(track.attrib.get("TRACK_ID", -1)))
        if track_id not in filtered_track_ids:
            continue
        for edge in track.findall("Edge"):
            if "SPOT_SOURCE_ID" in edge.attrib:
                tracked_spot_ids.add(int(float(edge.attrib["SPOT_SOURCE_ID"])))
            if "SPOT_TARGET_ID" in edge.attrib:
                tracked_spot_ids.add(int(float(edge.attrib["SPOT_TARGET_ID"])))
    all_spots["tracked"] = all_spots["id"].isin(tracked_spot_ids)

    candidates = all_spots[(all_spots["visibility"] == 1) & all_spots["tracked"]]
    if candidates.empty:
        print(f"  [UEBERSPRUNGEN] {settings_xml.name}: keine sichtbaren, getrackten Spots")
        return []

    counts_per_frame = candidates.groupby("frame").size()
    valid_frames = sorted(counts_per_frame[counts_per_frame >= 1].index.tolist())
    if not valid_frames:
        print(f"  [UEBERSPRUNGEN] {settings_xml.name}: kein Frame mit sichtbaren, getrackten Spots")
        return []
    frame_nr = random.choice(valid_frames)

    # Ganzer Kandidaten-Pool dieses Frames, zufaellig durchmischt -- nicht nur
    # exakt n_particles vorab ziehen: ein Spot kann am Bildrand liegen oder zu
    # wenige Hintergrund-Pixel haben, dann wird der naechste aus dem Pool
    # probiert, bis n_particles erfolgreich sind oder der Pool leer ist.
    frame_candidates = candidates[candidates["frame"] == frame_nr]
    pool_ids = list(frame_candidates["id"])
    random.shuffle(pool_ids)
    chosen = frame_candidates.set_index("id").loc[pool_ids].reset_index()

    image = tifffile.imread(tif_path, key=frame_nr).astype(np.float64)
    if image.ndim != 2:
        print(f"  [UEBERSPRUNGEN] {tif_path.name}, Frame {frame_nr}: kein 2D-Graustufenbild ({image.shape})")
        return []
    radius_yx = (default_radius_raw / pixelheight_xml, default_radius_raw / pixelwidth_xml) if needs_px_conversion else (default_radius_raw, default_radius_raw)
    dog_image, dog_small, dog_large = dog_filter(image, radius_yx)
    h_img, w_img = image.shape
    frame_all_spots = all_spots[all_spots["frame"] == frame_nr]

    mpp = MPP_BY_IMAGE_WIDTH_PX.get(w_img)
    if mpp is None:
        print(f"  [UEBERSPRUNGEN] {tif_path.name}: Bildbreite {w_img} px hat keine bekannte "
              f"mpp-Kalibrierung (siehe MPP_BY_IMAGE_WIDTH_PX)")
        return []

    tail_len_px = BACKGROUND_TAIL_FACTOR * (2 * SIGNAL_HALF_WIDTH_PX + 1)
    # Feste, spot-unabhaengige Offsets des Linienscans (Signal + Anhang) --
    # auch fuer die Isolationspruefung verwendet, siehe _is_isolated().
    signal_offsets_px = np.arange(-SIGNAL_HALF_WIDTH_PX, SIGNAL_HALF_WIDTH_PX + 1)
    tail_offsets_px = np.arange(SIGNAL_HALF_WIDTH_PX + 1, SIGNAL_HALF_WIDTH_PX + 1 + tail_len_px)
    offsets_px = np.concatenate([signal_offsets_px, tail_offsets_px])

    results = []
    for _, spot in chosen.iterrows():
        if len(results) >= n_particles:
            break
        xc, yc = spot["x_px"], spot["y_px"]
        r_px = spot["radius_px"] if np.isfinite(spot["radius_px"]) and spot["radius_px"] > 0 else 2.0
        inner_r, outer_r = INNER_RADIUS_FACTOR * r_px, OUTER_RADIUS_FACTOR * r_px

        half = int(np.ceil(outer_r)) + 2
        xi, yi = int(round(xc)), int(round(yc))
        if xi - half < 0 or yi - half < 0 or xi + half >= w_img or yi + half >= h_img:
            print(f"  [UEBERSPRUNGEN] {settings_xml.name} Spot {int(spot['id'])}: zu nah am Bildrand")
            continue
        if not _is_isolated(int(spot["id"]), xi, yi, frame_all_spots, offsets_px, ISOLATION_ROW_TOLERANCE_PX):
            print(f"  [UEBERSPRUNGEN] {settings_xml.name} Spot {int(spot['id'])}: "
                  f"nicht isoliert (Nachbarpartikel im Linienscan-Bereich)")
            continue

        yy, xx = np.mgrid[yi - half:yi + half + 1, xi - half:xi + half + 1]
        annulus_mask = (np.hypot(xx - xc, yy - yc) >= inner_r) & (np.hypot(xx - xc, yy - yc) <= outer_r)

        others = frame_all_spots[frame_all_spots["id"] != spot["id"]]
        for _, other in others.iterrows():
            other_r = other["radius_px"] if np.isfinite(other["radius_px"]) and other["radius_px"] > 0 else r_px
            annulus_mask &= np.hypot(xx - other["x_px"], yy - other["y_px"]) >= other_r

        crop = image[yi - half:yi + half + 1, xi - half:xi + half + 1]
        bg_pixels = crop[annulus_mask]
        if bg_pixels.size < 5:
            print(f"  [UEBERSPRUNGEN] {settings_xml.name} Spot {int(spot['id'])}: "
                  f"zu wenige Hintergrund-Pixel ({bg_pixels.size})")
            continue
        mu_bg, sigma_bg = float(bg_pixels.mean()), float(bg_pixels.std(ddof=1))

        # Signal: Linienscan durchs Zentrum. Hintergrund-Anhang: derselbe
        # Linienscan, direkt im Anschluss fortgesetzt (BACKGROUND_TAIL_FACTOR
        # x so lang wie der Signalbereich), an der Bildkante geclippt.
        # Truncate the tail instead of duplicating edge pixels.
        valid_offsets = offsets_px[(xi + offsets_px >= 0) & (xi + offsets_px < w_img)]
        if len(valid_offsets) < len(signal_offsets_px) or valid_offsets[0] != offsets_px[0]:
            continue
        xs = xi + valid_offsets
        profile = image[yi, xs]
        dog_profile = dog_image[yi, xs]
        dog_bg = dog_image[yi - half:yi + half + 1, xi - half:xi + half + 1][annulus_mask]
        dog_mu_bg, dog_sigma_bg = float(dog_bg.mean()), float(dog_bg.std(ddof=1))
        dog_peak_idx = int(np.argmax(dog_profile[:len(signal_offsets_px)]))
        dog_peak = float(dog_profile[dog_peak_idx])
        dog_snr = (dog_peak - dog_mu_bg) / dog_sigma_bg if dog_sigma_bg > 0 else np.nan
        signal_profile = profile[:len(signal_offsets_px)]

        peak_idx = int(np.argmax(signal_profile))
        peak = float(signal_profile[peak_idx])
        snr = (peak - mu_bg) / sigma_bg if sigma_bg > 0 else np.nan

        results.append({
            "spot_id": int(spot["id"]), "frame": frame_nr,
            "n_signal": len(signal_offsets_px),
            "offsets_px": valid_offsets, "offsets_nm": valid_offsets * mpp * 1000.0,
            "peak_offset_nm": float(signal_offsets_px[peak_idx] * mpp * 1000.0),
            "profile": profile, "mu_bg": mu_bg, "sigma_bg": sigma_bg, "snr": snr,
            "peak": peak, "signal": peak - mu_bg,
            "dog_profile": dog_profile, "dog_mu_bg": dog_mu_bg, "dog_sigma_bg": dog_sigma_bg,
            "dog_peak": dog_peak, "dog_signal": dog_peak - dog_mu_bg, "dog_snr": dog_snr,
            "dog_peak_offset_nm": float(signal_offsets_px[dog_peak_idx] * mpp * 1000.0),
            "snr_gain": dog_snr / snr if np.isfinite(snr) and snr > 0 else np.nan,
            "snr_change_percent": 100 * (dog_snr / snr - 1) if np.isfinite(snr) and snr > 0 else np.nan,
            "dog_snr_at_raw_peak": (float(dog_profile[peak_idx]) - dog_mu_bg) / dog_sigma_bg if dog_sigma_bg > 0 else np.nan,
            "source_tiff": str(tif_path), "settings_xml": str(settings_xml),
            "original_detector": detector_tag.attrib.get("DETECTOR_NAME", "unknown"),
            "median_prefilter_applied": False,
            "dog_sigma_small_y_px": dog_small[0], "dog_sigma_small_x_px": dog_small[1],
            "dog_sigma_large_y_px": dog_large[0], "dog_sigma_large_x_px": dog_large[1],
            "n_background_pixels": int(bg_pixels.size),
            "radius_px": r_px, "radius_nm": r_px * mpp * 1000.0, "mpp": mpp,
        })

    if len(results) < n_particles:
        print(f"  Warnung: nur {len(results)}/{n_particles} Spots aus {tracks_xml.name} auswertbar")
    return results


def particles_by_condition_for_size(size_nm: float, n_per_condition: int) -> dict[str, list[dict]]:
    """Fuer eine Groesse: pro verfuegbarem Fall (siehe FILES) bis zu
    n_per_condition isolierte Partikel -- die volle Quote je Fall, nicht
    ueber die Faelle aufgeteilt. Fehlende (Fall, Groesse)-Kombinationen
    werden uebersprungen (siehe FILES-Kommentare). Jedes Ergebnis traegt
    ein eindeutiges particle_label ("{Fall} #{n}")."""
    conditions = sorted(c for (c, s) in FILES if s == size_nm)
    if not conditions:
        print(f"  [UEBERSPRUNGEN] {size_nm:.0f} nm: keine Datenquelle in FILES")
        return {}

    by_condition: dict[str, list[dict]] = {}
    for condition in conditions:
        tracks_xml = Path(FILES[(condition, size_nm)])
        if not tracks_xml.exists():
            print(f"  [UEBERSPRUNGEN] {condition} {size_nm:.0f} nm: {tracks_xml} nicht gefunden")
            continue
        results = analyze_file(tracks_xml, n_per_condition)
        for i, r in enumerate(results, start=1):
            r["condition"] = condition
            r["particle_size_nm"] = size_nm
            r["particle_label"] = f"{condition} #{i}"
            print(f"  {size_nm:>6.0f} nm  {r['particle_label']:<16} Frame {r['frame']:>5}  "
                  f"Spot {r['spot_id']:>9}  SNR = {r['snr']:.1f}")
        if results:
            by_condition[condition] = results
        if len(results) < n_per_condition:
            print(f"  Warnung: nur {len(results)}/{n_per_condition} isolierte Partikel fuer "
                  f"{condition} {size_nm:.0f} nm gefunden")
    return by_condition


def main(output_dir=OUTPUT_DIR, open_output=True) -> None:
    random.seed(SEED)
    # ── Alle Groessen auswerten ─────────────────────────────────────────────────
    sizes = sorted({s for (_, s) in FILES})
    by_condition_by_size = {
        size_nm: particles_by_condition_for_size(size_nm, N_PARTICLES_PER_CONDITION) for size_nm in sizes
    }

    if not any(by_condition_by_size.values()):
        raise RuntimeError("Kein Spot aus keiner Datei auswertbar -- siehe [UEBERSPRUNGEN]-Meldungen oben.")

    # ── Eine Abbildung PRO PARTIKELGROESSE, Zeilen = Fall, Spalten = Partikel ────
    # Eine durchgehende Linie (Signal + Hintergrund-Anhang zusammen, kein
    # gestrichelter Abschnitt), der Peak (fuer die SNR-Berechnung verwendete
    # Maximalintensitaet) als Marker hervorgehoben. Nur isolierte Partikel
    # (siehe _is_isolated) werden gezeigt.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    with plt.rc_context(_RC):
        for size_nm in sizes:
            by_condition = by_condition_by_size[size_nm]
            if not by_condition:
                continue
            conditions = sorted(by_condition)
            nrows = len(conditions)
            ncols = N_PARTICLES_PER_CONDITION

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2.5),
                                      constrained_layout=True, squeeze=False)

            for row, condition in enumerate(conditions):
                particles = by_condition[condition]
                color = CONDITION_COLORS.get(condition, "#808080")
                for col in range(ncols):
                    ax = axes[row, col]
                    if col >= len(particles):
                        ax.set_visible(False)
                        continue
                    r = particles[col]
                    normalized = (
                        (r["profile"] - r["mu_bg"]) / r["sigma_bg"] if r["sigma_bg"] > 0 else r["profile"] * np.nan
                    )

                    ax.plot(r["offsets_nm"], normalized, color=color, linewidth=1.6, alpha=0.9)
                    # Peak-Y-Wert == SNR per Definition: (Peak - mu_bg) / sigma_bg.
                    ax.scatter([r["peak_offset_nm"]], [r["snr"]], color=color, s=32, zorder=5,
                               edgecolor="black", linewidth=0.6, marker="o")
                    dog_normalized = (r["dog_profile"] - r["dog_mu_bg"]) / r["dog_sigma_bg"] if r["dog_sigma_bg"] > 0 else r["dog_profile"] * np.nan
                    ax.lines[-1].set_label("Raw")
                    ax.plot(r["offsets_nm"], dog_normalized, color="#e69f00", linewidth=1.3, label="DoG")
                    ax.scatter([r["dog_peak_offset_nm"]], [r["dog_snr"]], color="#e69f00", s=24, marker="^", zorder=5)
                    ax.legend(loc="upper right", fontsize=6)
                    ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
                    ax.set_title(f"{r['particle_label']} | raw {r['snr']:.1f} / DoG {r['dog_snr']:.1f}\nSNR ratio {r['snr_gain']:.2f}x ({r['snr_change_percent']:+.0f}%)", fontsize=8)
                    if row == nrows - 1:
                        ax.set_xlabel("Position relativ\nzum Zentrum (nm)", fontsize=8)
                    if col == 0:
                        ax.set_ylabel(r"$(I - \mu_{bg}) / \sigma_{bg}$", fontsize=8)

            fig.suptitle(f"{size_nm:.0f} nm", fontsize=11)
            output_path = Path(output_dir) / f"snr_signal_profiles_{size_nm:.0f}nm.png"
            fig.savefig(output_path, dpi=600, bbox_inches="tight")
            plt.close(fig)
            output_paths.append(output_path)

    all_results = [r for groups in by_condition_by_size.values() for particles in groups.values() for r in particles]
    arrays = {"profile", "dog_profile", "offsets_px", "offsets_nm"}
    metrics = pd.DataFrame([{k: v for k, v in r.items() if k not in arrays} for r in all_results])
    metrics.to_csv(Path(output_dir) / "snr_dog_particle_metrics.csv", index=False)
    rows = []
    for r in all_results:
        for i, offset in enumerate(r["offsets_px"]):
            rows.append(dict(condition=r["condition"], particle_size_nm=r["particle_size_nm"],
                spot_id=r["spot_id"], frame=r["frame"], offset_px=offset, offset_nm=r["offsets_nm"][i],
                region="signal" if i < r["n_signal"] else "background_tail",
                raw_intensity=r["profile"][i], dog_intensity=r["dog_profile"][i]))
    pd.DataFrame(rows).to_csv(Path(output_dir) / "snr_dog_profiles.csv", index=False)
    summary = metrics.groupby(["condition", "particle_size_nm"]).agg(
        n=("spot_id", "size"), raw_snr_median=("snr", "median"), dog_snr_median=("dog_snr", "median"),
        paired_gain_median=("snr_gain", "median"), paired_change_percent_median=("snr_change_percent", "median"))
    summary.to_csv(Path(output_dir) / "snr_dog_summary.csv")
    print(summary.to_string())
    with plt.rc_context(_RC):
        for size_nm, groups in by_condition_by_size.items():
            if not groups:
                continue
            fig, axes = plt.subplots(2 * len(groups), N_PARTICLES_PER_CONDITION,
                figsize=(14, 3.6 * len(groups)), squeeze=False, constrained_layout=True)
            for group_idx, (condition, particles) in enumerate(sorted(groups.items())):
                for col in range(N_PARTICLES_PER_CONDITION):
                    for stage_idx, stage in enumerate(("raw", "dog")):
                        ax = axes[2 * group_idx + stage_idx, col]
                        if col >= len(particles):
                            ax.set_visible(False)
                            continue
                        r = particles[col]
                        prefix = "" if stage == "raw" else "dog_"
                        color = CONDITION_COLORS[condition] if stage == "raw" else "#e69f00"
                        ax.plot(r["offsets_nm"], r[prefix + "profile"], color=color, linewidth=1.2)
                        ax.axhline(r[prefix + "mu_bg"], color="gray", linestyle=":", linewidth=0.7)
                        ax.scatter([r[prefix + "peak_offset_nm"]], [r[prefix + "peak"]], color=color, s=16)
                        ax.set_title(f"{r['particle_label']} | {stage.upper()}", fontsize=8)
                        if col == 0:
                            ax.set_ylabel("Raw intensity (ADU)" if stage == "raw" else "DoG response (ADU)", fontsize=8)
                        if group_idx == len(groups) - 1 and stage == "dog":
                            ax.set_xlabel("Offset from center (nm)", fontsize=8)
            fig.suptitle(f"{size_nm:.0f} nm | absolute intensity before / after DoG", fontsize=11)
            out = Path(output_dir) / f"snr_absolute_profiles_{size_nm:.0f}nm.png"
            fig.savefig(out, dpi=600, bbox_inches="tight")
            plt.close(fig)
            output_paths.append(out)

    # ── Zusammenfassung ──────────────────────────────────────────────────────────
    total = sum(len(v) for by_condition in by_condition_by_size.values() for v in by_condition.values())
    print(f"\n{total} isolierte Partikel ueber {len(sizes)} Partikelgroessen verwendet, "
          f"{len(output_paths)} Abbildungsdateien (eine pro Partikelgroesse):")
    for size_nm in sizes:
        for condition, particles in by_condition_by_size[size_nm].items():
            for r in particles:
                print(f"  {size_nm:>6.0f} nm  {condition:<12} {r['particle_label']:<16} "
                      f"Frame {r['frame']:>5}  Spot-ID {r['spot_id']}")

    for output_path in output_paths:
        print(f"Gespeichert: {output_path}")

    if output_paths and open_output:
        os.startfile(Path(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired raw / DoG signal and local SNR profiles")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    main(args.output_dir, not args.no_open)
