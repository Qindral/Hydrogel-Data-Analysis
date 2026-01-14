import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import pandas as pd

# falls du die Pixelgröße aus den FEI-Metadaten holen willst:
# from read_fei_sem_metadata import summarize_sem_metadata

# ============================================================================
# PARTIKEL-GRÖSSENPARAMETER (in Nanometern - werden automatisch in Pixel umgerechnet)
# ============================================================================
EXPECTED_DIAM_NM_MIN = 225  # Minimale erwartete Partikelgröße in nm
EXPECTED_DIAM_NM_MAX = 300  # Maximale erwartete Partikelgröße in nm

# ============================================================================
# FILTERPARAMETER
# ============================================================================
# GAUSSIAN_SIGMA wird basierend auf Pixelgröße automatisch angepasst (siehe Funktion)
# Manuelles Override möglich, wenn nicht None:
GAUSSIAN_SIGMA_OVERRIDE = None  # Setze auf z.B. 7 für festen Wert, None für automatisch

MIN_CIRCULARITY = 0.28
MAX_CIRCULARITY = 1
MIN_INTENSITYMAX = 0.0  # max Intensität mindestens

# Qualitätsfilter für Partikel-Fitting
MIN_QUALITY_SCORE = 0.15  # Minimale Qualität des Profils (0-1)
MIN_CENTER_QUALITY = 0.3  # Minimale Zentrierungsqualität (0-1)
# MAX_CENTER_OFFSET_PX wird basierend auf Pixelgröße automatisch angepasst
MAX_CENTER_OFFSET_NM = 50.0  # Maximale Abweichung des Zentrums in nm (wird zu Pixel umgerechnet)
MAX_RADIUS_CHANGE_FACTOR = 0.7  # Maximale relative Änderung des Radius (z.B. 0.7 = 70%)

# Faktoren für Bereichsberechnung (können angepasst werden)
MIN_AREA_FACTOR = 3.0  # Faktor für minimale Fläche: factor * π * (min_radius)^2
MAX_AREA_FACTOR = 4.0  # Faktor für maximale Fläche: factor * π * (max_radius)^2

# Debug-Modus: Zeigt an, warum Partikel herausgefiltert werden
DEBUG_FILTERING = True

OUTPUT_CSV = "segmented_particles_watershed.csv"

def sample_circle(image, cx, cy, r, n_theta=64):
    """
    Intensitätsprofil auf einem Kreis mit Radius r um (cx, cy) samplen.
    Bilineare Interpolation via ndi.map_coordinates.
    """
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    xs = cx + r * np.cos(theta)
    ys = cy + r * np.sin(theta)
    # Achtung: map_coordinates erwartet Reihenfolge (y, x)
    values = ndi.map_coordinates(image, [ys, xs], order=1, mode="reflect")
    return values
def export_diameter_histogram_um(diam_um, out_path):
    """
    diam_um : 1D-Array / Liste der Partikeldurchmesser in µm
    out_path : Pfad zur Textdatei, z.B. 'hist_partikel.txt'

    - verwendet fixe Bins (in µm, unten definiert)
    - speichert 'bin_min;bin_max;count' mit Komma als Dezimaltrennzeichen
    """

    # --- deine vorgegebenen Bins in µm (mit Punkt als Dezimalpunkt) ---
    bins_um = np.array([
        2.09474E-4, 2.27147E-4, 2.4631E-4, 2.6709E-4, 2.89624E-4,
        3.14058E-4, 3.40554E-4, 3.69285E-4, 4.00440E-4, 4.34224E-4,
        4.70857E-4, 5.10582E-4, 5.53657E-4, 6.00367E-4, 6.51018E-4,
        7.05941E-4, 7.65499E-4, 8.30081E-4, 9.00112E-4, 9.76050E-4,
        0.00106, 0.00115, 0.00124, 0.00135, 0.00146, 0.00159, 0.00172,
        0.00187, 0.00202, 0.00219, 0.00238, 0.00258, 0.00280, 0.00303,
        0.00329, 0.00357, 0.00387, 0.00419, 0.00455, 0.00493, 0.00535,
        0.00580, 0.00629, 0.00682, 0.00739, 0.00802, 0.00869, 0.00943,
        0.01022, 0.01109, 0.01202, 0.01303, 0.01413, 0.01533, 0.01662,
        0.01802, 0.01954, 0.02119, 0.02298, 0.02492, 0.02702, 0.02930,
        0.03177, 0.03445, 0.03736, 0.04051, 0.04393, 0.04763, 0.05165,
        0.05601, 0.06074, 0.06586, 0.07142, 0.07744, 0.08397, 0.09106,
        0.09874, 0.10707, 0.11610, 0.12590, 0.13652, 0.14804, 0.16053,
        0.17407, 0.18876, 0.20468, 0.22195, 0.24068, 0.26098, 0.28300,
        0.30687, 0.33276, 0.36084, 0.39128, 0.42429, 0.46009, 0.49890,
        0.54099, 0.58663, 0.63613, 0.68979, 0.74799, 0.81109, 0.87952,
        0.95372, 1.03418, 1.12143, 1.21604, 1.31864, 1.42989, 1.55052,
        1.68133, 1.82318, 1.97699, 2.14378, 2.32464, 2.52077, 2.73343,
        2.96404, 3.21411, 3.48527, 3.77930, 4.09815, 4.44389, 4.81881,
        5.22535, 5.66619, 6.14423, 6.66259, 7.22469, 7.83420, 8.49514,
        9.21184, 9.98901, 10.83174, 11.74557, 12.73650, 13.81103,
        14.97621, 16.23969, 17.60977, 19.09543
    ])

    diam_um = np.asarray(diam_um, dtype=float)

    # Histogramm
    counts, edges = np.histogram(diam_um, bins=bins_um)

    # Textzeilen vorbereiten (Deutsch-Format: Komma, Semikolon)
    lines = []
    header = "bin_min_um;bin_max_um;count"
    lines.append(header)

    for i, c in enumerate(counts):
        bmin = edges[i]
        bmax = edges[i + 1]

        # in String mit Dezimal-Komma umwandeln
        s_min = f"{bmin:.6g}".replace(".", ",")
        s_max = f"{bmax:.6g}".replace(".", ",")
        line = f"{s_min};{s_max};{int(c)}"
        lines.append(line)

    # Datei schreiben
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Histogramm mit {len(counts)} Bins nach {out_path} geschrieben.")

def refine_radius(image,
                  cx, cy,
                  r_init,
                  factor_inner=0.6,
                  factor_outer=1.4,
                  n_r=40,
                  n_theta=64):
    """
    Lokale Radius-Optimierung um r_init herum.

    - nimmt an: Partikel heller als Hintergrund
    - sucht den Radius mit maximalem negativen dI/dr (stärkster Abfall)
    """
    r_min = r_init * factor_inner
    r_max = r_init * factor_outer
    radii = np.linspace(r_min, r_max, n_r)

    mean_I = []
    for r in radii:
        vals = sample_circle(image, cx, cy, r, n_theta=n_theta)
        mean_I.append(np.mean(vals))
    mean_I = np.array(mean_I)

    # radiale Ableitung dI/dr
    dI = np.diff(mean_I) / np.diff(radii)

    # stärkster negativer Gradient = Kante (hell -> dunkel)
    idx = np.argmin(dI)

    d_mean = (max(mean_I) + min(mean_I))/2
    idx_mean =np.where(mean_I<d_mean)[0]
    #print(idx, mean_I,d_mean,idx_mean)
    idx = max(idx, idx_mean[0])  # sicherstellen, dass Kante außerhalb des Mittelwerts liegt
    #print(idx)
    r_edge = 0.5 * (radii[idx] + radii[idx+1])

    return r_edge, radii, mean_I

def circularity(area, perimeter, eps=1e-9):
    return 4.0 * np.pi * area / (perimeter**2 + eps)

def refine_center(image, cx_init, cy_init, r_estimate, search_radius=3, n_theta=64):
    """
    Findet das optimale Zentrum eines Partikels durch Minimierung der Varianz
    im radialen Intensitätsprofil.
    
    Für ein perfekt zentriertes Partikel sollte das radiale Profil bei allen
    Winkeln ähnlich sein. Bei falscher Zentrierung variiert die Intensität
    je nach Winkelrichtung.
    """
    # Grid-Suche für bessere Robustheit
    best_var = np.inf
    best_cx = cx_init
    best_cy = cy_init
    
    # Durchsuche ein kleines Gitter um das initiale Zentrum
    search_steps = 7  # ungerade Zahl für Zentrum
    step_size = search_radius / (search_steps // 2)
    
    for dx in np.linspace(-search_radius, search_radius, search_steps):
        for dy in np.linspace(-search_radius, search_radius, search_steps):
            cx_try = cx_init + dx
            cy_try = cy_init + dy
            
            # Prüfe ob innerhalb des Bildes
            if (cx_try < r_estimate or cx_try > image.shape[1] - r_estimate or
                cy_try < r_estimate or cy_try > image.shape[0] - r_estimate):
                continue
            
            # Sample auf dem geschätzten Radius
            try:
                vals = sample_circle(image, cx_try, cy_try, r_estimate, n_theta=n_theta)
                # Berechne Varianz: je niedriger die Varianz, desto symmetrischer
                var = np.var(vals)
                if var < best_var:
                    best_var = var
                    best_cx = cx_try
                    best_cy = cy_try
            except:
                continue
    
    return best_cx, best_cy

def assess_profile_quality(radii, intensities, r_refined):
    """
    Bewertet die Qualität eines radialen Intensitätsprofils.
    
    Returns:
        dict mit verschiedenen Qualitätsmetriken
    """
    # 1. Gradient-Sharpness am refinierten Radius
    # Finde den Index des refinierten Radius
    idx_refined = np.argmin(np.abs(radii - r_refined))
    if idx_refined > 0 and idx_refined < len(radii) - 1:
        # Gradient am Edge
        dI_dr = (intensities[idx_refined+1] - intensities[idx_refined-1]) / (radii[idx_refined+1] - radii[idx_refined-1])
        gradient_sharpness = abs(dI_dr)
    else:
        gradient_sharpness = 0.0
    
    # 2. Profil-Symmetrie: Varianz der Intensitäten vor dem Peak
    # (sollten relativ konstant sein für ein gutes Partikel)
    idx_before_edge = max(0, idx_refined - 5)
    if idx_before_edge < idx_refined:
        intensities_inner = intensities[idx_before_edge:idx_refined]
        profile_symmetry = 1.0 / (1.0 + np.std(intensities_inner) / (np.mean(intensities_inner) + 1e-9))
    else:
        profile_symmetry = 0.0
    
    # 3. Konsistenz: Wie gut stimmt der refinine Radius mit der initialen Schätzung überein?
    # (wird extern berechnet, da wir den initialen Radius nicht haben)
    
    # 4. Signal-zu-Rausch-Verhältnis am Edge
    # Hoher Kontrast am Edge = gutes Partikel
    if idx_refined < len(intensities) - 3:
        intensity_before = np.mean(intensities[max(0, idx_refined-2):idx_refined])
        intensity_after = np.mean(intensities[idx_refined:min(len(intensities), idx_refined+3)])
        edge_contrast = abs(intensity_before - intensity_after) / (np.max(intensities) + 1e-9)
    else:
        edge_contrast = 0.0
    
    # Kombinierter Qualitäts-Score (0-1, höher ist besser)
    quality_score = 0.4 * min(profile_symmetry, 1.0) + 0.3 * min(gradient_sharpness / np.std(intensities), 1.0) + 0.3 * min(edge_contrast * 5, 1.0)
    
    return {
        'quality_score': quality_score,
        'gradient_sharpness': gradient_sharpness,
        'profile_symmetry': profile_symmetry,
        'edge_contrast': edge_contrast
    }

def check_center_quality(image, cx, cy, r_refined, n_theta=64):
    """
    Prüft die Qualität der Zentrierung durch Analyse der Winkelvarianz.
    
    Ein gut zentriertes Partikel sollte ähnliche Intensitätsprofile
    in allen Richtungen haben.
    """
    # Sample auf verschiedenen Winkeln
    vals = sample_circle(image, cx, cy, r_refined * 0.9, n_theta=n_theta)
    
    # Berechne Varianz: niedrige Varianz = gute Zentrierung
    angular_variance = np.var(vals)
    angular_std = np.std(vals)
    mean_val = np.mean(vals)
    
    # Normalisierte Varianz (Coefficient of Variation)
    cv = angular_std / (mean_val + 1e-9)
    
    # Score: je niedriger CV, desto besser die Zentrierung
    center_quality = 1.0 / (1.0 + cv * 10)
    
    return {
        'center_quality': center_quality,
        'angular_variance': angular_variance,
        'angular_cv': cv
    }
def summarize_sem_metadata(tif_path: str | Path) -> dict:
    """
    Lies ein FEI-SEM-TIF mit Pillow ein und fasse relevante Metadaten zusammen.
    Gibt ein Dictionary mit 'raw' (voll strukturierter FEI-Block)
    und 'summary' (wichtige physikalische Größen) zurück.
    """
    tif_path = Path(tif_path)
    img = Image.open(tif_path)
    tags = img.tag_v2

    # Grundlegende TIFF-Infos
    width, height = img.size
    mode = img.mode

    # FEI-Textblock (tag 34682)
    fei_raw = tags.get(34682, None)
    if fei_raw is None:
        raise RuntimeError("Kein FEI-Metadatenblock (Tag 34682) gefunden.")

    if isinstance(fei_raw, bytes):
        fei_text = fei_raw.decode("utf-8", errors="ignore")
    else:
        fei_text = str(fei_raw)

    sections = parse_fei_text_block(fei_text)

    # Häufig interessante Sektionen
    scan_sec = sections.get("Scan", {})
    ebeam_sec = sections.get("EBeam", {})
    beam_sec = sections.get("Beam", {})
    image_sec = sections.get("Image", {})
    system_sec = sections.get("System", {})

    # FEI arbeitet hier typischerweise in Metern
    # PixelWidth / PixelHeight / HFW / VFW etc. scheinen in m zu sein.
    px_width_m = safe_float(scan_sec, "PixelWidth")
    px_height_m = safe_float(scan_sec, "PixelHeight")

    hfw_m = safe_float(scan_sec, "HorFieldsize") or safe_float(ebeam_sec, "HFW")
    vfw_m = safe_float(scan_sec, "VerFieldsize") or safe_float(ebeam_sec, "VFW")

    dwell_s = safe_float(scan_sec, "Dwelltime") or safe_float(scan_sec, "Dwell")
    hv_V = safe_float(ebeam_sec, "HV") or safe_float(beam_sec, "HV")
    wd_m = safe_float(ebeam_sec, "WD")

    # Einheiten-Umrechnungen
    px_width_nm = px_width_m * 1e9 if px_width_m is not None else None
    px_height_nm = px_height_m * 1e9 if px_height_m is not None else None

    hfw_um = hfw_m * 1e6 if hfw_m is not None else None
    vfw_um = vfw_m * 1e6 if vfw_m is not None else None

    hv_kV = hv_V / 1000.0 if hv_V is not None else None
    wd_mm = wd_m * 1e3 if wd_m is not None else None

    summary = {
        "file": str(tif_path),
        "image_size_px": (width, height),
        "image_mode": mode,

        # Pixelgröße
        "pixel_width_m": px_width_m,
        "pixel_height_m": px_height_m,
        "pixel_width_nm": px_width_nm,
        "pixel_height_nm": px_height_nm,

        # Field of View
        "HFW_m": hfw_m,
        "VFW_m": vfw_m,
        "HFW_um": hfw_um,
        "VFW_um": vfw_um,

        # Elektronenstrahl-Parameter
        "HV_V": hv_V,
        "HV_kV": hv_kV,
        "WD_m": wd_m,
        "WD_mm": wd_mm,

        # Scanparameter
        "dwell_time_s": dwell_s,
        "frame_time_s": safe_float(scan_sec, "FrameTime"),

        # Sonstige interessante Angaben
        "magnification_correction": beam_sec.get("MagnificationCorrection", None),
        "HFW_from_EBeam_m": safe_float(ebeam_sec, "HFW"),
        "VFW_from_EBeam_m": safe_float(ebeam_sec, "VFW"),
        "scan_rotation_rad": safe_float(ebeam_sec, "ScanRotation"),
        "stage_X_m": safe_float(ebeam_sec, "StageX"),
        "stage_Y_m": safe_float(ebeam_sec, "StageY"),
        "stage_Z_m": safe_float(ebeam_sec, "StageZ"),
        "software_version": system_sec.get("Software", None),
        "microscope_type": system_sec.get("Type", None),
    }

    return {
        "raw_sections": sections,   # vollständig geparst, falls du später noch was brauchst
        "summary": summary,         # komprimierte, physikalisch relevante Infos
    }

def parse_fei_text_block(text: str) -> dict:
    """
    Parsen des FEI-Textblocks aus Tag 34682.
    Struktur:
        [Section]
        key=value
        ...

    Rückgabe:
        { "Section": { "key": "value", ... }, ... }
    """
    sections = {}
    current_section = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Neue Sektion
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            sections[current_section] = {}
            continue

        # key=value Zeilen
        if "=" in line and current_section is not None:
            key, value = line.split("=", 1)
            sections[current_section][key.strip()] = value.strip()

    return sections


def safe_float(d: dict, key: str):
    """Hilfsfunktion: float-Wert aus dict, oder None."""
    val = d.get(key, None)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None
def segment_particles_watershed(tif_path):
    tif_path = Path(tif_path)

    # --- Pixelgröße aus FEI-Metadaten (falls vorhanden) -------------
    meta = summarize_sem_metadata(tif_path)
    px_nm = 0.5*(meta["summary"]["pixel_width_nm"] +
                 meta["summary"]["pixel_height_nm"])
    
    if px_nm is None or px_nm <= 0:
        raise ValueError("Pixelgröße konnte nicht aus Metadaten extrahiert werden. "
                        "Bitte prüfe die TIFF-Metadaten.")
    
    print(f"\n{'='*60}")
    print(f"BILDINFORMATIONEN:")
    print(f"{'='*60}")
    print(f"Datei: {tif_path.name}")
    print(f"Pixelgröße: {px_nm:.4f} nm/pixel")
    print(f"Erwartete Partikelgröße: {EXPECTED_DIAM_NM_MIN:.0f} - {EXPECTED_DIAM_NM_MAX:.0f} nm")
    print(f"{'='*60}\n")
    
    # --- Automatische Berechnung der Pixel-basierten Parameter aus nm-Werten ---
    # Minimale und maximale Partikelradien in Pixel
    min_radius_nm = EXPECTED_DIAM_NM_MIN / 2.0
    max_radius_nm = EXPECTED_DIAM_NM_MAX / 2.0
    min_radius_px = min_radius_nm / px_nm
    max_radius_px = max_radius_nm / px_nm
    
    # Flächenberechnung in Pixel
    MIN_AREA_PX = int(MIN_AREA_FACTOR * np.pi * min_radius_px**2)
    MAX_AREA_PX = int(MAX_AREA_FACTOR * np.pi * max_radius_px**2)
    
    # Gaussian Sigma: sollte etwa 5-10% des minimalen Partikeldurchmessers entsprechen
    # Für bessere Anpassung: ca. 2-3% des Durchmessers oder min. 2-3 Pixel
    if GAUSSIAN_SIGMA_OVERRIDE is not None:
        GAUSSIAN_SIGMA = GAUSSIAN_SIGMA_OVERRIDE
    else:
        # Automatisch basierend auf Partikelgröße: ca. 2-3% des min. Durchmessers
        gaussian_sigma_nm = EXPECTED_DIAM_NM_MIN * 0.025  # 2.5% des Durchmessers
        GAUSSIAN_SIGMA = max(gaussian_sigma_nm / px_nm, 1.5)  # Mindestens 1.5 Pixel
        GAUSSIAN_SIGMA = min(GAUSSIAN_SIGMA, 10.0)  # Maximal 10 Pixel
    
    # MAX_CENTER_OFFSET in Pixel aus nm-Wert berechnen
    MAX_CENTER_OFFSET_PX = MAX_CENTER_OFFSET_NM / px_nm
    
    print(f"Berechnete Parameter (in Pixel):")
    print(f"  Min. Radius: {min_radius_px:.2f} px")
    print(f"  Max. Radius: {max_radius_px:.2f} px")
    print(f"  Min. Fläche: {MIN_AREA_PX:.0f} px²")
    print(f"  Max. Fläche: {MAX_AREA_PX:.0f} px²")
    print(f"  Gaussian Sigma: {GAUSSIAN_SIGMA:.2f} px")
    print(f"  Max. Zentrums-Offset: {MAX_CENTER_OFFSET_PX:.2f} px")
    print()

    # --- Bild laden & in Graustufen -------------------------------
    pil_img = Image.open(tif_path)
    if np.max(pil_img) < 255:
        img_gray = pil_img.convert("L")
    else:
        img_gray = pil_img
    image = np.array(img_gray, dtype=float)

    # --- 1) Denoising ---------------------------------------------
    smoothed = gaussian(image, sigma=GAUSSIAN_SIGMA)

    # --- 2) Threshold + Maske --------------------------------------
    thr = threshold_otsu(smoothed)
    mask = smoothed > thr

    # kleine Löcher schließen und Noise entfernen
    mask = closing(mask, footprint=disk(1))
    mask = remove_small_objects(mask, min_size=MIN_AREA_PX)

    # --- 3) Distance-Transform -------------------------------------
    distance = ndi.distance_transform_edt(mask)

    # --- 4) Peaks im Distance-Map (Seeds) --------------------------
    # min_distance sollte etwa 50% des minimalen Partikelradius sein
    # Verhindert, dass zwei Partikel zu nah beieinander als einer erkannt werden
    min_dist = max(int(min_radius_px * 0.5), 2)

    coords = peak_local_max(
        distance,
        min_distance=min_dist,
        labels=mask
    )

    markers = np.zeros_like(image, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    # --- 5) Watershed-Segmentierung -------------------------------
    labels_ws = watershed(-distance, markers, mask=mask)

    # --- 6) Analyse der Regionen ----------------------------------
    props = regionprops(labels_ws, intensity_image=image)
    s = 0
    results = []
    # Liste zum Sammeln von Profilen für kleine Partikel (< 120 nm)
    small_particle_profiles = []
    
    # Statistiken für Filterung
    filter_stats = {
        'total_detected': 0,
        'filtered_area': 0,
        'filtered_circularity': 0,
        'filtered_center_offset': 0,
        'filtered_radius_change': 0,
        'filtered_quality_score': 0,
        'filtered_center_quality': 0,
        'accepted': 0
    }
    
    for r in props:
        filter_stats['total_detected'] += 1
        area = r.area
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            filter_stats['filtered_area'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen Bereich (Area: {area:.1f}, erwartet: {MIN_AREA_PX:.1f}-{MAX_AREA_PX:.1f})")
            continue

        circ = circularity(area, r.perimeter)
        if circ < MIN_CIRCULARITY:
            filter_stats['filtered_circularity'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen Zirkularität ({circ:.3f} < {MIN_CIRCULARITY:.3f})")
            continue

        eq_diam_px = r.equivalent_diameter
        cy_init, cx_init = r.centroid
        
        # --- Zentrum-Refinement: Finde das optimale Zentrum vor Radius-Refinement ---
        r0_estimate = eq_diam_px / 2.0
        # Suchradius: entweder MAX_CENTER_OFFSET_PX oder 30% des geschätzten Radius, je nachdem was kleiner ist
        search_radius = min(MAX_CENTER_OFFSET_PX, r0_estimate * 0.3)
        cx_refined, cy_refined = refine_center(
            image, cx_init, cy_init, r0_estimate, 
            search_radius=search_radius, 
            n_theta=64
        )
        
        # Prüfe, ob Zentrum-Offset zu groß ist (möglicherweise schlechtes Partikel)
        center_offset = np.sqrt((cx_refined - cx_init)**2 + (cy_refined - cy_init)**2)
        if center_offset > MAX_CENTER_OFFSET_PX:
            filter_stats['filtered_center_offset'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen zu großem Zentrums-Offset ({center_offset:.2f} > {MAX_CENTER_OFFSET_PX:.2f} px)")
            continue  # Überspringe Partikel mit zu großem Zentrums-Offset
        
        # --- Radius-Refinement mit verbessertem Zentrum ---
        r_refined, rad_arr, prof = refine_radius(
            image, cx_refined, cy_refined, r0_estimate
        )
        radii, I = rad_arr, prof  # aus refine_radius zurückgegeben
        
        # Prüfe, ob Radius-Änderung zu groß ist (möglicherweise schlechtes Fitting)
        radius_change_factor = abs(r_refined - r0_estimate) / (r0_estimate + 1e-9)
        if radius_change_factor > MAX_RADIUS_CHANGE_FACTOR:
            filter_stats['filtered_radius_change'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen zu großer Radius-Änderung ({radius_change_factor:.2%} > {MAX_RADIUS_CHANGE_FACTOR:.2%})")
            continue  # Überspringe Partikel mit zu großer Radius-Änderung
        
        # --- Qualitätsbewertung ---
        profile_quality = assess_profile_quality(radii, I, r_refined)
        center_quality = check_center_quality(image, cx_refined, cy_refined, r_refined, n_theta=64)
        
        # Filtere Partikel mit schlechter Qualität
        if profile_quality['quality_score'] < MIN_QUALITY_SCORE:
            filter_stats['filtered_quality_score'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen niedrigem Quality Score ({profile_quality['quality_score']:.3f} < {MIN_QUALITY_SCORE:.3f})")
            continue  # Überspringe Partikel mit schlechter Profilqualität
            
        if center_quality['center_quality'] < MIN_CENTER_QUALITY:
            filter_stats['filtered_center_quality'] += 1
            if DEBUG_FILTERING:
                print(f"  Partikel {r.label}: Herausgefiltert wegen niedriger Zentrierungsqualität ({center_quality['center_quality']:.3f} < {MIN_CENTER_QUALITY:.3f})")
            continue  # Überspringe Partikel mit schlechter Zentrierung
        
        filter_stats['accepted'] += 1
        
        # Verwende die verfeinerten Koordinaten
        cx, cy = cx_refined, cy_refined
        
        # Umrechnung von Pixel zu nm
        radii_nm = radii * px_nm
        r_refined_nm = r_refined * px_nm
        eq_refined_nm = 2.0 * r_refined_nm  # Durchmesser in nm
        
        # Sammle Profile für kleine Partikel (< 120 nm) zum Overlay
        if eq_refined_nm < 300:
            small_particle_profiles.append({
                'radii_nm': radii_nm,
                'I': I,
                'r_refined_nm': r_refined_nm,
                'cx': cx,
                'cy': cy,
                'diameter_nm': eq_refined_nm,
                'label': r.label
            })
        else:
            # Detaillierte Plots für größere Partikel (>= 120 nm)
            if False :  # Nur für das erste große Partikel
                # Erstelle eine Figur mit zwei Subplots: Profil + Bildausschnitt
                fig = plt.figure(figsize=(14, 6))
                
                # Linker Subplot: Radiales Intensitätsprofil (in nm)
                ax1 = plt.subplot(1, 2, 1)
                ax1.plot(radii_nm, I, "-o", markersize=4, label="Intensitätsprofil")
                ax1.scatter([r_refined_nm], [np.interp(r_refined, radii, I)], 
                           color="red", s=100, zorder=5, label=f"Refinierter Radius: {r_refined_nm:.2f} nm")
                ax1.axvline(r_refined_nm, color="red", linestyle="--", alpha=0.5, linewidth=1)
                ax1.set_xlabel("Radius [nm]")
                ax1.set_ylabel("⟨I⟩ auf Kreis")
                ax1.set_title(f"Radiales Intensitätsprofil\nPosition: ({cx:.1f}, {cy:.1f}) px")
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Rechter Subplot: Bildausschnitt mit markierter Position
                ax2 = plt.subplot(1, 2, 2)
                
                # Definiere Ausschnitt um die Partikelposition (3x Radius in jede Richtung)
                crop_size = int(r_refined * 3)
                y_min = max(0, int(cy - crop_size))
                y_max = min(image.shape[0], int(cy + crop_size))
                x_min = max(0, int(cx - crop_size))
                x_max = min(image.shape[1], int(cx + crop_size))
                
                image_crop = image[y_min:y_max, x_min:x_max]
                ax2.imshow(image_crop, cmap="gray", interpolation="nearest")
                
                # Position relativ zum Ausschnitt
                cx_rel = cx - x_min
                cy_rel = cy - y_min
                
                # Zeichne den refinierten Kreis
                circle = plt.Circle((cx_rel, cy_rel), r_refined, fill=False, 
                                  color="red", linewidth=2, label=f"Radius: {r_refined_nm:.2f} nm")
                ax2.add_patch(circle)
                
                # Markiere den Mittelpunkt
                ax2.plot(cx_rel, cy_rel, "r+", markersize=15, markeredgewidth=2, 
                        label=f"Zentrum: ({cx:.1f}, {cy:.1f}) px")
                
                ax2.set_title(f"Bildausschnitt mit segmentiertem Kreis\nAbsolute Position: ({cx:.2f}, {cy:.2f}) px")
                ax2.legend(loc="upper right", fontsize=9)
                ax2.axis("off")
                
                # Zusätzliche Informationen als Text
                info_text = f"Radius: {r_refined_nm:.3f} nm ({r_refined:.2f} px)\n"
                info_text += f"Durchmesser: {eq_refined_nm:.3f} nm ({2*r_refined:.2f} px)\n"
                info_text += f"Position: ({cx:.2f}, {cy:.2f}) px"
                fig.text(0.5, 0.02, info_text, ha="center", va="bottom", 
                        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                
                plt.tight_layout()
                plt.show()
                
                s += 1
        
        eq_refined_px = 2.0 * r_refined
        eq_refined_nm = eq_refined_px * px_nm  # px_nm kommt aus deinen Metadaten

        results.append(
            dict(
                label=int(r.label),
                x_px=float(cx),
                y_px=float(cy),
                x_init_px=float(cx_init),  # Initiales Zentrum
                y_init_px=float(cy_init),
                center_offset_px=float(center_offset),
                area_px=float(area),
                circularity=float(circ),
                eq_diam_px=float(eq_diam_px),
                eq_diam_nm=float(eq_diam_px * px_nm),
                refined_diam_px=float(eq_refined_px),
                refined_diam_nm=float(eq_refined_nm),
                radius_change_factor=float(radius_change_factor),
                quality_score=float(profile_quality['quality_score']),
                gradient_sharpness=float(profile_quality['gradient_sharpness']),
                profile_symmetry=float(profile_quality['profile_symmetry']),
                edge_contrast=float(profile_quality['edge_contrast']),
                center_quality=float(center_quality['center_quality']),
                angular_variance=float(center_quality['angular_variance']),
                angular_cv=float(center_quality['angular_cv']),
                mean_intensity=float(r.mean_intensity),
                max_intensity=float(r.max_intensity),
            )
        )
    
    # --- Filterstatistik ausgeben ---
    print("\n" + "="*60)
    print("FILTERSTATISTIKEN:")
    print("="*60)
    print(f"Gesamt detektierte Partikel: {filter_stats['total_detected']}")
    print(f"  ✓ Akzeptiert: {filter_stats['accepted']}")
    print(f"  ✗ Herausgefiltert:")
    print(f"     - Bereich (Area): {filter_stats['filtered_area']}")
    print(f"     - Zirkularität: {filter_stats['filtered_circularity']}")
    print(f"     - Zentrums-Offset: {filter_stats['filtered_center_offset']}")
    print(f"     - Radius-Änderung: {filter_stats['filtered_radius_change']}")
    print(f"     - Quality Score: {filter_stats['filtered_quality_score']}")
    print(f"     - Zentrierungsqualität: {filter_stats['filtered_center_quality']}")
    print("="*60 + "\n")
    
    # --- Overlay-Plot für kleine Partikel (< 120 nm) ---
    if small_particle_profiles:
        fig_overlay = plt.figure(figsize=(14, 6))
        
        # Linker Subplot: Overlay aller radialen Intensitätsprofile
        ax1 = plt.subplot(1, 2, 1)
        
        # Farbschema für mehrere Partikel
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(small_particle_profiles), 10)))
        
        for idx, prof_data in enumerate(small_particle_profiles):
            color = colors[idx % len(colors)]
            label = f"Partikel {prof_data['label']}: {prof_data['diameter_nm']:.1f} nm"
            ax1.plot(prof_data['radii_nm'], prof_data['I'], "-o", markersize=3, 
                    alpha=0.7, color=color, label=label, linewidth=1.5)
            # Markiere den refinierten Radius für jedes Partikel
            # Finde den Index des refinierten Radius oder interpoliere
            r_refined_px = prof_data['r_refined_nm'] / px_nm
            radii_px = prof_data['radii_nm'] / px_nm
            I_at_refined = np.interp(r_refined_px, radii_px, prof_data['I'])
            ax1.scatter([prof_data['r_refined_nm']], [I_at_refined],
                       color=color, s=80, zorder=5, marker='x', linewidths=2)
            ax1.axvline(prof_data['r_refined_nm'], color=color, linestyle="--", 
                       alpha=0.4, linewidth=1)
        
        ax1.set_xlabel("Radius [nm]", fontsize=11)
        ax1.set_ylabel("⟨I⟩ auf Kreis", fontsize=11)
        ax1.set_title(f"Overlay: Radiale Intensitätsprofile\n{len(small_particle_profiles)} Partikel < 120 nm", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=8, ncol=1)
        
        # Rechter Subplot: Bild mit allen markierten kleinen Partikeln
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(image, cmap="gray", interpolation="nearest")
        
        # Zeichne alle kleinen Partikel mit ihren refinierten Kreisen
        for idx, prof_data in enumerate(small_particle_profiles):
            color = colors[idx % len(colors)]
            r_refined_px = prof_data['r_refined_nm'] / px_nm
            circle = plt.Circle((prof_data['cx'], prof_data['cy']), r_refined_px, 
                              fill=False, color=color, linewidth=1.5, alpha=0.8)
            ax2.add_patch(circle)
            # Markiere den Mittelpunkt
            ax2.plot(prof_data['cx'], prof_data['cy'], "+", color=color, 
                    markersize=10, markeredgewidth=2, alpha=0.9)
        
        ax2.set_title(f"Bild mit allen kleinen Partikeln markiert\n({len(small_particle_profiles)} Partikel < 120 nm)", fontsize=12)
        ax2.axis("off")
        
        # Zusammenfassungstext
        avg_diameter = np.mean([p['diameter_nm'] for p in small_particle_profiles])
        std_diameter = np.std([p['diameter_nm'] for p in small_particle_profiles])
        summary_text = f"Kleine Partikel (< 120 nm): {len(small_particle_profiles)}\n"
        summary_text += f"Durchschnittlicher Durchmesser: {avg_diameter:.2f} ± {std_diameter:.2f} nm"
        fig_overlay.text(0.5, 0.02, summary_text, ha="center", va="bottom", 
                        fontsize=10, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n{len(small_particle_profiles)} kleine Partikel (< 120 nm) gefunden und im Overlay-Plot dargestellt.")
    
    # --- 7) Filtern der Ergebnisse nach physikalisch sinnvollen Größen ---
    # Filteriere direkt die Liste `results` statt das DataFrame `df`
    filtered_results = [
        r for r in results
        if (r["circularity"] >= MIN_CIRCULARITY)
        and (r["circularity"] <= MAX_CIRCULARITY)
        and (r["max_intensity"] >= MIN_INTENSITYMAX)]

    if not results:
        print("Keine Partikel gefunden – Threshold/Parameter anpassen.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"{len(df)} Partikel gefunden. Ergebnisse in: {OUTPUT_CSV}")

    

    # Optional: als DataFrame speichern
    filtered_df = pd.DataFrame(filtered_results)
    filtered_csv = Path(OUTPUT_CSV).with_name("filtered_" + Path(OUTPUT_CSV).name)
    filtered_df.to_csv(filtered_csv, index=False)

    print(f"{len(filtered_results)} Partikel nach Filterung. Gefilterte Ergebnisse in: {filtered_csv}")

    # Für weitere Verarbeitung/Plotting die gefilterte Liste verwenden
    results = filtered_results
    # --- 8) Overlay + Histogram ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Binärmaske")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    for r in results:
        x = r["x_px"]
        y = r["y_px"]
        rad = r["eq_diam_px"] / 2.0
        circ_patch = plt.Circle((x, y), rad, fill=False, linewidth=1,color='red')
        axes[2].add_patch(circ_patch)
    axes[2].set_title("Segmentierte Partikel (Kreise = eq. Durchmesser)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", interpolation="nearest")

    for r in results:
        x = r["x_px"]
        y = r["y_px"]
        rad = r["refined_diam_px"] / 2.0   # statt eq_diam_px/2

        circ_patch = plt.Circle((x, y), radius=rad, fill=False, linewidth=1, color='blue'
                                )
        ax.add_patch(circ_patch)

    ax.set_title("Partikel mit lokal verfeinertem Radius (Kantenfit)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(5, 4))
    plt.hist(df["eq_diam_nm"], bins=30)
    plt.xlabel("Äquivalenter Durchmesser [nm]")
    plt.ylabel("Anzahl")
    plt.title("Durchmesserverteilung")
    plt.tight_layout()
    plt.show()


    # Durchmesser in µm ableiten
    diam_um = df["refined_diam_nm"] / 1000.0   # nm -> µm

    export_diameter_histogram_um(diam_um, "hist_partikel_um.txt")




if __name__ == "__main__":
    paths= r"Z:\Diffusion in Hydrogel Data\SEM Particles\20nm\Latex particle_T2_20nm_007.tif"
    paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\50nm\Latex particle_T2_50nm_005.tif"
    paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\200 nm\Latex particle_Cpad_200nm_001.tif"
    paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\200 nm\Latex particle_Cpad_200nm_003.tif"
    #paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\1000\1000nm_10kx_001.tif"
    #paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\1000\1000nm_20kx_003.tif"
    paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\100 nm\100nm_T1_03.tif"
    paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\100 nm\100nm_T1_04.tif"
    paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\100 nm\100nm_T2_03.tif"
    # paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\100 nm\100nm_T2_04.tif"
    paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\500 nm\500nm_T2_01.tif"
    paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\500 nm\500nm_T2_02.tif"

    segment_particles_watershed(paths)  # Pfad anpassen



