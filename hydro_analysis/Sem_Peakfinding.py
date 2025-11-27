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

EXPECTED_DIAM_NM_MIN = 33
EXPECTED_DIAM_NM_MAX = 233

MIN_AREA_PX = 3*(EXPECTED_DIAM_NM_MIN/2)**2     # zu kleine Flecken rauswerfen
MAX_AREA_PX = 4*(EXPECTED_DIAM_NM_MAX/2)**2     # zu große Flecken rauswerfen

#Filter
GAUSSIAN_SIGMA = 5
MIN_CIRCULARITY = 0.28
MAX_CIRCULARITY = 1
MIN_INTENSITYMAX = 0.0  # max Intensität mindestens

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
    # oder vorerst annehmen:
    #px_nm = 1.0  # Dummy, wenn du noch keine Skala hast

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
    # min_distance grob ~ erwarteter Radius in Pixel
    # wenn du px_nm kennst, kannst du das aus 10–40 nm ableiten
    approx_radius_px = (EXPECTED_DIAM_NM_MAX / 2.0) / px_nm
    min_dist = max(int(approx_radius_px * 0.5), 2)

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
    for r in props:
        area = r.area
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            continue

        circ = circularity(area, r.perimeter)
        if circ < MIN_CIRCULARITY:
            continue

        eq_diam_px = r.equivalent_diameter
        cy, cx = r.centroid

        # --- NEUER TEIL: lokaler Radius-Refinement ---
        r0 = eq_diam_px / 2.0
        r_refined, rad_arr, prof = refine_radius(
            image, cx, cy, r0
        )
        radii, I = rad_arr, prof  # aus refine_radius zurückgegeben
        # Plot des Profils zur Kontrolle (optional)
        if s == 0:  # nur für den ersten Partikel

            plt.plot(radii, I, "-o")
            plt.scatter([r_refined], [np.interp(r_refined, radii, I)], color="red", label="Refinierter Radius")

            plt.xlabel("Radius [px]")
            plt.ylabel("⟨I⟩ auf Kreis")
            plt.title("Radiales Intensitätsprofil eines Partikels")
            plt.show()
            s += 1
        eq_refined_px = 2.0 * r_refined
        eq_refined_nm = eq_refined_px * px_nm  # px_nm kommt aus deinen Metadaten

        results.append(
            dict(
                label=int(r.label),
                x_px=float(cx),
                y_px=float(cy),
                area_px=float(area),
                circularity=float(circ),
                eq_diam_px=float(eq_diam_px),
                eq_diam_nm=float(eq_diam_px * px_nm),
                refined_diam_px=float(eq_refined_px),
                refined_diam_nm=float(eq_refined_nm),
                mean_intensity=float(r.mean_intensity),
                max_intensity=float(r.max_intensity),
            )
        )
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


    segment_particles_watershed(paths)  # Pfad anpassen



