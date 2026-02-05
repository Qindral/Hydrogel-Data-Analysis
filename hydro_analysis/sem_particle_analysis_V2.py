import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from skimage.filters import difference_of_gaussians
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_filter


# WICHTIG:
# Entweder hier direkt deine summarize_sem_metadata-Funktion einfügen
# oder aus dem vorherigen Script importieren, z.B.:
# from read_fei_sem_metadata import summarize_sem_metadata

# ---------- Parameter ---------------------------------------------
# Erwartete Partikelgröße (aus deinem Wissen)
EXPECTED_DIAM_NM_MIN = 5.0
EXPECTED_DIAM_NM_MAX = 50.0

# Filter für die Segmentation
MIN_AREA_PX = 20          # sehr klein (SEM, kleine Partikel)
MAX_AREA_PX = 5000       # Artefakt-Deckel
MIN_CIRCULARITY = 0.6    # 0..1, höher = runder

# Wo sollen Ergebnisse gespeichert werden?
OUTPUT_CSV = "sem_particles_results.csv"

# ------------------------------------------------------------------
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


def circularity(area, perimeter, eps=1e-9):
    """Circularity = 4πA / P², robust gegen perimeter≈0."""
    return 4.0 * np.pi * area / (perimeter**2 + eps)


def estimate_dog_sigmas_from_nm(px_nm: float,
                                dmin_nm: float,
                                dmax_nm: float) -> tuple[float, float]:
    """
    Grobe Heuristik: Partikel-Durchmesser (10–40 nm) in Pixel → DoG-Skalen.

    Annahme: Blob-Radius r ~ sqrt(2)*sigma => sigma ~ r / sqrt(2).
    """
    rmin_nm = dmin_nm / 2.0
    rmax_nm = dmax_nm / 2.0

    rmin_px = rmin_nm / px_nm
    rmax_px = rmax_nm / px_nm

    sigma1 = max(rmin_px / np.sqrt(2), 0.3)
    sigma2 = max(rmax_px / np.sqrt(2), sigma1 + 0.3)

    return sigma1, sigma2, rmin_px, rmax_px


def analyze_sem_particles(tif_path: str | Path):
    tif_path = Path(tif_path)

    # ---------- 1) Metadaten auslesen (Pixelgröße) -------------------
    meta = summarize_sem_metadata(tif_path)
    summary = meta["summary"]

    px_nm_w = summary["pixel_width_nm"]
    px_nm_h = summary["pixel_height_nm"]

    if px_nm_w is None or px_nm_h is None:
        raise RuntimeError("Konnte Pixelgröße in nm nicht aus Metadaten lesen.")

    # evtl. anisotrope Pixel mitteln (typisch bei SEM aber gleich)
    px_nm = 0.5 * (px_nm_w + px_nm_h)
    px_um = px_nm * 1e-3

    print(f"Pixelgröße: {px_nm:.4f} nm/px (~{px_um:.4f} µm/px)")

    # ---------- 2) Bild laden und in Graustufen konvertieren --------
    img = Image.open(tif_path)
    # SEM-Bild ist meistens "pseudo-RGB", Kanal-Mittel reicht:
    img_gray = img.convert("L")
    image = np.array(img_gray, dtype=float)
    image = gaussian_filter(image, sigma=4.0)  # leicht glätten

    print(f"Bildgröße: {image.shape[1]} x {image.shape[0]} px")

    # ---------- 3) DoG-Skalen aus 10–40 nm ableiten -----------------
    sigma1, sigma2, rmin_px, rmax_px = estimate_dog_sigmas_from_nm(
        px_nm, EXPECTED_DIAM_NM_MIN, EXPECTED_DIAM_NM_MAX
    )

    print(f"Erwartete Partikel: {EXPECTED_DIAM_NM_MIN:.1f}–{EXPECTED_DIAM_NM_MAX:.1f} nm")
    print(f"→ geschätzter Radius in Pixel: {rmin_px:.2f}–{rmax_px:.2f} px")
    print(f"→ DoG-Sigmas: sigma1 = {sigma1:.2f}, sigma2 = {sigma2:.2f}")

    # ---------- 4) Difference of Gaussians ---------------------------
    dog = difference_of_gaussians(image, low_sigma=sigma1, high_sigma=sigma2)
    # nur positive Peaks
    dog = np.clip(dog, a_min=0, a_max=None)

    # DoG-Preview
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(dog, cmap="magma", interpolation="nearest")
    ax.set_title("DoG-Response")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    # ---------- 5) Thresholding (einfach über Quantil) ---------------
    # Du kannst das Quantil je nach Kontrast variieren:
    quantile = 0.5
    thr = np.quantile(dog, quantile)
    print(f"Threshold via DoG-Quantil Q={quantile:.2f}: thr = {thr:.4g}")

    mask = dog > thr
    mask = remove_small_objects(mask, min_size=MIN_AREA_PX)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mask, cmap="gray", interpolation="nearest")
    ax.set_title("Binärmaske nach DoG + Threshold + Remove small objects")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    # ---------- 6) Labeling & regionprops ----------------------------
    labeled = label(mask)
    props = regionprops(labeled, intensity_image=image)

    results = []
    for r in props:
        area = r.area
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            continue

        circ = circularity(area, r.perimeter)
        if circ < MIN_CIRCULARITY:
            continue

        eq_diam_px = r.equivalent_diameter
        eq_diam_nm = eq_diam_px * px_nm

        cy, cx = r.centroid  # (y,x) in Pixeln

        results.append(
            dict(
                label=int(r.label),
                x_px=float(cx),
                y_px=float(cy),
                area_px=float(area),
                circularity=float(circ),
                eq_diam_px=float(eq_diam_px),
                eq_diam_nm=float(eq_diam_nm),
                mean_intensity=float(r.mean_intensity),
                max_intensity=float(r.max_intensity),
            )
        )

    if not results:
        print("⚠ Keine Partikel nach den gesetzten Kriterien gefunden.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"{len(df)} Partikel gefunden. Ergebnisse gespeichert in: {OUTPUT_CSV}")

    # ---------- 7) Histogram der Partikel-Durchmesser (nm) ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["eq_diam_nm"], bins=30)
    ax.axvspan(EXPECTED_DIAM_NM_MIN, EXPECTED_DIAM_NM_MAX,
               alpha=0.2, label="erwarteter Bereich")
    ax.set_xlabel("Äquivalenter Durchmesser [nm]")
    ax.set_ylabel("Anzahl Partikel")
    ax.legend()
    ax.set_title("Verteilung der detektierten Partikelgrößen")
    plt.tight_layout()
    plt.show()

    # ---------- 8) Overlay (Ringe als äquivalente Durchmesser) ------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", interpolation="nearest")

    for r in results:
        x = r["x_px"]
        y = r["y_px"]
        rad = r["eq_diam_px"] / 2.0

        circ_patch = plt.Circle((x, y), radius=rad, fill=False, linewidth=0.8)
        ax.add_patch(circ_patch)

    ax.set_title("Detektierte Partikel (Ringe = äquivalente Durchmesser)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    # Quick-Check: statische Kenngrößen
    print("\n--- Kurze Statistik (nm) ---")
    print(f"Median: {df['eq_diam_nm'].median():.2f} nm")
    print(f"Mean  : {df['eq_diam_nm'].mean():.2f} nm")
    print(f"Std   : {df['eq_diam_nm'].std():.2f} nm")


if __name__ == "__main__":
    # HIER deinen Datei-Pfad einsetzen:
    tif_file = r"D:\SEM\2025_12-17\50nm_T2_03.tif"
    analyze_sem_particles(tif_file)
