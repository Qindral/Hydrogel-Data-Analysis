from pathlib import Path
from pprint import pprint

from PIL import Image
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

EXPECTED_DIAM_NM_MIN = 33
EXPECTED_DIAM_NM_MAX = 333

MIN_AREA_PX = EXPECTED_DIAM_NM_MIN**2     # zu kleine Flecken rauswerfen
MAX_AREA_PX = EXPECTED_DIAM_NM_MAX**2     # zu große Flecken rauswerfen

#Filter
GAUSSIAN_SIGMA = 3.0
MIN_CIRCULARITY = 0.28
MAX_CIRCULARITY = 1
MIN_INTENSITYMAX = 0.0  # max Intensität mindestens


















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

paths = r"Z:\Diffusion in Hydrogel Data\SEM Particles\200 nm\Latex particle_Cpad_200nm_004.tif"
paths =r"Z:\Diffusion in Hydrogel Data\SEM Particles\20nm\Latex particle_T2_20nm_007.tif"

tif_path = Path(paths)

# --- Pixelgröße aus FEI-Metadaten (falls vorhanden) -------------
meta = summarize_sem_metadata(tif_path)
px_nm = 0.5*(meta["summary"]["pixel_width_nm"] +
                meta["summary"]["pixel_height_nm"])
# oder vorerst annehmen:
#px_nm = 1.0  # Dummy, wenn du noch keine Skala hast

# --- Bild laden & in Graustufen -------------------------------
pil_img = Image.open(tif_path)
img_gray = pil_img.convert("L")
image = np.array(img_gray, dtype=float)
print(pil_img.size, img_gray.size, image.shape)

print(np.max(pil_img), np.max(img_gray), np.max(image))
print(type(pil_img), type(img_gray), type(image))

# --- 1) Denoising ---------------------------------------------
smoothed = gaussian(image, sigma=GAUSSIAN_SIGMA)

# --- 2) Threshold + Maske --------------------------------------
thr = threshold_otsu(smoothed)
mask = smoothed > thr

# kleine Löcher schließen und Noise entfernen
mask = closing(mask, footprint=disk(1))
mask = remove_small_objects(mask, min_size=MIN_AREA_PX)
ax, fig = plt.subplots(1, 3, figsize=(15, 5))
fig[0].imshow(pil_img, cmap="gray")
fig[0].set_title("Original")
fig[1].imshow(img_gray, cmap="gray")
fig[1].set_title("Geglättet")
fig[2].imshow(image, cmap="gray")
fig[2].set_title("Segmentierungs-Maske")
plt.show()
ax, fig = plt.subplots(1, 3, figsize=(15, 5))
fig[0].imshow(image, cmap="gray")
fig[0].set_title("Original")
fig[1].imshow(smoothed, cmap="gray")
fig[1].set_title("Geglättet")
fig[2].imshow(mask, cmap="gray")
fig[2].set_title("Segmentierungs-Maske")
plt.show()
if __name__ == "__main__":


    



    # Beispielaufruf: Pfad anpassen
    tif_file = r"Z:\Diffusion in Hydrogel Data\SEM Particles\20nm\Latex particle_T2_20nm_007.tif"

    meta = summarize_sem_metadata(tif_file)

    print("\n=== Zusammenfassung (wichtige physikalische Größen) ===")
    pprint(meta["summary"])

    print("\n=== Verfügbare Sektionen im FEI-Block ===")
    print(list(meta["raw_sections"].keys()))

    # Beispiel: wenn du später z.B. Scan-Parameter sehen willst:
    # pprint(meta["raw_sections"]["Scan"])
