"""
Max-Frame Trajectory Visualization
====================================
Wertet ein TIFF-Video Frame für Frame aus:
  - Jeder Pixel behält die Farbe des Frames, in dem er seine maximale
    Helligkeit erreicht hat (Jet-Colormap, skaliert auf Gesamtanzahl Frames).
  - Pixel unterhalb des Hintergrundschwellwerts werden schwarz dargestellt.
  - Pixelhelligkeit im Output wird durch die relative Maximalintensität
    moduliert, sodass helle Partikel klar hervorstechen.
  - Ergebnis: ein RGB-PNG mit farbigem Schweif der Partikel.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# Konfiguration
# =============================================================================
VIDEO_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\2026.01.19\50 nm water_07.tif")   # <-- TIFF-Stack hier eintragen

# Pixel, deren Maximalintensität unter diesem Bruchteil des globalen Maximums
# liegt, werden als Hintergrund gewertet und schwarz dargestellt.
BG_THRESHOLD_FRAC = 0.05     # 5 % des globalen Maximums

COLORMAP = "jet"             # Matplotlib-Colormap für Frame-Index


# =============================================================================
# TIFF-Stack laden
# =============================================================================
def load_tif_stack(path: Path) -> np.ndarray:
    """Lädt einen Multi-Frame-TIFF als (N, H, W) float32-Array."""
    frames = []
    img = Image.open(path)
    try:
        while True:
            frame = np.array(img)
            if frame.ndim == 3:
                # RGB/RGBA → Grauwert (mittlere Helligkeit)
                frame = frame[..., :3].mean(axis=2)
            frames.append(frame.astype(np.float32))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return np.stack(frames, axis=0)


# =============================================================================
# Max-Frame-Algorithmus
# =============================================================================
def max_frame_trajectory(
    stack: np.ndarray,
    bg_threshold_frac: float = 0.05,
    colormap: str = "jet",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Erzeugt ein Frame-indiziertes Max-Intensity-Bild.

    Für jeden Pixel wird gespeichert, in welchem Frame er seine maximale
    Helligkeit hatte. Die Farbe des Pixels entspricht diesem Frame-Index
    (Jet-Colormap). Die Pixelhelligkeit wird durch die relative
    Maximalintensität moduliert.

    Rückgabe
    --------
    rgb_image : (H, W, 3) uint8   – fertige Trajektoriendarstellung
    max_map   : (H, W) float32    – Maximalintensität je Pixel
    owner_map : (H, W) int32      – Frame-Index des Maximums je Pixel
    """
    N, H, W = stack.shape

    max_map   = stack[0].copy()
    owner_map = np.zeros((H, W), dtype=np.int32)

    for i in range(1, N):
        brighter = stack[i] > max_map
        max_map[brighter]   = stack[i][brighter]
        owner_map[brighter] = i

    # Hintergrundmaske
    global_max = float(max_map.max())
    bg_mask = max_map < (bg_threshold_frac * global_max)

    # Frame-Index → Farbe (Jet)
    cmap = matplotlib.colormaps[colormap]
    norm_owner = owner_map / max(N - 1, 1)          # 0.0 … 1.0
    rgba = cmap(norm_owner)                          # (H, W, 4) float64

    # Helligkeit durch relative Maximalintensität modulieren
    brightness = (max_map / global_max)             # 0.0 … 1.0
    rgb = (rgba[..., :3] * brightness[..., np.newaxis]).clip(0.0, 1.0)

    # Hintergrund schwärzen
    rgb[bg_mask] = 0.0

    rgb_image = (rgb * 255).astype(np.uint8)
    return rgb_image, max_map, owner_map


# =============================================================================
# Main
# =============================================================================
def main():
    if not VIDEO_PATH.exists():
        print(f"FEHLER: Datei nicht gefunden: {VIDEO_PATH}")
        sys.exit(1)

    print(f"Lade {VIDEO_PATH.name} …")
    stack = load_tif_stack(VIDEO_PATH)
    stack = stack[20:120,:,:]
    N, H, W = stack.shape
    print(f"  {N} Frames, {W}×{H} px")
    print(f"  Intensitätsbereich: {stack.min():.1f} – {stack.max():.1f}")

    print("Berechne Max-Frame-Trajektorie …")
    rgb_image, max_map, owner_map = max_frame_trajectory(
        stack,
        bg_threshold_frac=BG_THRESHOLD_FRAC,
        colormap=COLORMAP,
    )

    # Ergebnis speichern
    out_path = VIDEO_PATH.with_stem(VIDEO_PATH.stem + "_maxtraj").with_suffix(".png")
    Image.fromarray(rgb_image).save(out_path)
    print(f"Gespeichert: {out_path}")

    # --- Darstellung ---
    _, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Max-Intensity-Projektion (Graustufen)
    axes[0].imshow(max_map, cmap="gray", interpolation="nearest")
    axes[0].set_title("Max-Intensity-Projektion")
    axes[0].axis("off")

    # Frame-Owner-Map (alle Pixel, roh)
    im = axes[1].imshow(owner_map, cmap=COLORMAP, vmin=0, vmax=N - 1,
                        interpolation="nearest")
    axes[1].set_title("Frame-Owner (alle Pixel)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Frame-Index")

    # Ergebnis
    axes[2].imshow(rgb_image, interpolation="nearest")
    axes[2].set_title("Trajektorie (Farbe = Frame-Index)")
    axes[2].axis("off")
    sm = plt.cm.ScalarMappable(
        cmap=COLORMAP, norm=matplotlib.colors.Normalize(vmin=0, vmax=N - 1)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=axes[2], label="Frame-Index")

    overview_path = out_path.with_stem(out_path.stem + "_overview")
    plt.savefig(overview_path, dpi=200)
    print(f"Übersicht gespeichert: {overview_path}")
    plt.show()


if __name__ == "__main__":
    main()
