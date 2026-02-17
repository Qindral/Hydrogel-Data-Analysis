
ROOT_DIR = r"E:\PhD Data Analysis\SPT 2025 II\2025.12.09\Calibration_time\Files"  # <<-- HIER anpassen
"""
Fit der effektiven Frame-Zeit dt pro Ordner:
- Nominelle dt (ms) wird aus Ordnernamen gelesen, z.B. '10hz_25k' -> 25 ms.
- dt wird im Bereich [dt_nom-1, dt_nom+1] ms abgefahren.
- Für jedes dt wird eine ideale 10 Hz Stufenfunktion (60 ms hell, 40 ms dunkel)
  erstellt und gegen das gemessene Binärsignal gefittet (MSE).
- Ausgabe: beste dt_ms pro Ordner und ein Kontrollplot (gemessen vs. ideal).
"""

from pathlib import Path
from typing import List, Tuple, Dict
import re

import numpy as np
import matplotlib.pyplot as plt
from pco_image import PCOImage


# ============ PARAMETER ANPASSEN ================================

MAX_FRAMES_PER_FOLDER = None               # z.B. 5000 oder None = alle
DT_SCAN_RANGE_MS =2.0                     # +/- 1 ms
DT_SCAN_STEP_MS = 0.0001                     # Raster (0.01 ms -> 200 Schritte)


# ============ STROBO-PARAMETER =================================

F_STROBE = 10.28            # Hz
T_CYCLE = 1.0 / F_STROBE        # 0.1 s = 100 ms
T_ON = 0.060                    # 60 ms hell
T_OFF = T_CYCLE - T_ON          # 40 ms dunkel


# ============ HILFSFUNKTIONEN ==================================

def parse_dt_from_folder_ms(folder_name: str) -> float:
    """
    Liest nominelle Frame-Zeit in Millisekunden aus Ordnernamen.
    Erwartet Pattern: ..._<zahl>k  -> <zahl> ms

    Beispiele:
        '10hz_14k' -> 14 ms
        '10hz_25k' -> 25 ms
    """
    m = re.search(r'(\d+(?:\.\d+)?)k', folder_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Kann keine '..._Xk' Angabe aus Ordnernamen '{folder_name}' lesen.")
    return float(m.group(1))  # ms


def load_frame_means(folder: Path,
                     max_frames: int | None = None,
                     ignore_first_row: bool = True) -> np.ndarray:
    """Lädt alle .b16 in einem Ordner, gibt Mittelwert pro Frame zurück."""
    files = sorted(folder.glob("*.b16"))
    if not files:
        raise FileNotFoundError(f"Keine .b16-Dateien in {folder}")

    if max_frames is not None:
        files = files[:max_frames]

    means: List[float] = []
    for fn in files:
        pco_img = PCOImage(fn)
        img = pco_img.img  # (H, W)

        if ignore_first_row and img.shape[0] > 1:
            roi = img[1:, :]
        else:
            roi = img

        means.append(float(roi.mean()))

    return np.array(means)


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Normiert auf [0, 1]; bei konstantem Signal -> Null-Array."""
    if x.max() > x.min():
        return (x - x.min()) / (x.max() - x.min())
    else:
        return np.zeros_like(x)


def to_binary(x_norm: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Normiertes Signal -> 0/1 mittels Threshold."""
    return (x_norm > thr).astype(np.uint8)


def crop_to_first_rising(binary: np.ndarray,
                         x_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Schneidet binary und x_norm so, dass sie am ersten 0->1 Übergang beginnen.
    """
    if binary.size < 2:
        return binary, x_norm

    start_idx = None
    for i in range(1, len(binary)):
        if binary[i-1] == 0 and binary[i] == 1:
            start_idx = i
            break

    if start_idx is None:
        return binary, x_norm

    return binary[start_idx:], x_norm[start_idx:]


def ideal_square_wave(t: np.ndarray,
                      period: float = T_CYCLE,
                      t_on: float = T_ON) -> np.ndarray:
    """Ideale 10 Hz Stufenfunktion (60 ms hell, Rest dunkel)."""
    t_mod = np.mod(t, period)
    return (t_mod < t_on).astype(float)


def fit_dt_for_signal(binary: np.ndarray,
                      dt_nom_ms: float,
                      scan_range_ms: float,
                      scan_step_ms: float) -> Tuple[float, float]:
    """
    Sucht dt (in ms) im Bereich [dt_nom_ms - scan_range_ms, dt_nom_ms + scan_range_ms],
    das den MSE zwischen gemessenem Binärsignal und idealer 10 Hz-Stufe minimiert.

    Rückgabe:
        best_dt_ms : beste Frame-Zeit in ms
        best_mse   : zugehöriger mittlerer quadratischer Fehler
    """
    n = len(binary)
    if n < 5:
        raise ValueError("Signal zu kurz für Fit.")

    # Scan-Bereich vorbereiten
    dt_min = max(0.1, dt_nom_ms - scan_range_ms)  # dt > 0.1 ms sicherheitshalber
    dt_max = dt_nom_ms + scan_range_ms
    dt_values_ms = np.arange(dt_min, dt_max + 1e-9, scan_step_ms)

    best_mse = np.inf
    best_dt_ms = dt_nom_ms

    for dt_ms in dt_values_ms:
        dt_s = dt_ms / 1000.0
        t = np.arange(n) * dt_s
        ideal = ideal_square_wave(t)
        mse = np.mean((binary - ideal)**2)
        if mse < best_mse:
            best_mse = mse
            best_dt_ms = dt_ms

    return best_dt_ms, best_mse


# ============ HAUPTFUNKTION =====================================

def main(root_dir: str | Path,
         max_frames_per_folder: int | None = None):
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} ist kein gültiger Ordner.")

    subfolders = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subfolders:
        raise FileNotFoundError(f"Keine Unterordner in {root}")

    print(f"Gefundene Experimente in {root}:")
    for d in subfolders:
        print("  -", d.name)
    print()
    dt_nom_ms_list = []
    dt_fit_ms_list = []
    for folder in subfolders:
        print(f"=== Ordner: {folder.name} ===")
        try:
            dt_nom_ms = parse_dt_from_folder_ms(folder.name)
        except Exception as e:
            print(f"  -> Übersprungen (dt_nom konnte nicht gelesen werden): {e}")
            continue

        try:
            means = load_frame_means(folder,
                                     max_frames=max_frames_per_folder,
                                     ignore_first_row=True)
        except Exception as e:
            print(f"  -> Übersprungen (Fehler beim Laden): {e}")
            continue

        means_norm = normalize_signal(means)
        binary = to_binary(means_norm, thr=0.5)

        # am ersten Zyklusstart (0->1) beginnen
        binary, means_norm = crop_to_first_rising(binary, means_norm)

        if len(binary) < 20:
            print("  -> Zu wenige Frames nach Cropping, Ordner ausgelassen.")
            continue

        # dt im Bereich dt_nom ± 1 ms fitten
        best_dt_ms, best_mse = fit_dt_for_signal(
            binary,
            dt_nom_ms=dt_nom_ms,
            scan_range_ms=DT_SCAN_RANGE_MS,
            scan_step_ms=DT_SCAN_STEP_MS
        )

        dt_nom_s = dt_nom_ms / 1000.0
        dt_fit_s = best_dt_ms / 1000.0
        fps_nom = 1.0 / dt_nom_s
        fps_fit = 1.0 / dt_fit_s

        print(f"  nominell:  dt = {dt_nom_ms:.3f} ms  (fps ≈ {fps_nom:.2f})")
        print(f"  best fit:  dt = {best_dt_ms:.3f} ms  (fps ≈ {fps_fit:.4f}),  MSE = {best_mse:.4f}")

        # Kontrollplot gemessen vs. ideal mit best_dt_ms
        n = len(binary)
        t_fit = np.arange(n) * dt_fit_s
        ideal_fit = ideal_square_wave(t_fit)

        # plt.figure(figsize=(8, 4))
        # plt.step(t_fit * 1e3, binary, where="post", label="gemessen (0/1)")
        # plt.plot(t_fit * 1e3, ideal_fit, "k--", label="ideal 10 Hz (60/40)")
        # plt.ylim(-0.2, 1.2)
        # plt.xlabel("Zeit [ms]")
        # plt.ylabel("Helligkeit (0/1)")
        # plt.title(f"{folder.name}: dt_nom={dt_nom_ms:.2f} ms, dt_fit={best_dt_ms:.3f} ms")
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        dt_nom_ms_list.append(dt_nom_ms)
        dt_fit_ms_list.append(best_dt_ms)

    plt.figure(figsize=(8, 4))
    plt.scatter(dt_nom_ms_list, (np.array(dt_fit_ms_list)-np.array(dt_nom_ms_list)), color="blue")
    plt.xlabel("Nominelle Frame-Zeit dt_nom [ms]")
    plt.ylabel("Gefittete Frame-Zeit dt_fit [ms]")
    plt.title("Gefittete vs. nominelle Frame-Zeit")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# ============ SCRIPT ENTRY ======================================

if __name__ == "__main__":
    main(ROOT_DIR, max_frames_per_folder=MAX_FRAMES_PER_FOLDER)
