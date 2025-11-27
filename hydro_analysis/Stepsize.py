import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage.data import binary_blobs
from skimage.util import img_as_float
import trackpy as tp
import pandas as pd

# ---------------------------------------------------------
# 1. SETUP & BEISPIELDATEN GENERIEREN
# (Diesen Teil kannst du durch das Laden deiner eigenen Bilder ersetzen)
# ---------------------------------------------------------

def create_fake_microscopy_stack(n_frames=10, shape=(256, 256)):
    # Wir erzeugen ein festes Bild mit "Partikeln" (Blobs)
    background = binary_blobs(length=512, blob_size_fraction=0.05, volume_fraction=0.05, seed=42)
    background = img_as_float(background)
    
    frames = []
    shifts_true = []
    
    # Simuliere Stage-Bewegung: Wir schneiden immer ein (256,256) Fenster aus
    # Startposition in der Mitte
    current_y, current_x = 128, 128 
    
    for i in range(n_frames):
        # Zufällige Bewegung des Tisches (dx, dy)
        dx = np.random.randint(-5, 6)
        dy = np.random.randint(-5, 6)
        
        current_x += dx
        current_y += dy
        
        # Ausschneiden des Frames (FOV)
        frame = background[current_y:current_y+shape[0], current_x:current_x+shape[1]]
        
        # Rauschen hinzufügen für Realismus
        noise = np.random.normal(0, 0.05, frame.shape)
        frames.append(frame + noise)
        shifts_true.append((dy, dx))
        
    return np.array(frames)

print("Generiere Beispieldaten...")
frames = create_fake_microscopy_stack(n_frames=20)
print(f"Stack-Größe: {frames.shape}")

# ---------------------------------------------------------
# 2. ANALYSE DER STAGE-BEWEGUNG (Shift Calculation)
# ---------------------------------------------------------

print("Berechne Verschiebungen (dx, dy)...")

# Array um die kumulative Verschiebung zu speichern
# Format: [Frame, y_shift, x_shift]
drifts = [] 
cumulative_drift = np.array([0.0, 0.0]) # Start bei 0

# Referenz ist das erste Bild
# Alternativ: Immer Frame t mit Frame t-1 vergleichen (besser bei großen Bewegungen)
reference_frame = frames[0]

for i in range(1, len(frames)):
    moving_frame = frames[i]
    prev_frame = frames[i-1]
    
    # Berechne Versatz zwischen Frame t und Frame t-1
    # upsample_factor=100 gibt uns 1/100 Pixel Genauigkeit
    shift, error, diffphase = phase_cross_correlation(prev_frame, moving_frame, upsample_factor=100)
    
    # shift ist (y, x). 
    # Wenn sich das Bild nach rechts bewegt, ist der Shift negativ.
    # Wir addieren den Shift zur kumulativen Drift.
    cumulative_drift += shift
    
    drifts.append({
        'frame': i,
        'dy_step': shift[0],
        'dx_step': shift[1],
        'y_cum': cumulative_drift[0],
        'x_cum': cumulative_drift[1]
    })

drift_df = pd.DataFrame(drifts)
print("Analyse der Bewegung abgeschlossen.")
print(drift_df.head())

# ---------------------------------------------------------
# 3. PARTIKEL DETEKTIEREN UND VERLINKEN
# ---------------------------------------------------------

print("Starte Partikel-Tracking...")

# A. Partikel finden (Locate) in jedem Frame
# diameter: Ungefähre Größe der Partikel in Pixeln (muss ungerade sein)
# minmass: Helligkeitsschwelle (muss angepasst werden an deine Bilder)
f = tp.locate(frames, diameter=11, minmass=0.1, invert=False)

# B. Koordinaten korrigieren
# Wir müssen die Drift von den gefundenen Positionen abziehen (oder addieren, je nach Perspektive),
# um die "absoluten" Koordinaten auf dem Substrat zu erhalten.

def correct_coordinates(features, drift_data):
    # Erstelle ein Dictionary für schnellen Zugriff auf Drifts pro Frame
    # Frame 0 hat Drift (0,0)
    drift_map = {0: (0, 0)}
    for _, row in drift_data.iterrows():
        drift_map[row['frame']] = (row['y_cum'], row['x_cum'])
    
    features_corrected = features.copy()
    
    # Korrektur anwenden
    # Wenn sich die Kamera nach rechts bewegt (dx > 0), wandert das Bild nach links.
    # Um die absolute Position zu bekommen, müssen wir die Kamerabewegung addieren.
    # Hinweis: phase_cross_correlation gibt den Shift zurück, um Bild B auf Bild A zu mappen.
    
    for i, row in features.iterrows():
        frame_idx = int(row['frame'])
        dy, dx = drift_map.get(frame_idx, (0,0))
        
        # Hier ist das Vorzeichen wichtig! 
        features_corrected.at[i, 'x'] = row['x'] - dx 
        features_corrected.at[i, 'y'] = row['y'] - dy
        
    return features_corrected

f_corrected = correct_coordinates(f, drift_df)

# C. Verlinken (Linken)
# Da wir die Koordinaten korrigiert haben, bewegen sich die Partikel kaum noch
# (nur noch durch Messrauschen/Brownsche Bewegung).
# search_range: Wie viele Pixel darf sich ein Partikel max bewegen (nach Korrektur sehr wenig).
t = tp.link(f_corrected, search_range=5, memory=3)

print(f"Anzahl gefundener Trajektorien: {t['particle'].nunique()}")

# ---------------------------------------------------------
# 4. VISUALISIERUNG
# ---------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Die berechnete Stage-Bewegung
ax1.plot(drift_df['frame'], drift_df['x_cum'], label='X Drift (Pixel)')
ax1.plot(drift_df['frame'], drift_df['y_cum'], label='Y Drift (Pixel)')
ax1.set_title("Erkannte Stage-Bewegung")
ax1.set_xlabel("Frame")
ax1.set_ylabel("Verschiebung (px)")
ax1.legend()
ax1.grid(True)

# Plot 2: Die Trajektorien (Korrigierte Koordinaten)
# Das zeigt die Positionen der Partikel "auf dem Dia", also stationär.
tp.plot_traj(t, ax=ax2, label=True)
ax2.set_title("Partikel-Trajektorien (Drift-korrigiert)")

plt.tight_layout()
plt.show()