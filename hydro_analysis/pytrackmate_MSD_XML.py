import pytrackmate
import pandas as pd
import trackpy as tp
import numpy as np
import matplotlib.pyplot as plt

def calculate_msd_from_trackmate_xml(xml_file_path, frame_rate, particle_size=1):
    """
    Importiert Trajektorien aus einer TrackMate XML-Datei und berechnet die MSD.

    Args:
        xml_file_path (str): Pfad zur TrackMate XML-Datei.
        frame_rate (float): Bildrate des Experiments (z.B. 1.0/dt) in Hz.
        particle_size (int): Numerische Kennung für das Partikel, wenn TrackMate 
                             mehrere Partikeltypen hat (meistens 1).

    Returns:
        pd.DataFrame: DataFrame mit den berechneten MSD-Werten ('msd', 'lagt' usw.).
    """
    
    # 1. Trajektorien aus XML importieren
    print(f"Lese Trajektorien aus: {xml_file_path}")
    tracks = pytrackmate.get_tracks_as_pandas(xml_file_path, 
                                            particle_size=particle_size, 
                                            columns=['POSITION_X', 'POSITION_Y', 'FRAME'])
    
    # TrackMate speichert Frames als Integer (0, 1, 2, ...), 
    # trackpy benötigt eine Spalte 'frame'.
    tracks.rename(columns={'FRAME': 'frame', 
                          'POSITION_X': 'x', 
                          'POSITION_Y': 'y',
                          'TRACK_ID': 'particle'}, 
                 inplace=True)
    
    # In trackpy ist die Spalte 'particle' die ID der Trajektorie.
    tracks['particle'] = tracks['particle'].astype(int)
    
    # 2. MSD-Berechnung mit trackpy
    print("Berechne Mean Squared Displacement (MSD)...")
    
    # Konvertieren Sie die Frame-Nummer in die tatsächliche Zeit (Tau).
    # t ist die Zeit in Sekunden.
    # Die Spalte 'frame' wird für 'trackpy' verwendet, 
    # aber die Einheiten müssen mit der Frame-Rate korrekt sein.
    
    # Setzen Sie die Spaltenreihenfolge für trackpy
    tracks = tracks[['frame', 'x', 'y', 'particle']]
    
    # Berechne die Ensemble-MSD (gemittelt über alle Partikel)
    # msd = tp.emsd(trajectories, mpp, fps)
    # mpp (micron per pixel) ist hier 1, da TrackMate oft schon in physikalischen
    # Einheiten (z.B. µm) exportiert. Wenn nicht, muss es angepasst werden.
    # Wir nehmen an, dass 'x' und 'y' in µm sind, daher mpp=1.
    
    # Wenn Ihre TrackMate-Daten Pixelkoordinaten enthalten, 
    # ersetzen Sie 1.0 durch Ihre µm/Pixel-Kalibrierung:
    microns_per_pixel = 1.0 
    
    im_msd = tp.imsd(tracks, 
                     mpp=microns_per_pixel, 
                     fps=frame_rate, 
                     max_lagtime=100) # Berechnet die individuelle MSD
    
    # Ensemble MSD (Durchschnitt über alle Partikel)
    # Die MSD-Werte sind in mpp^2 (z.B. µm^2)
    msd = tp.emsd(tracks, 
                  mpp=microns_per_pixel, 
                  fps=frame_rate) 
    
    print("MSD-Berechnung abgeschlossen.")
    
    return msd, im_msd, tracks

# --- ANWENDUNG ---
# !!! PASSEN SIE DIESE WERTE AN IHRE DATEN AN !!!

XML_DATEI = r"Z:\Diffusion in Hydrogel Data\20mg_20nm\Trajektorien\ResultofB1_20nm_20mg_1d_nichtzentral_1_ohne_Tracks.xml"
XML_DATEI = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml"
dx = 0.150  # µm per pixel


XML_DATEI = 'Ihr_TrackMate_Ergebnis.xml' 
# Angenommene Bildrate: z.B. 10 Frames pro Sekunde (fps)
EXPERIMENT_FRAME_RATE = 20.0  # Hz (entspricht 50 ms pro Frame)
# Angenommene Einheiten: 'x' und 'y' sind in Mikrometern (µm)

try:
    msd_ensemble, msd_individual, all_tracks = calculate_msd_from_trackmate_xml(
        xml_file_path=XML_DATEI, 
        frame_rate=EXPERIMENT_FRAME_RATE
    )

    ## 3. Visualisierung der Ergebnisse
    
    plt.figure(figsize=(10, 6))
    
    # MSD-Ensemble-Plot
    plt.loglog(msd_ensemble.index, msd_ensemble.values, 'o-', label='Ensemble MSD')
    
    # Fit für Diffusionskoeffizienten (optional)
    # MSD = 4*D*tau (für 2D)
    
    # Linearen Teil der Kurve auswählen (z.B. von Lag-Zeit 0.1s bis 1.0s)
    # Diese Auswahl muss eventuell an Ihre Daten angepasst werden!
    fit_range = msd_ensemble.loc[0.1:1.0]
    
    if not fit_range.empty:
        # Führen Sie einen linearen Fit im log-log-Plot durch (für den Exponenten alpha)
        # log(MSD) = log(C) + alpha * log(tau)
        p = np.polyfit(np.log(fit_range.index), np.log(fit_range.values), 1)
        alpha = p[0] # MSD-Exponent
        log_C = p[1]
        
        # Lineare Anpassung im Plot
        plt.loglog(fit_range.index, np.exp(log_C) * fit_range.index**alpha, 
                   '--', color='red', label=f'Fit (α={alpha:.2f})')
        
        # Diffusionskoeffizient D berechnen (D = MSD / 4*tau für 2D)
        # Genauer: D = msd_ensemble.loc[tau] / (4 * tau) 
        # aus dem ersten linearen Bereich.
        # Vereinfachte Annahme für D aus dem ersten fit_range Punkt
        tau_fit = fit_range.index[0]
        msd_fit = fit_range.values[0]
        diffusion_coefficient = msd_fit / (4 * tau_fit)
        
        plt.title(f'MSD-Analyse (2D) - D ≈ {diffusion_coefficient:.2e} $\mu m^2/s$')
    else:
        plt.title('MSD-Analyse (2D)')

    
    plt.xlabel('Lag-Zeit $\\tau$ (s)')
    plt.ylabel('MSD $\\langle \\Delta r^2 \\rangle$ ($\mu m^2$)')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    
    plt.show()
    
    # Anzeigen der ersten Zeilen des Ensemble-MSD-DataFrames
    print("\nErste Zeilen des Ensemble-MSD-Ergebnisses:")
    print(msd_ensemble.head())
    
except FileNotFoundError:
    print(f"\nFEHLER: Datei nicht gefunden. Bitte prüfen Sie den Pfad: {XML_DATEI}")
except Exception as e:
    print(f"\nEin Fehler ist aufgetreten: {e}")