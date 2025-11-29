#!/usr/bin/env python3
"""
Load TIFF(s), find spots with trackpy, link trajectories, compute MSD.
Save/display MSD vs lag (in frames). Inline descriptions/comments included.
"""

# Minimal dependencies: pims, trackpy, pandas, numpy, matplotlib
import argparse
import pims
import trackpy as tp
import numpy as np
import pandas as pd
from types import SimpleNamespace
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


def main(tif_path, diameter, minmass, mpp, fps, plot=False):

    # 1. Daten laden (pims öffnet den Stack 'lazy', also speicherschonend)
    # Ersetze 'dein_stack.tif' mit deinem Pfad
    frames = pims.open(tif_path)

    # compute per-pixel background as the mean over the whole stack (float64 accumulator for precision)
    n_frames = len(frames)
    acc = None
    for fr in frames:
        arr = np.asarray(fr, dtype=np.float64)
        if acc is None:
            acc = np.zeros_like(arr, dtype=np.float64)
        acc += arr
    background = (acc / float(n_frames)).astype(np.float32)

    # subtract background from each frame, clip negatives to zero and keep float32
    frames = [np.clip(np.asarray(fr, dtype=np.float32) - background, 0.0, None).astype(np.float32)
              for fr in frames]
    print(f"Anzahl der Frames: {len(frames)}")
    print(f"Bildgröße: {frames[0].shape}")

    # 2. Parameter-Einstellung (WICHTIGSTER SCHRITT FÜR KONTROLLE)
    # diameter: Geschätzter Durchmesser eines Partikels in Pixeln (Muss ungerade sein!)
    # minmass: Mindest-integrierte Helligkeit, um Rauschen zu filtern.
    DIAMETER = diameter  # <-- Hier anpassen (ungerade Zahl, z.B. 9, 11, 13)

    # Testlauf auf dem ersten Frame (Frame 0)
    f = tp.locate(frames[0], diameter=DIAMETER, invert=False, minmass=minmass) 
    # invert=True nutzen, falls Partikel dunkel auf hellem Grund sind

    # 3. Visualisierung der Erkennung
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(frames[0], cmap='gray')
        tp.annotate(f, frames[0], ax=ax)
        plt.show()
        #tp.annotate(f, frames[0])

        fig, ax = plt.subplots()
        ax.hist(f['mass'], bins=21)

        #Optionally, label the axes.
        ax.set(xlabel='mass', ylabel='count')
        plt.show()

        tp.subpx_bias(f)
        plt.show()
    tp.quiet()
    f = tp.batch(frames, 17, minmass=2000)
    t = tp.link(f, 5, memory=3)
    t1 = tp.filter_stubs(t, 25)

    d = tp.compute_drift(t1)
    if plot:
        print(d)
        plt.figure()
        tp.plot_traj(t1);
        plt.show()
    tm = tp.subtract_drift(t1, d)
    
    if plot:
        plt.figure()
        tp.plot_traj(tm);
        plt.show()
    
    # Avoid ValueError: "cannot insert particle, already exists"
    # If columns duplicate index level names, drop those columns first so reset_index can insert index levels as columns.
    if tm.index.nlevels > 0:
        dup_levels = [n for n in tm.index.names if n in tm.columns]
        if dup_levels:
            tm = tm.drop(columns=dup_levels)
    tm = tm.reset_index()
    # Remove any accidental duplicate column names (keeps first occurrence)
    tm = tm.loc[:, ~tm.columns.duplicated()]
    im = tp.imsd(tm, mpp, fps) # Pixel zu µm (100 nm/Pixel), fps=24
    em = tp.emsd(tm, mpp, fps)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
        ax.plot(em.index, em, 'o')
        ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
            xlabel='lag time $t$')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.figure()
        plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        plt.xlabel('lag time $t$');
    
    tp.quiet()
    params = tp.utils.fit_powerlaw(em,plot=plot)
    if not plot: 
        plt.close('all')
    if plot:
        print(params)
    D = params.A/4
    print("D =", D ,'µm²/s')
    return D
if __name__ == "__main__":
    # On Windows, multiprocessing used by trackpy/pims requires the freeze_support guard.
    # This prevents the "Safe importing of main module" / freeze_support warning/error.
    import multiprocessing
    multiprocessing.freeze_support()
    #Parameter
    mpp = 0.15
    fps = 22
    diamter = 7
    print("diameter",diamter*mpp,"µm")

    kb = 1.380649e-23  # Boltzmann-Konstante in J/K
    T = 293.15  # Temperatur in Kelvin (25 °C)
    nu = 0.001002  # Dynamische Viskosität von Wasser bei 25 °C in Pa·s


    print(f'Theoretischer D ({diamter*mpp*1000} nm Partikel):', (kb * T)/(6 * np.pi * diamter * mpp/2*nu*(1e-6)*1e-12) ,'µm²/s')

    paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm_2.tif",
             r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm_3.tif",
             r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm_4.tif",
             r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm.tif"]  
    #D_1000 = []
    #for path in paths:
    #    minmass = 2000
    #    print("Processing:", path)
    #    D = main(path,diamter,mpp, fps,plot=False)
    #    D_1000.append(D)
    #print("Mittelwert D 1000 nm:", np.mean(D_1000), "µm²/s ±", np.std(D_1000), "µm²/s")


    diamter = 5
    print("diameter",diamter*mpp,"µm")


    print(f'Theoretischer D ({diamter*mpp*1000} nm Partikel):', (kb * T)/(6 * np.pi * diamter * mpp/2*nu*(1e-6)*1e-12) ,'µm²/s')

    paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\500 nm.tif",
             r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\500 nm _3.tif",
             r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\500 nm _4.tif",
             r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\500 nm _2.tif"]  
    D_500 = []
    minmass = 250
    for path in paths:
        D = main(path,diamter, minmass, mpp, fps,plot=True)
        D_500.append(D)
    print("Mittelwert D 500 nm:", np.mean(D_500), "µm²/s ±", np.std(D_500), "µm²/s")

    #return
    diamter = 3
    print("diameter",diamter*mpp,"µm")

    print(f'Theoretischer D ({diamter*mpp*1000} nm Partikel):', (kb * T)/(6 * np.pi * diamter * mpp/2*nu*(1e-6)*1e-12) ,'µm²/s')

    paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\200 nm.tif",
             r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\200 nm_2.tif"]  
    D_1000 = []
    for path in paths:
        D = main(path,diamter,mpp, fps,plot=False)
        D_1000.append(D)
    print("Mittelwert D 1000 nm:", np.mean(D_1000), "µm²/s ±", np.std(D_1000), "µm²/s")

    

    diamter = 3
    print("diameter",diamter*mpp,"µm")

    print(f'Theoretischer D ({diamter*mpp*1000} nm Partikel):', (kb * T)/(6 * np.pi * diamter * mpp/2*nu*(1e-6)*1e-12) ,'µm²/s')

    paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm_5.tif",
    r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm_4.tif",
    r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm_3.tif",
    r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm_2.tif",
    r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm.tif"]  
    D_1000 = []
    for path in paths:
        D = main(path,diamter,mpp, fps,plot=False)
        D_1000.append(D)
    print("Mittelwert D 1000 nm:", np.mean(D_1000), "µm²/s ±", np.std(D_1000), "µm²/s")