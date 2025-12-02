#!/usr/bin/env python3
"""
Load TIFF(s), find spots with trackpy, link trajectories, compute MSD.
Save/display MSD vs lag (in frames). Inline descriptions/comments included.
"""

# Minimal dependencies: pims, trackpy, pandas, numpy, matplotlib
import argparse
from email.mime import image
import pims
import trackpy as tp
import numpy as np
import pandas as pd
from types import SimpleNamespace
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import os
import pickle
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import tqdm as tqdm


def subtract_background(image, radius=50):
    # img, background = subtract_background_rolling_ball(image, radius=radius, light_background=light_bg)
    # out = image - background
    # 1. Create a "flat" kernel (disk) instead of a 3D ball
    # Adjust the size (50, 50) to match your rolling ball diameter (not radius!)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

    # 2. Run the background estimation (Morphological Opening)
    # This is the "Background" that rolling ball would normally give you
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # 3. Subtract
    out = cv2.subtract(image, background)
    return out

if 'fit_powerlaw_with_errors' not in globals():
    def fit_powerlaw_with_errors(em_series, points=10, ax=None, plot=False):
        """
        Fit y = A * x^n to the first `points` of em_series (pandas Series).
        Returns a SimpleNamespace similar to trackpy.utils.fit_powerlaw with error estimates.
        """
        xs = em_series.iloc[0:points].index.values.astype(float)
        ys = em_series.iloc[0:points].values.astype(float)
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
        if mask.sum() < 2:
            # fallback to trackpy's fitter if not enough valid points
            return tp.utils.fit_powerlaw(em_series.iloc[0:points], plot=plot, ax=ax)
        lx = np.log(xs[mask])
        ly = np.log(ys[mask])
        coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
        n_fit = float(coeffs[0])
        logA_fit = float(coeffs[1])
        se = np.sqrt(np.diag(cov))
        se_n = float(se[0])
        se_logA = float(se[1])
        A_fit = float(np.exp(logA_fit))
        se_A = A_fit * se_logA
        return SimpleNamespace(
            A = np.array([A_fit]),
            n = np.array([n_fit]),
            A_err = np.array([se_A]),
            n_err = np.array([se_n]),
            logA = np.array([logA_fit]),
            logA_err = np.array([se_logA]),
            cov = cov
        )

def main(tif_path, diameter,distance, minmass, mpp, fps, plot=False,smooth = False):

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
    if smooth:
        # rolling-ball (morphological opening with a disk) background subtraction (radius=50 px)
        if plot:
            plt.imshow(frames[0], cmap='gray')
            plt.show()
        r = 50
        frames_copy = np.array(frames.copy()).astype('uint8')
        for i, fr in enumerate(tqdm.tqdm(frames_copy)):
            frames_copy[i] = subtract_background(fr, radius=50, light_bg=False)
        frames = frames_copy
        from scipy.ndimage import gaussian_filter
        frames = [gaussian_filter(fr, sigma=2) for fr in frames]
        if plot:
            plt.imshow(frames[0], cmap='gray')
            plt.show()
    if plot:
        print(f"Anzahl der Frames: {len(frames)}")
        print(f"Bildgröße: {frames[0].shape}")

    # 2. Parameter-Einstellung (WICHTIGSTER SCHRITT FÜR KONTROLLE)
    # diameter: Geschätzter Durchmesser eines Partikels in Pixeln (Muss ungerade sein!)
    # minmass: Mindest-integrierte Helligkeit, um Rauschen zu filtern.
    DIAMETER = diameter  # <-- Hier anpassen (ungerade Zahl, z.B. 9, 11, 13)

    # Testlauf auf dem ersten Frame (Frame 0)
    # invert=True nutzen, falls Partikel dunkel auf hellem Grund sind

    # 3. Visualisierung der Erkennung
    if plot:
        f = tp.locate(fr:=frames[50], diameter=DIAMETER, invert=False, minmass=minmass) 
        fig, ax = plt.subplots()
        ax.imshow(fr, cmap='gray')
        tp.annotate(f, fr, ax=ax)
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
    f = tp.batch(frames, diameter, minmass=minmass)
    if plot:
        print("Partikel in allen Frames gefunden.",len(f.iloc[:]))
        fig, ax = plt.subplots()
        ax.hist(f['mass'], bins=21)

        #Optionally, label the axes.
        ax.set(xlabel='mass', ylabel='count')
        plt.show()

        tp.subpx_bias(f)
        plt.show()
    # sanity check: abort early if no particles detected or detections span fewer frames than the movie
    if f is None or getattr(f, "empty", False):
        raise RuntimeError("No particles detected in any frame. Adjust diameter/minmass or check the image stack.")
    if "frame" in f.columns:
        n_detected_particles = int(len(f))
        if n_detected_particles < len(frames):
            raise RuntimeError(
                f"Particles detected in {n_detected_particles}/{len(frames)} frames — aborting. "
                f"Consider lowering minmass, changing diameter, or checking image quality."
            )
    else:
        if len(f) == 0:
            raise RuntimeError("No particle data available; aborting.")
        

    t = tp.link(f, distance , memory=3)
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
    #tm = tm.loc[:, ~tm.columns.duplicated()]
    im = tp.imsd(tm, mpp, fps) 


    # -- simple physical filters for iMSD to remove tracking artifacts --
    # Tunable thresholds
    max_jump_px = max(5, DIAMETER * 4)   # allow up to ~3 particle diameters per frame (pixels)
    min_track_length = 10                # minimum frames per trajectory
    fit_n = min(5, max(2, im.shape[0] - 1))  # number of short-lag points to fit for D
    #max_D = 10.0                         # upper bound for single-particle D [µm^2/s]

    # 1) per-step jump filter + minimum length
    tm = tm.sort_values(['particle', 'frame'])
    steps = tm.groupby('particle')[['x', 'y']].diff().pow(2).sum(axis=1)   # squared step (pixels^2)
    max_step_sq = steps.groupby(tm['particle']).max()
    track_len = tm.groupby('particle').size()

    good_particles = set(track_len[track_len >= min_track_length].index)
    good_particles &= set(max_step_sq[max_step_sq <= (max_jump_px ** 2)].index)

    # 2) drop iMSD columns for bad particles (if im is per-particle DataFrame)
    if isinstance(im, pd.DataFrame):
        # keep only columns that correspond to surviving particles
        keep_cols = [c for c in im.columns if c in good_particles]
        im = im[keep_cols].copy()
    else:
        # if im is Series (rare), ensure its name is allowed
        if im.name not in good_particles:
            im = im.iloc[0:0].copy()

    # 3) sanity checks on each remaining iMSD curve (negative values, decreasing MSD, unphysical D)
    bad_cols = []
    for col in list(im.columns):
        s = im[col].astype(float).dropna()
        if s.empty:
            bad_cols.append(col); continue
        # no negative MSD values
        if (s < 0).any():
            bad_cols.append(col); continue
        # grossly decreasing MSD (final << initial) -> likely artifact
        if s.iloc[-1] < 0.5 * max(1e-12, s.iloc[0]):  # guard against zero first point
            bad_cols.append(col); continue


    # apply final pruning
    if bad_cols:
        im = im.drop(columns=bad_cols, errors='ignore')
    good_particles = set(im.columns)
    tm= tm[tm['particle'].isin(good_particles)].copy()

    # (Optional) compute and log number removed for diagnostics
    if plot:
        print(f"Filtered trajectories: removed {len(set(track_len.index) - good_particles)} / {len(track_len)}")
    em = tp.emsd(tm, mpp, fps)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
        ax.plot(em.index, em, 'o', markersize=8, color='blue')
        ax.plot(em.iloc[0:40].index, em.iloc[0:40], 'o', markersize=3, color='red')
        ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
            xlabel='lag time $t$')
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.figure()
        plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        plt.xlabel('lag time $t$');
    
    tp.quiet()
    if plot: 
        fig, ax = plt.subplots()
        params = tp.utils.fit_powerlaw(em.iloc[0:40], plot=plot, ax=ax)
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        plt.show()
    else:
        params = tp.utils.fit_powerlaw(em.iloc[0:40],plot=plot)
    if not plot: 
        plt.close('all')
    if plot:
        print(params)
    D = params.A/4
    print("D =", D ,'µm²/s')
    return D, im





if __name__ == "__main__":
    # On Windows, multiprocessing used by trackpy/pims requires the freeze_support guard.
    # This prevents the "Safe importing of main module" / freeze_support warning/error.
    import multiprocessing
    multiprocessing.freeze_support()
    #Parameter
    mpp = 0.15
    fps = 22
    diamter = 7
    #print("diameter",diamter*mpp,"µm")

    kb = 1.380649e-23  # Boltzmann-Konstante in J/K
    T = 293.15  # Temperatur in Kelvin (25 °C)
    nu = 0.001002  # Dynamische Viskosität von Wasser bei 25 °C in Pa·s



    # paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm_2.tif",
    #          r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm_3.tif",
    #          r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm_4.tif",
    #          r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm.tif"]  
    # D_1000 = []
    # imsd_1000 = []
    # for path in paths:
    #     print(os.path.basename(path))  
    #     minmass = 2000
    #     distance = 8
    #     diamter = 7 
    #     D, imsd = main(path,diamter,distance,minmass,mpp, fps,plot=False)
    #     D_1000.append(D)
    #     imsd_1000.append(imsd)
    # print(f'Theoretischer D ({1000} nm Partikel):', (kb * T)/(6 * np.pi * 1/2*nu*(1e-6)*1e-12) ,'µm²/s')
    # print("Mittelwert D 1000 nm:", np.mean(D_1000), "µm²/s ±", np.std(D_1000), "µm²/s")
    

    # diamter = 5
    # print("diameter",diamter*mpp,"µm")



    # paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\500 nm.tif",
    #          r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\500 nm _3.tif",
    #          r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\500 nm _4.tif",
    #          r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\500 nm _2.tif"]  
    # D_500 = []
    # imsd_500 = []
    # for path in paths:
        
    #     minmass = 250
    #     distance = 12
    #     print(os.path.basename(path))   
    #     D, imsd = main(path,diamter,distance,minmass, mpp, fps,plot=False)
    #     D_500.append(D)
    #     imsd_500.append(imsd)
    
    # print(f'Theoretischer D ({500} nm Partikel):', (kb * T)/(6 * np.pi * 0.5/2*nu*(1e-6)*1e-12) ,'µm²/s')
    # print("Mittelwert D 500 nm:", np.mean(D_500), "µm²/s ±", np.std(D_500), "µm²/s")

    # #return
    # print("diameter",diamter*mpp,"µm")

    # print(f'Theoretischer D ({diamter*mpp*1000} nm Partikel):',(kb * T)/(6 * np.pi * diamter * mpp/2*nu*(1e-6)*1e-12) ,'µm²/s')
    
    # paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\200 nm.tif",
    #          r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\200 nm_2.tif"]
    # D_200 = []
    # imsd_200 = []
    
    # diamter = 5
    # for path in paths:
    #     print(os.path.basename(path))   
    #     minmass = 200
    #     distance = 15
    #     D, imsd = main(path,diamter,distance,minmass,mpp, fps,plot=False)
    #     D_200.append(D)
    #     imsd_200.append(imsd)
    
    # print(f'Theoretischer D ({200} nm Partikel):', (kb * T)/(6 * np.pi * 0.2/2*nu*(1e-6)*1e-12) ,'µm²/s')
    # print("Mittelwert D 200 nm:", np.mean(D_200), "µm²/s ±", np.std(D_200), "µm²/s")

    

    print("diameter",diamter*mpp,"µm")


    paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm_2.tif", 
             r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm_4.tif",
    r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm_3.tif", 
    r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm.tif"]  
    # r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\50 nm_5.tif", bad data
    D_50 = []
    imsd_50 = []
    
    diamter = 3
    for path in paths:
        minmass = 22
        distance = 23
        print(os.path.basename(path))   
        D, imsd = main(path,diamter,distance,minmass,mpp, fps,plot=True,smooth=True)
        D_50.append(D)
        imsd_50.append(imsd)
    
    print(f'Theoretischer D ({50} nm Partikel):', (kb * T)/(6 * np.pi * 0.05/2*nu*(1e-6)*1e-12) ,'µm²/s')
    print("Mittelwert D 50 nm:", np.mean(D_50), "µm²/s ±", np.std(D_50), "µm²/s")
    
    
    print("diameter",diamter*mpp,"µm")

    paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\20 nm_2.tif",
            r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\20 nm_3.tif",
            r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\20 nm_2.tif",
            r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\20 nm.tif",
            r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\20 nm_5.tif",
            r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\20 nm_4.tif"]
    # paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\preprocess\20 nm_processed.tif",
            # r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\preprocess\20 nm_5_processed.tif",
            # r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\preprocess\20 nm_4_processed.tif",
            # r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\preprocess\20 nm_3_processed.tif",
            # r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\preprocess\20 nm_2_processed.tif"]
    D_20 = []
    imsd_20 = []
    for path in paths:
        diamter = 7
        minmass = 1100
        distance = 27
        print(os.path.basename(path))   
        D, imsd = main(path,diamter,distance,minmass,mpp, fps,plot=False)
        D_20.append(D)
        imsd_20.append(imsd)
    
    print(f'Theoretischer D ({40} nm Partikel):', (kb * T)/(6 * np.pi * 0.04/2*nu*(1e-6)*1e-12) ,'µm²/s')
    print("Mittelwert D 20 nm:", np.mean(D_20), "µm²/s ±", np.std(D_20), "µm²/s")


    imsds = {'D_1000': imsd_1000, 'D_500': imsd_500, 'D_200': imsd_200, 'D_50': imsd_50, 'D_20': imsd_20}
    
    with open(r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\imsds.pkl", 'wb') as f:
        pickle.dump(imsds, f)
    #%% Plot all MSDs and fit ensemble MSDs

    import argparse
    import pims
    import trackpy as tp
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt
    import pickle
    import os
    from scipy.ndimage import grey_opening
    save_path = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\MSD_Plots"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if 'fit_powerlaw_with_errors' not in globals():
        def fit_powerlaw_with_errors(em_series, points=10, ax=None, plot=False):
            """
            Fit y = A * x^n to the first `points` of em_series (pandas Series).
            Returns a SimpleNamespace similar to trackpy.utils.fit_powerlaw with error estimates.
            """
            xs = em_series.iloc[0:points].index.values.astype(float)
            ys = em_series.iloc[0:points].values.astype(float)
            mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
            if mask.sum() < 2:
                # fallback to trackpy's fitter if not enough valid points
                return tp.utils.fit_powerlaw(em_series.iloc[0:points], plot=plot, ax=ax)
            lx = np.log(xs[mask])
            ly = np.log(ys[mask])
            coeffs, cov = np.polyfit(lx, ly, 1, cov=True)
            n_fit = float(coeffs[0])
            logA_fit = float(coeffs[1])
            se = np.sqrt(np.diag(cov))
            se_n = float(se[0])
            se_logA = float(se[1])
            A_fit = float(np.exp(logA_fit))
            se_A = A_fit * se_logA
            return SimpleNamespace(
                A = np.array([A_fit]),
                n = np.array([n_fit]),
                A_err = np.array([se_A]),
                n_err = np.array([se_n]),
                logA = np.array([logA_fit]),
                logA_err = np.array([se_logA]),
                cov = cov
            )

    mpp = 0.15
    fps = 22
    diamter = 7
    #print("diameter",diamter*mpp,"µm")
    points = 40 # Range of fits
    kb = 1.380649e-23  # Boltzmann-Konstante in J/K
    T = 293.15  # Temperatur in Kelvin (25 °C)
    nu = 0.001002  # Dynamische Viskosität von Wasser bei 25 °C in Pa·s

    imsds = pickle.load(open(r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\imsds.pkl", 'rb'))
    print(type(imsds))
    D_values = {}
    for key in imsds:
        print(key)
        imsd = imsds[key]
        
        imsds_df_list = []
        for i in range(len(imsd)):
            imsd_i = imsd[i].copy()
            # imsd_i.particle
            # imsd_i['particle'] = imsd_i.columns
            # imsd[i]['particle'] = imsd[i].columns
            # imsd[i] = imsd[i].melt(id_vars=None, value_name='msd', var_name='particle')
            # imsd[i]['size'] = key
            imsds_df_list.append(imsd_i)
        
        imsd_all = pd.concat(imsds_df_list, ignore_index=True, axis=1)
        # print(imsd_all.head())
        
        plt.show()
        # convert wide per-particle MSDs to long form with a 'particle' column
        imsd_all = imsd_all.rename_axis('lag').reset_index()
        imsd_all = imsd_all.melt(id_vars='lag', var_name='particle', value_name='msd')
        # try to cast particle ids to integers when possible
        try:
            imsd_all['particle'] = imsd_all['particle'].astype(int)
        except Exception:
            pass


        


        # em = tp.emsd(imsd_all, mpp, fps)
        # convert long-form per-particle MSDs back to a wide table and compute ensemble MSD
        imsd_all = imsd_all.dropna(subset=['msd']).copy()
        # pivot to wide form (rows: lag, cols: particle)
        try:
            im = imsd_all.pivot(index='lag', columns='particle', values='msd')
        except Exception:
            im = imsd_all.pivot_table(index='lag', columns='particle', values='msd', aggfunc='mean')

        im = im.sort_index()
        # ensemble MSD: mean over particles for each lag
        em = im.mean(axis=1)
        
        
        tp.quiet()


        # call the reusable fitter (no ax available here by default)
        params = fit_powerlaw_with_errors(em, points=points, ax=None, plot=False)
        A = float(params.A[0])
        n = float(params.n[0])
        print(A,type(A), n,type(n))


        fig, ax = plt.subplots()
        # ax.plot(im.index, im[0], 'k-', alpha=0.1,label='Individual particles')
        cols = list(im.columns)
        
        ax.plot(im.index, im[cols[0]], 'k-', alpha=0.2, label='Individual MSDs')
        for c in cols[1:]:
            ax.plot(im.index, im[c], 'k-', alpha=0.08)
        ax.plot(em.index, em, 'o', markersize=8, color='blue', label='Ensemble MSD')
        ax.plot(em.iloc[0:points].index, em.iloc[0:points], 'o', markersize=3, color='red', label='Fitting range')
        ax.plot(em.iloc[0:points].index, A*np.array(em.iloc[0:points].index)**n, 'g--', linewidth=4, alpha = 0.8, label='Fit')
        ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
            xlabel='lag time $t$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.title.set_text(f'Ensemble MSD for {key} nm particles')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'MSD_{key}nm.png'), dpi=300)
        plt.show()

        # plt.figure()
 
    
 
        # fig, ax = plt.subplots()
        # # ax.set_xscale('linear')
        # # ax.set_yscale('linear')
        # ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        # ax.set_xlabel('lag time $t$');
        # ax.title.set_text(f'Fit of Ensemble MSD for {key} nm particles')
        # plt.show()


        print(params,key)
        D = params.A/4
        print("D =", D ,'µm²/s' )
        print(f'Theoretischer Durchmesser:', 2*(kb * T)/(D*nu*(6 * np.pi))*1e6*1e12 ,'µm')
        D_values[key] = D
        D_values[key+'error'] = params.A_err/(4)
    fig, ax = plt.subplots()
    print(D_values)
    d_val=[]
    d_val_err = []
    for key in imsds:
        d_val.append(D_values[key][0])
        d_val_err.append(D_values[key+'error'][0])
    print(d_val, d_val_err)
    x_vals = [20, 50, 200, 500, 1000][::-1]
    # ensure x and y lengths match
    x_vals = x_vals[:len(d_val)]
    ax.errorbar(x_vals, d_val, yerr=d_val_err, fmt='o', color='blue',
                ecolor='black', elinewidth=1, capsize=3,label='Gemessener D')
    ax.scatter([20, 50, 200, 500, 1000], [(kb * T)/(6 * np.pi * d/2*nu*(1e-6)*1e-12) for d in [0.02, 0.05, 0.2, 0.5, 1.0]], color='black', marker='x', label='Theoretischer D')
    ax.scatter([20, 50, 200, 500, 1000], [13.48,8.294646849,1.783746311,0.621773811,0.394612505], color='gray', marker='*', label='DSL Messung')
    ax.legend(['Gemessener D', 'Theoretischer D', 'DSL Messung D'])
    ax.set_ylabel('Diffusionskoeffizient D [µm²/s]')
    ax.set_xlabel('Partikelgröße [nm]')
    ax.set_title('Übersicht der Diffusionskoeffizienten')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'Diffusionskoeffizienten_Übersicht.png'), dpi=300)
    plt.show()

# %%
