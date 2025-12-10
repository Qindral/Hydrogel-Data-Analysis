import xml.etree.ElementTree as ET
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
import re
from collections import defaultdict
from collections import defaultdict as _defaultdict
save_path = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\trackmate_MSD_results"
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
paths =[] # Dummy
kb = 1.380649e-23  # Boltzmann-Konstante in J/K


def combine_and_analyze(paths_list, save_path=save_path, mpp=mpp, fps=fps, points=10):
    """
    Combine tracks per nominal particle size (detected from filename),
    compute ensemble MSD on the combined tracks, fit a power-law and
    plot/save comparison to theory.
    Returns dicts: combined_D, combined_D_err (both keyed by size_nm).
    """
    # collect raw DataFrames per size
    dfs_by_size = _defaultdict(list)
    for p in paths_list:
        name = os.path.basename(p)
        m = re.search(r'(\d+(?:\.\d+)?)\s*nm', name, re.I)
        if m:
            size_nm = int(float(m.group(1)))
        else:
            nums = [int(x) for x in re.findall(r'(\d+)', name)]
            size_nm = next((n for n in nums if 20 <= n <= 1000), None)
        if size_nm is None:
            continue
        df = read_trackmate_xml(p)
        if df is None or df.empty:
            continue
        dfs_by_size[size_nm].append(df)

    combined_D = {}
    combined_D_err = {}

    for size_nm, dflist in dfs_by_size.items():
        # Concatenate tracks, ensuring unique particle IDs across files
        if not dflist:
            continue
        # renumber particle ids to be unique across files
        offset = 0
        reindexed = []
        for df in dflist:
            df = df.copy()
            df['particle'] = df['particle'].astype(int) + offset
            max_id = df['particle'].max()
            offset = max(offset, max_id + 1)
            reindexed.append(df)
        df_combined = pd.concat(reindexed, ignore_index=True)

        # ensure dtypes
        df_combined['frame'] = df_combined['frame'].astype(int)
        df_combined['particle'] = df_combined['particle'].astype(int)

        # compute MSDs on combined dataset
        tp.quiet()
        try:
            # linking optional; if tracks already labeled this will be skipped gracefully
            tp.link(df_combined, 12, memory=8)
        except Exception:
            pass

        im = tp.imsd(df_combined, mpp, fps)
        em = tp.emsd(df_combined, mpp, fps)

        params = fit_powerlaw_with_errors(em, points=points, plot=False)
        A = float(params.A[0])
        A_err = float(params.A_err[0]) if hasattr(params, 'A_err') else np.nan
        D = A / 4.0
        D_err = A_err / 4.0

        combined_D[size_nm] = D
        combined_D_err[size_nm] = D_err

        # plot MSD for this combined size
        fig, ax = plt.subplots()
        cols = list(im.columns)
        if cols:
            ax.plot(im.index, im[cols[0]], 'k-', alpha=0.2, label='Individual MSDs')
            for c in cols[1:]:
                ax.plot(im.index, im[c], 'k-', alpha=0.08)
        ax.plot(em.index, em, 'o', markersize=6, color='blue', label='Ensemble MSD (combined)')
        ax.plot(em.iloc[0:points].index, em.iloc[0:points], 'o', markersize=4, color='red', label='Fitting range')
        ax.plot(em.iloc[0:points].index, A*np.array(em.iloc[0:points].index)**float(params.n[0]), 'g--', linewidth=2, alpha=0.8, label=f'Fit: A={A:.2e}, n={float(params.n[0]):.2f}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('lag time [frames]')
        ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
        ax.set_title(f'Combined ensemble MSD for {size_nm} nm (Nfiles={len(dflist)})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'combined_MSD_{size_nm}nm.png'), dpi=300)
        plt.close(fig)

    # optional: plot combined D vs theory like earlier
    sizes = sorted(combined_D.keys())
    measured_mean = [combined_D[s] for s in sizes]
    measured_err = [combined_D_err[s] for s in sizes]
    theory_aligned = []
    for s in sizes:
        if s in [20,50,200,500,1000]:
            d_m = s * 1e-9
            R = d_m / 2.0
            D_m2_s = kb * T / (6 * np.pi * nu * R)
            D_um2_s = D_m2_s * 1e12
            theory_aligned.append(D_um2_s)
        else:
            theory_aligned.append(np.nan)

    if sizes:
        fig, ax = plt.subplots()
        ax.errorbar(sizes, measured_mean, yerr=measured_err, fmt='o', color='tab:blue', label='Combined measured D')
        ax.plot(sizes, theory_aligned, 'x', color='black', label='Theoretical D (canonical sizes)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Particle size [nm]')
        ax.set_ylabel('Diffusion coefficient D [µm²/s]')
        ax.set_title('Combined measured vs theoretical D')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'combined_D_measured_vs_theoretical.png'), dpi=300)
        plt.close(fig)

    return combined_D, combined_D_err
T = 293.15  # Temperatur in Kelvin (20 °C)
nu = 0.001002  # Dynamische Viskosität von Wasser bei 25 °C in Pa·s
def read_trackmate_xml(xml_file_path):
    """
    Liest eine XML-Datei (TrackMate Format) und konvertiert sie in ein Pandas DataFrame.
    Extrahiert nur frame, particle, x und y.
    """
    try:
        # 1. XML Datei parsen
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        data_rows = []
        
        # 2. Durch alle 'particle' Elemente iterieren
        # Wir nutzen enumerate, um eine ID für das Partikel zu erzeugen (0, 1, 2...)
        for particle_id, particle in enumerate(root.findall('particle')):
            
            # 3. Innerhalb jedes Partikels durch alle 'detection' Elemente iterieren
            for detection in particle.findall('detection'):
                # Attribute extrahieren
                t_raw = detection.get('t')
                x_raw = detection.get('x')
                y_raw = detection.get('y')
                
                # In Dictionary speichern und Typen umwandeln
                row = {
                    'frame': int(float(t_raw)),  # float->int, falls t als "40.0" gespeichert ist
                    'particle': particle_id + 1, # +1 damit Partikel bei 1 starten (optional)
                    'x': float(x_raw),
                    'y': float(y_raw)
                }
                data_rows.append(row)
        
        # 4. DataFrame erstellen
        df = pd.DataFrame(data_rows)
        
        # Sicherstellen, dass die Spalten in der gewünschten Reihenfolge sind
        if not df.empty:
            df = df[['frame', 'particle', 'x', 'y']]
            
            # Optional: Sortieren nach Frame und Partikel für bessere Lesbarkeit
            df = df.sort_values(by=['frame', 'particle']).reset_index(drop=True)
            
        return df

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None

paths = [r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\Resultof50nm_3_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\Resultof50nm_2_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\1000nm.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20nm_2.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20nm_2_tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\Resultof20nm_5_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\Resultof20nm_4_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\Resultof20nm_3_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\Resultof20nm_2_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\50 nm_5_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\50 nm_4_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\50 nm_3_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\50 nm_2_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\50 nm_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\200 nm_2_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\200 nm_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\500 nm_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\500 nm _4_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\500 nm _3_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\500 nm _2_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\1000 nm_4_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\1000 nm_3_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\1000 nm_2_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\1000 nm_Tracks.xml",
r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\ResultofResultof20nm_2_Tracks.xml"]
# Dateipfad hier anpassen
# group measured D (µm^2/s) and errors by nominal size (nm)
measured_by_size = defaultdict(list)
measured_err_by_size = defaultdict(list)

for p in paths:
    name = os.path.basename(p)
    # try to extract "<number>nm" first, otherwise pick first number between 20 and 1000
    m = re.search(r'(\d+(?:\.\d+)?)\s*nm', name, re.I)
    if m:
        size_nm = int(float(m.group(1)))
    else:
        nums = [int(x) for x in re.findall(r'(\d+)', name)]
        size_nm = next((n for n in nums if 20 <= n <= 1000), None)
    if size_nm is None:
        print(f"Could not determine size from filename, skipping: {name}")
        continue

    df = read_trackmate_xml(p)
    if df is None or df.empty:
        print(f"No data in {name}, skipping")
        continue

    try:
        # ensure correct dtypes
        df['frame'] = df['frame'].astype(int)
        df['particle'] = df['particle'].astype(int)

        # compute MSDs and fit
        tp.quiet()
        # linking is optional if particles already labeled; keep as in original script
        try:
            tp.link(df, 12, memory=8)
        except Exception:
            # linking may fail or be unnecessary; continue anyway
            pass

        im = tp.imsd(df, mpp, fps)
        em = tp.emsd(df, mpp, fps)
        params = fit_powerlaw_with_errors(em, points=10, plot=False)
        A = float(params.A[0])
        A_err = float(params.A_err[0]) if hasattr(params, 'A_err') else np.nan
        D = A / 4.0              # µm^2/s (same units as emsd when mpp in µm)
        D_err = A_err / 4.0

        measured_by_size[size_nm].append(D)
        measured_err_by_size[size_nm].append(D_err)

        print(f"{name}: size={size_nm} nm, D={D:.3e} µm^2/s")

    except Exception as exc:
        print(f"Error processing {name}: {exc}")
        continue

# aggregate results: mean and standard error (or mean reported error if only one file)
sizes = sorted(measured_by_size.keys())
measured_mean = []
measured_sem = []
for s in sizes:
    vals = np.array(measured_by_size[s], dtype=float)
    errs = np.array(measured_err_by_size[s], dtype=float)
    if len(vals) == 0:
        measured_mean.append(np.nan); measured_sem.append(np.nan); continue
    measured_mean.append(np.nanmean(vals))
    if len(vals) > 1:
        measured_sem.append(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
    else:
        # fall back to reported fit error if only one file
        measured_sem.append(np.nanmean(errs) if errs.size>0 else np.nan)

# theoretical D for canonical diameters (20,50,200,500,1000 nm) in µm^2/s
canonical = [20, 50, 200, 500, 1000]
theory_D = {}
for d_nm in canonical:
    d_m = d_nm * 1e-9
    R = d_m / 2.0
    D_m2_s = kb * T / (6 * np.pi * nu * R)        # m^2/s
    D_um2_s = D_m2_s * 1e12                       # convert to µm^2/s
    theory_D[d_nm] = D_um2_s

# prepare theory array aligned with measured sizes
theory_aligned = [theory_D.get(s, np.nan) for s in sizes]

# plot comparison
fig, ax = plt.subplots()
ax.errorbar(sizes, measured_mean, yerr=measured_sem, fmt='o', color='tab:blue', label='Measured D (mean ± SEM)')
ax.plot(sizes, theory_aligned, 'x', color='black', label='Theoretical D')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Particle size [nm]')
ax.set_ylabel('Diffusion coefficient D [µm²/s]')
ax.set_title('Measured vs theoretical diffusion coefficients')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'D_measured_vs_theoretical.png'), dpi=300)
plt.show()

# print summarized table
print("Size (nm) | Measured D (µm²/s) ± SEM | Theory D (µm²/s)")
for s, m, e, t in zip(sizes, measured_mean, measured_sem, theory_aligned):
    print(f"{s:8d} | {m: .3e} ± {e: .3e} | {t: .3e}")
fig, ax = plt.subplots()
points=10
# Funktion aufrufen
df_tracks = read_trackmate_xml(file_path)
t = tp.link(df_tracks, 12 , memory=8)
im = tp.imsd(df_tracks, mpp, fps) 
em = tp.emsd(df_tracks, mpp, fps)
print(em.head())
params = fit_powerlaw_with_errors(em, points=points,  plot=True)
A = float(params.A[0])
n = float(params.n[0])
print(A,type(A), n,type(n))
# ax.plot(im.index, im[0], 'k-', alpha=0.1,label='Individual particles')
cols = list(im.columns)

ax.plot(im.index, im[cols[0]], 'k-', alpha=0.2, label='Individual MSDs')
for c in cols[1:]:
    ax.plot(im.index, im[c], 'k-', alpha=0.08)
ax.plot(em.index, em, 'o', markersize=8, color='blue', label='Ensemble MSD')
ax.plot(em.iloc[0:points].index, em.iloc[0:points], 'o', markersize=3, color='red', label='Fitting range')
ax.plot(em.iloc[0:points].index, A*np.array(em.iloc[0:points].index)**n, 'g--', linewidth=4, alpha = 0.8, label='Fit')

#ax.plot(em.iloc[0:points].index, 4*Dthr*np.array(em.iloc[0:points].index)**1, 'p--', linewidth=4, alpha = 0.8, label='Theory')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
    xlabel='lag time $t$')
ax.set_xscale('log')
ax.set_yscale('log')
# ax.title.set_text(f'Ensemble MSD for {key} nm particles')
ax.legend()
plt.tight_layout()
# plt.savefig(os.path.join(save_path, f'MSD_{key}nm.png'), dpi=300)
plt.show()
D = params.A/4
print("D =", D ,'µm²/s' )
print(f'Theoretischer Durchmesser:', 2*(kb * T)/(D*nu*(6 * np.pi))*1e6*1e12 ,'µm')
D_values[key] = D
D_values[key+'error'] = params.A_err/(4)
# # Ergebnis anzeigen
# D_values = {}
# for j,key in enumerate(imsds):
#     print(key)
#     imsd = imsds[key]
#     Dthr = [13.48,8.294646849,1.783746311,0.621773811,0.394612505][::-1][j]
#     imsds_df_list = []
#     for i in range(len(imsd)):
#         imsd_i = imsd[i].copy()
#         imsds_df_list.append(imsd_i)
    
#     imsd_all = pd.concat(imsds_df_list, ignore_index=True, axis=1)
#     # print(imsd_all.head())
    
#     plt.show()
#     # convert wide per-particle MSDs to long form with a 'particle' column
#     imsd_all = imsd_all.rename_axis('lag').reset_index()
#     imsd_all = imsd_all.melt(id_vars='lag', var_name='particle', value_name='msd')
#     # try to cast particle ids to integers when possible
#     try:
#         imsd_all['particle'] = imsd_all['particle'].astype(int)
#     except Exception:
#         pass


    


#     # em = tp.emsd(imsd_all, mpp, fps)
#     # convert long-form per-particle MSDs back to a wide table and compute ensemble MSD
#     imsd_all = imsd_all.dropna(subset=['msd']).copy()
#     # pivot to wide form (rows: lag, cols: particle)
#     try:
#         im = imsd_all.pivot(index='lag', columns='particle', values='msd')
#     except Exception:
#         im = imsd_all.pivot_table(index='lag', columns='particle', values='msd', aggfunc='mean')

#     im = im.sort_index()
#     # ensemble MSD: mean over particles for each lag
#     em = im.mean(axis=1)
    
    
#     tp.quiet()


#     # call the reusable fitter (no ax available here by default)
#     params = fit_powerlaw_with_errors(em, points=points, ax=None, plot=False)
#     A = float(params.A[0])
#     n = float(params.n[0])
#     print(A,type(A), n,type(n))


#     fig, ax = plt.subplots()
#     # ax.plot(im.index, im[0], 'k-', alpha=0.1,label='Individual particles')
#     cols = list(im.columns)
    
#     ax.plot(im.index, im[cols[0]], 'k-', alpha=0.2, label='Individual MSDs')
#     for c in cols[1:]:
#         ax.plot(im.index, im[c], 'k-', alpha=0.08)
#     ax.plot(em.index, em, 'o', markersize=8, color='blue', label='Ensemble MSD')
#     ax.plot(em.iloc[0:points].index, em.iloc[0:points], 'o', markersize=3, color='red', label='Fitting range')
#     ax.plot(em.iloc[0:points].index, A*np.array(em.iloc[0:points].index)**n, 'g--', linewidth=4, alpha = 0.8, label='Fit')
    
#     ax.plot(em.iloc[0:points].index, 4*Dthr*np.array(em.iloc[0:points].index)**1, 'p--', linewidth=4, alpha = 0.8, label='Theory')
#     ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
#         xlabel='lag time $t$')
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.title.set_text(f'Ensemble MSD for {key} nm particles')
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f'MSD_{key}nm.png'), dpi=300)
#     plt.show()

#     plt.figure()



#     fig, ax = plt.subplots()
#     # ax.set_xscale('linear')
#     # ax.set_yscale('linear')
#     ax.set_ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
#     ax.set_xlabel('lag time $t$');
#     ax.title.set_text(f'Fit of Ensemble MSD for {key} nm particles')
#     plt.show()


#     print(params,key)
#     D = params.A/4
#     print("D =", D ,'µm²/s' )
#     print(f'Theoretischer Durchmesser:', 2*(kb * T)/(D*nu*(6 * np.pi))*1e6*1e12 ,'µm')
#     D_values[key] = D
#     D_values[key+'error'] = params.A_err/(4)
# fig, ax = plt.subplots()
# print(D_values)
# d_val=[]
# d_val_err = []
# for key in imsds:
#     d_val.append(D_values[key][0])
#     d_val_err.append(D_values[key+'error'][0])
# # print(d_val, d_val_err)
# x_vals = [20, 50, 200, 500, 1000][::-1]
# # ensure x and y lengths match
# x_vals = x_vals[:len(d_val)]
# ax.errorbar(x_vals, d_val, yerr=d_val_err, fmt='o', color='blue',
#             ecolor='black', elinewidth=1, capsize=3,label='Gemessener D')
# ax.scatter([20, 50, 200, 500, 1000], [(kb * T)/(6 * np.pi * d/2*nu*(1e-6)*1e-12) for d in [0.02, 0.05, 0.2, 0.5, 1.0]], color='black', marker='x', label='Theoretischer D')
# ax.scatter([20, 50, 200, 500, 1000], [13.48,8.294646849,1.783746311,0.621773811,0.394612505], color='gray', marker='*', label='DSL Messung')
# ax.legend(['Gemessener D', 'Theoretischer D', 'DSL Messung D'])
# ax.set_ylabel('Diffusionskoeffizient D [µm²/s]')
# ax.set_xlabel('Partikelgröße [nm]')
# ax.set_title('Übersicht der Diffusionskoeffizienten')
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.tight_layout()
# plt.savefig(os.path.join(save_path, f'Diffusionskoeffizienten_Übersicht.png'), dpi=300)
# plt.show()
