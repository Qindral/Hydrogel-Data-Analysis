import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
# python -m venv .venv   
#pip install numpy scipy scikit-image matplotlib
# Load XML file
file_path = r"Z:\Diffusion in Hydrogel Data\20mg_500nm\SPT\B1_500nm_water_Tracks.xml"
title = "500nm in water"

tree = ET.parse(file_path)
root = tree.getroot()

# Constants
frame_interval = 0.05  # seconds (50 ms per frame)
dim = 2

# Parse tracks
tracks = []
for particle in root.findall(".//particle"):
    detections = []
    for d in particle.findall(".//detection"):
        t = int(float(d.get("t")))
        x = float(d.get("x"))
        y = float(d.get("y"))
        detections.append((t, x, y))
    detections.sort(key=lambda x: x[0])
    if len(detections) > 1:
        tracks.append(np.array(detections))

# Compute MSD per track
def compute_msd(track):
    t, x, y = track[:,0], track[:,1], track[:,2]
    n = len(t)
    msd = np.zeros(n-1)
    for lag in range(1, n):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd[lag-1] = np.mean(dx**2 + dy**2)
    return msd

all_msds = [compute_msd(track) for track in tracks if len(track) > 1]

# Align MSD lengths
min_len = min(len(m) for m in all_msds)
aligned_msds = np.array([m[:min_len] for m in all_msds])
ensemble_msd = np.mean(aligned_msds, axis=0)
time_lags = np.arange(1, min_len + 1) * frame_interval

# Linear fit in early regime
fit_end = min_len // 2
slope, intercept, r, p, se = linregress(time_lags[:fit_end], ensemble_msd[:fit_end])
D = slope / (4)  # µm²/s for 2D diffusion

# Plot MSDs
plt.figure(figsize=(7,5))
for msd in all_msds[:]:  # plot some single MSDs
    n_points = min(len(msd), len(time_lags))
    plt.plot(time_lags[:n_points], msd[:n_points], color='lightgray', alpha=0.6)
plt.plot(time_lags, ensemble_msd, 'o-', color='blue', label='Ensemble MSD')
plt.plot(time_lags[:fit_end], intercept + slope*time_lags[:fit_end], 'r--', label=f'Linear fit\nD={D:.3e} µm²/s')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Δt [s]')
plt.ylabel('MSD [µm²]')
plt.legend()
plt.title(f'MSD Analysis: {title}')
plt.tight_layout()
plt.show()

D
