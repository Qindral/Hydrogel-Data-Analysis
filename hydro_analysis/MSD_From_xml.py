import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
# python -m venv .venv   
#PS C:\Users\Jonas\Documents\GitHub\Hydrogel-Data-Analysis> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#pip install numpy scipy scikit-image matplotlib

plt.rcParams.update({ 'font.family': 'serif', 'font.serif': ['Arial'], 'font.size': 12, 'axes.linewidth': 3, 'axes.labelsize': 12, 'axes.edgecolor': 'black', 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.width': 3, 'ytick.major.width': 2, 'xtick.major.size': 5, 'ytick.major.size': 5, 'xtick.top': True, 'ytick.right': True, 'legend.frameon': True, 'legend.fontsize': 12, 'legend.title_fontsize': 12, 'lines.linewidth': 2.5, 'lines.markersize': 8, 'figure.figsize': [6, 6 / np.sqrt(2)], 'savefig.bbox': 'tight', 'figure.autolayout': True, 'axes.grid': False })


# Load XML file
file_path = r"Z:\Diffusion in Hydrogel Data\20mg_500nm\SPT\B1_500nm_water_Tracks.xml"
file_path = r"Z:\Diffusion in Hydrogel Data\20mg_50nm\A2_50nm_20mg_4d_newpos_Tracks.xml"
file_path = r"Z:\Diffusion in Hydrogel Data\20mg_20nm\Trajektorien\ResultofB1_20nm_20mg_1d_nichtzentral_1_ohne_Tracks.xml"
file_path = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml"
#file_path = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\preprocess\tracks\20 nm_2_processed_Tracks.xml"

title = "20nm 20mg 1d nichtzentral 1"
print(title)
tree = ET.parse(file_path)
root = tree.getroot()

# Constants
frame_interval = 0.05  # seconds (50 ms per frame)
dim = 2
dx = 0.150  # µm per pixel
# Parse tracks
tracks = []
for particle in root.findall(".//particle"):
    detections = []
    for d in particle.findall(".//detection"):
        t = int(float(d.get("t")))
        x = float(d.get("x")) * dx
        y = float(d.get("y")) * dx
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
print(  f"Total tracks: {len(tracks)}")
all_msds = [compute_msd(track) for track in tracks if len(track) > 1]
print(  f"Tracks with MSD computed: {len(all_msds)}")
#all_msds = [m for m in all_msds if m[-1] < 14]
print(  f"Tracks after filtering long MSDs: {len(all_msds)}")
# Align MSD lengths
min_len = min(len(m) for m in all_msds)
aligned_msds = np.array([m[:min_len] for m in all_msds])
ensemble_msd = np.mean(aligned_msds, axis=0)
time_lags = np.arange(1, min_len + 1) * frame_interval

# Linear fit in early regime
fit_end = min_len // 2
slope, intercept, r, p, se = linregress(time_lags[:fit_end], ensemble_msd[:fit_end])
D = slope / (4)  # µm²/s for 2D diffusion

print(f"Diffusion Coefficient D: {D} µm²/s")


# Plot MSDs
plt.figure(figsize=(7,5))
for msd in all_msds[:]:  # plot some single MSDs
    n_points = min(len(msd), len(time_lags))
    plt.plot(time_lags[:n_points], msd[:n_points], color='lightgray', alpha=0.6)
plt.plot(time_lags, ensemble_msd, 'o-', color='blue', label='Ensemble MSD')
plt.plot(time_lags[:fit_end], intercept + slope*time_lags[:fit_end], 'r--', label=f'Linear fit\nD={D:.3} µm²/s')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Δt [s]')
plt.ylabel('MSD [µm²]')
plt.legend()
plt.title(f'MSD Analysis: {title}')
plt.tight_layout()
plt.show()


