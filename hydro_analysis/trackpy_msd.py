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

# 1. Daten laden (pims öffnet den Stack 'lazy', also speicherschonend)
# Ersetze 'dein_stack.tif' mit deinem Pfad
frames = pims.open(r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\1000 nm_4.tif")

print(f"Anzahl der Frames: {len(frames)}")
print(f"Bildgröße: {frames[0].shape}")

# 2. Parameter-Einstellung (WICHTIGSTER SCHRITT FÜR KONTROLLE)
# diameter: Geschätzter Durchmesser eines Partikels in Pixeln (Muss ungerade sein!)
# minmass: Mindest-integrierte Helligkeit, um Rauschen zu filtern.
DIAMETER = 7  # <-- Hier anpassen (ungerade Zahl, z.B. 9, 11, 13)

# Testlauf auf dem ersten Frame (Frame 0)
f = tp.locate(frames[0], diameter=DIAMETER, invert=False, minmass=2000) 
# invert=True nutzen, falls Partikel dunkel auf hellem Grund sind

# 3. Visualisierung der Erkennung

tp.annotate(f, frames[0])

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=6)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count')
plt.show()

#tp.subpx_bias(f)
#plt.show()
f = tp.batch(frames[:], 17, minmass=2000)
t = tp.link(f, 5, memory=3)
t1 = tp.filter_stubs(t, 25)

d = tp.compute_drift(t1)

plt.figure()
tp.plot_traj(t2);

# Prepare figure
n_frames = len(frames)
h, w = frames[0].shape
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(frames[0], cmap='gray', interpolation='nearest')
ax.set_xlim(0, w)
ax.set_ylim(h, 0)  # invert y-axis so image coords match trackpy (y increases downward)
ax.set_xlabel('x (px)')
ax.set_ylabel('y (px)')
title = ax.set_title("Frame 0")

# Prepare particles (one color per particle)
pids = np.unique(t1['particle'])
cmap = plt.cm.get_cmap('tab20', len(pids))
pid_to_color = {pid: cmap(i % cmap.N) for i, pid in enumerate(pids)}

# Pre-create Line2D objects (trajectories) and Circle patches (current spot rings)
lines = {}
circles = {}
radius = DIAMETER / 2.0
for pid in pids:
    line, = ax.plot([], [], lw=1.5, color=pid_to_color[pid], alpha=0.9)
    circ = Circle((0, 0), radius=radius, edgecolor=pid_to_color[pid],
                  facecolor='none', lw=1.2, visible=False)
    ax.add_patch(circ)
    lines[pid] = line
    circles[pid] = circ

# Try to make drift lookup function (works if d is indexable by frame)
def get_drift_at(frame_idx):
    try:
        # If d is a DataFrame with index=frame and columns ['x','y']
        dx = float(d.loc[frame_idx, 'x'])
        dy = float(d.loc[frame_idx, 'y'])
        return dx, dy
    except Exception:
        try:
            # If d is numpy-like array or list of (x,y)
            dx, dy = d[frame_idx]
            return float(dx), float(dy)
        except Exception:
            return 0.0, 0.0

# Animation update function
def update(frame_idx):
    # update image
    im.set_array(frames[frame_idx])
    title.set_text(f"Frame {frame_idx + 1}/{n_frames}")

    # drift to subtract (if available)
    dx, dy = get_drift_at(frame_idx)

    # update each particle
    for pid in pids:
        hist = t1[(t1['particle'] == pid) & (t1['frame'] <= frame_idx)].sort_values('frame')
        if hist.empty:
            lines[pid].set_data([], [])
            circles[pid].set_visible(False)
            continue

        x = hist['x'].to_numpy() - dx
        y = hist['y'].to_numpy() - dy
        lines[pid].set_data(x, y)

        # current position (at this exact frame)
        current = hist[hist['frame'] == frame_idx]
        if not current.empty:
            cx = float(current['x'].iloc[-1] - dx)
            cy = float(current['y'].iloc[-1] - dy)
            circles[pid].center = (cx, cy)
            circles[pid].set_visible(True)
        else:
            circles[pid].set_visible(False)

    return [im, title] + list(lines.values()) + list(circles.values())

# Create and run animation
anim = FuncAnimation(fig, update, frames=range(n_frames), interval=100, blit=False, repeat=True)

plt.show()




