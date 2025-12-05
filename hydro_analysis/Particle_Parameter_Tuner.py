import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.collections import LineCollection
import pims
import trackpy as tp
import cv2
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import filedialog
import pandas as pd

# --- Helper: Fast Rolling Ball ---
def fast_rolling_ball(image, radius):
    if radius < 1: return image
    k_size = int(radius * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(image, background)

class ParticleTrackerUI:
    def __init__(self, tif_path):
        print("Loading frames...")
        self.frames = pims.open(tif_path)
        self.n_frames = len(self.frames)
        
        # 1. Pre-calculate Static Background
        print("Calculating static background...")
        # Robust background: compute per-pixel median across all frames to avoid
        # per-frame scaling artifacts that can lead to over-subtraction (black images).
        frames_stack = [np.asarray(fr, dtype=np.float32) for fr in self.frames]
        if len(frames_stack) == 0:
            # Fallback in the unlikely event there are no frames
            self.static_background = np.zeros_like(np.asarray(self.frames[0], dtype=np.float32))
        else:
            # Stack frames for vectorized ops
            stack = np.stack(frames_stack, axis=0)  # shape (N, H, W)
            # Per-frame mean intensity
            frame_means = stack.mean(axis=(1, 2))
            # Use the first frame as reference mean (can be changed if desired)
            ref_mean = float(frame_means[0])
            # Avoid division by zero: if a frame mean is zero, keep scale 1.0
            scales = (ref_mean / frame_means).astype(np.float32)
            # Apply normalization (broadcast scales over H,W)
            normalized_stack = stack * scales
            # Compute static background from normalized frames (median)
            self.static_background = np.median(normalized_stack, axis=0).astype(np.float32)

        # --- Internal State ---
        self.tracks = None  # Will hold the DataFrame after linking
        self.has_tracks = False
        # Default Parameters
        self.params = {
            'frame': 12,
            'diameter': 9,
            'minmass': 420,
            'radius': 0,    # Rolling ball
            'smooth': 0.0,  # Gaussian
            'search_range': 12.0, # Linking distance
            'memory': 4     # Linking gap
        }

        # --- UI Layout ---
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Particle Tracker UI')
        
        # Main Image (Top Left), Plots (Right)
        gs = self.fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.6])
        self.ax_img = self.fig.add_subplot(gs[0:2, 0:3])
        self.ax_hist = self.fig.add_subplot(gs[0, 3])
        self.ax_bias = self.fig.add_subplot(gs[1, 3])
        plt.subplots_adjust(bottom=0.35)

        # --- Controls Generation ---
        # Helper to create Slider + Textbox pairs
        self.widgets = {}
        
        def add_control(label, key, val_min, val_max, y_pos, val_step=1, is_int=False):
            # Slider Axis
            ax_s = plt.axes([0.1, y_pos, 0.35, 0.03])
            # Textbox Axis
            ax_t = plt.axes([0.48, y_pos, 0.06, 0.03])
            
            s = Slider(ax_s, label, val_min, val_max, valinit=self.params[key], valstep=val_step)
            t = TextBox(ax_t, '', initial=str(self.params[key]))
            
            # Update Logic
            def update_from_slider(val):
                self.params[key] = int(val) if is_int else val
                t.set_val(str(round(val, 2))) # Sync Text
                self.update_plot()
                
            def update_from_text(text):
                try:
                    val = float(text)
                    if is_int: val = int(val)
                    # Update slider (this triggers update_from_slider, which updates plot)
                    s.set_val(val) 
                except ValueError:
                    pass

            s.on_changed(update_from_slider)
            t.on_submit(update_from_text)
            
            self.widgets[key] = {'slider': s, 'text': t}

        # Row 1: Image Processing
        add_control('Frame', 'frame', 0, self.n_frames-1, 0.28, 1, True)
        add_control('Diameter (odd)', 'diameter', 3, 31, 0.24, 2, True)
        add_control('Min Mass', 'minmass', 0, 5000, 0.20, 10, False)
        add_control('RB Radius', 'radius', 0, 50, 0.16, 1, True)
        add_control('Smooth Sigma', 'smooth', 0, 5, 0.12, 0.1, False)

        # Row 2: Linking Parameters (Right side)
        add_control('Search Range (px)', 'search_range', 0, 50, 0.24, 0.5, False)
        # Manually move the linking widgets to the right
        self.widgets['search_range']['slider'].ax.set_position([0.60, 0.24, 0.25, 0.03])
        self.widgets['search_range']['text'].ax.set_position([0.87, 0.24, 0.06, 0.03])
        
        add_control('Memory (Gap)', 'memory', 0, 10, 0.20, 1, True)
        self.widgets['memory']['slider'].ax.set_position([0.60, 0.20, 0.25, 0.03])
        self.widgets['memory']['text'].ax.set_position([0.87, 0.20, 0.06, 0.03])

        # Buttons
        ax_btn_track = plt.axes([0.60, 0.12, 0.15, 0.05])
        self.btn_track = Button(ax_btn_track, 'Find & Draw Trajectories', color='lightblue', hovercolor='0.9')
        self.btn_track.on_clicked(self.run_tracking)
        
        ax_btn_print = plt.axes([0.80, 0.12, 0.1, 0.05])
        self.btn_print = Button(ax_btn_print, 'Print Params', hovercolor='0.9')
        self.btn_print.on_clicked(self.print_params)

        # Initial Draw
        self.update_plot()
        plt.show()

    def process_frame(self, frame_idx):
        raw = np.asarray(self.frames[frame_idx], dtype=np.float32)
        img = np.clip(raw - self.static_background, 0.0, None).astype(np.float32)
        if self.params['radius'] > 0:
            img = fast_rolling_ball(img, self.params['radius'])
        if self.params['smooth'] > 0:
            img = gaussian_filter(img, sigma=self.params['smooth'])
        return img

    def run_tracking(self, event):
        print("\n--- Starting Batch Process & Linking ---")
        print(f"1. Locating features in {len(self.frames)} frames (this may take time)...")
        
        # We need to perform the same preprocessing as the UI
        # Since pims is lazy, we iterate and process
        frame_process = []
        for i in range(len(self.frames)):
            frame_process.append(self.process_frame(i))
        
        # tp.batch can accept a generator or list
        # To ensure diameter is odd
        d = int(self.params['diameter'])
        if d % 2 == 0:
            d += 1
        
        # Run Batch
        tp.quiet()
        f = tp.batch(frame_process, diameter=d, minmass=self.params['minmass'], invert=False)
        # Ensure 'frame' is a column (tp.batch may put frame into the index)
        if 'frame' not in f.columns:
            try:
                f = f.reset_index()
            except Exception:
                pass
        
        print(f"2. Found {len(f)} features. Linking trajectories...")
        print(f"   Range: {self.params['search_range']}, Memory: {self.params['memory']}")
        
        # Run Link
        t1 = tp.link(f, int(self.params['search_range']), memory=int(self.params['memory']))
        
        # Ensure 'frame' and 'particle' are columns after linking; if they are in the index, bring them out
        if ('frame' not in t1.columns) or ('particle' not in t1.columns):
            try:
                t1 = t1.reset_index()
            except Exception:
                pass

        # Save linked tracks to the UI and enable tracked mode
        try:
            # store the linked DataFrame so update_plot can use it
            self.tracks = t1
            self.has_tracks = True
            print(f"3. Linked features into {t1['particle'].nunique()} unique trajectories.")
        except Exception as e:
            print("Warning: could not save tracks:", e)
        
    def update_plot(self):
        curr_frame = self.params['frame']
        
        # 1. Image Processing
        img = self.process_frame(curr_frame)
        
        self.ax_img.clear()
        self.ax_img.imshow(img, cmap='gray', origin='upper')
        self.ax_img.axis('off')

        current_particles = pd.DataFrame()  # default empty

        # 2. Particle Detection (Current Frame)
        if self.has_tracks and self.tracks is not None and len(self.tracks) > 0:
            # Ensure the tracked DataFrame exposes 'frame' and 'particle' as columns
            if ('frame' not in self.tracks.columns) or ('particle' not in self.tracks.columns):
                try:
                    self.tracks = self.tracks.reset_index()
                except Exception:
                    pass

            # Slicing the pre-calculated dataframe for the current frame
            curr_frame_int = int(curr_frame)
            if 'frame' in self.tracks.columns:
                current_particles = self.tracks[self.tracks['frame'] == curr_frame_int].copy()
            else:
                # If no frame information, treat as empty
                current_particles = pd.DataFrame()

            print(f"Displaying {len(current_particles)} particles from tracked data.")
            # --- DRAW TRAILS (Fading 30 steps) ---
            # Get data from (Current - 30) to Current
            start_lookback = max(0, int(curr_frame) - 30)

            if 'frame' in self.tracks.columns:
                history = self.tracks[
                    (self.tracks['frame'] >= start_lookback) &
                    (self.tracks['frame'] <= curr_frame_int)
                ]
            else:
                history = pd.DataFrame()

            if current_particles.empty:
                self.ax_img.set_title(f"Frame {curr_frame_int} | Tracking Mode | Particles: 0")
            else:
                # Only proceed if 'particle' column exists
                if 'particle' not in current_particles.columns:
                    print("Warning: 'particle' column missing in tracked data; skipping trail drawing.")
                    # Just draw current positions if x,y exist
                    if 'x' in current_particles.columns and 'y' in current_particles.columns:
                        self.ax_img.plot(current_particles.x, current_particles.y, 'o', color='red', markersize=4)
                        self.ax_img.set_title(f"Frame {curr_frame_int} | Tracking Mode | Particles: {len(current_particles)}")
                else:
                    # We only care about particles that exist in the CURRENT frame
                    active_ids = current_particles['particle'].unique()
                    active_history = history[history['particle'].isin(active_ids)]
                    print(f"Drawing trails for {len(active_ids)} active particles, {len(active_history)} total points in history.")
                    if not active_history.empty:
                        # Group by particle to draw lines
                        for pid, group in active_history.groupby('particle'):
                            # Plot the line (trail)
                            # Alpha 0.6 makes it look slightly transparent/faded
                            self.ax_img.plot(group.x, group.y, '-', color='lime', linewidth=1.5, alpha=0.6)

                    # Draw current heads
                    if 'x' in current_particles.columns and 'y' in current_particles.columns:
                        self.ax_img.plot(current_particles.x, current_particles.y, 'o', color='red', markersize=4)
                    self.ax_img.set_title(f"Frame {curr_frame_int} | Tracking Mode | Particles: {len(current_particles)}")
                    print(f"Plotted {len(current_particles.index)} current particles.")
        else:
            # LIVE PREVIEW MODE
            d = int(self.params['diameter'])
            if d % 2 == 0:
                d += 1
            f = tp.locate(img, diameter=d, minmass=self.params['minmass'], invert=False)
            current_particles = f
            if len(f) > 0:
                tp.annotate(f, img, ax=self.ax_img)
            self.ax_img.set_title(f"Frame {curr_frame} | Preview Mode | Particles: {len(f)}")

        # 3. Update Histograms
        self.ax_hist.clear()
        self.ax_bias.clear()
        
        try:
            cp_len = len(current_particles)
        except Exception:
            cp_len = 0

        if cp_len > 0 and 'mass' in current_particles.columns:
            self.ax_hist.hist(current_particles['mass'], bins=20, color='skyblue', edgecolor='black')
            self.fig.canvas.draw_idle()
            self.ax_hist.axvline(self.params['minmass'], color='r', linestyle='--')
            self.ax_hist.set_title('Mass')
            
            # Subpixel Bias
            self.ax_bias.hist(current_particles['x'] % 1, bins=10, alpha=0.5, color='red', label='x')
            self.ax_bias.hist(current_particles['y'] % 1, bins=10, alpha=0.5, color='blue', label='y')
            self.ax_bias.set_title('Subpx Bias')
        
        self.fig.canvas.draw_idle()

    def print_params(self, event):
        print("\n--- Final Parameters ---")
        for k, v in self.params.items():
            print(f"{k}: {v}")
        print(f"diameter = {self.p_diameter}\nminmass = {self.p_minmass}\nradius = {self.p_radius}\nsigma = {self.p_smooth}\n")
        print({"diameter":self.p_diameter,"minmass":self.p_minmass,"radius":self.p_radius,"sigma":self.p_smooth})

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select TIFF Stack", filetypes=[("TIFF files", "*.tif *.tiff")])
    
    if file_path:
        app = ParticleTrackerUI(file_path)
    else:
        print("No file selected.")