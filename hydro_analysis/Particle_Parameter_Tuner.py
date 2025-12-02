import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pims
import trackpy as tp
import cv2
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import filedialog

# --- Helper Function for Fast Rolling Ball (using OpenCV) ---
def fast_rolling_ball(image, radius):
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

class ParticleTuner:
    def __init__(self, tif_path):
        print("Loading frames...")
        self.frames = pims.open(tif_path)
        self.n_frames = len(self.frames)
        
        # --- 1. Pre-calculate Static Mean Background (Your Logic) ---
        print("Calculating static mean background (this may take a moment)...")
        acc = None
        for fr in self.frames:
            arr = np.asarray(fr, dtype=np.float64)
            if acc is None:
                acc = np.zeros_like(arr, dtype=np.float64)
            acc += arr
        self.static_background = (acc / float(self.n_frames)).astype(np.float32)
        print("Static background calculated.")

        # --- Initial Parameters ---
        self.current_frame_idx = 0
        self.p_diameter = 11  # Must be odd
        self.p_minmass = 100
        self.p_radius = 0     # Rolling ball radius (0 = off)
        self.p_smooth = 0.0   # Gaussian sigma
        
        # --- Setup UI Layout ---
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('Particle Tracking Tuner')
        
        # Grid layout: Image (Left), Plots (Right), Sliders (Bottom)
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.4])
        
        # Axes
        self.ax_img = self.fig.add_subplot(gs[0:2, 0:2]) # Main Image
        self.ax_hist = self.fig.add_subplot(gs[0, 2])    # Mass Histogram
        self.ax_bias = self.fig.add_subplot(gs[1, 2])    # Subpixel Bias
        
        # Adjust layout for sliders
        plt.subplots_adjust(bottom=0.25)

        # --- Sliders ---
        # Format: [left, bottom, width, height]
        ax_frame = plt.axes([0.15, 0.15, 0.65, 0.03])
        ax_diam  = plt.axes([0.15, 0.11, 0.25, 0.03])
        ax_mass  = plt.axes([0.55, 0.11, 0.25, 0.03])
        ax_rad   = plt.axes([0.15, 0.07, 0.25, 0.03])
        ax_smooth= plt.axes([0.55, 0.07, 0.25, 0.03])
        ax_btn   = plt.axes([0.85, 0.07, 0.1, 0.07])

        self.s_frame = Slider(ax_frame, 'Frame', 0, self.n_frames-1, valinit=0, valstep=1)
        self.s_diam  = Slider(ax_diam, 'Diameter (px)', 3, 51, valinit=self.p_diameter, valstep=2)
        self.s_mass  = Slider(ax_mass, 'Min Mass', 0, 5000, valinit=self.p_minmass, valstep=10)
        self.s_rad   = Slider(ax_rad, 'RB Radius', 0, 100, valinit=self.p_radius, valstep=1)
        self.s_smooth= Slider(ax_smooth, 'Smooth Sigma', 0, 5, valinit=self.p_smooth, valstep=0.1)
        
        self.btn_run = Button(ax_btn, 'Print Params', hovercolor='0.975')

        # Connect events
        self.s_frame.on_changed(self.update)
        self.s_diam.on_changed(self.update)
        self.s_mass.on_changed(self.update)
        self.s_rad.on_changed(self.update)
        self.s_smooth.on_changed(self.update)
        self.btn_run.on_clicked(self.print_params)

        # Initial Draw
        self.update(None)
        plt.show()

    def get_processed_image(self, frame_idx):
        # 1. Get Raw Frame
        raw = np.asarray(self.frames[frame_idx], dtype=np.float32)
        
        # 2. Subtract Static Mean Background (Clip negatives)
        img = np.clip(raw - self.static_background, 0.0, None).astype(np.float32)
        
        # 3. Rolling Ball (Optional)
        # Note: We convert to uint8 temporarily if needed by cv2, or keep float if cv2 supports it.
        # CV2 morphology usually wants defined ranges. Let's normalize safely or keep float.
        if self.p_radius > 0:
            img = fast_rolling_ball(img, self.p_radius)
            
        # 4. Gaussian Smoothing (Optional)
        if self.p_smooth > 0:
            img = gaussian_filter(img, sigma=self.p_smooth)
            
        return img

    def update(self, val):
        # Update Parameter Variables
        self.current_frame_idx = int(self.s_frame.val)
        self.p_diameter = int(self.s_diam.val)
        
        # Force odd diameter
        if self.p_diameter % 2 == 0: 
            self.p_diameter += 1
            
        self.p_minmass = self.s_mass.val
        self.p_radius = int(self.s_rad.val)
        self.p_smooth = self.s_smooth.val

        # --- Process Image ---
        img = self.get_processed_image(self.current_frame_idx)

        # --- Trackpy Locate ---
        # invert=False based on your code (particles are bright on dark after subtraction)
        tp.quiet()
        f = tp.locate(img, diameter=self.p_diameter, minmass=self.p_minmass, invert=False)

        # --- Update Image Plot ---
        self.ax_img.clear()
        self.ax_img.imshow(img, cmap='gray', origin='upper')
        
        # Annotate manually (faster than tp.annotate for UI)
        if len(f) > 0:
            self.ax_img.plot(f['x'], f['y'], 'o', markerfacecolor='none', markeredgecolor='r', markersize=10, alpha=0.7)
        self.ax_img.set_title(f"Frame {self.current_frame_idx} | Detected: {len(f)}")
        self.ax_img.axis('off')

        # --- Update Histogram (MinMass) ---
        self.ax_hist.clear()
        if len(f) > 0:
            self.ax_hist.hist(f['mass'], bins=20, color='skyblue', edgecolor='black')
            self.ax_hist.axvline(self.p_minmass, color='r', linestyle='--', label='Threshold')
            self.ax_hist.set_title('Mass Distribution')
            self.ax_hist.set_xlabel('Mass')
        else:
            self.ax_hist.text(0.5, 0.5, "No Particles", ha='center')

        # --- Update Subpixel Bias ---
        self.ax_bias.clear()
        if len(f) > 0:
            # Manually plotting subpx bias to fit in subplot
            # tp.subpx_bias(f, ax=self.ax_bias) generates its own figure usually, 
            # so we reimplement the histogram logic briefly:
            self.ax_bias.hist(f['x'] % 1, bins=10, alpha=0.5, label='x', color='red')
            self.ax_bias.hist(f['y'] % 1, bins=10, alpha=0.5, label='y', color='blue')
            self.ax_bias.legend(loc='upper right', fontsize='small')
            self.ax_bias.set_title('Subpixel Bias')
        else:
            self.ax_bias.text(0.5, 0.5, "No Particles", ha='center')

        self.fig.canvas.draw_idle()

    def print_params(self, event):
        print("\n--- Current Parameters ---")
        print(f"Diameter: {self.p_diameter}")
        print(f"Min Mass: {self.p_minmass}")
        print(f"Rolling Ball Radius: {self.p_radius}")
        print(f"Gaussian Sigma: {self.p_smooth}")
        print("-" * 26)
        print(f"diameter = {self.p_diameter}\nminmass = {self.p_minmass}\nradius = {self.p_radius}\nsigma = {self.p_smooth}\n")
        print({"diameter":self.p_diameter,"minmass":self.p_minmass,"radius":self.p_radius,"sigma":self.p_smooth})


# --- Main Entry Point ---
if __name__ == "__main__":
    # Select file
    root = tk.Tk()
    root.withdraw() # Hide small tkinter window
    file_path = filedialog.askopenfilename(title="Select TIFF Stack", filetypes=[("TIFF files", "*.tif *.tiff")])
    
    if file_path:
        app = ParticleTuner(file_path)
    else:
        print("No file selected.")