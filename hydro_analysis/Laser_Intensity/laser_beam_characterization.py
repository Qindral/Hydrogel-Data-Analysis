"""
Irradiance E (W/m^2) of the laser beam in the sample plane for every wavelength and
power setting, its 2D Gaussian fit and the total power.

Measurement (per setting, both in the sample plane, 19.02.2026):
  (1) a mirror slide reflected the beam onto the SPT camera (pco.pixelfly USB,
      4 x 4 binning, 0.30 µm per pixel) -> TIFF stack with the relative spatial profile;
  (2) a Thorlabs S170C slide power sensor -> total power P, noted in the Comment
      section of the matching .tif.rec file.

The power P is distributed over the visible area only, i.e. the SPT camera image
(sensor-centred window of VISIBLE_PX, 60 x 45 µm), in proportion to the
background-corrected camera counts I:
    E(x, y) = P I(x, y) / (sum_visible I * A_px),
so the irradiance integrated over the visible area equals P. Full-sensor recordings
are cropped to the same window. The beam overfills the image, so power outside the
visible area is assigned to it and E is an upper bound.
Background: the camera offset, taken from the darkest flat plateau of all recordings
(beam-free corners of the full-sensor recordings; identical camera settings).
Single-frame outliers (e.g. one over-range pixel) are replaced by the temporal median,
static hot pixels by the 5 x 5 median of the mean image.

An axis-aligned 2D Gaussian, E0 exp(-2 ((x - x0)^2 / wx^2 + (y - y0)^2 / wy^2)), with
1/e^2 radii wx, wy, is fitted to E. The x and y profiles through the fitted centre
are averaged over PROFILE_AVG_PX rows / columns; the fit is averaged identically.

Power per particle in the focus: P_particle = E_focus * pi d^2 / 4, the power
incident on the geometric cross-section of a particle with the DLS diameter d
(core.io.get_dls_labels). E_focus is the maximum of the fitted Gaussian inside the
visible area: the beam centre (E0) if it lies inside, else the brightest point a
particle in the image can reach. This is incident, not absorbed, power.

Stage: compute stage. Reads the raw TIFF stacks and .rec files in DATA_DIR on every
run and overwrites in OUTPUT_DIR (no pickle cache, no consumers):
  irradiance_results.csv              P (W), mean, focus and fitted peak irradiance (W/m^2), Gaussian parameters
  irradiance_power_per_particle.csv   per setting: P (W), E_mean and E_focus (W/m^2, W/cm^2),
                                      incident power per particle (W) for every DLS particle size
  irradiance_<file stem>.pdf          irradiance map with 1/e^2 contour, x and y profiles with the fit
Related: laser_intensity_calibration.py (same raw data); not read or modified here.
"""
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Ellipse
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

from hydro_analysis.core.io import get_dls_labels, parse_rec_comment_metadata, parse_rec_file

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Laser_Intensity")
FILE_PATTERN = "Mirror_SizeofFWHM_*.tif"
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
                  r"\Experiments and Results - Data\Laser_Beam_Characterization")
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"

VISIBLE_PX = (200, 150)                 # SPT camera image (x, y) at 4 x 4 binning, sensor-centred
PROFILE_AVG_PX = 5
TEMPORAL_OUTLIER_SIGMA = 8.0
HOT_PIXEL_SIGMA = 8.0
HOT_PIXEL_BOX_PX = 5
BACKGROUND_SMOOTH_PX = 15
BACKGROUND_PERCENTILE = 1.0

FIG_SIZE_IN = (6.30, 3.4)
COLOR_X = ("#3B8C8C", "#2A6666")        # category A
COLOR_Y = ("#D98C3D", "#A6672D")        # category B
COLOR_ACCENT = "#da00bd"
PANEL_LABEL_KW = dict(fontsize=10, fontweight="bold", va="bottom", ha="left")
SAVE_RETRIES = 5


def load_measurement(tif_path: Path) -> dict:
    rec_path = tif_path.with_name(tif_path.name + ".rec")
    frames = tifffile.imread(tif_path).astype(np.float64)
    med = np.median(frames, axis=0)
    mad = 1.4826 * np.median(np.abs(frames - med), axis=0)
    frames = np.where(np.abs(frames - med) > TEMPORAL_OUTLIER_SIGMA * np.maximum(mad, 1.0), med, frames)
    img = frames.mean(axis=0)
    local = median_filter(img, size=HOT_PIXEL_BOX_PX)
    diff = img - local
    hot = np.abs(diff) > HOT_PIXEL_SIGMA * 1.4826 * np.median(np.abs(diff - np.median(diff)))
    img = np.where(hot, local, img)
    return {"file": tif_path.name, "stem": tif_path.stem.lower(),
            "wavelength_nm": int(re.search(r"_(\d{3})(?:_\d+)?\.tif$", tif_path.name).group(1)),
            "P_W": parse_rec_comment_metadata(rec_path)["laser_power_mW"] * 1e-3,
            "pixel_um": float(parse_rec_file(rec_path)["mpp"]), "img": img,
            "dark_counts": float(np.percentile(median_filter(img, BACKGROUND_SMOOTH_PX), BACKGROUND_PERCENTILE))}


def irradiance_map(m: dict, background: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E (W/m^2) in the visible area and the pixel-centre coordinates (µm, image centre = 0)."""
    ny, nx = m["img"].shape
    x0, y0 = (nx - VISIBLE_PX[0]) // 2, (ny - VISIBLE_PX[1]) // 2
    I = m["img"][y0:y0 + VISIBLE_PX[1], x0:x0 + VISIBLE_PX[0]] - background
    E = m["P_W"] * I / (I.sum() * (m["pixel_um"] * 1e-6) ** 2)
    x = (np.arange(VISIBLE_PX[0]) - (VISIBLE_PX[0] - 1) / 2.0) * m["pixel_um"]
    y = (np.arange(VISIBLE_PX[1]) - (VISIBLE_PX[1] - 1) / 2.0) * m["pixel_um"]
    return E, x, y


def gauss2d(xy, x0, y0, wx, wy, E0):
    x, y = xy
    return E0 * np.exp(-2.0 * ((x - x0) ** 2 / wx ** 2 + (y - y0) ** 2 / wy ** 2))


def fit_gaussian(E: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    xx, yy = np.meshgrid(x, y)
    p0 = [x[np.argmax(E.mean(axis=0))], y[np.argmax(E.mean(axis=1))], np.ptp(x) / 2.0, np.ptp(y) / 2.0, E.max()]
    p, cov = curve_fit(gauss2d, (xx.ravel(), yy.ravel()), E.ravel(), p0=p0,
                       bounds=([-np.inf, -np.inf, 0.0, 0.0, 0.0], np.inf), x_scale="jac", max_nfev=5000)
    model = gauss2d((xx, yy), *p)
    return {**dict(zip(("x0", "y0", "wx", "wy", "E0"), p)),
            **dict(zip(("x0_err", "y0_err", "wx_err", "wy_err", "E0_err"), np.sqrt(np.diag(cov)))),
            "R2": float(1.0 - np.sum((E - model) ** 2) / np.sum((E - E.mean()) ** 2)), "model": model}


def line_profiles(E: np.ndarray, x: np.ndarray, y: np.ndarray, fit: dict) -> dict:
    """Measured and fitted E along x and y through the fitted centre, averaged over PROFILE_AVG_PX rows / columns."""
    h = PROFILE_AVG_PX // 2
    iy = int(np.clip(np.argmin(np.abs(y - fit["y0"])), h, len(y) - 1 - h))
    ix = int(np.clip(np.argmin(np.abs(x - fit["x0"])), h, len(x) - 1 - h))
    rows, cols = slice(iy - h, iy + h + 1), slice(ix - h, ix + h + 1)
    return {"x": (x, E[rows].mean(axis=0), fit["model"][rows].mean(axis=0), y[rows]),
            "y": (y, E[:, cols].mean(axis=1), fit["model"][:, cols].mean(axis=1), x[cols])}


def _sci(value: float) -> str:
    exponent = int(np.floor(np.log10(value)))
    return rf"{value / 10.0 ** exponent:.2f} $\times$ $10^{{{exponent}}}$"


def plot_measurement(m: dict, E: np.ndarray, x: np.ndarray, y: np.ndarray, fit: dict, prof: dict) -> Path:
    px = m["pixel_um"]
    exponent = int(np.floor(np.log10(max(E.max(), fit["E0"]))))
    scale = 10.0 ** exponent
    unit = rf"($10^{{{exponent}}}$ W m$^{{-2}}$)"
    ext = (x[0] - px / 2, x[-1] + px / 2, y[-1] + px / 2, y[0] - px / 2)

    fig, axs = plt.subplot_mosaic([["A", "B"], ["A", "C"]], figsize=FIG_SIZE_IN, width_ratios=[1.15, 1.0])
    ax = axs["A"]
    im = ax.imshow(E / scale, extent=ext, cmap="viridis", vmin=0.0, interpolation="nearest", origin="upper")
    ax.add_patch(Ellipse((fit["x0"], fit["y0"]), 2 * fit["wx"], 2 * fit["wy"], fill=False, ec=COLOR_ACCENT, lw=1.2))
    band_y, band_x = prof["x"][3], prof["y"][3]
    ax.axhspan(band_y[0] - px / 2, band_y[-1] + px / 2, color=COLOR_X[1], alpha=0.35, lw=0)
    ax.axvspan(band_x[0] - px / 2, band_x[-1] + px / 2, color=COLOR_Y[1], alpha=0.35, lw=0)
    ax.set_xlim(ext[:2])
    ax.set_ylim(ext[2:])
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.text(0.03, 0.04, rf"$P_\mathrm{{total}}$ = {_sci(m['P_W'])} W", transform=ax.transAxes, fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    cb = fig.colorbar(im, ax=ax, location="bottom", fraction=0.06, pad=0.02)
    cb.set_label(f"Irradiance $E$ {unit}")

    y_top = 1.15 * max(prof["x"][1].max(), prof["y"][1].max()) / scale
    for key, colors in (("x", COLOR_X), ("y", COLOR_Y)):
        ax = axs["B" if key == "x" else "C"]
        pos, meas, model, _ = prof[key]
        ax.plot(pos, meas / scale, color=colors[0], lw=1.0, label=f"Measured, {PROFILE_AVG_PX} px mean")
        ax.plot(pos, model / scale, color=colors[1], lw=1.5,
                label=rf"2D Gaussian fit, $w_{key}$ = {fit['w' + key]:.1f} µm")
        ax.set_xlim(pos[0] - px / 2, pos[-1] + px / 2)
        ax.set_ylim(0.0, y_top)
        ax.set_xlabel(f"Position {key} (µm)")
        ax.set_ylabel(f"$E$ {unit}")
        ax.legend(loc="lower center", handlelength=1.6)
    for label, ax in axs.items():
        ax.text(0.0, 1.02, label, transform=ax.transAxes, **PANEL_LABEL_KW)

    path = OUTPUT_DIR / f"irradiance_{m['stem']}.pdf"
    # The external drive occasionally refuses to open a file that a viewer holds; retry briefly.
    for attempt in range(SAVE_RETRIES):
        try:
            fig.savefig(path)
            break
        except OSError:
            if attempt == SAVE_RETRIES - 1:
                raise
            time.sleep(1.0)
    plt.close(fig)
    return path


def power_per_particle(results: pd.DataFrame) -> pd.DataFrame:
    table = results[["file", "wavelength_nm", "P_total_W", "E_mean_W_m2", "E_focus_W_m2"]].copy()
    table["E_mean_W_cm2"] = table["E_mean_W_m2"] * 1e-4
    table["E_focus_W_cm2"] = table["E_focus_W_m2"] * 1e-4
    for d_nm in sorted(get_dls_labels().values()):
        table[f"P_particle_{d_nm}nm_W"] = table["E_focus_W_m2"] * np.pi * (d_nm * 1e-9) ** 2 / 4.0
    return table


def print_power_per_particle(table: pd.DataFrame) -> None:
    sizes = [c for c in table.columns if c.startswith("P_particle_")]
    print("\nIncident power per particle in the focus (nW), E_focus x geometric cross-section:")
    print(f"{'nm':>4} {'P_total (W)':>12} {'E_mean (W/m^2)':>15} {'E_focus (W/m^2)':>16} {'E_focus (W/cm^2)':>17}"
          + "".join(f"{c.split('_')[2]:>10}" for c in sizes))
    for r in table.itertuples(index=False):
        row = r._asdict()
        print(f"{r.wavelength_nm:>4} {r.P_total_W:>12.3e} {r.E_mean_W_m2:>15.3e} {r.E_focus_W_m2:>16.3e} "
              f"{r.E_focus_W_cm2:>17.1f}" + "".join(f"{row[c] * 1e9:>10.3g}" for c in sizes))


def main() -> None:
    ms = [load_measurement(p) for p in sorted(DATA_DIR.glob(FILE_PATTERN))]
    if not ms:
        raise FileNotFoundError(f"no {FILE_PATTERN} stacks in {DATA_DIR}")
    background = min(m["dark_counts"] for m in ms)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with plt.style.context(STYLE_PATH):
        for m in sorted(ms, key=lambda m: (m["wavelength_nm"], m["P_W"])):
            E, x, y = irradiance_map(m, background)
            fit = fit_gaussian(E, x, y)
            plot_measurement(m, E, x, y, fit, line_profiles(E, x, y, fit))
            A_px = (m["pixel_um"] * 1e-6) ** 2
            rows.append({"file": m["file"], "wavelength_nm": m["wavelength_nm"], "P_total_W": m["P_W"],
                         "visible_area_m2": E.size * A_px, "E_mean_W_m2": float(E.mean()),
                         "E_focus_W_m2": float(fit["model"].max()), "E_peak_fit_W_m2": fit["E0"], "E_peak_fit_err_W_m2": fit["E0_err"],
                         "x0_um": fit["x0"], "y0_um": fit["y0"], "wx_um": fit["wx"], "wx_err_um": fit["wx_err"],
                         "wy_um": fit["wy"], "wy_err_um": fit["wy_err"], "R2": fit["R2"],
                         "P_fit_visible_W": float(fit["model"].sum() * A_px), "background_counts": background})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "irradiance_results.csv", index=False)
    print(f"Background {background:.1f} counts; visible area {VISIBLE_PX[0]} x {VISIBLE_PX[1]} px")
    for r in results.itertuples():
        print(f"{r.file:<30} {r.wavelength_nm} nm  P_total {r.P_total_W:.3e} W  E_mean {r.E_mean_W_m2:.3e} W/m^2  "
              f"E_peak(fit) {r.E_peak_fit_W_m2:.3e} W/m^2  w_x {r.wx_um:.1f} µm  w_y {r.wy_um:.1f} µm  R2 {r.R2:.3f}")
    particles = power_per_particle(results)
    particles.to_csv(OUTPUT_DIR / "irradiance_power_per_particle.csv", index=False)
    print_power_per_particle(particles)
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
