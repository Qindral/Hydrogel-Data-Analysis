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
1/e^2 radii wx, wy, is fitted to E. Profiles (user, 2026-10-07): the single pixel row and
column nearest the fitted centre (x0, y0), without averaging over neighbouring rows or
columns; a centre outside the visible area is clipped to the nearest edge row / column
(514 nm and 514_2: x0 at about -31 / -33 µm, image +-30 µm). The fit is evaluated on the
same row and column.

Power per particle in the focus: P_particle = E_focus * pi d^2 / 4, the power
incident on the geometric cross-section of a particle with the DLS diameter d
(core.io.get_dls_labels). E_focus is the maximum of the fitted Gaussian inside the
visible area: the beam centre (E0) if it lies inside, else the brightest point a
particle in the image can reach. This is incident, not absorbed, power.

Stage: compute stage. Reads the raw TIFF stacks and .rec files in DATA_DIR on every
run and overwrites in OUTPUT_DIR (no pickle cache; irradiance_results.csv is read by
spt_irradiance_overview.py):
  irradiance_results.csv              P (W), mean, focus and fitted peak irradiance (W/m^2), Gaussian parameters
  irradiance_power_per_particle.csv   per setting: P (W), E_mean and E_focus (W/m^2, W/cm^2),
                                      incident power per particle (W) for every DLS particle size
  beam_profiles_through_centre.csv    the plotted profiles: E measured and fitted per pixel
  laser_beam_exclusions.csv           per file: temporal outliers and hot pixels replaced
  beam_profiles_all_settings.pdf      x and y profiles through the centre, all wavelengths and powers
  beam_profiles_all_settings_fit.pdf  the same with the 2D Gaussian fit on the same row / column
Related: laser_intensity_calibration.py (same raw data); not read or modified here.
"""
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

from hydro_analysis.core.io import get_dls_labels, parse_rec_comment_metadata, parse_rec_file

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Laser_Intensity")
FILE_PATTERN = "Mirror_SizeofFWHM_*.tif"
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
                  r"\Experiments and Results - Data\Laser_Characterization\Laser_Beam_Characterization")
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"

VISIBLE_PX = (200, 150)                 # SPT camera image (x, y) at 4 x 4 binning, sensor-centred
TEMPORAL_OUTLIER_SIGMA = 8.0
HOT_PIXEL_SIGMA = 8.0
HOT_PIXEL_BOX_PX = 5
BACKGROUND_SMOOTH_PX = 15
BACKGROUND_PERCENTILE = 1.0

FIG_SIZE_IN = (6.30, 3.6)               # width class "full"; two panels side by side, legend below
# per wavelength, ordered by increasing power (light to dark): shades of the laser colours,
# 488 nm cyan-blue, 514 nm green (user, 2026-10-07; deviates from the style guide palette)
SETTING_COLORS = {488: ["#1c78c4", "#005db4", "#0B2A92"], 514: ["#2ad232", "#3ba521", "#499136"]}
MEASURED_ALPHA_WITH_FIT = 0.45
PANEL_LABEL_KW = dict(fontsize=10, fontweight="bold", va="bottom", ha="left")
SAVE_RETRIES = 5


def load_measurement(tif_path: Path) -> dict:
    rec_path = tif_path.with_name(tif_path.name + ".rec")
    frames = tifffile.imread(tif_path).astype(np.float64)
    med = np.median(frames, axis=0)
    mad = 1.4826 * np.median(np.abs(frames - med), axis=0)
    outlier = np.abs(frames - med) > TEMPORAL_OUTLIER_SIGMA * np.maximum(mad, 1.0)
    frames = np.where(outlier, med, frames)
    img = frames.mean(axis=0)
    local = median_filter(img, size=HOT_PIXEL_BOX_PX)
    diff = img - local
    hot = np.abs(diff) > HOT_PIXEL_SIGMA * 1.4826 * np.median(np.abs(diff - np.median(diff)))
    img = np.where(hot, local, img)
    return {"file": tif_path.name, "stem": tif_path.stem.lower(),
            "wavelength_nm": int(re.search(r"_(\d{3})(?:_\d+)?\.tif$", tif_path.name).group(1)),
            "P_W": parse_rec_comment_metadata(rec_path)["laser_power_mW"] * 1e-3,
            "pixel_um": float(parse_rec_file(rec_path)["mpp"]), "img": img,
            "dark_counts": float(np.percentile(median_filter(img, BACKGROUND_SMOOTH_PX), BACKGROUND_PERCENTILE)),
            "n_values": int(frames.size), "n_temporal_outliers": int(outlier.sum()),
            "n_pixels": int(img.size), "n_hot_pixels": int(hot.sum())}


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


def centre_profiles(E: np.ndarray, x: np.ndarray, y: np.ndarray, fit: dict) -> dict:
    """Measured and fitted E along the single row and column nearest the fitted centre (no averaging).

    The nearest index is always inside the image, so a centre outside it is clipped to the edge.
    """
    iy = int(np.argmin(np.abs(y - fit["y0"])))
    ix = int(np.argmin(np.abs(x - fit["x0"])))
    return {"x": (x, E[iy], fit["model"][iy], float(y[iy])),
            "y": (y, E[:, ix], fit["model"][:, ix], float(x[ix]))}


def setting_label(m: dict) -> str:
    return f"{m['wavelength_nm']} nm, {m['P_W'] * 1e3:g} mW"


def save_figure(fig: plt.Figure, path: Path) -> None:
    # The external drive occasionally refuses to open a file that a viewer holds; retry briefly.
    for attempt in range(SAVE_RETRIES):
        try:
            fig.savefig(path)
            return
        except OSError:
            if attempt == SAVE_RETRIES - 1:
                raise
            time.sleep(1.0)


def plot_profiles(entries: list[tuple[dict, dict]], with_fit: bool) -> plt.Figure:
    """x and y profiles of all settings; entries = [(measurement, centre_profiles), ...] in plot order."""
    exponent = int(np.floor(np.log10(max(max(pr[k][1].max(), pr[k][2].max()) for _, pr in entries for k in "xy"))))
    scale = 10.0 ** exponent
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE_IN, sharey=True, layout="constrained")
    n_seen = {}
    for m, prof in entries:
        i = n_seen[m["wavelength_nm"]] = n_seen.get(m["wavelength_nm"], -1) + 1
        color = SETTING_COLORS[m["wavelength_nm"]][i]
        for ax, key in zip(axs, "xy"):
            pos, meas, model, _ = prof[key]
            ax.plot(pos, meas / scale, color=color, lw=0.8 if with_fit else 1.0,
                    alpha=MEASURED_ALPHA_WITH_FIT if with_fit else 1.0,
                    label=None if with_fit else setting_label(m))
            if with_fit:
                ax.plot(pos, model / scale, color=color, lw=1.5, label=setting_label(m))
    for ax, key, label in zip(axs, "xy", "AB"):
        pos = entries[0][1][key][0]
        px = pos[1] - pos[0]
        ax.set_xlim(pos[0] - px / 2, pos[-1] + px / 2)
        ax.set_ylim(0.0, None)
        ax.set_xlabel(f"Position {key} (µm)")
        ax.text(0.0, 1.02, label, transform=ax.transAxes, **PANEL_LABEL_KW)
    axs[0].set_ylabel(rf"Irradiance $E$ ($10^{{{exponent}}}$ W m$^{{-2}}$)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False,
               title="2D Gaussian fit (lines), measured (pale)" if with_fit else None, title_fontsize=8)
    return fig


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

    rows, entries, profile_rows = [], [], []
    with plt.style.context(STYLE_PATH):
        for m in sorted(ms, key=lambda m: (m["wavelength_nm"], m["P_W"])):
            E, x, y = irradiance_map(m, background)
            fit = fit_gaussian(E, x, y)
            prof = centre_profiles(E, x, y, fit)
            entries.append((m, prof))
            for key, other in (("x", "y"), ("y", "x")):
                pos, meas, model, at = prof[key]
                profile_rows += [{"file": m["file"], "wavelength_nm": m["wavelength_nm"], "P_total_W": m["P_W"],
                                  "profile": key, f"{other}_um_of_profile": at, "position_um": float(q),
                                  "E_measured_W_m2": float(e), "E_fit_W_m2": float(f)}
                                 for q, e, f in zip(pos, meas, model)]
            A_px = (m["pixel_um"] * 1e-6) ** 2
            rows.append({"file": m["file"], "wavelength_nm": m["wavelength_nm"], "P_total_W": m["P_W"],
                         "visible_area_m2": E.size * A_px, "E_mean_W_m2": float(E.mean()),
                         "E_focus_W_m2": float(fit["model"].max()), "E_peak_fit_W_m2": fit["E0"], "E_peak_fit_err_W_m2": fit["E0_err"],
                         "x0_um": fit["x0"], "y0_um": fit["y0"], "wx_um": fit["wx"], "wx_err_um": fit["wx_err"],
                         "wy_um": fit["wy"], "wy_err_um": fit["wy_err"], "R2": fit["R2"],
                         "P_fit_visible_W": float(fit["model"].sum() * A_px), "background_counts": background})

        for stem, with_fit in (("beam_profiles_all_settings", False), ("beam_profiles_all_settings_fit", True)):
            fig = plot_profiles(entries, with_fit)
            save_figure(fig, OUTPUT_DIR / f"{stem}.pdf")
            plt.close(fig)

    pd.DataFrame(profile_rows).to_csv(OUTPUT_DIR / "beam_profiles_through_centre.csv", index=False)
    pd.DataFrame([row for m in ms for row in (
        {"file": m["file"], "step": "temporal outlier", "parameter": f"{TEMPORAL_OUTLIER_SIGMA:g} x MAD",
         "n_before": m["n_values"], "n_after": m["n_values"] - m["n_temporal_outliers"],
         "reason": "pixel value replaced by its temporal median"},
        {"file": m["file"], "step": "hot pixel", "parameter": f"{HOT_PIXEL_SIGMA:g} x MAD, {HOT_PIXEL_BOX_PX} px box",
         "n_before": m["n_pixels"], "n_after": m["n_pixels"] - m["n_hot_pixels"],
         "reason": "mean-image pixel replaced by the local median"}
    )]).to_csv(OUTPUT_DIR / "laser_beam_exclusions.csv", index=False)
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
