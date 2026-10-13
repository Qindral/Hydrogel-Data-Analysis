"""
Irradiance E (W/m^2) in the SPT image for every laser setting: the absolute power of the
Thorlabs S170C distributed over the pixels in proportion to the normalised camera profile.
The camera gives only the relative spatial distribution; no absolute power is derived from it.

Measurement (per setting, both in the sample plane, 19.02.2026): a mirror slide reflected the
beam onto the SPT camera (pco.pixelfly USB) -> TIFF stack; a Thorlabs S170C slide power sensor
-> total power P_total, noted in the Comment section of the matching .tif.rec file.

Steps (user, 2026-10-07; ENTSCHEIDUNGEN.md E015):
  image area: the sensor-centred SPT window VISIBLE_PX (60 x 45 µm) for every recording; the
    full-sensor recordings (514, 514_2) are cropped to it;
  no pixel corrections (no temporal-outlier or hot-pixel replacement); the frames are averaged;
  background: the camera offset, darkest flat plateau of all recordings (1st percentile of the
    BACKGROUND_SMOOTH_PX median-filtered mean images; beam-free corners of the full-sensor
    recordings), as in laser_beam_characterization.py;
  S = sum I_corr, f = I_corr / S, P_pixel = P_total f, E = P_pixel / A_pixel. All of P_total is
    assigned to the window although the beam is larger than the window: E is an upper bound.
    fit_power_fraction_in_window reports how much of the fitted Gaussian lies in the window
    (check only, not applied);
  rotated elliptical 2D Gaussian E0 exp(-2 (u^2 / w_major^2 + v^2 / w_minor^2)) on E; beam centre,
    1/e^2 radii, FWHM = w sqrt(2 ln 2), x and y widths through the centre, ellipticity
    w_minor / w_major, residuals: all from this fit. No D4sigma (needs the complete beam);
  E_max: brightest pixel of the measured map; E_mean_illuminated: mean of E over the window
    pixels inside the fitted 1/e^2 ellipse;
  saturation: fraction of raw values >= SATURATION_COUNTS in the window;
  profiles: single pixel row and column nearest the fitted centre, clipped to the window
    (as laser_beam_characterization.py, E013).

Stage: compute stage. Reads the raw TIFF stacks and .rec files in DATA_DIR (read-only) on every
run and overwrites in OUTPUT_DIR; no pickle; no other script reads these outputs:
  laser_beam_norm_detector_results.csv           all measured and fitted parameters per setting
  laser_beam_norm_detector_power_comparison.csv  per wavelength: parameters vs. P_total, range
  laser_beam_norm_detector_profiles.csv          the plotted profiles (E measured and fitted)
  laser_beam_norm_detector_exclusions.csv        files found and used; cropping per file
  laser_beam_norm_detector_profiles.pdf          x and y irradiance profiles with the fit
  laser_beam_norm_detector_geometry.pdf          S170C active area, fitted 1/e^2 ellipses, SPT window and
                                                 full camera sensor (largest recorded frame) to scale
Related: laser_beam_characterization.py (same raw data, W/m^2, axis-aligned fit, pixel corrections).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
import pandas as pd
import tifffile
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

from hydro_analysis.core.io import parse_rec_comment_metadata, parse_rec_file

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Laser_Intensity")
FILE_PATTERN = "Mirror_SizeofFWHM_*.tif"
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
                  r"\Experiments and Results - Data\Laser_Beam_Norm_Detector")
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
STEM = "laser_beam_norm_detector"

VISIBLE_PX = (200, 150)        # SPT window (x, y) at 4 x 4 binning, sensor-centred; decided by user, 2026-10-07
BACKGROUND_SMOOTH_PX = 15      # as laser_beam_characterization.py; method decided by user, 2026-10-07
BACKGROUND_PERCENTILE = 1.0    # as laser_beam_characterization.py; method decided by user, 2026-10-07
# pixelfly USB: 14-bit ADC written as multiples of 4 into 16 bit (checked at run time)
ADC_STEP_COUNTS = 4
SATURATION_COUNTS = ADC_STEP_COUNTS * (2 ** 14 - 1)

FIG_SIZE_IN = (6.30, 3.6)      # width class "full"; two panels side by side, legend below
# per wavelength, ordered by increasing power (light to dark): shades of the laser colours,
# 488 nm cyan-blue, 514 nm green (user, 2026-10-07; deviates from the style guide palette)
SETTING_COLORS = {488: ["#1c78c4", "#005db4", "#0B2A92"], 514: ["#2ad232", "#3ba521", "#499136"]}
MEASURED_ALPHA = 0.45
# Thorlabs S170C: "18 mm x 18 mm active sensor area" (Thorlabs catalog page); confirmed by user, 2026-10-07
S170C_ACTIVE_AREA_MM = (18.0, 18.0)
GEOMETRY_FIG_SIZE_IN = (6.30, 3.9)
GEOMETRY_MARGIN = 1.08         # zoom panels: extent of the largest ellipse / camera sensor times this factor
PANEL_LABEL_KW = dict(fontsize=10, fontweight="bold", va="bottom", ha="left")


# ── Data ───────────────────────────────────────────────────────────────────────
def rec_text(rec_path: Path) -> str:
    raw = rec_path.read_bytes()
    return raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "cp1252")


def rec_field(text: str, name: str, rec_path: Path) -> str:
    match = re.search(re.escape(name) + r"\s*:\s*(.+)", text)
    if match is None:
        raise ValueError(f"{rec_path.name}: field '{name}' missing")
    return match.group(1).strip()


def load_measurement(tif_path: Path) -> dict:
    """Raw frames, their mean and the metadata of one recording; errors name the file and field."""
    rec_path = tif_path.with_name(tif_path.name + ".rec")
    if not rec_path.is_file():
        raise FileNotFoundError(f"{tif_path.name}: .rec file missing")
    with tifffile.TiffFile(tif_path) as tif:
        frames = tif.asarray()
        bits = int(tif.pages[0].bitspersample)
    if frames.ndim != 3 or frames.dtype != np.uint16:
        raise ValueError(f"{tif_path.name}: expected a uint16 stack, got {frames.dtype} {frames.shape}")
    if np.any(frames % ADC_STEP_COUNTS):
        raise ValueError(f"{tif_path.name}: values are not multiples of {ADC_STEP_COUNTS}; saturation level unknown")
    text = rec_text(rec_path)
    rec = parse_rec_file(rec_path)
    power_mw = parse_rec_comment_metadata(rec_path)["laser_power_mW"]
    if not np.isfinite(power_mw):
        raise ValueError(f"{rec_path.name}: no laser power (mW) in the comment")
    match = re.search(r"_(\d{3})(?:_\d+)?\.tif$", tif_path.name)
    if match is None:
        raise ValueError(f"{tif_path.name}: no wavelength in the file name")
    stated = {int(v) for v in re.findall(r"(?<!\d)(488|514)(?!\d)", text.split("Comment:", 1)[-1])}
    if stated and stated != {int(match.group(1))}:
        raise ValueError(f"{tif_path.name}: wavelength in the file name and the .rec comment differ")
    img = frames.mean(axis=0)
    return {"file": tif_path.name, "wavelength_nm": int(match.group(1)), "P_total_W": float(power_mw) * 1e-3,
            "pixel_um": float(rec["mpp"]), "exposure_ms": float(rec["exposure_ms"]),
            "binning": rec_field(text, "Binning horz./vert.", rec_path),
            "conversion_factor": rec_field(text, "Conversion factor", rec_path),
            "ir_sensitivity": rec_field(text, "IR-Sensitivity", rec_path), "bit_depth_tiff": bits,
            "n_frames": frames.shape[0], "sensor_px": f"{frames.shape[2]} x {frames.shape[1]}",
            "frames": frames, "img": img,
            "dark_counts": float(np.percentile(median_filter(img, BACKGROUND_SMOOTH_PX), BACKGROUND_PERCENTILE))}


def window(a: np.ndarray) -> np.ndarray:
    """Sensor-centred SPT window of the last two axes."""
    ny, nx = a.shape[-2:]
    if nx < VISIBLE_PX[0] or ny < VISIBLE_PX[1]:
        raise ValueError(f"Image {nx} x {ny} px smaller than the SPT window {VISIBLE_PX}")
    x0, y0 = (nx - VISIBLE_PX[0]) // 2, (ny - VISIBLE_PX[1]) // 2
    return a[..., y0:y0 + VISIBLE_PX[1], x0:x0 + VISIBLE_PX[0]]


# ── Analysis ───────────────────────────────────────────────────────────────────
def irradiance(img_window: np.ndarray, background: float, P_total_W: float, pixel_um: float) -> dict:
    """S, f, P_pixel (W) and E (W/m^2) with pixel-centre coordinates (µm, window centre = 0)."""
    I_corr = img_window - background
    S = float(I_corr.sum())
    f = I_corr / S
    A_px_m2 = (pixel_um * 1e-6) ** 2
    ny, nx = I_corr.shape
    return {"S_counts": S, "f": f, "P_pixel_W": P_total_W * f, "E": P_total_W * f / A_px_m2,
            "A_px_m2": A_px_m2,
            "x": (np.arange(nx) - (nx - 1) / 2.0) * pixel_um, "y": (np.arange(ny) - (ny - 1) / 2.0) * pixel_um}


def gauss2d_rot(xy, x0, y0, w_major, w_minor, theta, E0):
    x, y = xy
    u = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    v = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    return E0 * np.exp(-2.0 * (u ** 2 / w_major ** 2 + v ** 2 / w_minor ** 2))


def fit_gaussian_rot(E: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Rotated elliptical 2D Gaussian; w_major >= w_minor, theta (rad) of the major axis in (-pi/2, pi/2]."""
    xx, yy = np.meshgrid(x, y)
    p0 = [x[np.argmax(E.mean(axis=0))], y[np.argmax(E.mean(axis=1))], np.ptp(x) / 2.0, np.ptp(y) / 2.0, 0.0, E.max()]
    p, cov = curve_fit(gauss2d_rot, (xx.ravel(), yy.ravel()), E.ravel(), p0=p0,
                       bounds=([-np.inf, -np.inf, 0.0, 0.0, -np.pi, 0.0], np.inf), x_scale="jac", max_nfev=20000)
    err = np.sqrt(np.diag(cov))
    x0, y0, w1, w2, theta, E0 = p
    if w1 < w2:
        w1, w2, theta = w2, w1, theta + np.pi / 2
        err[[2, 3]] = err[[3, 2]]
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    if theta == -np.pi / 2:
        theta = np.pi / 2
    model = gauss2d_rot((xx, yy), x0, y0, w1, w2, theta, E0)
    resid = E - model
    return {"x0": x0, "y0": y0, "w_major": w1, "w_minor": w2, "theta": theta, "E0": E0,
            "x0_err": err[0], "y0_err": err[1], "w_major_err": err[2], "w_minor_err": err[3],
            "theta_err": err[4], "E0_err": err[5], "model": model, "resid": resid,
            "R2": float(1.0 - np.sum(resid ** 2) / np.sum((E - E.mean()) ** 2)),
            "resid_rms": float(np.sqrt(np.mean(resid ** 2))), "resid_max_abs": float(np.abs(resid).max())}


def beam_parameters(m: dict, irr: dict, fit: dict) -> dict:
    E, x, y = irr["E"], irr["x"], irr["y"]
    xx, yy = np.meshgrid(x, y)
    c, s = np.cos(fit["theta"]), np.sin(fit["theta"])
    u, v = (xx - fit["x0"]) * c + (yy - fit["y0"]) * s, -(xx - fit["x0"]) * s + (yy - fit["y0"]) * c
    inside = u ** 2 / fit["w_major"] ** 2 + v ** 2 / fit["w_minor"] ** 2 <= 1.0
    # 1/e^2 radius along the x and y axes through the centre of the rotated Gaussian
    wx = 1.0 / np.sqrt(c ** 2 / fit["w_major"] ** 2 + s ** 2 / fit["w_minor"] ** 2)
    wy = 1.0 / np.sqrt(s ** 2 / fit["w_major"] ** 2 + c ** 2 / fit["w_minor"] ** 2)
    fwhm = np.sqrt(2.0 * np.log(2.0))
    half_x, half_y = (x[-1] - x[0] + m["pixel_um"]) / 2.0, (y[-1] - y[0] + m["pixel_um"]) / 2.0
    total_fit_W = np.pi / 2.0 * fit["E0"] * fit["w_major"] * fit["w_minor"] * 1e-12     # µm^2 -> m^2
    raw = window(m["frames"])
    return {"S_counts": irr["S_counts"], "sum_f": float(irr["f"].sum()), "sum_P_pixel_W": float(irr["P_pixel_W"].sum()),
            "E_max_measured_W_m2": float(E.max()),
            "E_mean_illuminated_W_m2": float(E[inside].mean()) if inside.any() else np.nan,
            "n_px_illuminated": int(inside.sum()), "n_px_window": int(E.size),
            "E_mean_window_W_m2": float(E.mean()),
            "x0_um": fit["x0"], "x0_err_um": fit["x0_err"], "y0_um": fit["y0"], "y0_err_um": fit["y0_err"],
            "centre_in_window": bool(abs(fit["x0"]) <= half_x and abs(fit["y0"]) <= half_y),
            "w_major_um": fit["w_major"], "w_major_err_um": fit["w_major_err"],
            "w_minor_um": fit["w_minor"], "w_minor_err_um": fit["w_minor_err"],
            "theta_deg": np.degrees(fit["theta"]), "theta_err_deg": np.degrees(fit["theta_err"]),
            "w_x_um": wx, "w_y_um": wy, "FWHM_major_um": fwhm * fit["w_major"], "FWHM_minor_um": fwhm * fit["w_minor"],
            "FWHM_x_um": fwhm * wx, "FWHM_y_um": fwhm * wy, "ellipticity": fit["w_minor"] / fit["w_major"],
            "E0_fit_W_m2": fit["E0"], "E0_fit_err_W_m2": fit["E0_err"],
            "E_max_fit_in_window_W_m2": float(fit["model"].max()),
            "R2": fit["R2"], "resid_rms_W_m2": fit["resid_rms"], "resid_rms_rel_E0": fit["resid_rms"] / fit["E0"],
            "resid_max_abs_W_m2": fit["resid_max_abs"],
            "fit_power_fraction_in_window": float(fit["model"].sum() * irr["A_px_m2"] / total_fit_W),
            "ellipse_inside_window": bool(abs(fit["x0"]) + np.hypot(fit["w_major"] * c, fit["w_minor"] * s) <= half_x
                                          and abs(fit["y0"]) + np.hypot(fit["w_major"] * s, fit["w_minor"] * c) <= half_y),
            "max_raw_counts_window": int(raw.max()), "saturation_counts": SATURATION_COUNTS,
            "saturated_fraction": float(np.mean(raw >= SATURATION_COUNTS))}


def centre_profiles(E: np.ndarray, model: np.ndarray, x: np.ndarray, y: np.ndarray, fit: dict) -> dict:
    """Measured and fitted E along the single row and column nearest the fitted centre (clipped)."""
    iy = int(np.argmin(np.abs(y - fit["y0"])))
    ix = int(np.argmin(np.abs(x - fit["x0"])))
    return {"x": (x, E[iy], model[iy], float(y[iy])), "y": (y, E[:, ix], model[:, ix], float(x[ix]))}


def power_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Per wavelength and parameter: the value at each power level (ascending), the range (max - min)
    and, for positive quantities, the range relative to the mean."""
    signed = ["x0_um", "y0_um", "theta_deg"]
    positive = ["E0_fit_W_m2", "E_max_measured_W_m2", "E0_fit_per_P_total_per_m2", "w_major_um",
                "w_minor_um", "ellipticity", "FWHM_x_um", "FWHM_y_um", "R2"]
    results = results.assign(E0_fit_per_P_total_per_m2=results["E0_fit_W_m2"] / results["P_total_W"])
    rows = []
    for wl, g in results.sort_values("P_total_W").groupby("wavelength_nm"):
        for col in positive + signed:
            vals = g[col].to_numpy(dtype=float)
            row = {"wavelength_nm": wl, "parameter": col}
            for k, (p, v) in enumerate(zip(g["P_total_W"], vals), start=1):
                row[f"P{k}_W"], row[f"value_P{k}"] = p, v
            row["range"] = float(np.ptp(vals))
            row["range_rel_to_mean"] = float(np.ptp(vals) / vals.mean()) if col in positive else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_profiles(entries: list[tuple[dict, dict]]) -> plt.Figure:
    """x and y irradiance profiles of all settings, measured (pale) and fit (line)."""
    exponent = int(np.floor(np.log10(max(max(pr[k][1].max(), pr[k][2].max()) for _, pr in entries for k in "xy"))))
    scale = 10.0 ** exponent
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE_IN, sharey=True, layout="constrained")
    n_seen = {}
    for m, prof in entries:
        i = n_seen[m["wavelength_nm"]] = n_seen.get(m["wavelength_nm"], -1) + 1
        color = SETTING_COLORS[m["wavelength_nm"]][i]
        for ax, key in zip(axs, "xy"):
            pos, meas, model, _ = prof[key]
            ax.plot(pos, meas / scale, color=color, lw=0.8, alpha=MEASURED_ALPHA)
            ax.plot(pos, model / scale, color=color, lw=1.5, label=f"{m['wavelength_nm']} nm, {m['P_total_W'] * 1e3:g} mW")
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
               title="2D Gaussian fit (lines), measured (pale)", title_fontsize=8)
    return fig


def ellipse_xy(fit: dict, n: int = 361) -> tuple[np.ndarray, np.ndarray]:
    """Fitted 1/e^2 contour (µm)."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    c, s = np.cos(fit["theta"]), np.sin(fit["theta"])
    u, v = fit["w_major"] * np.cos(t), fit["w_minor"] * np.sin(t)
    return fit["x0"] + u * c - v * s, fit["y0"] + u * s + v * c


def plot_geometry(maps: list[tuple[dict, dict, dict]], sensor_um: tuple[float, float]) -> plt.Figure:
    """A: S170C active area to scale with the beam; B, C: per wavelength the measured E in the SPT window
    (highest power), the fitted 1/e^2 ellipses of all powers, the SPT window and the full camera sensor.
    maps = [(measurement, irradiance dict, fit), ...] ordered by wavelength and power."""
    wavelengths = sorted({m["wavelength_nm"] for m, _, _ in maps})
    ellipses = {id(fit): ellipse_xy(fit) for _, _, fit in maps}
    half_x = GEOMETRY_MARGIN * max([sensor_um[0] / 2] + [np.abs(ex).max() for ex, _ in ellipses.values()])
    half_y = GEOMETRY_MARGIN * max([sensor_um[1] / 2] + [np.abs(ey).max() for _, ey in ellipses.values()])
    vmax = max(irr["E"].max() for _, irr, _ in maps)
    exponent = int(np.floor(np.log10(vmax)))
    scale = 10.0 ** exponent

    fig, axd = plt.subplot_mosaic([["A", "B"], ["A", "C"]], figsize=GEOMETRY_FIG_SIZE_IN, layout="constrained",
                                  width_ratios=[1.0, 1.25])
    ax = axd["A"]
    sx, sy = S170C_ACTIVE_AREA_MM
    ax.add_patch(Rectangle((-sx / 2, -sy / 2), sx, sy, facecolor="0.9", edgecolor="0.3", lw=0.8))
    for m, _, fit in maps:
        ex, ey = ellipses[id(fit)]
        ax.fill(ex * 1e-3, ey * 1e-3, color=SETTING_COLORS[m["wavelength_nm"]][-1], lw=0)
    ax.add_patch(Rectangle((-half_x * 1e-3, -half_y * 1e-3), 2 * half_x * 1e-3, 2 * half_y * 1e-3,
                           fill=False, edgecolor="black", lw=0.6))
    ax.set_xlim(-0.6 * sx, 0.6 * sx)
    ax.set_ylim(0.6 * sy, -0.6 * sy)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-10, 11, 5))
    ax.set_yticks(np.arange(-10, 11, 5))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.text(0.0, -0.53 * sy, f"S170C active area {sx:g} mm × {sy:g} mm", ha="center", va="center", fontsize=7)
    ax.text(0.0, 0.53 * sy, "beam position on the sensor: schematic", ha="center", va="center",
            fontsize=6, color="0.35")

    for key, wl in zip("BC", wavelengths):
        ax = axd[key]
        group = [(m, irr, fit) for m, irr, fit in maps if m["wavelength_nm"] == wl]
        m_top, irr_top, _ = group[-1]
        px = m_top["pixel_um"]
        x, y = irr_top["x"], irr_top["y"]
        im = ax.imshow(irr_top["E"] / scale, extent=(x[0] - px / 2, x[-1] + px / 2, y[-1] + px / 2, y[0] - px / 2),
                       cmap="gray", vmin=0.0, vmax=vmax / scale, interpolation="nearest", zorder=1)
        win_w, win_h = x[-1] - x[0] + px, y[-1] - y[0] + px
        ax.add_patch(Rectangle((-win_w / 2, -win_h / 2), win_w, win_h, fill=False, edgecolor="black", lw=1.0,
                               zorder=3, label="SPT camera window"))
        ax.add_patch(Rectangle((-sensor_um[0] / 2, -sensor_um[1] / 2), sensor_um[0], sensor_um[1], fill=False,
                               edgecolor="black", lw=0.8, ls=(0, (4, 3)), zorder=3, label="Full camera sensor"))
        for i, (m, _, fit) in enumerate(group):
            ex, ey = ellipses[id(fit)]
            color = SETTING_COLORS[wl][i]
            ax.plot(ex, ey, color=color, lw=1.2, zorder=4, label=f"1/e², {wl} nm, {m['P_total_W'] * 1e3:g} mW")
            ax.plot(fit["x0"], fit["y0"], "+", color=color, ms=4, mew=0.8, zorder=4)
        ax.set_xlim(-half_x, half_x)
        ax.set_ylim(half_y, -half_y)
        ax.set_aspect("equal")
        ax.set_ylabel("y (µm)")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, frameon=False,
                  handlelength=1.4, borderaxespad=0.0, labelspacing=0.3)
    axd["C"].set_xlabel("x (µm)")
    axd["B"].tick_params(labelbottom=False)
    cb = fig.colorbar(im, ax=axd["A"], location="bottom", fraction=0.05, pad=0.02, aspect=25)
    cb.set_label(rf"Measured $E$ in the window, highest power ($10^{{{exponent}}}$ W m$^{{-2}}$)", fontsize=7)
    for label, ax in axd.items():
        ax.text(0.0, 1.02, label, transform=ax.transAxes, **PANEL_LABEL_KW)
    for key, corner_y in (("B", -1), ("C", 1)):
        fig.add_artist(ConnectionPatch(xyA=(half_x * 1e-3, corner_y * half_y * 1e-3), coordsA=axd["A"].transData,
                                       xyB=(-half_x, corner_y * half_y), coordsB=axd[key].transData,
                                       color="0.4", lw=0.5))
    return fig


# ── Test ───────────────────────────────────────────────────────────────────────
def test_synthetic_beam() -> None:
    """Synthetic rotated beam with known parameters, background and noise; prints recovered vs. true.
    Writes nothing to disk."""
    rng = np.random.default_rng(7)
    true = {"x0": -4.0, "y0": 2.5, "w_major": 70.0, "w_minor": 45.0, "theta": np.radians(25.0)}
    pixel_um, background, P_W = 0.3, 833.0, 1.5e-3
    x = (np.arange(VISIBLE_PX[0]) - (VISIBLE_PX[0] - 1) / 2.0) * pixel_um
    y = (np.arange(VISIBLE_PX[1]) - (VISIBLE_PX[1] - 1) / 2.0) * pixel_um
    xx, yy = np.meshgrid(x, y)
    counts = background + gauss2d_rot((xx, yy), *true.values(), 3000.0) + rng.normal(0, 5.0, xx.shape)
    irr = irradiance(counts, background, P_W, pixel_um)
    fit = fit_gaussian_rot(irr["E"], irr["x"], irr["y"])
    E_true = P_W * (counts - background) / (counts - background).sum() / irr["A_px_m2"]
    print(f"sum f = {irr['f'].sum():.12f}, sum P_pixel = {irr['P_pixel_W'].sum():.12e} W (P = {P_W} W), "
          f"max |E - E_true| / max E = {np.abs(irr['E'] - E_true).max() / E_true.max():.2e}")
    for k, v in true.items():
        got = np.degrees(fit[k]) if k == "theta" else fit[k]
        ref = np.degrees(v) if k == "theta" else v
        print(f"{k:>8}: true {ref:8.3f}  recovered {got:8.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    paths = sorted(DATA_DIR.glob(FILE_PATTERN))
    tifs_in_folder = sorted(DATA_DIR.glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"no {FILE_PATTERN} stacks in {DATA_DIR}")
    ms = [load_measurement(p) for p in paths]
    background = min(m["dark_counts"] for m in ms)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exclusions = [{"file": p.name, "step": "file pattern", "parameter": FILE_PATTERN, "n_before": 1, "n_after": 0,
                   "reason": "TIFF in DATA_DIR not matching the beam-profile pattern"}
                  for p in tifs_in_folder if p not in paths]
    rows, entries, profile_rows, maps = [], [], [], []
    for m in sorted(ms, key=lambda m: (m["wavelength_nm"], m["P_total_W"])):
        img_w = window(m["img"])
        exclusions.append({"file": m["file"], "step": "SPT window", "parameter": f"{VISIBLE_PX[0]} x {VISIBLE_PX[1]} px",
                           "n_before": int(m["img"].size), "n_after": int(img_w.size),
                           "reason": "pixels outside the sensor-centred SPT window not used (user, 2026-10-07)"})
        irr = irradiance(img_w, background, m["P_total_W"], m["pixel_um"])
        fit = fit_gaussian_rot(irr["E"], irr["x"], irr["y"])
        row = {k: m[k] for k in ("file", "wavelength_nm", "P_total_W", "pixel_um", "exposure_ms", "binning",
                                 "conversion_factor", "ir_sensitivity", "bit_depth_tiff", "n_frames", "sensor_px")}
        row.update(background_counts=background, A_pixel_m2=irr["A_px_m2"], **beam_parameters(m, irr, fit))
        rows.append(row)
        prof = centre_profiles(irr["E"], fit["model"], irr["x"], irr["y"], fit)
        entries.append((m, prof))
        maps.append((m, irr, fit))
        for key, other in (("x", "y"), ("y", "x")):
            pos, meas, model, at = prof[key]
            profile_rows += [{"file": m["file"], "wavelength_nm": m["wavelength_nm"], "P_total_W": m["P_total_W"],
                              "profile": key, f"{other}_um_of_profile": at, "position_um": float(q),
                              "E_measured_W_m2": float(e), "E_fit_W_m2": float(f)}
                             for q, e, f in zip(pos, meas, model)]

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / f"{STEM}_results.csv", index=False)
    power_comparison(results).to_csv(OUTPUT_DIR / f"{STEM}_power_comparison.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(OUTPUT_DIR / f"{STEM}_profiles.csv", index=False)
    pd.DataFrame(exclusions, columns=["file", "step", "parameter", "n_before", "n_after", "reason"]).to_csv(
        OUTPUT_DIR / f"{STEM}_exclusions.csv", index=False)
    with plt.style.context(STYLE_PATH):
        fig = plot_profiles(entries)
        fig.savefig(OUTPUT_DIR / f"{STEM}_profiles.pdf")
        plt.close(fig)
        # full camera sensor: the largest recorded frame (full-sensor recordings 514, 514_2)
        largest = max(ms, key=lambda m: m["img"].size)
        sensor_um = (largest["img"].shape[1] * largest["pixel_um"], largest["img"].shape[0] * largest["pixel_um"])
        fig = plot_geometry(maps, sensor_um)
        fig.savefig(OUTPUT_DIR / f"{STEM}_geometry.pdf")
        plt.close(fig)

    print(f"{len(paths)} of {len(tifs_in_folder)} TIFFs used; background {background:.1f} counts")
    show = ["file", "P_total_W", "E_max_measured_W_m2", "E0_fit_W_m2", "E_mean_illuminated_W_m2",
            "x0_um", "y0_um", "w_major_um", "w_minor_um", "theta_deg", "ellipticity", "R2",
            "fit_power_fraction_in_window", "saturated_fraction"]
    print(results[show].round(3).to_string(index=False))
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
