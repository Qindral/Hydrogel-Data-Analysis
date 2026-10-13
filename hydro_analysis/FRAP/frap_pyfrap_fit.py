"""
FRAP with PyFRAP, stage 2 (fit): D of FITC-dextran 150 kDa from PyFRAP (mueller-lab/PyFRAP 1.1.13,
simulation of the diffusion equation with the first post-bleach image as initial condition and fit
of D by time scaling). Run with the PyFRAP environment, not with .venv:
    C:\\Users\\Jonas\\venvs\\pyfrap\\Scripts\\python.exe hydro_analysis\\FRAP\\frap_pyfrap_fit.py
(Python 3.11, scipy 1.13.1, setuptools < 81, trackpy; gmsh 2.14 in PyFRAP/pyfrp/executables/gmsh).

Compute stage. Reads cache/frap_pyfrap_inputs.pkl (frap_pyfrap_inputs.py, run that first: drift-tracked
ROI centre per frame, bleach radius r_n from the Leica XML, times) and, read-only, the raw TIF series
(frap_analysis_v2.load_measurement). Writes cache/frap_pyfrap_result_<version>.pkl, read by
frap_pyfrap_figures.py, and tables in OUTPUT_DIR.

Versions (user, 2026-10-08, E024 and follow-up), differing only in geometry and initial condition:
  2d      PyFRAP custom 2D disk; bleach and measurement in one focal plane.
  thin3d  cylinder of height Z_HEIGHT_UM = 1 um; the bleach fills the whole height (no gradient in z).
  slab03  cylinder of height Z_HEIGHT_UM = 1 um; the bleach is a SLAB_UM = 0.3 um layer in the middle
          and diffuses in z within the 1 um; data and fitted ROIs are taken in the same 0.3 um layer.
Both cylinders are PyFRAP custom geometries made by extruding the 2D disk mesh in z into prisms
(gmsh Layers + Recombine; layers aligned with the 0.3 um layer). Synthetic check of thin3d, true
D = 25 um2/s: PyFRAP's cylinder template -14 %, extruded tetrahedra -6.6 %, extruded prisms +1.0 %
(same as 2d, as it must be without gradients in z); FiPy's finite volumes need near-orthogonal cells.
Normalisation (E022, user 2026-10-08): PyFRAP standard, each frame divided pixelwise by the pre-bleach
image with pyfrp_img_module.normImg: (F + DATA_OFFSET) / (F_pre + DATA_OFFSET).
Data ROIs: "Bleached" = disk of radius r_n around the stage-1 centre of each frame (E019, E020, E023);
"Slice" = whole image. PyFRAP pins data and simulation with "Bleached" (background) and "Slice" (norm)
(computeIdealFRAPPinVals) and fits only "Bleached". The simulated "Bleached" ROI sits at the centre of
the first post-bleach frame, where the initial condition was taken. Initial condition: PyFRAP ICmode 3
(first normalised post-bleach image, rim concentration outside the image); for slab03 the cells
outside the 0.3 um layer are set to the rim concentration (wrapper around applyInterpolatedICs).
All other settings are chosen by Claude (user: "den Rest darf alles geraten werden", 2026-10-08) and
marked in the configuration; PyFRAP defaults are kept where they exist.
Before the real data, synthetic_check fits images with a known D in every version (2D Gaussian bleach;
for slab03 multiplied by the analytic z-average over the 0.3 um layer in a closed 1 um column).
PyFRAP issues worked around without changing PyFRAP: mesh.addBoxField fails under Python 3
(bytes/str), so the refinement is written into the .geo file; radialROI.computeMeshIdx passes the
radius twice (NameError), so radialSliceROI is used.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# PyFRAP lives only in its own environment; started with another interpreter (e.g. .venv from the IDE),
# the script restarts itself there.
PYFRAP_PYTHON = Path(r"C:\Users\Jonas\venvs\pyfrap\Scripts\python.exe")
try:
    import pyfrp  # noqa: F401
except ModuleNotFoundError:
    if __name__ == "__main__" and PYFRAP_PYTHON.is_file() and Path(sys.executable).resolve() != PYFRAP_PYTHON.resolve():
        print(f"pyfrp not found in {sys.executable}; restarting with {PYFRAP_PYTHON}")
        sys.exit(subprocess.call([str(PYFRAP_PYTHON), str(Path(__file__).resolve())]))
    raise ModuleNotFoundError(f"pyfrp is not installed in {sys.executable}; run this script with {PYFRAP_PYTHON}")

from pyfrp.modules import pyfrp_img_module, pyfrp_sim_module
from pyfrp.subclasses import pyfrp_embryo

from hydro_analysis.FRAP.frap_analysis_v2 import load_measurement
from hydro_analysis.FRAP.frap_pyfrap_inputs import disk_mean

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_CACHE = Path(__file__).resolve().parent / "cache" / "frap_pyfrap_inputs.pkl"
RESULT_CACHE = Path(__file__).resolve().parent / "cache" / "frap_pyfrap_result_{version}.pkl"
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\FRAP_v2\pyfrap_fit")
VERSIONS = ("2d", "thin3d", "slab03")             # decided by user, 2026-10-08 (E024 and follow-up)
Z_HEIGHT_UM = 1.0                                 # decided by user, 2026-10-08 (E022, E024)
SLAB_UM = 0.3                                     # decided by user, 2026-10-08: bleach height of slab03

# Chosen by Claude (E024: "den Rest darf alles geraten werden"), not decided by the user
DATA_OFFSET = 1.0                # PyFRAP default of analysis.dataOffset
RIM_FACTOR = 0.66                # PyFRAP default of genDefaultROIs: rim = ring 0.66-1.0 of the half image width
DOMAIN_RADIUS_FIELDS = 2.0       # domain radius in image widths (2 x 144.7 um), far from the spreading bleach
VOL_SIZE_IMAGE_PX = 8.0          # mesh size inside the image square (2.3 um)
VOL_SIZE_OUTSIDE_PX = 40.0       # mesh size outside the image square
LAYERS = {"thin3d": (1,), "slab03": (3, 3, 3)}   # extrusion layers: slab03 = 0.35 / 0.3 / 0.35 um, 0.1-0.12 um each
STEPS_SIM = 3000                 # PyFRAP default; log time scale (PyFRAP toLogTimeScale)
LBD_PX, UBD_PX = 0.01, 3000.0    # fit bounds in px^2/s; PyFRAP default UBD = 300 px^2/s = 24 um^2/s is below water
X0_D_PX = 10.0                   # PyFRAP default start value
EQU_ON, FIT_PINNED = True, True  # PyFRAP defaults (equalisation factor 0.1-3, pinned fit)
QC_BOUND_MARGIN = 0.01           # qc_pyfrap: fit converged and D not within 1 % of a bound

SYNTHETIC_TOL = 0.05             # max relative deviation of the fitted D from the true D
SYNTHETIC = {"D_um2_s": 25.0, "w0_um": 35.0, "K0": 0.8, "intensity": 60.0, "texture_rel": 0.1,
             "r_n_um": 16.29, "mpp_um": 0.283, "n_pre": 7, "schedule": ((0.373, 60), (2.0, 20)), "seed": 5}


# ── Geometry ───────────────────────────────────────────────────────────────────
def _z_layers(version: str, mpp_um: float) -> list[tuple[float, int]]:
    """(thickness in px, number of mesh layers) from the top (z = 0) downwards."""
    h = Z_HEIGHT_UM / mpp_um
    if version == "thin3d":
        return [(h, LAYERS["thin3d"][0])]
    outer = (Z_HEIGHT_UM - SLAB_UM) / 2 / mpp_um
    return [(outer, LAYERS["slab03"][0]), (SLAB_UM / mpp_um, LAYERS["slab03"][1]), (outer, LAYERS["slab03"][2])]


def write_geo(path: Path, version: str, mpp_um: float):
    """Disk of DOMAIN_RADIUS_FIELDS image widths, refined inside the image, extruded in z for 3D versions."""
    r = DOMAIN_RADIUS_FIELDS * 512
    lines = [f"volSize_px={VOL_SIZE_OUTSIDE_PX};",
             "Point(1) = {256, 256, 0, volSize_px};",
             f"Point(2) = {{{256 - r}, 256, 0, volSize_px}};", f"Point(3) = {{256, {256 + r}, 0, volSize_px}};",
             f"Point(4) = {{{256 + r}, 256, 0, volSize_px}};", f"Point(5) = {{256, {256 - r}, 0, volSize_px}};",
             "Circle(1) = {2, 1, 3};", "Circle(2) = {3, 1, 4};", "Circle(3) = {4, 1, 5};", "Circle(4) = {5, 1, 2};",
             "Line Loop(1) = {1, 2, 3, 4};", "Plane Surface(1) = {1};",
             "Field[1] = Box;", f"Field[1].VIn = {VOL_SIZE_IMAGE_PX};", f"Field[1].VOut = {VOL_SIZE_OUTSIDE_PX};",
             "Field[1].XMin = 0;", "Field[1].XMax = 512;", "Field[1].YMin = 0;", "Field[1].YMax = 512;",
             "Field[1].ZMin = -1e6;", "Field[1].ZMax = 1e6;", "Background Field = 1;"]
    if version != "2d":
        surface = "1"
        for i, (dz, n) in enumerate(_z_layers(version, mpp_um)):
            lines.append(f"ex{i}[] = Extrude {{0, 0, {-dz}}} {{ Surface{{{surface}}}; Layers{{{n}}}; Recombine; }};")
            surface = f"ex{i}[0]"
    path.write_text("\n".join(lines) + "\n")


def slab_z_range_px(mpp_um: float) -> tuple[float, float]:
    top = -(Z_HEIGHT_UM - SLAB_UM) / 2 / mpp_um
    return top - SLAB_UM / mpp_um, top


# ── PyFRAP ─────────────────────────────────────────────────────────────────────
def fit_pyfrap(version, post_norm_first, t, data_bleached, data_slice, centre0, r_n_um, mpp_um, conc_rim):
    """One PyFRAP simulation and fit; returns a result dict and the fitted curves."""
    original_ics = pyfrp_sim_module.applyInterpolatedICs
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            t0 = time.time()
            emb = pyfrp_embryo.embryo(f"frap_{version}")
            emb.setDataResPx(512)
            emb.setDataResMu(512 * mpp_um)
            geo = Path(tmp) / f"{version}.geo"
            write_geo(geo, version, mpp_um)
            emb.setGeometry2Custom([256.0, 256.0], fnGeo=str(geo), dim=2 if version == "2d" else 3)
            emb.tStart, emb.tEnd, emb.nFrames = 0.0, float(t[-1]), len(t)
            emb.tvecData = np.asarray(t, float)
            emb.newAnalysis()
            emb.analysis.concRim = float(conc_rim)
            if version == "slab03":
                z_lo, z_hi = slab_z_range_px(mpp_um)
                height, width = (z_lo + z_hi) / 2, z_hi - z_lo
            else:
                height, width = 0.0, 1e6
            emb.newSquareROI("Image", 0, [0, 0], 512, asMaster=True)
            emb.newSquareSliceROI("Slice", 1, [0, 0], 512, height, width, False)
            # radialSliceROI: PyFRAP radialROI.computeMeshIdx passes the radius twice (NameError)
            emb.newRadialSliceROI("Bleached", 2, [float(centre0[0]), float(centre0[1])], r_n_um / mpp_um,
                                  height, width, False)
            sim = emb.newSimulation()
            sim.mesh.setVolSizePx(VOL_SIZE_OUTSIDE_PX)
            sim.mesh.genMesh()
            z_cells = np.asarray(sim.mesh.getCellCenters()[2])
            emb.computeROIIdxs()
            for name, vec in (("Bleached", data_bleached), ("Slice", data_slice), ("Image", data_slice)):
                emb.getROIByName(name).dataVec = list(vec)
            if version == "slab03":
                outside = (z_cells < z_lo) | (z_cells > z_hi)

                def slab_ics(phi, simulation, *args, **kwargs):
                    phi = original_ics(phi, simulation, *args, **kwargs)
                    values = np.asarray(phi.value).copy()
                    values[outside] = conc_rim
                    phi.setValue(values)
                    return phi

                pyfrp_sim_module.applyInterpolatedICs = slab_ics
            sim.setICMode(3)
            sim.setICimg(np.asarray(post_norm_first, float))
            sim.setD(UBD_PX)
            sim.setTimesteps(STEPS_SIM)
            sim.getOptTvecSim(UBD_PX)
            sim.toLogTimeScale()
            sim.run(showProgress=False)
            pyfrp_sim_module.applyInterpolatedICs = original_ics
            bkgd, norm, bkgd_sim, norm_sim = emb.computeIdealFRAPPinVals(bkgdName="Bleached", normName="Slice")
            emb.pinAllROIs(bkgdVal=bkgd, normVal=norm, bkgdValSim=bkgd_sim, normValSim=norm_sim)
            fit = emb.newFit("fit")
            fit.addROIByName("Bleached")
            fit.setLBD(LBD_PX)
            fit.setUBD(UBD_PX)
            fit.setX0D(X0_D_PX)
            fit.setEqu(EQU_ON)
            fit.setFitPinned(FIT_PINNED)
            fit.run()
            roi = emb.getROIByName("Bleached")
            d_px = float(fit.DOptPx)
            qc = bool(fit.success) and LBD_PX * (1 + QC_BOUND_MARGIN) < d_px < UBD_PX * (1 - QC_BOUND_MARGIN)
            res = {"version": version, "D_pyfrap_um2_s": float(fit.DOptMu), "D_pyfrap_px2_s": d_px,
                   "fit_success": bool(fit.success), "qc_pyfrap": qc, "Rsq": float(fit.MeanRsq),
                   "SSD": float(fit.SSD), "equ_factor": float(fit.equFacts[0]) if len(fit.equFacts) else np.nan,
                   "n_cells": len(z_cells), "n_cells_bleached_roi": len(roi.meshIdx),
                   "sim_bleached_t0": float(roi.simVec[0]), "data_bleached_t0": float(data_bleached[0]),
                   "pin_data": (bkgd, norm), "pin_sim": (bkgd_sim, norm_sim), "runtime_s": time.time() - t0}
            curves = {"t": np.asarray(fit.tvecFit, float), "data_pinned": np.asarray(fit.dataVecsFitted[0], float),
                      "fit_pinned": np.asarray(fit.fittedVecs[0], float)}
        finally:
            pyfrp_sim_module.applyInterpolatedICs = original_ics
            os.chdir(cwd)
    return res, curves


# ── Data ───────────────────────────────────────────────────────────────────────
def prepare(pre_frames, post_frames, centres, r_n_um, mpp_um):
    """PyFRAP-normalised frames -> data vectors, first frame (initial condition) and rim concentration."""
    pre = pre_frames.mean(axis=0)
    norm = [pyfrp_img_module.normImg(np.asarray(f, float), pre, dataOffset=DATA_OFFSET) for f in post_frames]
    radius_px = r_n_um / mpp_um
    bleached = np.array([disk_mean(n, c, radius_px)[0] for n, c in zip(norm, centres)])
    slice_ = np.array([n.mean() for n in norm])
    rows, cols = np.indices(pre.shape)
    r = np.hypot(rows - 255.5, cols - 255.5)
    rim = (r >= RIM_FACTOR * 256) & (r <= 256)
    return bleached, slice_, norm[0], float(norm[0][rim].mean())


# ── Synthetic check ────────────────────────────────────────────────────────────
def slab_average_factor(t, d_um2_s, n_terms=4000):
    """Mean over the central SLAB_UM layer of a layer bleached to 1 at t = 0, closed column Z_HEIGHT_UM."""
    h, a = Z_HEIGHT_UM, SLAB_UM
    z1, z2 = (h - a) / 2, (h + a) / 2
    n = np.arange(1, n_terms + 1)[:, None]
    s = np.sin(n * np.pi * z2 / h) - np.sin(n * np.pi * z1 / h)
    terms = 2 * h / (a * (n * np.pi) ** 2) * s ** 2 * np.exp(-(n * np.pi / h) ** 2 * d_um2_s * np.asarray(t)[None, :])
    return a / h + terms.sum(axis=0)


def synthetic_inputs(p: dict, version: str) -> dict:
    rng = np.random.default_rng(p["seed"])
    steps = np.repeat([dt for dt, _ in p["schedule"]], [n for _, n in p["schedule"]])[:-1]
    t = np.concatenate([[0.0], np.cumsum(steps)])
    axial = slab_average_factor(t, p["D_um2_s"]) if version == "slab03" else np.ones_like(t)
    axial[0] = 1.0
    rows, cols = np.indices((512, 512))
    truth_pre = p["intensity"] * (1 + p["texture_rel"] * rng.standard_normal((512, 512)))
    pre = np.clip(rng.poisson(np.tile(truth_pre, (p["n_pre"], 1, 1))), 0, 255).astype(float)
    centre = (250.3, 258.7)
    r2_um = ((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2) * p["mpp_um"] ** 2
    post = []
    for ti, ai in zip(t, axial):
        w2 = p["w0_um"] ** 2 + 8 * p["D_um2_s"] * ti
        k = p["K0"] * p["w0_um"] ** 2 / w2 * ai
        post.append(np.clip(rng.poisson(truth_pre * (1 - k * np.exp(-2 * r2_um / w2))), 0, 255).astype(float))
    return {"pre": pre, "post": np.array(post), "t": t, "centres": [centre] * len(t), "axial": axial}


def synthetic_check(versions=VERSIONS) -> pd.DataFrame:
    rows = []
    for version in versions:
        s = synthetic_inputs(SYNTHETIC, version)
        bleached, slice_, first, rim = prepare(s["pre"], s["post"], s["centres"], SYNTHETIC["r_n_um"], SYNTHETIC["mpp_um"])
        res, _ = fit_pyfrap(version, first, s["t"], bleached, slice_, s["centres"][0], SYNTHETIC["r_n_um"],
                            SYNTHETIC["mpp_um"], rim)
        rel = res["D_pyfrap_um2_s"] / SYNTHETIC["D_um2_s"] - 1
        rows.append({**{k: v for k, v in res.items() if k not in ("pin_data", "pin_sim")},
                     "D_true_um2_s": SYNTHETIC["D_um2_s"], "rel_error": rel, "passed": abs(rel) <= SYNTHETIC_TOL})
        print(f"Synthetic {version}: D = {res['D_pyfrap_um2_s']:.2f} um2/s (true {SYNTHETIC['D_um2_s']}), "
              f"error {100 * rel:+.1f} %, cells {res['n_cells']}, {res['runtime_s']:.0f} s")
    df = pd.DataFrame(rows)
    if not df.passed.all():
        raise RuntimeError(f"Synthetic check failed (tolerance {SYNTHETIC_TOL:.0%}); real data are not fitted.\n{df}")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_check().to_csv(OUTPUT_DIR / "frap_pyfrap_fit_synthetic.csv", index=False)
    if not INPUT_CACHE.is_file():
        raise FileNotFoundError(f"{INPUT_CACHE} missing: run frap_pyfrap_inputs.py (with .venv) first.")
    inputs = pickle.loads(INPUT_CACHE.read_bytes())["data"]
    prepared = {}
    for mid, d in inputs.items():
        meas = load_measurement(Path(d["folder"]))
        if len(meas["post"]) != len(d["t_s"]) or not np.allclose(meas["post_t_s"], d["t_s"]):
            raise ValueError(f"{mid}: frames or times differ from {INPUT_CACHE.name}")
        centres = list(zip(d["row"], d["col"]))
        prepared[mid] = (centres, *prepare(meas["pre"], meas["post"], centres, d["r_n_um"], d["mpp_um"]))
    for version in VERSIONS:      # one version after the other; the cache is rewritten after every measurement
        results, curves = [], {}
        for mid, d in inputs.items():
            centres, bleached, slice_, first, rim = prepared[mid]
            res, cur = fit_pyfrap(version, first, d["t_s"], bleached, slice_, centres[0], d["r_n_um"],
                                  d["mpp_um"], rim)
            results.append({"measurement_id": mid, "condition": d["condition"], "r_n_um": d["r_n_um"],
                            "conc_rim": rim, **res})
            curves[mid] = {**cur, "data_bleached": bleached, "data_slice": slice_, "t_s": d["t_s"]}
            print(f"{version} {mid}: D = {res['D_pyfrap_um2_s']:.2f} um2/s, Rsq {res['Rsq']:.3f}, "
                  f"qc {res['qc_pyfrap']}, {res['runtime_s']:.0f} s", flush=True)
            df = pd.DataFrame(results)
            Path(str(RESULT_CACHE).format(version=version)).write_bytes(pickle.dumps(
                {"created": f"{datetime.now():%Y-%m-%d %H:%M}", "results": df, "curves": curves,
                 "input_cache": str(INPUT_CACHE), "complete": len(results) == len(inputs)}))
            df.drop(columns=["pin_data", "pin_sim"]).to_csv(OUTPUT_DIR / f"frap_pyfrap_fit_{version}.csv", index=False)
        print(df[["measurement_id", "D_pyfrap_um2_s", "Rsq", "equ_factor", "qc_pyfrap"]].round(3).to_string(index=False))
    print(f"Caches: {RESULT_CACHE.parent}\nTables: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
