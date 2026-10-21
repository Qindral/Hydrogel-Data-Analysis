"""
Key figures and tables of the oscillatory rheology of the 20 mg/mL hydrogel:
viscoelastic behaviour (frequency and amplitude sweep), stability of G' over gel
age, network mesh size, and pore sizes (cavity and throat) of phantom networks
with junction functionality f = 4, 6 and 8 for fibril diameters of 9-13 nm.
Consumer stage: reads only cache/rheo_hyd20_results.pkl written by
Rheo_Hyd20_Compute.py (run that first); raw .tri files are never touched and
nothing is refitted here.

Data basis (decisions by the user, 2026-10-08, ENTSCHEIDUNGEN.md E008, E025-E028):
  - only runs of the "qc_filtered" selection of Rheo_Hyd20_Compute.py are used: runs
    with the QC codes axial_force, presheared, cracks or group_outlier and FAIL runs are
    left out of figures and statistics; they stay in the run table (table 1);
  - the unit of every statistic is the specimen (cast + position); technical
    repeats are averaged per specimen in Rheo_Hyd20_Compute.py (stability_specimen_level);
  - uncertainty = two-sided 95 % confidence interval of the mean over specimens,
    t distribution with N - 1 degrees of freedom; no interval for N = 1;
  - figure 1: amplitude sweep of one 9 d run drawn at random (fixed seed; second draw of
    the generator, the first draw picks a 4 d specimen whose frequency sweep is no longer
    shown) from the qc_filtered amplitude sweeps of 9 d; black, y range 10-600 Pa (points outside are counted in the console output);
    legend G' (filled) / G'' (open) and no vertical gamma_5% / gamma_10% lines (user,
    2026-10-14, replaces "Legende weglassen" of E028; the strains stay in table 1);
  - figure 2: G' at 0.5 Hz (Gp_rep_Pa) of every qc_filtered run vs. gel age, no means,
    no legend;
  - mesh size (value for the caption, no figure) and pore sizes pool all qc_filtered
    specimens younger than 10 d (4 d, 9 d);
  - pore-size error bars combine the 95 % CI over specimens with the fibril diameter,
    taken as uniform on 9-13 nm (standard uncertainty (13 - 9) / sqrt(12) nm, expanded
    with 1.96), in quadrature; the point is the value at d_f = 11 nm.
Upstream data handling (Rheo_Hyd20_Compute.py, unchanged): inertia-dominated and
torque-limited points are left out of every fit (E026; shown grey in figure 1); G'_rep is
the log-log window fit evaluated at 0.5 Hz; G'' and tan(delta) at 0.5 Hz are log-log
interpolations; gamma_5% / gamma_10% are the strains where G' leaves the low-strain
reference by 5 % / 10 %.

Mesh size (independent of f): affine network, xi = (k_B T / G')^(1/3)
(core.physics.calculate_network_mesh_size with front factor 1).

Pore geometry (phantom network, Rubinstein & Colby 2003, ch. 7; network_geometry.py):
  strand density nu = G' / ((1 - 2/f) k_B T); mesh size xi_f = nu^(-1/3) (identical to
  core.physics.calculate_network_mesh_size with front factor 1 - 2/f). The junctions sit
  on the regular lattice of coordination f: f = 4 diamond (16 strands per cubic cell),
  f = 6 simple cubic (3), f = 8 body-centred cubic (8, strands from the corners to the
  body centre). Cell edge a = (strands per cell)^(1/3) xi_f. Strands are cylinders of
  diameter d_f; cavity diameter = kappa_cav a - d_f (largest inscribed sphere), throat
  diameter = kappa_thr a - d_f (largest sphere passing from one cavity to the next).
  kappa_cav is the exact maximum of the distance to the strand axes; kappa_thr is the
  exact minimum of that distance on the straight line between neighbouring cavity
  centres. The grid percolation of network_geometry.py is a lower bound of the throat
  and is used here only as a check (verification table); at its default 48 points per
  edge it gives 0.678 a for the diamond lattice instead of the exact 0.7071 a.

Writes to OUTPUT_DIR (figures as PDF, tables as CSV and one workbook):
  fig1_viscoelasticity.pdf          G'(gamma) filled, G''(gamma) open of the drawn 9 d
                                    amplitude sweep, with legend
  fig2_stability_gp_vs_age.pdf      G' at 0.5 Hz of every qc_filtered run vs. gel age
  fig3a_lattice_cells.pdf           shaded renderings (pyvista/VTK) of the periodic lattices
                                    f = 4, 6, 8 around one cavity, one scale, 10 nm bar;
                                    strands as thin tubes, cavity sphere and throat ring with
                                    the axis diameters (fibril diameter not drawn); neighbouring
                                    cells fade out, strands in front of the cavity are faint
  fig3a_lattice_cells_f<f>.png      the three renderings (embedded in the PDF)
  fig3b_cavity_vs_f.pdf             cavity diameter vs. f (d_f = 11 nm), combined uncertainty
  table1_viscoelastic_all_runs.csv  key values of every run (frequency + amplitude sweeps)
  table1b_viscoelastic_summary.csv  mean, SD, 95 % CI per age and pooled < 10 d
  table2_stability_gp_by_age.csv    G' at 0.5 Hz per age (mean, SD, 95 % CI, N)
  table3a_pore_geometry_factors.csv geometric factors per f
  table3b_pore_size.csv             xi_f, a, b, cavity, throat per f and d_f (95 % CI)
  table3c_pore_size_per_specimen.csv
  table3d_pore_size_combined.csv    cavity and throat at d_f = 11 nm with combined uncertainty
  table4_mesh_size.csv              xi per specimen and pooled < 10 d (value for the caption)
  verification_synthetic.csv        checks against known answers (run before the data)
  rheo_hyd20_key_figures_exclusions.csv
  rheo_hyd20_key_figures.xlsx       all tables above as sheets
No downstream consumer.
"""
from __future__ import annotations

import pickle
from itertools import product
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import optimize, stats

from hydro_analysis.core.physics import BOLTZMANN_CONSTANT, calculate_network_mesh_size
from hydro_analysis.Rheology import network_geometry as ng

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "rheo_hyd20_results.pkl"
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
OUTPUT_DIR = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
    r"\Rheology"
)
SELECTION = "qc_filtered"                 # decided by user, 2026-10-08 (E008, E025)
CI_LEVEL = 0.95                           # decided by user, 2026-10-08 (E025): t interval over specimens
REPRESENTATIVE_AGE_D = 4                  # decided by user, 2026-10-08 (E025): "random eine 4d auswählen"
AMPLITUDE_SWEEP_AGE_D = 9                 # decided by user, 2026-10-08 (E027): amplitude sweep drawn at random from 9 d
RANDOM_SEED = 20261008                    # fixed seed for both draws, set by Claude
POOL_MAX_AGE_D = 10                       # decided by user, 2026-10-08 (E025): ages below 10 d pooled
FUNCTIONALITIES = (4, 6, 8)               # decided by user, 2026-10-08
STRAND_DIAMETER_RANGE_NM = (9.0, 13.0)    # decided by user, 2026-10-08: fibril diameter "ca. 9 bis 13 nm"
STRAND_DIAMETERS_TABLE_NM = (9.0, 11.0, 13.0)   # table rows: range limits and midpoint, set by Claude
STRAND_DIAMETER_DRAWING_NM = 11.0         # point value of the pore figures (midpoint of the range), set by Claude
STRAND_DIAMETER_COVERAGE = 1.96           # decided by user, 2026-10-08 (E027): d_f uniform, u = width/sqrt(12) x 1.96
VERIFY_GRID_POINTS = (48, 96)             # grid percolation check of the throat (lower bound)
AS_YLIM_PA = (10.0, 600.0)               # decided by user, 2026-10-08 (E028): y range of figure 1
STABILITY_MARKERSIZE = 3.0               # decided by user, 2026-10-08 (E028): "etwas kleiner" than 4
# Rendering of the lattices (figure 3a; pyvista/VTK, shading without cast shadows); all set by Claude.
RENDER_PX = 1200                          # pixels per panel edge (about 600 dpi at the printed size)
RENDER_VIEW_DIRECTION = (1.0, 0.62, 0.42) # camera direction from the cavity centre, same for every f
RENDER_HALF_HEIGHT_NM = 27.0              # half height of the shown field (parallel projection, one scale)
RENDER_SHOW_RADIUS_NM = 27.0              # strands drawn within this distance of the cavity centre
RENDER_FADE_START_NM = 13.0               # full opacity inside, fading to zero at RENDER_SHOW_RADIUS_NM
RENDER_FRONT_DEPTH_NM = 4.0               # strands more than this in front of the cavity centre are faint
RENDER_FRONT_ALPHA = 0.3
RENDER_TUBE_RADIUS_NM = 0.55              # visual strand radius only, not the fibril diameter
RENDER_NODE_RADIUS_NM = 1.1
RENDER_CAVITY_OPACITY = 0.5
RENDER_SCALE_BAR_NM = 10.0

LATTICE_NAME = {4: "diamond", 6: "simple cubic", 8: "body-centred cubic"}
STRANDS_PER_CELL = {4: 16, 6: 3, 8: 8}
STRAND_LENGTH_PER_EDGE = {4: np.sqrt(3.0) / 4.0, 6: 1.0, 8: np.sqrt(3.0) / 2.0}

WIDTH_IN = {"full": 6.30, "narrow": 4.72, "half": 3.07, "third": 2.01}
RATIO = 1.42
COLOR_GP = ("#3B8C8C", "#2A6666")         # style guide §11, category A
COLOR_GPP = ("#D98C3D", "#A6672D")        # style guide §11, category B
COLOR_F = {4: ("#3B8C8C", "#2A6666"), 6: ("#D98C3D", "#A6672D"), 8: ("#3B6E8C", "#2A4F66")}
COLOR_SINGLE = ("#3B6E8C", "#2A4F66")     # style guide §11, single series
COLOR_EXCLUDED = "#9a9a9a"
COLOR_REFERENCE = "#555555"
POINT_ALPHA = 0.7
ERR_KW = dict(elinewidth=0.8, capsize=2.0, capthick=0.8)


# ── Statistics ─────────────────────────────────────────────────────────────────

def ci_t(values, level: float = CI_LEVEL) -> Dict[str, float]:
    """N, mean, SD and the two-sided t confidence interval of the mean of the finite values."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    n = len(v)
    out = {"N": n, "mean": float(v.mean()) if n else np.nan, "sd": np.nan, "t": np.nan,
           "ci_half": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if n >= 2:
        out["sd"] = float(v.std(ddof=1))
        out["t"] = float(stats.t.ppf(0.5 + level / 2.0, n - 1))
        out["ci_half"] = out["t"] * out["sd"] / np.sqrt(n)
        out["ci_low"], out["ci_high"] = out["mean"] - out["ci_half"], out["mean"] + out["ci_half"]
    return out


# ── Lattice geometry ───────────────────────────────────────────────────────────

_SHIFTS = np.array(list(product((-1, 0, 1), repeat=3)), float)


def lattice_segments(functionality: int) -> np.ndarray:
    """Strand axes of one cubic cell (edge 1); f = 4 and 6 from network_geometry.py, f = 8 bcc."""
    if functionality == 8:
        centre = np.full(3, 0.5)
        return np.array([[centre, centre + 0.5 * np.array(s)] for s in product((1.0, -1.0), repeat=3)])
    return ng.lattice_segments(functionality)


def axis_distance(segments: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact distance of arbitrary points (cell units) to the nearest strand axis of the periodic lattice."""
    pts = np.atleast_2d(points)
    best = np.full(len(pts), np.inf)
    for p0, p1 in segments:
        d = p1 - p0
        for s in _SHIFTS:
            rel = pts - (p0 + s)
            t = np.clip(rel @ d / (d @ d), 0.0, 1.0)
            np.minimum(best, np.linalg.norm(rel - t[:, None] * d, axis=1), out=best)
    return best


def lattice_pore_geometry(functionality: int, n: int = 48) -> Dict[str, object]:
    """Cavity and throat axis diameters (units of a), cavity centre and throat position.

    Cavity centres: maxima of the distance field on the grid, refined by a local maximisation
    of the exact distance. Throat: for each neighbouring cavity centre the minimum of the exact
    distance along the straight connection; the largest of these minima.
    """
    segments = lattice_segments(functionality)
    field = ng.axis_distance_field(segments, n)
    candidates = np.argwhere(field >= field.max() - 1e-9) / n
    refined = []
    for c in candidates:
        res = optimize.minimize(lambda p: -axis_distance(segments, p)[0], c, method="Nelder-Mead",
                                options={"xatol": 1e-10, "fatol": 1e-12})
        refined.append((res.x % 1.0, -res.fun))
    r_cav = max(r for _, r in refined)
    centres = np.array([c for c, r in refined if r >= r_cav - 1e-8])
    centre = centres[np.argmin(np.linalg.norm(centres - 0.5, axis=1))]

    images = np.concatenate([centres + s for s in _SHIFTS])
    dist = np.linalg.norm(images - centre, axis=1)
    nearest = dist[dist > 1e-6].min()
    neighbours = images[(dist > 1e-6) & (dist < nearest * 1.001)]
    best = (-np.inf, None, None)
    for q in neighbours:
        res = optimize.minimize_scalar(lambda t: axis_distance(segments, centre + t * (q - centre))[0],
                                       bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10})
        coarse = axis_distance(segments, centre + np.linspace(0, 1, 2001)[:, None] * (q - centre)).min()
        r = min(res.fun, coarse)
        mid = centre + res.x * (q - centre)
        # Among equivalent throats keep the one closest to the cell centre (for the drawing only).
        if r > best[0] + 1e-9 or (abs(r - best[0]) <= 1e-9 and np.linalg.norm(mid - 0.5) < np.linalg.norm(best[1] - 0.5)):
            best = (r, mid, (q - centre) / np.linalg.norm(q - centre))
    return {
        "functionality": functionality,
        "lattice": LATTICE_NAME[functionality],
        "segments": segments,
        "cavity_axis_diameter": 2.0 * r_cav,
        "throat_axis_diameter": 2.0 * best[0],
        "cavity_centre": centre,
        "throat_centre": best[1],
        "throat_normal": best[2],
        "cavity_centres_per_cell": len(centres),
        "strand_length": STRAND_LENGTH_PER_EDGE[functionality],
        "strands_per_cell": STRANDS_PER_CELL[functionality],
    }


def cell_edge_nm(storage_modulus_Pa, temperature_K, functionality: int):
    """Cell edge a (nm) of the lattice whose phantom-network modulus equals G'."""
    xi = calculate_network_mesh_size(storage_modulus_Pa, temperature_K, 1.0 - 2.0 / functionality)
    return STRANDS_PER_CELL[functionality] ** (1.0 / 3.0) * xi


# ── Verification on known answers ──────────────────────────────────────────────

def verify_on_synthetic(geometries: Dict[int, dict]) -> pd.DataFrame:
    """Checks with known answers; raises if one fails. Run before any real data are used."""
    rows = []

    def check(name: str, recovered: float, true: float, tol: float) -> None:
        ok = abs(recovered - true) <= tol
        rows.append({"check": name, "true": true, "recovered": recovered, "abs_error": recovered - true,
                     "tolerance": tol, "passed": ok})

    # 1) Confidence interval against scipy for a synthetic sample.
    sample = np.array([700.0, 760.0, 820.0, 880.0])
    lo, hi = stats.t.interval(0.95, len(sample) - 1, loc=sample.mean(), scale=stats.sem(sample))
    ci = ci_t(sample)
    check("95 % CI lower limit, synthetic sample", ci["ci_low"], lo, 1e-9)
    check("95 % CI upper limit, synthetic sample", ci["ci_high"], hi, 1e-9)

    # 2) Geometry: analytic simple cubic lattice; exact values sqrt(2/3) a and a / sqrt(2) of diamond and bcc.
    analytic = {6: (np.sqrt(2.0), 1.0), 4: (np.sqrt(2.0 / 3.0), 1.0 / np.sqrt(2.0)), 8: (np.sqrt(2.0 / 3.0), 1.0 / np.sqrt(2.0))}
    for f, (cav, thr) in analytic.items():
        check(f"cavity axis diameter / a, {LATTICE_NAME[f]} (analytic)", geometries[f]["cavity_axis_diameter"], cav, 1e-6)
        check(f"throat axis diameter / a, {LATTICE_NAME[f]} (analytic)", geometries[f]["throat_axis_diameter"], thr, 1e-6)

    # 3) Grid percolation (network_geometry.py) is a lower bound of the throat and converges to it.
    for f in FUNCTIONALITIES:
        previous = 0.0
        for n in VERIFY_GRID_POINTS:
            grid_thr = 2.0 * ng.throat_axis_radius(ng.axis_distance_field(lattice_segments(f), n))
            exact = geometries[f]["throat_axis_diameter"]
            rows.append({"check": f"grid percolation throat / a, {LATTICE_NAME[f]}, n = {n}", "true": exact,
                         "recovered": grid_thr, "abs_error": grid_thr - exact, "tolerance": np.nan,
                         "passed": bool(grid_thr <= exact + 2e-4 and grid_thr >= previous - 2e-4)})
            previous = grid_thr

    # 4) Round trip: G' of a synthetic lattice with known cell edge, recovered cell edge.
    temperature, a_true_nm = 298.15, 30.0
    for f in FUNCTIONALITIES:
        nu = STRANDS_PER_CELL[f] / (a_true_nm * 1e-9) ** 3
        g_prime = (1.0 - 2.0 / f) * nu * BOLTZMANN_CONSTANT * temperature
        check(f"cell edge from G' (nm), f = {f}", float(cell_edge_nm(g_prime, temperature, f)), a_true_nm, 1e-9)
        if f in (4, 6):
            check(f"cell edge vs. network_geometry.cell_edge_from_modulus (nm), f = {f}",
                  float(cell_edge_nm(g_prime, temperature, f)),
                  float(ng.cell_edge_from_modulus(g_prime, temperature, f)) * 1e9, 1e-9)

    table = pd.DataFrame(rows)
    if not table["passed"].all():
        raise RuntimeError("Verification failed:\n" + table[~table["passed"]].to_string(index=False))
    return table


# ── Tables ─────────────────────────────────────────────────────────────────────

RUN_COLUMNS = {
    "run": "run", "file": "file", "measurement_type": "measurement type", "run_date": "run date",
    "cast": "cast", "age_d": "gel age (d)", "specimen_id": "specimen", "gap_um": "gap (µm)",
    "temperature_K": "temperature (K)", "qc_status": "QC status", "qc_codes": "QC codes",
    "in_statistics": f"in statistics ({SELECTION})", "representative_specimen": "figure 1 specimen",
    "Gp_rep_Pa": "G' at 0.5 Hz (Pa)", "Gpp_0p5Hz_interp_Pa": "G'' at 0.5 Hz (Pa)",
    "tan_delta_0p5Hz_interp": "tan delta at 0.5 Hz", "slope_m": "slope dlog G'/dlog f in window",
    "slope_m_se": "slope SE", "window_f_min_Hz": "window f min (Hz)", "window_f_max_Hz": "window f max (Hz)",
    "Gp_ref_Pa": "G' low-strain plateau (Pa)", "Gpp_ref_Pa": "G'' low-strain plateau (Pa)",
    "tan_delta_ref": "tan delta low strain", "strain_5pct_pct": "strain at 5 % drop of G' (%)",
    "strain_10pct_pct": "strain at 10 % drop of G' (%)", "crossover_strain_pct": "crossover strain (%)",
    "crossover_stress_Pa": "crossover stress (Pa)", "crossover_status": "crossover status",
    "strain_max_pct": "max strain (%)", "n_points": "points", "n_valid_points": "points used in fits",
}
SUMMARY_VALUES = {
    "frequency_sweep": {"Gp_rep_Pa": "G' at 0.5 Hz (Pa)", "Gpp_0p5Hz_interp_Pa": "G'' at 0.5 Hz (Pa)",
                        "tan_delta_0p5Hz_interp": "tan delta at 0.5 Hz", "slope_m": "slope dlog G'/dlog f"},
    "amplitude_sweep": {"Gp_ref_Pa": "G' low-strain plateau (Pa)", "strain_5pct_pct": "strain at 5 % drop of G' (%)",
                        "strain_10pct_pct": "strain at 10 % drop of G' (%)",
                        "crossover_strain_pct": "crossover strain (%)", "crossover_stress_Pa": "crossover stress (Pa)"},
}


def run_table(cache: dict, rep_runs: List[str]) -> pd.DataFrame:
    fs = cache["tables"]["frequency_sweep_summary"].assign(measurement_type="frequency sweep")
    amp = cache["tables"]["amplitude_sweep_summary"].assign(measurement_type="amplitude sweep")
    runs = pd.concat([fs, amp], ignore_index=True)
    meta = cache["metadata"]
    runs["file"] = runs["run"].map(lambda r: meta[r]["file"])
    runs["run_date"] = runs["run"].map(lambda r: meta[r]["run_date"])
    runs["in_statistics"] = (runs["qc_status"] != "FAIL") & ~runs["excluded_by_qc_codes"].astype(bool) & \
        ~runs["run"].map(lambda r: meta[r]["exclude_from_stats"])
    runs["representative_specimen"] = runs["run"].isin(rep_runs)
    order = {r: i for i, r in enumerate(cache["results"])}
    runs = runs.sort_values("run", key=lambda c: c.map(order)).reset_index(drop=True)
    return runs[[c for c in RUN_COLUMNS if c in runs.columns]].rename(columns=RUN_COLUMNS)


def _age_groups(ages) -> List[tuple]:
    groups = [(f"{a:g} d", [a]) for a in sorted(ages)]
    pooled = [a for a in sorted(ages) if a < POOL_MAX_AGE_D]
    groups.append((f"< {POOL_MAX_AGE_D} d pooled ({', '.join(f'{a:g}' for a in pooled)} d)", pooled))
    return groups


def summary_table(specimens: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and 95 % CI over specimens per age and pooled below POOL_MAX_AGE_D."""
    rows = []
    for sweep, values in SUMMARY_VALUES.items():
        sub = specimens[specimens["sweep_type"] == sweep]
        for label, ages in _age_groups(specimens["age_d"].unique()):
            g = sub[sub["age_d"].isin(ages)]
            for col, name in values.items():
                s = ci_t(g[col].tolist())
                rows.append({"measurement type": sweep.replace("_", " "), "group": label, "quantity": name,
                             "N specimens": s["N"], "N runs": int(g.loc[g[col].notna(), "n_repeats"].sum()),
                             "mean": s["mean"], "SD": s["sd"], "t": s["t"], "95 % CI half-width": s["ci_half"],
                             "95 % CI low": s["ci_low"], "95 % CI high": s["ci_high"],
                             "specimens": "; ".join(g.loc[g[col].notna(), "specimen_id"])})
    return pd.DataFrame(rows)


def stability_table(specimens: pd.DataFrame) -> pd.DataFrame:
    fs = specimens[specimens["sweep_type"] == "frequency_sweep"]
    rows = []
    for label, ages in _age_groups(fs["age_d"].unique())[:-1]:
        g = fs[fs["age_d"].isin(ages)]
        s = ci_t(g["Gp_rep_Pa"].tolist())
        rows.append({"gel age (d)": ages[0], "N specimens": s["N"], "N runs": int(g["n_repeats"].sum()),
                     "casts": ", ".join(sorted(g["cast"].unique())), "G' mean (Pa)": s["mean"], "G' SD (Pa)": s["sd"],
                     "t": s["t"], "95 % CI half-width (Pa)": s["ci_half"], "95 % CI low (Pa)": s["ci_low"],
                     "95 % CI high (Pa)": s["ci_high"], "runs": "; ".join(g["runs"])})
    return pd.DataFrame(rows)


def pore_tables(pool: pd.DataFrame, geometries: Dict[int, dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Geometric factors, pooled pore sizes (mean, 95 % CI over specimens) and per-specimen values."""
    factors, per_spec = [], []
    for f in FUNCTIONALITIES:
        geo = geometries[f]
        edge = STRANDS_PER_CELL[f] ** (1.0 / 3.0)
        factors.append({
            "f": f, "lattice": geo["lattice"], "front factor 1 - 2/f": 1.0 - 2.0 / f,
            "strands per cell": STRANDS_PER_CELL[f], "a / xi_f": edge, "b / a": geo["strand_length"],
            "cavity axis diameter / a": geo["cavity_axis_diameter"], "throat axis diameter / a": geo["throat_axis_diameter"],
            "cavity axis diameter / xi_f": geo["cavity_axis_diameter"] * edge,
            "throat axis diameter / xi_f": geo["throat_axis_diameter"] * edge,
            "relation": "d_cavity = (cavity/xi_f) xi_f - d_f;  d_throat = (throat/xi_f) xi_f - d_f",
        })
        xi = calculate_network_mesh_size(pool["Gp_rep_Pa"], pool["temperature_K"], 1.0 - 2.0 / f)
        a = edge * np.asarray(xi)
        for d_f in STRAND_DIAMETERS_TABLE_NM:
            frame = pool[["specimen_id", "cast", "age_d", "runs", "Gp_rep_Pa", "temperature_K"]].copy()
            frame["xi_affine_nm"] = calculate_network_mesh_size(pool["Gp_rep_Pa"], pool["temperature_K"], 1.0)
            frame["f"], frame["lattice"], frame["d_f_nm"] = f, geo["lattice"], d_f
            frame["xi_f_nm"], frame["a_nm"], frame["b_nm"] = np.asarray(xi), a, a * geo["strand_length"]
            for kind in ("cavity", "throat"):
                d = a * geo[f"{kind}_axis_diameter"] - d_f
                frame[f"{kind}_nm"] = np.where(d > 0, d, np.nan)
            per_spec.append(frame)
    per_spec = pd.concat(per_spec, ignore_index=True)

    rows = []
    for (f, d_f), g in per_spec.groupby(["f", "d_f_nm"]):
        row = {"f": f, "lattice": LATTICE_NAME[f], "d_f (nm)": d_f, "N specimens": len(g),
               "ages (d)": ", ".join(f"{x:g}" for x in sorted(g["age_d"].unique()))}
        for col, name in (("Gp_rep_Pa", "G' at 0.5 Hz (Pa)"), ("xi_affine_nm", "xi affine (nm)"), ("xi_f_nm", "xi_f (nm)"),
                          ("a_nm", "cell edge a (nm)"), ("b_nm", "strand length b (nm)"),
                          ("cavity_nm", "cavity diameter (nm)"), ("throat_nm", "throat diameter (nm)")):
            s = ci_t(g[col].tolist())
            row.update({f"{name} mean": s["mean"], f"{name} SD": s["sd"], f"{name} 95 % CI half-width": s["ci_half"]})
            if col in ("cavity_nm", "throat_nm"):
                row[f"{name} N valid"] = s["N"]
        rows.append(row)
    return pd.DataFrame(factors), pd.DataFrame(rows), per_spec


def exclusion_log(cache: dict, runs: pd.DataFrame) -> pd.DataFrame:
    meta = cache["metadata"]
    rows = []
    for run, res in cache["results"].items():
        file = meta[run]["file"]
        data = res["data"]
        if len(data):
            n_valid = int((data["point_flag"] == "").sum())
            if n_valid < len(data):
                reasons = sorted({r for fl in data["point_flag"] if fl for r in fl.split(";")})
                rows.append({"file": file, "step": "points left out of fits (Rheo_Hyd20_Compute.py)",
                             "parameter": "min torque 5e-8 N m, max inertia ratio 1.0", "n_before": len(data),
                             "n_after": n_valid, "reason": ", ".join(reasons)})
    for _, r in runs.iterrows():
        if not r[f"in statistics ({SELECTION})"]:
            rows.append({"file": r["file"], "step": f"run left out of statistics ({SELECTION})",
                         "parameter": "QC status FAIL or codes axial_force, presheared, cracks, group_outlier",
                         "n_before": 1, "n_after": 0, "reason": f"{r['QC status']}: {r['QC codes']}"})
        elif r["measurement type"] == "frequency sweep" and r["gel age (d)"] >= POOL_MAX_AGE_D:
            rows.append({"file": r["file"], "step": "pore size (figure 3, table 3)",
                         "parameter": f"gel age < {POOL_MAX_AGE_D} d", "n_before": 1, "n_after": 0,
                         "reason": f"gel age {r['gel age (d)']:g} d not in the pooled group"})
    return pd.DataFrame(rows, columns=["file", "step", "parameter", "n_before", "n_after", "reason"])


# ── Figures ────────────────────────────────────────────────────────────────────

def _log_minor(ax: plt.Axes, axis: str = "both") -> None:
    for a in ([ax.xaxis, ax.yaxis] if axis == "both" else [getattr(ax, f"{axis}axis")]):
        a.set_minor_locator(mticker.LogLocator(subs=np.arange(2, 10), numticks=100))
        a.set_minor_formatter(mticker.NullFormatter())


def plot_viscoelasticity(cache: dict, as_run: str, path: Path) -> int:
    """Amplitude sweep in black: G' filled, G'' open, legend (user, 2026-10-14).

    Returns the number of points outside AS_YLIM_PA (not visible in the figure).
    """
    data = cache["results"][as_run]["data"]
    x, gp, gpp = data["strain_pct"], data["Gp_Pa"], data["Gpp_Pa"]
    if ((gp <= 0) | (gpp <= 0)).any():
        raise ValueError(f"{as_run}: non-positive modulus cannot be shown on log axes")
    valid = data["point_flag"] == ""
    fig, ax = plt.subplots(figsize=(WIDTH_IN["half"], WIDTH_IN["half"] / RATIO))
    ax.plot(x[valid], gp[valid], "o", mfc="black", mec="black", zorder=3)
    ax.plot(x[valid], gpp[valid], "s", mfc="white", mec="black", zorder=3)
    if (~valid).any():
        ax.plot(x[~valid], gp[~valid], "o", mfc=COLOR_EXCLUDED, mec=COLOR_EXCLUDED, zorder=2)
        ax.plot(x[~valid], gpp[~valid], "s", mfc="white", mec=COLOR_EXCLUDED, zorder=2)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="none", mfc="black", mec="black", label="Storage modulus $G'$"),
        plt.Line2D([], [], marker="s", ls="none", mfc="white", mec="black", label="Loss modulus $G''$"),
    ], loc="lower left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _log_minor(ax)
    ax.set_ylim(*AS_YLIM_PA)
    ax.set_xlabel("Strain amplitude γ (%)")
    ax.set_ylabel("Modulus G', G'' (Pa)")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    lo, hi = AS_YLIM_PA
    return int(((gp < lo) | (gp > hi)).sum() + ((gpp < lo) | (gpp > hi)).sum())


def plot_stability(fs_runs: pd.DataFrame, path: Path) -> None:
    """G' at 0.5 Hz of every qc_filtered run at its real age; no means, no legend."""
    base, dark = COLOR_SINGLE
    fig, ax = plt.subplots(figsize=(WIDTH_IN["half"], WIDTH_IN["half"] / RATIO))
    ax.plot(fs_runs["age_d"], fs_runs["Gp_rep_Pa"], "o", ms=STABILITY_MARKERSIZE, mfc=base, mec=dark,
            alpha=POINT_ALPHA, zorder=3)
    ax.set_xlim(0, fs_runs["age_d"].max() + 2)
    ax.set_ylim(0, 1.2 * fs_runs["Gp_rep_Pa"].max())
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_xlabel("t (days) after Gel Cast")
    ax.set_ylabel("G' (Pa) at 0.5 Hz")
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _clip_to_sphere(p0: np.ndarray, p1: np.ndarray, centre: np.ndarray, radius: float):
    """Part of the segment p0-p1 inside the sphere (centre, radius), or None."""
    d, rel = p1 - p0, p0 - centre
    qa, qb, qc = d @ d, 2.0 * rel @ d, rel @ rel - radius ** 2
    disc = qb * qb - 4.0 * qa * qc
    if disc <= 0:
        return None
    t0 = max((-qb - np.sqrt(disc)) / (2.0 * qa), 0.0)
    t1 = min((-qb + np.sqrt(disc)) / (2.0 * qa), 1.0)
    return None if t1 - t0 < 1e-6 else (p0 + t0 * d, p0 + t1 * d)


def render_lattice(geo: dict, a: float, colors: tuple, path: Path) -> float:
    """Shaded rendering (pyvista/VTK) of the periodic lattice around one cavity; returns pixels per nm.

    Strands are drawn as thin tubes of fixed visual radius (fibril diameter not drawn); the cavity
    sphere and the throat ring have the axis diameters kappa_cav a and kappa_thr a (d_f = 0).
    Strand opacity falls off with the distance from the cavity centre (neighbouring cells only
    hinted), and strands in front of the cavity are drawn faint. Parallel projection with the same
    scale for every lattice, so all panels are to one scale.
    """
    import pyvista as pv

    base, dark = colors
    view = np.asarray(RENDER_VIEW_DIRECTION, float)
    view /= np.linalg.norm(view)
    centre = geo["cavity_centre"] * a
    n = int(np.ceil(RENDER_SHOW_RADIUS_NM / a)) + 2
    tubes: Dict[float, list] = {}
    nodes: Dict[tuple, float] = {}
    for shift in product(range(-n, n + 1), repeat=3):
        s = np.array(shift, float)
        for p0, p1 in geo["segments"]:
            seg = _clip_to_sphere((p0 + s) * a, (p1 + s) * a, centre, RENDER_SHOW_RADIUS_NM)
            if seg is None:
                continue
            mid = 0.5 * (seg[0] + seg[1])
            alpha = np.clip((RENDER_SHOW_RADIUS_NM - np.linalg.norm(mid - centre))
                            / (RENDER_SHOW_RADIUS_NM - RENDER_FADE_START_NM), 0.0, 1.0)
            if (mid - centre) @ view > RENDER_FRONT_DEPTH_NM:
                alpha *= RENDER_FRONT_ALPHA
            level = round(alpha * 4) / 4
            if level <= 0:
                continue
            tubes.setdefault(level, []).append(pv.Tube(pointa=seg[0], pointb=seg[1], radius=RENDER_TUBE_RADIUS_NM,
                                                       n_sides=24))
            for p in ((p0 + s) * a, (p1 + s) * a):
                if np.linalg.norm(p - centre) < RENDER_SHOW_RADIUS_NM:
                    key = tuple(np.round(p, 4))
                    nodes[key] = max(nodes.get(key, 0.0), level)

    pl = pv.Plotter(off_screen=True, window_size=(RENDER_PX, RENDER_PX), lighting="none")
    pl.set_background("white")
    for level, meshes in tubes.items():
        pl.add_mesh(pv.merge(meshes), color=dark, opacity=level, smooth_shading=True, specular=0.3, specular_power=20)
    by_level: Dict[float, list] = {}
    for key, level in nodes.items():
        by_level.setdefault(level, []).append(pv.Sphere(radius=RENDER_NODE_RADIUS_NM, center=key,
                                                        theta_resolution=24, phi_resolution=24))
    for level, meshes in by_level.items():
        pl.add_mesh(pv.merge(meshes), color=dark, opacity=level, smooth_shading=True, specular=0.3)
    pl.add_mesh(pv.Sphere(radius=geo["cavity_axis_diameter"] * a / 2, center=centre, theta_resolution=160,
                          phi_resolution=160), color=base, opacity=RENDER_CAVITY_OPACITY, ambient=0.25,
                smooth_shading=True, specular=0.5, specular_power=30)
    normal = geo["throat_normal"]
    e1 = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    phi = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    ring = geo["throat_centre"] * a + (geo["throat_axis_diameter"] * a / 2) * (
        np.outer(np.cos(phi), e1) + np.outer(np.sin(phi), e2))
    pl.add_mesh(pv.lines_from_points(ring, close=True).tube(radius=0.45, n_sides=20), color="black",
                smooth_shading=True)
    pl.add_light(pv.Light(position=tuple(centre + 200 * np.array([0.6, 0.2, 1.0])), focal_point=tuple(centre),
                          intensity=0.9, light_type="scene light"))
    pl.add_light(pv.Light(position=tuple(centre + 200 * view), focal_point=tuple(centre), intensity=0.35,
                          light_type="scene light"))
    pl.enable_depth_peeling(number_of_peels=12)
    pl.camera.position = tuple(centre + 400 * view)
    pl.camera.focal_point = tuple(centre)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = RENDER_HALF_HEIGHT_NM
    pl.screenshot(str(path))
    pl.close()
    return RENDER_PX / (2.0 * RENDER_HALF_HEIGHT_NM)


def plot_cells(geometries: Dict[int, dict], per_spec: pd.DataFrame, path: Path) -> None:
    """Rendered lattices f = 4, 6, 8 side by side, same scale, 10 nm scale bar in every panel."""
    sub = per_spec[per_spec["d_f_nm"] == STRAND_DIAMETER_DRAWING_NM]
    a_mean = {f: sub.loc[sub["f"] == f, "a_nm"].mean() for f in FUNCTIONALITIES}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN["full"], 2.45))
    for ax, f, letter in zip(axes, FUNCTIONALITIES, "ABC"):
        png = path.with_name(f"{path.stem}_f{f}.png")
        px_per_nm = render_lattice(geometries[f], a_mean[f], COLOR_F[f], png)
        ax.imshow(plt.imread(png), interpolation="lanczos")
        x0, y0 = 0.04 * RENDER_PX, 0.985 * RENDER_PX
        ax.plot([x0, x0 + RENDER_SCALE_BAR_NM * px_per_nm], [y0, y0], color="black", lw=1.5, solid_capstyle="butt")
        ax.text(x0 + 0.5 * RENDER_SCALE_BAR_NM * px_per_nm, y0 - 0.015 * RENDER_PX, f"{RENDER_SCALE_BAR_NM:g} nm",
                ha="center", va="bottom", fontsize=8)
        ax.set_axis_off()
        ax.set_title(f"f = {f}, {geometries[f]['lattice']}")
        ax.text(0.5, -0.02, f"a = {a_mean[f]:.1f} nm", transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.savefig(path, format="pdf", dpi=600)
    plt.close(fig)


def plot_cavity_vs_f(combined: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_IN["half"], WIDTH_IN["half"] / RATIO))
    ax.errorbar(combined["f"], combined["cavity diameter (nm)"], yerr=combined["cavity combined 95 % uncertainty (nm)"],
                fmt="o", mfc="black", mec="black", ecolor="black", zorder=3, **ERR_KW)
    ax.set_xlim(3, 9)
    ax.set_xticks(list(FUNCTIONALITIES))
    ax.set_ylim(0, 1.25 * (combined["cavity diameter (nm)"] + combined["cavity combined 95 % uncertainty (nm)"]).max())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.set_xlabel("Junction functionality f")
    ax.set_ylabel("Cavity diameter (nm)")
    fig.savefig(path, format="pdf")
    plt.close(fig)


def combined_pore_table(per_spec: pd.DataFrame) -> pd.DataFrame:
    """Cavity and throat at the midpoint d_f with the 95 % CI and the d_f contribution in quadrature."""
    width = STRAND_DIAMETER_RANGE_NM[1] - STRAND_DIAMETER_RANGE_NM[0]
    u_df = STRAND_DIAMETER_COVERAGE * width / np.sqrt(12.0)
    rows = []
    for f, g in per_spec[per_spec["d_f_nm"] == STRAND_DIAMETER_DRAWING_NM].groupby("f"):
        row = {"f": f, "lattice": LATTICE_NAME[f], "d_f (nm)": STRAND_DIAMETER_DRAWING_NM, "N specimens": len(g),
               "d_f contribution (nm)": u_df,
               "d_f contribution definition": f"{STRAND_DIAMETER_COVERAGE} x ({width:g} nm / sqrt(12)), uniform "
                                              f"{STRAND_DIAMETER_RANGE_NM[0]:g}-{STRAND_DIAMETER_RANGE_NM[1]:g} nm"}
        for kind in ("cavity", "throat"):
            s = ci_t(g[f"{kind}_nm"].tolist())
            row.update({f"{kind} diameter (nm)": s["mean"], f"{kind} 95 % CI half-width (nm)": s["ci_half"],
                        f"{kind} combined 95 % uncertainty (nm)": float(np.hypot(s["ci_half"], u_df))})
        rows.append(row)
    return pd.DataFrame(rows)


def mesh_size_tables(pool: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Affine mesh size per specimen and the pooled mean with its 95 % CI."""
    spec = pool[["specimen_id", "cast", "age_d", "runs", "Gp_rep_Pa", "temperature_K"]].copy()
    spec["xi_affine_nm"] = np.asarray(calculate_network_mesh_size(spec["Gp_rep_Pa"], spec["temperature_K"], 1.0))
    s = ci_t(spec["xi_affine_nm"].tolist())
    summary = pd.DataFrame([{"level": f"pooled < {POOL_MAX_AGE_D} d", "N specimens": s["N"],
                             "ages (d)": ", ".join(f"{a:g}" for a in sorted(spec["age_d"].unique())),
                             "xi_affine_nm": s["mean"], "SD (nm)": s["sd"], "t": s["t"],
                             "95 % CI half-width (nm)": s["ci_half"], "95 % CI low (nm)": s["ci_low"],
                             "95 % CI high (nm)": s["ci_high"]}])
    return pd.concat([spec.assign(level="specimen"), summary], ignore_index=True), s


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"{CACHE_PATH} not found; run Rheo_Hyd20_Compute.py first.")

    geometries = {f: lattice_pore_geometry(f) for f in FUNCTIONALITIES}
    verification = verify_on_synthetic(geometries)
    print("Verification on known answers (all passed):")
    print(verification[["check", "true", "recovered"]].to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    with open(CACHE_PATH, "rb") as fh:
        cache = pickle.load(fh)
    if cache["config"]["stats_exclude_qc_codes"] != ("axial_force", "presheared", "cracks", "group_outlier"):
        raise ValueError(f"Cache uses QC codes {cache['config']['stats_exclude_qc_codes']}, not the ones decided in E025.")
    specimens = cache["tables"]["stability_specimen_level"]
    specimens = specimens[specimens["selection"] == SELECTION]

    # Both draws come from one generator: first the 4 d specimen (no longer plotted, the draw is kept so
    # that the amplitude sweep drawn second stays the one decided in E027), then the 9 d amplitude sweep.
    rng = np.random.default_rng(RANDOM_SEED)
    fs_spec = specimens[specimens["sweep_type"] == "frequency_sweep"].dropna(subset=["Gp_rep_Pa"])
    candidates = sorted(fs_spec.loc[fs_spec["age_d"] == REPRESENTATIVE_AGE_D, "specimen_id"])
    if not candidates:
        raise ValueError(f"No {SELECTION} specimen of age {REPRESENTATIVE_AGE_D} d in {CACHE_PATH}")
    rep_specimen = candidates[rng.integers(len(candidates))]
    rep_runs = fs_spec.loc[(fs_spec["specimen_id"] == rep_specimen) &
                           (fs_spec["age_d"] == REPRESENTATIVE_AGE_D), "runs"].iloc[0].split(", ")

    runs = run_table(cache, rep_runs)
    in_stats = runs.set_index("run")[f"in statistics ({SELECTION})"]
    order = list(cache["results"])
    as_candidates = sorted((r for r, res in cache["results"].items()
                            if res["as"] is not None and "error" not in res["as"] and in_stats[r]
                            and cache["metadata"][r]["age_d"] == AMPLITUDE_SWEEP_AGE_D), key=order.index)
    if not as_candidates:
        raise ValueError(f"No {SELECTION} amplitude sweep of age {AMPLITUDE_SWEEP_AGE_D} d in {CACHE_PATH}")
    as_run = as_candidates[rng.integers(len(as_candidates))]
    runs["figure 1"] = runs["run"].map(lambda r: "amplitude sweep" if r == as_run else "")
    runs = runs.drop(columns=["figure 1 specimen"])

    summary = summary_table(specimens)
    stab = stability_table(specimens)
    fs_all = cache["tables"]["frequency_sweep_summary"]
    fs_runs = fs_all[fs_all["run"].map(in_stats) & fs_all["Gp_rep_Pa"].notna()]
    pool = fs_spec[fs_spec["age_d"] < POOL_MAX_AGE_D]
    if pool["temperature_K"].isna().any():
        raise ValueError(f"Missing temperature for {list(pool.loc[pool['temperature_K'].isna(), 'specimen_id'])}")
    factors, pores, per_spec = pore_tables(pool, geometries)
    combined = combined_pore_table(per_spec)
    mesh, mesh_pooled = mesh_size_tables(pool)
    exclusions = exclusion_log(cache, runs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with plt.style.context(str(STYLE_PATH)):
        n_outside = plot_viscoelasticity(cache, as_run, OUTPUT_DIR / "fig1_viscoelasticity.pdf")
        plot_stability(fs_runs, OUTPUT_DIR / "fig2_stability_gp_vs_age.pdf")
        plot_cells(geometries, per_spec, OUTPUT_DIR / "fig3a_lattice_cells.pdf")
        plot_cavity_vs_f(combined, OUTPUT_DIR / "fig3b_cavity_vs_f.pdf")

    tables = {
        "table1_viscoelastic_all_runs": runs,
        "table1b_viscoelastic_summary": summary,
        "table2_stability_gp_by_age": stab,
        "table3a_pore_geometry_factors": factors,
        "table3b_pore_size": pores,
        "table3c_pore_size_per_specimen": per_spec,
        "table3d_pore_size_combined": combined,
        "table4_mesh_size": mesh,
        "verification_synthetic": verification,
        "rheo_hyd20_key_figures_exclusions": exclusions,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    with pd.ExcelWriter(OUTPUT_DIR / "rheo_hyd20_key_figures.xlsx") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name.replace("rheo_hyd20_key_figures_", "")[:31], index=False)

    print(f"\nSeed {RANDOM_SEED}: frequency sweep {rep_specimen} {rep_runs} (from {candidates}); "
          f"amplitude sweep {as_run} (from {as_candidates})")
    print(f"Figure 1: {n_outside} points of {as_run} outside the y range {AS_YLIM_PA} Pa (not visible)")
    print(f"\nFigure 2: {len(fs_runs)} qc_filtered runs: "
          + ", ".join(f"{r} ({a:g} d, {g:.0f} Pa)" for r, a, g in fs_runs[["run", "age_d", "Gp_rep_Pa"]].to_numpy()))
    print("\nPore sizes at d_f = 11 nm, combined 95 % uncertainty:")
    print(combined.drop(columns=["d_f contribution definition"]).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nMesh size xi (affine, for the caption), pooled < {POOL_MAX_AGE_D} d: {mesh_pooled['mean']:.2f} "
          f"+- {mesh_pooled['ci_half']:.2f} nm (95 % CI, N = {mesh_pooled['N']} specimens)")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
