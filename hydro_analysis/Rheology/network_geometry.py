"""
Pore geometry of idealised periodic polymer networks built from the strand and
junction densities of a phantom network. Numerical geometry only; here wont be
stored any data loading, rheological analysis or visualization methods.

Phantom network with junction functionality f (Rubinstein & Colby, Polymer
Physics, 2003, ch. 7): G' = (nu - mu) k_B T = (1 - 2/f) nu k_B T, with the
strand number density nu and the junction number density mu = 2 nu / f.
The junctions are placed on the regular lattice matching f:
  f = 4  diamond lattice (tetrahedral junctions): 8 junctions and 16 strands
         per cubic cell, a = (8 / mu)^(1/3), strand length b = sqrt(3) a / 4;
  f = 6  simple cubic lattice: 1 junction and 3 strands per cell, a = mu^(-1/3), b = a.
Strands are straight cylinders of diameter d_f between junctions.

Pore measures (all in units of the cell edge a, computed once on a periodic grid
from the exact distance to the strand axes, then scaled to nm):
  cavity diameter  largest sphere that fits into the pore space
                   (2 * max distance to the nearest strand axis - d_f);
  throat diameter  largest sphere that can traverse the network, i.e. the
                   largest R for which {distance to axes >= R + d_f/2}
                   percolates (constriction between cavities).
In a regular lattice all cavities are identical, so the pore-size distribution
is monodisperse and its mean equals the cavity diameter; a real gel is
heterogeneous and has a broad distribution around these values.
Cavity and throat diameters are exactly linear in d_f (axis value minus d_f).
Grid resolution: GRID_POINTS per cell edge; validate_cubic_lattice() checks the
algorithm against the analytic simple-cubic results (cavity sqrt(2) a, throat a).
"""
from __future__ import annotations

from itertools import product
from typing import Dict

import numpy as np
from scipy import ndimage

GRID_POINTS = 48           # grid points per cell edge (multiple of 8: diamond sites lie on the grid)

_DIAMOND_FCC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
_DIAMOND_BONDS = 0.25 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


def lattice_segments(functionality: int) -> np.ndarray:
    """Strand axes of one cubic cell (edge 1) as array (n, 2, 3) of start/end points."""
    if functionality == 4:
        return np.array([[p, p + d] for p in _DIAMOND_FCC for d in _DIAMOND_BONDS])
    if functionality == 6:
        eye = np.eye(3)
        return np.array([[np.zeros(3), eye[i]] for i in range(3)])
    raise ValueError(f"No lattice implemented for junction functionality f = {functionality}.")


def cell_edge_from_modulus(storage_modulus_Pa, temperature_K, functionality: int):
    """Cell edge a (m) of the lattice whose phantom-network modulus equals G'."""
    k_B = 1.380649e-23
    nu = np.asarray(storage_modulus_Pa, float) / ((1.0 - 2.0 / functionality) * k_B * np.asarray(temperature_K, float))
    mu = 2.0 * nu / functionality
    junctions_per_cell = {4: 8, 6: 1}[functionality]
    return (junctions_per_cell / mu) ** (1.0 / 3.0)


def strand_length_per_edge(functionality: int) -> float:
    """Junction-junction distance b in units of the cell edge a."""
    return {4: np.sqrt(3.0) / 4.0, 6: 1.0}[functionality]


def axis_distance_field(segments: np.ndarray, n: int = GRID_POINTS) -> np.ndarray:
    """Exact distance of every grid point (i/n, j/n, k/n) to the nearest strand axis (periodic)."""
    g = np.arange(n) / n
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    shifts = np.array(list(product((-1, 0, 1), repeat=3)), float)
    best = np.full(len(pts), np.inf)
    for p0, p1 in segments:
        d = p1 - p0
        dd = d @ d
        for s in shifts:
            rel = pts - (p0 + s)
            t = np.clip(rel @ d / dd, 0.0, 1.0)
            dist = np.linalg.norm(rel - t[:, None] * d, axis=1)
            np.minimum(best, dist, out=best)
    return best.reshape(n, n, n)


def _percolates(mask: np.ndarray) -> bool:
    """True if a connected region of the periodic mask spans three tiled cells along x."""
    tiled = np.tile(mask, (3, 3, 3))
    labels, count = ndimage.label(tiled)
    if count == 0:
        return False
    first, last = np.unique(labels[0]), np.unique(labels[-1])
    return bool(np.intersect1d(first[first > 0], last[last > 0]).size)


def throat_axis_radius(field: np.ndarray, tol: float = 1e-4) -> float:
    """Largest R (cell units) for which {distance to axes >= R} percolates (bisection)."""
    lo, hi = 0.0, float(field.max())
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _percolates(field >= mid) else (lo, mid)
    return lo


def lattice_geometry(functionality: int, n: int = GRID_POINTS) -> Dict[str, object]:
    """Dimensionless cavity and throat axis diameters (units of a) and the distance field."""
    field = axis_distance_field(lattice_segments(functionality), n)
    # All grid points at the maximum are equivalent cavity centres; keep the one nearest the cell centre.
    centres = np.argwhere(field >= field.max() - 1e-9) / n
    return {
        "functionality": functionality,
        "field": field,
        "cavity_centre": centres[np.argmin(np.linalg.norm(centres - 0.5, axis=1))],
        "cavity_axis_diameter": 2.0 * float(field.max()),
        "throat_axis_diameter": 2.0 * throat_axis_radius(field),
        "strand_length": strand_length_per_edge(functionality),
    }


def validate_cubic_lattice(n: int = GRID_POINTS) -> Dict[str, float]:
    """Relative errors of the numerical cavity and throat diameters for the simple cubic lattice."""
    geo = lattice_geometry(6, n)
    return {"cavity_rel_error": geo["cavity_axis_diameter"] / np.sqrt(2.0) - 1.0,
            "throat_rel_error": geo["throat_axis_diameter"] / 1.0 - 1.0}
