from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from hydro_analysis.core.io import get_dls_reference_maps

# ── Configuration ──────────────────────────────────────────────────
CACHE_20MG = Path(__file__).parent / "cache" / "msd_20mg_results.pkl"
CACHE_13MG = Path(__file__).parent / "cache" / "msd_13mg_results.pkl"
CACHE_D0   = Path(__file__).parent / "cache" / "msd_d0_results.pkl"

# Smallest two nominal particle sizes to display
SIZES_OF_INTEREST = [20.0, 50.0]


def _load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        results = pickle.load(f)
    print(f"Cache geladen: {path.name} ({len(results)} Dateien)")
    return results


def _extract_summary(results: dict, size_map: dict[float, float],
                     sizes: list[float]) -> pd.DataFrame:
    """Extract D, D_error, exponent, exponent_error per file, filtered to sizes."""
    rows = []
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        if size_nm not in sizes:
            continue
        fit = r.get("fit_results_MSD")
        if fit is None:
            continue
        tracks_df = r.get("tracks_df")
        n_points = len(tracks_df) if tracks_df is not None else 0
        rows.append({
            "size_nom":   float(size_nm),
            "size_dls":   float(size_map.get(float(size_nm), float(size_nm))),
            "D":          fit.get("D_um2_per_s"),
            "D_err":      fit.get("D_error"),
            "n":          fit.get("exponent"),
            "n_err":      fit.get("exponent_error"),
            "n_points":   n_points,
            "file":       r.get("base_name", ""),
        })
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per nominal particle size: mean ± std of D and n."""
    rows = []
    for size, grp in df.groupby("size_nom"):
        D_vals = grp["D"].dropna()
        n_vals = grp["n"].dropna()
        rows.append({
            "Größe nominal (nm)": int(size),
            "Größe DLS (nm)":     f"{grp['size_dls'].iloc[0]:.1f}",
            "D (µm²/s)":          f"{D_vals.mean():.4f}",
            "± D":                f"{D_vals.std():.4f}" if len(D_vals) > 1
                                  else f"{grp['D_err'].dropna().mean():.4f}",
            "n":                  f"{n_vals.mean():.3f}",
            "± n":                f"{n_vals.std():.3f}"  if len(n_vals) > 1
                                  else f"{grp['n_err'].dropna().mean():.3f}",
            "N Datenpunkte":      int(grp["n_points"].sum()),
            "N Dateien":          len(grp),
        })
    return pd.DataFrame(rows).set_index("Größe nominal (nm)")


def main() -> None:
    dls = get_dls_reference_maps()
    size_map = dls["size_override_nm"]

    datasets = {
        "Wasser (D0)":   CACHE_D0,
        "Hydrogel 20mg": CACHE_20MG,
        "Hydrogel 13mg": CACHE_13MG,
    }

    for label, cache_path in datasets.items():
        if not cache_path.exists():
            print(f"\n[FEHLT] {label}: {cache_path.name} nicht gefunden")
            continue

        results = _load_cache(cache_path)
        df = _extract_summary(results, size_map, SIZES_OF_INTEREST)

        if df.empty:
            print(f"\n{label}: Keine Daten für Partikelgrößen {SIZES_OF_INTEREST}")
            continue

        agg = _aggregate(df)
        print(f"\n── {label} ──")
        print(agg.to_string())


if __name__ == "__main__":
    main()
