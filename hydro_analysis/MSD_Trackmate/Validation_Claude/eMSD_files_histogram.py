"""
Pro-File-Verteilungen (ein Datenpunkt = ein Film/Movie), je Partikelgroesse
SEPARAT ausgewertet (nie gepoolt), Wasser vs. Hydrogel als KDE-Histogramme
innerhalb jedes Groessen-Panels ueberlagert: mittlere Tracklaenge (Frames),
mittlere Schrittweite (µm), D (µm²/s) und Anomalie-Exponent n. Identische
Abbildungsstruktur wie iMSD_histograms.py (Grid: Spalte je Groesse, Zeile
je Metrik, gleiche Farb-/KDE-Konvention), aber pro Film statt pro Track
aggregiert.

Reiner Konsument, kein Neufitten: liest cache/msd_d0_results.pkl (Wasser)
und cache/msd_20mg_files.pkl (Hydrogel) unveraendert. D und n kommen direkt
aus fit_results_MSD["D_um2_per_s"]/["exponent"] (die bereits im
Compute-Stage-Skript gefittete Ensemble-eMSD-Powerlaw-Fit dieses Files).
Tracklaenge und Schrittweite werden je File aus tracks_df/mpp gemittelt
(core.analysis.calculate_step_sizes fuer die Schrittweite, wie in
iMSD_histograms.py, hier aber ueber alle Tracks eines Files gemittelt statt
pro Track einzeln ausgegeben).

Ausgabe (Auswertungsbilder\\eMSD_Histograms\\): eine eigene Abbildungsdatei
je Metrik statt einer gemeinsamen Uebersichtsgrafik -- emsd_tracklength_by_
size.png, emsd_stepsize_by_size.png, emsd_D_by_size.png, emsd_n_by_size.png.

Vor Ausfuehrung muessen MSD_FromTrackmate_D0.py und MSD_FromTrackmate_20mg.py
bereits gelaufen sein.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from hydro_analysis.core.analysis import calculate_step_sizes
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_MSD_D0 = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_MSD_20MG = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "eMSD_Histograms"

N_BINS = 30
STYLE_WATER = dict(face="#009900", edge="#006600", label="Wasser")
STYLE_HYDROGEL = dict(face="#0000da", edge="#000099", label="Hydrogel")


def _load_pickle(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def build_file_table() -> pd.DataFrame:
    """Pro-File-Tabelle: mittlere Tracklaenge/Schrittweite je File + D/n aus fit_results_MSD."""
    rows = []
    sources = (
        ("water", CACHE_MSD_D0, "MSD_FromTrackmate_D0.py"),
        ("hydrogel", CACHE_MSD_20MG, "MSD_FromTrackmate_20mg.py"),
    )
    for condition, cache_path, compute_script in sources:
        results = _load_pickle(cache_path, compute_script)
        for r in results.values():
            tracks_df = r.get("tracks_df")
            mpp = r.get("mpp")
            fit = r.get("fit_results_MSD")
            if tracks_df is None or tracks_df.empty or mpp is None or fit is None:
                continue
            lengths = tracks_df.groupby("particle").size()
            steps = calculate_step_sizes(tracks_df, step_interval=1, sliding=True)
            step_um_mean = (
                float(np.hypot(steps["dx"].to_numpy(), steps["dy"].to_numpy()).mean() * mpp)
                if not steps.empty else np.nan
            )
            rows.append({
                "xml_path": r.get("xml_path"), "base_name": r.get("base_name"),
                "particle_size_nm": r.get("particle_size_nm"), "condition": condition,
                "track_length_frames_mean": float(lengths.mean()) if len(lengths) else np.nan,
                "step_um_mean": step_um_mean,
                "D_um2_per_s": fit.get("D_um2_per_s"), "exponent": fit.get("exponent"),
            })
    return pd.DataFrame(rows)


def _hist_kde_panel(ax, water_vals, hydrogel_vals, xlabel: str, log_x: bool = False,
                     n_bins: int = N_BINS) -> None:
    water_vals = np.asarray(water_vals, dtype=float)
    water_vals = water_vals[np.isfinite(water_vals)]
    hydrogel_vals = np.asarray(hydrogel_vals, dtype=float)
    hydrogel_vals = hydrogel_vals[np.isfinite(hydrogel_vals)]
    if log_x:
        water_vals = water_vals[water_vals > 0]
        hydrogel_vals = hydrogel_vals[hydrogel_vals > 0]
    pooled = np.concatenate([water_vals, hydrogel_vals])
    if pooled.size == 0:
        ax.text(0.5, 0.5, "keine Daten", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        return
    if log_x:
        bins = np.logspace(np.log10(pooled.min()), np.log10(pooled.max()), n_bins + 1)
    else:
        bins = np.linspace(pooled.min(), pooled.max(), n_bins + 1)

    for vals, style in ((water_vals, STYLE_WATER), (hydrogel_vals, STYLE_HYDROGEL)):
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, density=True, alpha=0.5, color=style["face"],
                edgecolor=style["edge"], linewidth=0.4, label=style["label"])
        if vals.size >= 2 and np.ptp(vals) > 0:
            if log_x:
                x_grid = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 300)
                density = gaussian_kde(np.log10(vals))(np.log10(x_grid)) / (x_grid * np.log(10))
            else:
                x_grid = np.linspace(bins[0], bins[-1], 300)
                density = gaussian_kde(vals)(x_grid)
            ax.plot(x_grid, density, color=style["edge"], linewidth=1.6)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Dichte")
    ax.minorticks_on()


_METRICS = [
    ("track_length_frames_mean", "Mittlere Tracklänge je File (Frames)", False),
    ("step_um_mean", "Mittlere Schrittweite je File (µm)", False),
    ("D_um2_per_s", r"$D$ (µm²/s)", True),
    ("exponent", r"Anomalie-Exponent $n$", False),
]


_FILENAMES = {
    "track_length_frames_mean": "emsd_tracklength_by_size.png",
    "step_um_mean": "emsd_stepsize_by_size.png",
    "D_um2_per_s": "emsd_D_by_size.png",
    "exponent": "emsd_n_by_size.png",
}


def plot_emsd_metric_grid(df: pd.DataFrame, metric: str, xlabel: str, log_x: bool) -> plt.Figure:
    """Ein Grid pro Metrik (eigene Abbildungsdatei): ein Panel je Partikelgröße
    (dynamisches 3-Spalten-Raster), Wasser vs. Hydrogel je Panel überlagert."""
    sizes = sorted(df["particle_size_nm"].dropna().unique())
    n = len(sizes)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2.2),
                              constrained_layout=True, squeeze=False)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    for ax, size_nm in zip(ax_flat, sizes):
        water = df.loc[(df["condition"] == "water") & (df["particle_size_nm"] == size_nm), metric]
        hydrogel = df.loc[(df["condition"] == "hydrogel") & (df["particle_size_nm"] == size_nm), metric]
        _hist_kde_panel(ax, water, hydrogel, xlabel, log_x=log_x)
        ax.set_title(f"{size_nm:.0f} nm", fontsize=9)
    ax_flat[0].legend(loc="upper right", fontsize=7, frameon=False)
    return fig


def main() -> None:
    df = build_file_table()
    n_water = int((df["condition"] == "water").sum())
    n_hydrogel = int((df["condition"] == "hydrogel").sum())
    print(f"Pro-File-Tabelle: {len(df)} Files ({n_water} Wasser, {n_hydrogel} Hydrogel)")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        for col_name, ylabel, log_x in _METRICS:
            fig = plot_emsd_metric_grid(df, col_name, ylabel, log_x)
            fig_path = SAVE_PATH / _FILENAMES[col_name]
            safe_savefig(fig, fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot gespeichert: {fig_path}")


if __name__ == "__main__":
    main()
