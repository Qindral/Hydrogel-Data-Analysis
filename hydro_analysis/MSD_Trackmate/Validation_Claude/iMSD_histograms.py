"""
Pro-Track-Verteilungen (ein Datenpunkt = ein Track), je Partikelgroesse
SEPARAT ausgewertet (nie gepoolt) -- Wasser vs. Hydrogel als KDE-Histogramme
innerhalb jedes Groessen-Panels ueberlagert: Tracklaenge (Frames),
Schrittweite (µm, Mittel je Track), D (µm²/s) und Anomalie-Exponent n.

Reiner Konsument, keine Rohdaten/kein Neufitten: Tracklaenge, D und n
stammen unveraendert aus cache/track_length_bias_d0.pkl bzw.
cache/track_length_bias_20mg.pkl (produziert von
TaskB1_TrackLengthBias_Compute.py::per_track_fits, welches bereits pro
Track fit_powerlaw_with_errors auf den iMSD-Spalten ausfuehrt). Die
Schrittweite wird zusaetzlich pro Track aus tracks_df/mpp in
cache/msd_d0_results.pkl bzw. cache/msd_20mg_files.pkl ueber
core.analysis.calculate_step_sizes() berechnet (Mittel der Euklidischen
Schrittlaenge je Track, in µm) und per (xml_path, particle_id) an die
Track-Tabelle gemerged.

Zentraler Output fuer den Handoff an Loc_Error_Analyse_immob_particle.py:
cache/imsd_median_stepsize_by_size.pkl -- {particle_size_nm: median
Schrittweite (µm)}, gepoolt ueber Wasser+Hydrogel je Partikelgroesse (6
Standardgroessen 20/50/100/200/500/1000 nm, je nachdem welche tatsaechlich
Daten haben). Loc_Error_Analyse_immob_particle.py laedt diese Datei anstatt
selbst eine Referenz-Schrittweite zu scannen.

Ausgabe (Auswertungsbilder\\iMSD_Histograms\\): eine eigene Abbildungsdatei
je Metrik statt einer gemeinsamen Uebersichtsgrafik -- imsd_tracklength_by_
size.png, imsd_stepsize_by_size.png, imsd_D_by_size.png, imsd_n_by_size.png.
Jede Datei: dynamisches Raster mit einem Panel je Partikelgroesse,
Wasser/Hydrogel je Panel ueberlagert, KDE in log10(D)-Raum fuer die
D-Abbildung wie in Dissertation_Figures/Deff_Histogram.py.

Vor Ausfuehrung muessen TaskB1_TrackLengthBias_Compute.py sowie
MSD_FromTrackmate_D0.py/_20mg.py bereits gelaufen sein.
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
CACHE_TRACKLENGTH_D0 = Path(__file__).parent.parent / "cache" / "track_length_bias_d0.pkl"
CACHE_TRACKLENGTH_20MG = Path(__file__).parent.parent / "cache" / "track_length_bias_20mg.pkl"
CACHE_MSD_D0 = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_MSD_20MG = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
CACHE_MEDIAN_STEPSIZE = Path(__file__).parent.parent / "cache" / "imsd_median_stepsize_by_size.pkl"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "iMSD_Histograms"

N_BINS = 30
STYLE_WATER = dict(face="#009900", edge="#006600", label="Wasser")
STYLE_HYDROGEL = dict(face="#0000da", edge="#000099", label="Hydrogel")


def _load_pickle(path: Path, compute_script: str):
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _step_sizes_per_track(results: dict) -> dict[tuple[str, int], float]:
    """{(xml_path, particle_id): mittlere Euklidische Schrittweite (µm)} je Track."""
    out: dict[tuple[str, int], float] = {}
    for r in results.values():
        tracks_df = r.get("tracks_df")
        mpp = r.get("mpp")
        xml_path = r.get("xml_path")
        if tracks_df is None or tracks_df.empty or mpp is None:
            continue
        steps = calculate_step_sizes(tracks_df, step_interval=1, sliding=True)
        if steps.empty:
            continue
        step_um = np.hypot(steps["dx"].to_numpy(), steps["dy"].to_numpy()) * mpp
        means = pd.Series(step_um, index=steps["particle"]).groupby(level=0).mean()
        for particle_id, value in means.items():
            out[(xml_path, particle_id)] = float(value)
    return out


def build_track_table() -> pd.DataFrame:
    """Pro-Track-Tabelle: Tracklaenge/D/n aus TaskB1-Cache + Schrittweite aus MSD-Caches gemerged."""
    df_water = _load_pickle(CACHE_TRACKLENGTH_D0, "TaskB1_TrackLengthBias_Compute.py")
    df_hydrogel = _load_pickle(CACHE_TRACKLENGTH_20MG, "TaskB1_TrackLengthBias_Compute.py")
    df = pd.concat([df_water, df_hydrogel], ignore_index=True)

    d0_results = _load_pickle(CACHE_MSD_D0, "MSD_FromTrackmate_D0.py")
    hydrogel_results = _load_pickle(CACHE_MSD_20MG, "MSD_FromTrackmate_20mg.py")
    step_lookup = _step_sizes_per_track(d0_results)
    step_lookup.update(_step_sizes_per_track(hydrogel_results))

    df["step_um"] = [step_lookup.get((row.xml_path, row.particle_id), np.nan) for row in df.itertuples()]
    return df


def _median_stepsize_by_size(df: pd.DataFrame) -> dict[float, float]:
    return df.groupby("particle_size_nm")["step_um"].median().dropna().to_dict()


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
    ("track_length_frames", "Tracklänge (Frames)", False),
    ("step_um", "Schrittweite (µm)", False),
    ("D_um2_per_s", r"$D$ (µm²/s)", True),
    ("exponent", r"Anomalie-Exponent $n$", False),
]


_FILENAMES = {
    "track_length_frames": "imsd_tracklength_by_size.png",
    "step_um": "imsd_stepsize_by_size.png",
    "D_um2_per_s": "imsd_D_by_size.png",
    "exponent": "imsd_n_by_size.png",
}


def plot_imsd_metric_grid(df: pd.DataFrame, metric: str, xlabel: str, log_x: bool) -> plt.Figure:
    """Ein Grid pro Metrik (eigene Abbildungsdatei): ein Panel je Partikelgröße
    (dynamisches 3-Spalten-Raster), Wasser vs. Hydrogel je Panel überlagert.
    Separate Abbildung pro Metrik statt einer gemeinsamen Übersichtsgrafik --
    besser geeignet als Einzelabbildung im Hauptkapitel."""
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
    df = build_track_table()
    n_water = int((df["condition"] == "water").sum())
    n_hydrogel = int((df["condition"] == "hydrogel").sum())
    print(f"Pro-Track-Tabelle: {len(df)} Tracks ({n_water} Wasser, {n_hydrogel} Hydrogel)")
    missing_step = int(df["step_um"].isna().sum())
    if missing_step:
        print(f"Hinweis: {missing_step} Tracks ohne zugeordnete Schrittweite (kein Match in tracks_df).")

    median_by_size = _median_stepsize_by_size(df)
    CACHE_MEDIAN_STEPSIZE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_MEDIAN_STEPSIZE, "wb") as f:
        pickle.dump(median_by_size, f)
    print(f"Cache gespeichert: {CACHE_MEDIAN_STEPSIZE}")
    print(pd.Series(median_by_size, name="median_step_um").sort_index().to_string())

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(_RC):
        for col_name, ylabel, log_x in _METRICS:
            fig = plot_imsd_metric_grid(df, col_name, ylabel, log_x)
            fig_path = SAVE_PATH / _FILENAMES[col_name]
            safe_savefig(fig, fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot gespeichert: {fig_path}")


if __name__ == "__main__":
    main()
