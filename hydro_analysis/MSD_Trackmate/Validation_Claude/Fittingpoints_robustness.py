"""
Robustheit von D und n gegenueber der Anzahl der in den MSD-Powerlaw-Fit
einbezogenen Lag-Zeit-Punkte (MSD(tau) = 4D * tau^n).

Neues Skript, ersetzt keine bestehenden Skripte. Reiner Konsument: liest
cache/msd_d0_results.pkl (Bedingung "Wasser") und cache/msd_20mg_files.pkl
(Bedingung ueber core.io.condition_label_from_filename -> "Hydrogel surface
loading" / "Hydrogel injection", sonst Fallback "Hydrogel"). Kein Neuladen
von Rohdaten, kein Neuberechnen der Trajektorien/iMSD -- fit_results_MSD
["imsd"] (eine Spalte pro Track, bereits ueber genuegend Lag-Zeiten von
trackpy berechnet) ist in beiden Caches bereits vorhanden.

Fuer jeden Track (jede imsd-Spalte) wird core.analysis.fit_powerlaw_with_
errors() -- unveraendert wiederverwendet, bereits ueber den Parameter
`points` parametrisiert -- fuer jede der 5 Datenpunktanzahlen (4, 5, 6, 7, 8)
separat aufgerufen (Fenster beginnend bei der ersten Lag-Zeit). Ein Track
wird fuer eine gegebene Punktanzahl uebersprungen (nicht stillschweigend
weggelassen, siehe Konsolen-Ausgabe), wenn das Fit-Fenster NaN- oder
<=0-Werte enthaelt -- gleiche Ausschlussregel wie
TaskB1_TrackLengthBias_Compute.py::per_track_fits().

Aggregation pro (Partikelgroesse x Bedingung x Datenpunktanzahl): Median und
Interquartilsabstand (25./75. Perzentil) von D und n.

Ausgabe 1: fittingpoints_robustness.csv (Spalten: partikelgröße_nm,
bedingung, fittingpoints, D_median, D_IQR_low, D_IQR_high, n_median,
n_IQR_low, n_IQR_high, n_tracks).

Ausgabe 2: fittingpoints_robustness.png -- zwei Subplots uebereinander
(oben D, log-Skala; unten n, linear, gestrichelte Referenzlinie bei n=1),
eine Linie (Median) + schattiertes Band (IQR) pro (Groesse x Bedingung)-
Kombination. Farbe = Partikelgroesse, Linienstil = Bedingung (durchgezogen
= Wasser, gestrichelt = Hydrogel surface loading, gepunktet = Hydrogel
injection).

Vor Ausfuehrung muessen MSD_FromTrackmate_D0.py und MSD_FromTrackmate_20mg.py
bereits gelaufen sein.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.analysis import fit_powerlaw_with_errors
from hydro_analysis.core.io import condition_label_from_filename, get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0 = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_20MG = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Fittingpoints_Robustness"

FITTINGPOINTS = (4, 5, 6, 7, 8)

SIZE_COLORS = {
    20.0: "#0000da", 50.0: "#004cff", 100.0: "#00c4ff",
    200.0: "#49ffad", 500.0: "#adff49", 1000.0: "#ffd700",
}
CONDITION_LINESTYLE = {
    "Wasser": "solid",
    "Hydrogel surface loading": (0, (4, 2)),
    "Hydrogel injection": (0, (1, 1.5)),
    "Hydrogel": (0, (3, 1, 1, 1)),
}


def _load_pickle(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _condition_for(base_name: str, source: str) -> str:
    if source == "water":
        return "Wasser"
    label = condition_label_from_filename(base_name)
    return f"Hydrogel {label.lower()}" if label else "Hydrogel"


def per_track_fits_by_points(results: dict, source: str) -> pd.DataFrame:
    """Pro Track x pro Datenpunktanzahl: D, n (oder NaN, wenn das Fit-Fenster
    fuer diese Punktanzahl NaN/<=0-Werte enthaelt)."""
    rows = []
    n_total = 0
    n_excluded = 0
    for r in results.values():
        fit = r.get("fit_results_MSD")
        size_nm = r.get("particle_size_nm")
        if fit is None or fit.get("imsd") is None or size_nm is None:
            continue
        imsd = fit["imsd"]
        condition = _condition_for(r.get("base_name", ""), source)
        for col in imsd.columns:
            n_total += 1
            series = imsd[col]
            for points in FITTINGPOINTS:
                window = series.iloc[0:points]
                if len(window) < points or window.isna().any() or (window <= 0).any():
                    n_excluded += 1
                    continue
                track_fit = fit_powerlaw_with_errors(window, points=points)
                rows.append({
                    "particle_size_nm": float(size_nm), "condition": condition,
                    "fittingpoints": points, "xml_path": r.get("xml_path"), "particle_id": col,
                    "D_um2_per_s": float(track_fit["A"][0] / 4.0), "exponent": float(track_fit["n"][0]),
                })
    print(f"  {source}: {n_total} Tracks x {len(FITTINGPOINTS)} Punktanzahlen geprüft, "
          f"{n_excluded} Kombinationen ausgeschlossen (NaN/<=0 im Fit-Fenster oder Track zu kurz)")
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (size_nm, condition, points), g in df.groupby(["particle_size_nm", "condition", "fittingpoints"]):
        D = g["D_um2_per_s"].to_numpy(dtype=float)
        n = g["exponent"].to_numpy(dtype=float)
        rows.append({
            "partikelgröße_nm": size_nm, "bedingung": condition, "fittingpoints": points,
            "D_median": float(np.median(D)), "D_IQR_low": float(np.percentile(D, 25)),
            "D_IQR_high": float(np.percentile(D, 75)),
            "n_median": float(np.median(n)), "n_IQR_low": float(np.percentile(n, 25)),
            "n_IQR_high": float(np.percentile(n, 75)),
            "n_tracks": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values(["partikelgröße_nm", "bedingung", "fittingpoints"]).reset_index(drop=True)


def plot_robustness(summary: pd.DataFrame, dls_labels: dict[float, int]) -> plt.Figure:
    fig, (ax_d, ax_n) = plt.subplots(2, 1, figsize=(7.15, 5.00), constrained_layout=True, sharex=True)
    combos = summary[["partikelgröße_nm", "bedingung"]].drop_duplicates().itertuples(index=False)
    legend_size = {}
    legend_condition = {}
    for size_nm, condition in combos:
        g = summary[(summary["partikelgröße_nm"] == size_nm) & (summary["bedingung"] == condition)]
        g = g.sort_values("fittingpoints")
        color = SIZE_COLORS.get(size_nm, "#888888")
        linestyle = CONDITION_LINESTYLE.get(condition, "solid")
        x = g["fittingpoints"].to_numpy()

        ax_d.plot(x, g["D_median"], color=color, linestyle=linestyle, linewidth=1.6, marker="o", markersize=3)
        ax_d.fill_between(x, g["D_IQR_low"], g["D_IQR_high"], color=color, alpha=0.12, linewidth=0)

        ax_n.plot(x, g["n_median"], color=color, linestyle=linestyle, linewidth=1.6, marker="o", markersize=3)
        ax_n.fill_between(x, g["n_IQR_low"], g["n_IQR_high"], color=color, alpha=0.12, linewidth=0)

        legend_size.setdefault(size_nm, color)
        legend_condition.setdefault(condition, linestyle)

    ax_n.axhline(1.0, color="grey", linewidth=1.0, linestyle="--", zorder=1)
    ax_d.set_yscale("log")
    ax_d.set_ylabel(r"$D$ (µm²/s)")
    ax_n.set_ylabel(r"$n$")
    ax_n.set_xlabel("Anzahl Fitting-Datenpunkte")
    ax_n.set_xticks(list(FITTINGPOINTS))

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=color, linewidth=1.6, label=f"{dls_labels.get(s, int(s))} nm")
               for s, color in sorted(legend_size.items())]
    handles += [Line2D([0], [0], color="black", linewidth=1.6, linestyle=ls, label=cond)
                for cond, ls in legend_condition.items()]
    handles.append(Line2D([0], [0], color="grey", linewidth=1.0, linestyle="--", label="n = 1 (Brownsche Diffusion)"))
    ax_d.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)

    fig.suptitle("Robustheit von D und n gegenüber der Anzahl der Fit-Datenpunkte",
                 fontsize=11, fontweight="semibold")
    return fig


def main() -> None:
    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    hydrogel_results = _load_pickle(CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    print(f"Cache geladen: {CACHE_20MG} ({len(hydrogel_results)} Dateien)")

    df_water = per_track_fits_by_points(d0_results, "water")
    df_gel = per_track_fits_by_points(hydrogel_results, "hydrogel")
    df = pd.concat([df_water, df_gel], ignore_index=True)
    print(f"Gesamt: {len(df)} (Track x Fitpunkte)-Kombinationen erfolgreich gefittet.")

    summary = build_summary(df)
    print(summary.to_string(index=False))

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    csv_path = SAVE_PATH / "fittingpoints_robustness.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Tabelle gespeichert: {csv_path}")

    dls_labels = get_dls_labels()
    with plt.rc_context(_RC):
        fig = plot_robustness(summary, dls_labels)
        fig_path = SAVE_PATH / "fittingpoints_robustness.png"
        safe_savefig(fig, fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {fig_path}")


if __name__ == "__main__":
    main()
