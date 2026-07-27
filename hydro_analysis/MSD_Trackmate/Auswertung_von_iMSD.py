"""
Auswertung von iMSD – per-Track Diffusionskonstante.

Lädt die Cache-Dateien (20 mg, 13 mg, Wasser) und berechnet für jeden
einzelnen iMSD-Track einen Diffusionskoeffizienten D über einen linearen
Fit der ersten FIT_POINTS Lag-Zeit-Punkte (MSD = 4D·τ + c).
Für jede Partikelgröße wird die Verteilung der D-Werte als Histogramm
(log-skalierte x-Achse) dargestellt.
"""

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Style (Style_guide.txt) ─────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.family": "Source Sans 3",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
})

# ── Configuration ───────────────────────────────────────────────────
CACHE_20MG = Path(__file__).parent / "cache" / "msd_20mg_files.pkl"
CACHE_13MG = Path(__file__).parent / "cache" / "msd_13mg_results.pkl"
CACHE_D0   = Path(__file__).parent / "cache" / "msd_d0_results.pkl"

# Auf False setzen um einzelne Datensätze aus der Darstellung zu entfernen
INCLUDE_13MG = True
INCLUDE_D0   = True

FIT_POINTS = 6   # Anzahl Lag-Zeit-Punkte für den per-Track-Linearfit
N_BINS     = 40  # Histogramm-Bins (log-verteilt)

SAVE_PATH = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")

# Farben pro Datensatz (Style-Guide: Blau für 20 mg, Orange für 13 mg, Grün für Wasser)
STYLE: dict[str, dict] = {
    "20mg": dict(face="#0000da", edge="#000099", label="20 mg/mL"),
    "13mg": dict(face="#da6000", edge="#994500", label="13 mg/mL"),
    "d0":   dict(face="#009900", edge="#006600", label="Wasser (D\u2080)"),
}


# ── Helpers ─────────────────────────────────────────────────────────

def _load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _fit_track_D(lag_times: np.ndarray, msd_values: np.ndarray, n_points: int) -> float:
    """
    Berechnet D für einen einzelnen iMSD-Track via Linearfit.
    Modell: MSD = 4D·τ + c  →  Steigung / 4 = D  (2D-Diffusion).
    Gibt NaN zurück wenn der Fit fehlschlägt oder D ≤ 0.
    """
    mask = (
        np.isfinite(lag_times) & np.isfinite(msd_values)
        & (lag_times > 0) & (msd_values > 0)
    )
    lt = lag_times[mask][:n_points]
    mv = msd_values[mask][:n_points]
    if len(lt) < 2:
        return np.nan
    slope, _ = np.polyfit(lt, mv, 1)
    D = slope / 4.0
    return float(D) if D > 0 else np.nan


def _extract_track_D_per_size(results: dict, n_points: int) -> dict[float, np.ndarray]:
    """
    Extrahiert D-Werte für alle Tracks aus den Cache-Ergebnissen.
    Gibt {particle_size_nm: array_of_D_values} zurück.
    """
    size_D: dict[float, list[float]] = {}
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        if size_nm is None or fit is None:
            continue
        imsd: pd.DataFrame | None = fit.get("imsd")
        if imsd is None or imsd.empty:
            continue
        lag_times = np.asarray(imsd.index.values, dtype=float)
        for col in imsd.columns:
            msd_vals = np.asarray(imsd[col].values, dtype=float)
            D = _fit_track_D(lag_times, msd_vals, n_points)
            if np.isfinite(D):
                size_D.setdefault(float(size_nm), []).append(D)
    return {s: np.array(v) for s, v in size_D.items()}


def _plot_size_histograms(
    size_D: dict[float, np.ndarray],
    style: dict,
    save_dir: Path | None,
) -> None:
    """
    Erstellt ein Figure mit einem Subplot pro Partikelgröße (max. 3 Spalten).
    X-Achse log-skaliert, Bins log-verteilt. Median als vertikale Linie.
    """
    sizes = sorted(size_D.keys())
    n = len(sizes)
    if n == 0:
        print(f"  Keine iMSD-Daten für {style['label']}.")
        return

    # Layout: ≤3 Größen → 1 Zeile; >3 → 2 Zeilen à max. 3 Spalten
    if n <= 3:
        nrows, ncols = 1, n
    else:
        ncols = 3
        nrows = int(np.ceil(n / ncols))

    subplot_w = 7.15 / 3          # ~2.38 inch pro Spalte
    subplot_h = 5.00 / 2          # ~2.50 inch pro Zeile
    fig_w = ncols * subplot_w
    fig_h = nrows * subplot_h

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        squeeze=False,
    )
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:         # überzählige Subplots ausblenden
        ax.set_visible(False)

    for ax, size_nm in zip(ax_flat, sizes):
        D_vals = size_D[size_nm]
        D_pos  = D_vals[D_vals > 0]

        if D_pos.size < 2:
            ax.text(0.5, 0.5, "keine Daten", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{int(size_nm)} nm")
            continue

        bins = np.logspace(np.log10(D_pos.min()), np.log10(D_pos.max()), N_BINS + 1)
        ax.hist(D_pos, bins=bins,
                color=style["face"], edgecolor=style["edge"],
                linewidth=0.5, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel(r"$D$ (µm²/s)")
        ax.set_ylabel("Anzahl Tracks")

        median = float(np.median(D_pos))
        mean   = float(np.mean(D_pos))
        ax.axvline(median, color=style["edge"], linewidth=1.2,
                   label=f"Median = {median:.3g}")
        ax.legend(loc="best")
        ax.set_title(
            f"{int(size_nm)} nm  |  N = {D_pos.size:,}\n"
            f"Mean = {mean:.3g}  µm²/s"
        )
        ax.minorticks_on()

    fig.suptitle(
        f"Per-Track Diffusionskonstante (iMSD) – {style['label']}",
        fontsize=12, fontweight="semibold",
    )
    plt.show()

    if save_dir is not None:
        tag = style["label"].replace("/", "-").replace(" ", "_").replace("\u2080", "0")
        base = f"imsd_track_D_hist_{tag}"
        fig.savefig(save_dir / f"{base}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  Gespeichert: {save_dir / base}.png")


def _print_summary(size_D: dict[float, np.ndarray], label: str) -> None:
    rows = []
    for s in sorted(size_D.keys()):
        v = size_D[s]
        rows.append({
            "size_nm": int(s),
            "N": len(v),
            "median (µm²/s)": np.median(v),
            "mean (µm²/s)": np.mean(v),
            "std (µm²/s)": np.std(v),
        })
    df = pd.DataFrame(rows)
    print(f"\n── {label} ──")
    print(df.to_string(index=False))


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path(SAVE_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 20 mg/mL  (immer aktiv)
    print("Lade 20 mg/mL Cache …")
    size_D_20 = _extract_track_D_per_size(_load_cache(CACHE_20MG), FIT_POINTS)
    _print_summary(size_D_20, STYLE["20mg"]["label"])
    _plot_size_histograms(size_D_20, STYLE["20mg"], out_dir)

    # 13 mg/mL  (optional: INCLUDE_13MG)
    if INCLUDE_13MG:
        print("\nLade 13 mg/mL Cache …")
        size_D_13 = _extract_track_D_per_size(_load_cache(CACHE_13MG), FIT_POINTS)
        _print_summary(size_D_13, STYLE["13mg"]["label"])
        _plot_size_histograms(size_D_13, STYLE["13mg"], out_dir)

    # Wasser / D0  (optional: INCLUDE_D0)
    if INCLUDE_D0:
        print("\nLade Wasser (D\u2080) Cache …")
        size_D_d0 = _extract_track_D_per_size(_load_cache(CACHE_D0), FIT_POINTS)
        _print_summary(size_D_d0, STYLE["d0"]["label"])
        _plot_size_histograms(size_D_d0, STYLE["d0"], out_dir)


if __name__ == "__main__":
    main()
