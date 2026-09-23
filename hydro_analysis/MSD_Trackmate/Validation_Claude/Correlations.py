"""
Konsolidierte MSD-Korrelationsauswertung.

Arbeitet ausschliesslich mit Endresultaten pro Datei (eMSD-Ensemble-Fits aus
fit_results_MSD, NIE per-Track/iMSD-Zwischenwerte) und wertet Partikelgroessen
immer separat aus (nie gepoolt). Reiner Konsument der bereits gefitteten
Caches: cache/msd_d0_results.pkl (Wasser), cache/msd_20mg_files.pkl
(Hydrogel), cache/localization_error_immobilized.pkl
(Loc_Error_Analyse_immob_particle.py). Kein TrackMate-Parameter-Sweep, keine
Intercept-Diagnostik (beides auf Wunsch entfernt) -- fuer letzteres siehe
weiterhin TaskA2_A4_Intercept_Comparison.py.

Plots:
  A_correction_deviation_per_file.png
      Prozentuale Abweichung D/n durch die loc_err-Korrektur
      (MSD_corrected = MSD_measured - 4*sigma_loc^2, Refit nur ueber das
      positive Fenster, unveraendert aus TaskA3_MSD_Correction.py's
      correct_and_refit() uebernommen), angewendet auf die Datei-eMSD (nicht
      auf einzelne Tracks) -- ein Punkt pro Datei, nur fuer
      TARGET_SIZES_NM=(20, 50) nm (einzige Groessen mit vertrauenswuerdiger
      sigma_loc-Referenz). Je Groesse eine eigene Spalte (nie gepoolt), oben
      Delta D, unten Delta n, Medianlinie je Panel.
  B_length_bin_comparison.png
      Dataset in Tracklaengen-Bins geteilt (10-20, 20-40, 40-70, 70+ Frames);
      je Datei und Bin wird das Ensemble-D/n NEU gefittet (core.analysis.
      perform_msd_analysis, unveraendert, auf eine auf den Bin gefilterte
      Tracks-Teilmenge), dann die Dateien untereinander verglichen (ein Punkt
      je Datei und Bin, farblich nach Partikelgroesse). Separat fuer Wasser
      und Hydrogel.
  C_D_vs_n_scatter.png
      D vs. n, ein Punkt pro Datei (Endresultat aus fit_results_MSD) -- ein
      eigenes Panel je Partikelgroesse (nie gepoolt), Farbe = Wasser/Hydrogel
      innerhalb jedes Panels. Keine loc_err-korrigierte Variante hier (siehe
      stattdessen Plot A fuer die Korrektur selbst).
  D_correlations_quality_d0.png / _20mg.png
      2xN-Korrelationsgrids D/D_error vs. Datei-/Prozessierungsmetriken
      (Helligkeit, Trajektorienlaenge, Edge-Filter, sigma_loc), unveraendert
      aus MSD_Diffusion_Correlations_Test_D0.py/_20mg.py uebernommen (ohne
      den TrackMate-Settings-Teil).

CSV-Ausgaben (Auswertungsbilder\\Correlations\\): msd_correction_comparison.csv
(pro Datei), length_bin_comparison.csv (pro Datei x Bin) -- Validation_Summary.py
liest von hier.

Vor Ausfuehrung muessen MSD_FromTrackmate_D0.py, MSD_FromTrackmate_20mg.py und
Loc_Error_Analyse_immob_particle.py bereits gelaufen sein.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter
import tifffile

from hydro_analysis.core.analysis import fit_powerlaw_with_errors, perform_msd_analysis, DEFAULT_MSD_FIT_POINTS
from hydro_analysis.core.io import condition_label_from_filename
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0 = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_20MG = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
CACHE_LOCERR = Path(__file__).parent.parent / "cache" / "localization_error_immobilized.pkl"

SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "Correlations"

TARGET_SIZES_NM = (20.0, 50.0)   # einzige Groessen mit vertrauenswuerdiger sigma_loc-Referenz

STYLE_WATER = dict(face="#3B6E8C", edge="#2A4F66", label="Wasser")
STYLE_HYDROGEL = dict(face="#da0000", edge="#990000", label="Hydrogel")
SIZE_COLORS = {
    # Style_guide.txt Abschnitt 9 (Jet-angelehnte 8-Akzentfarben-Palette),
    # groesste Partikel = Index 1 (blau) bis kleinste = Index 7 (orange) --
    # "universelles Muster" 1000 nm blau ... 35/20 nm orange.
    1000.0: ("#0000da", "#000099"), 500.0: ("#004cff", "#0035b2"), 200.0: ("#00c4ff", "#0089b2"),
    100.0: ("#49ffad", "#33b279"), 50.0: ("#ffd700", "#b29600"), 20.0: ("#ff6800", "#b24900"),
}

LENGTH_BINS = [(10, 20), (20, 40), (40, 70), (70, np.inf)]
LENGTH_BIN_LABELS = ["10-20", "20-40", "40-70", "70+"]

BRIGHTNESS_MAX_FRAMES = 20
TARGET_NOMINAL_SIZES_20MG = {20.0}
SIZE_MARKER_20MG = {20.0: "s", 50.0: "o"}
SIZE_LABEL_20MG = {20.0: "35 nm", 50.0: "50 nm"}
COLOR_SURFACE, COLOR_SURFACE_DARK = "#3B8C8C", "#2A6666"
COLOR_INJECTION, COLOR_INJECTION_DARK = "#D98C3D", "#A6672D"
CONDITION_STYLE_20MG = {
    "Surface loading": {"face": COLOR_SURFACE, "edge": COLOR_SURFACE_DARK},
    "Injection": {"face": COLOR_INJECTION, "edge": COLOR_INJECTION_DARK},
}
POINT_ALPHA = 0.65
METRICS = [
    ("num_tracks", "Anzahl Trajektorien"),
    ("num_points", "Anzahl Datenpunkte"),
    ("edge_removed", "Prozessierte Punkte (Edge-Filter entfernt)"),
    ("mean_track_length", "Mittlere Trajektorienlänge (Frames)"),
    ("mean_brightness", "Mittlere Kamera-Helligkeit (a.u.)"),
    ("sigma_loc_nm", "Lokalisationsunsicherheit $\\sigma_{loc}$ (nm)"),
]


def _load_pickle(path: Path, compute_script: str):
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _representative_sigma_loc_nm(locerr_df: pd.DataFrame, size_nm: float) -> float:
    """Gepoolter Mittelwert von x- & y-static-SD (nm) ueber beide immobilisierte
    Datensaetze, fuer eine nominelle Groesse."""
    sub = locerr_df[locerr_df["particle_size_nm"] == size_nm]
    if sub.empty:
        return np.nan
    vals = np.concatenate([sub["sigma_x_static_nm"].dropna().to_numpy(),
                            sub["sigma_y_static_nm"].dropna().to_numpy()])
    return float(np.mean(vals)) if vals.size else np.nan


# ── Plot A: loc_err-Korrektur, pro Datei (Endresultat, eMSD) ───────────────────

def correct_and_refit(fit: dict, sigma_loc_um: float) -> dict:
    """MSD_corrected(tau) = MSD_measured(tau) - 4*sigma_loc^2, Refit nur ueber
    das positive Fenster -- unveraendert aus TaskA3_MSD_Correction.py."""
    emsd = fit["emsd"]
    fit_points = fit["fit_points"]
    offset_um2 = 4.0 * sigma_loc_um ** 2
    corrected_full = emsd - offset_um2
    window = corrected_full.iloc[0:fit_points]
    positive_window = window[window > 0]
    n_excluded = int(fit_points - len(positive_window))
    corrected_fit = None
    if len(positive_window) >= 2:
        corrected_fit = fit_powerlaw_with_errors(positive_window, points=len(positive_window))
    return {"corrected_emsd_full": corrected_full, "offset_um2": offset_um2,
            "n_excluded_from_fit": n_excluded, "corrected_fit": corrected_fit}


def build_file_correction_table(d0_results: dict, hydrogel_results: dict, locerr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for results, condition in ((d0_results, "water"), (hydrogel_results, "hydrogel")):
        for r in results.values():
            size_nm = r.get("particle_size_nm")
            if size_nm not in TARGET_SIZES_NM:
                continue
            fit = r.get("fit_results_MSD")
            if fit is None or fit.get("emsd") is None:
                continue
            sigma_loc_um = _representative_sigma_loc_nm(locerr_df, float(size_nm)) / 1000.0
            if not np.isfinite(sigma_loc_um):
                continue
            correction = correct_and_refit(fit, sigma_loc_um)
            corrected_fit = correction["corrected_fit"]
            D_before, n_before = fit["D_um2_per_s"], fit["exponent"]
            D_after = (corrected_fit["A"][0] / 4.0) if corrected_fit is not None else np.nan
            n_after = corrected_fit["n"][0] if corrected_fit is not None else np.nan
            rows.append({
                "xml_path": r.get("xml_path"), "base_name": r.get("base_name"),
                "particle_size_nm": float(size_nm), "condition": condition, "num_tracks": r.get("num_tracks"),
                "D_before": D_before, "D_after": D_after,
                "D_rel_change": (D_after - D_before) / D_before if np.isfinite(D_after) and D_before else np.nan,
                "n_before": n_before, "n_after": n_after,
                "n_rel_change": (n_after - n_before) / n_before if np.isfinite(n_after) and n_before else np.nan,
            })
    return pd.DataFrame(rows)


def plot_correction_deviation(table: pd.DataFrame) -> plt.Figure:
    sizes = sorted(table["particle_size_nm"].dropna().unique())
    fig, axes = plt.subplots(2, len(sizes), figsize=(4.5 * len(sizes), 6.0), constrained_layout=True, squeeze=False)
    for col, size_nm in enumerate(sizes):
        sub = table[table["particle_size_nm"] == size_nm].dropna(subset=["D_rel_change", "n_rel_change"])
        sub = sub.sort_values(["condition", "xml_path"]).reset_index(drop=True)
        x = np.arange(len(sub))
        colors = [STYLE_WATER["face"] if c == "water" else STYLE_HYDROGEL["face"] for c in sub["condition"]]
        for row, (col_name, ylabel) in enumerate((("D_rel_change", r"$\Delta D$ (%)"),
                                                    ("n_rel_change", r"$\Delta n$ (%)"))):
            ax = axes[row][col]
            vals = sub[col_name].to_numpy(dtype=float) * 100.0
            ax.scatter(x, vals, s=24, alpha=0.75, c=colors, edgecolor="none")
            if len(vals):
                median = float(np.median(vals))
                ax.axhline(median, color="black", linewidth=1.2, linestyle="--", label=f"Median = {median:.2f}%")
            ax.axhline(0.0, color="grey", linewidth=0.8)
            if row == 0:
                ax.set_title(f"{size_nm:.0f} nm  |  N={len(sub)} Dateien", fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Datei-Index (sortiert nach Bedingung)")
    handles = [Patch(facecolor=STYLE_WATER["face"], label="Wasser"),
               Patch(facecolor=STYLE_HYDROGEL["face"], label="Hydrogel"),
               Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Median")]
    axes[0][0].legend(handles=handles, loc="best", fontsize=7, frameon=False)
    fig.suptitle("Abweichung durch loc_err-Korrektur pro Datei (Endresultate, eMSD)",
                 fontsize=11, fontweight="semibold")
    return fig


# ── Plot B: Tracklaengen-Bins, Dateien im Vergleich ─────────────────────────────

def _fit_key(fit_points: int) -> str:
    return "fit_results_MSD" if fit_points == DEFAULT_MSD_FIT_POINTS else f"fit_MSD_fp_{fit_points}"


def build_length_bin_table(results: dict, condition: str) -> pd.DataFrame:
    """Pro Datei x Tracklaengen-Bin: Ensemble-D/n neu gefittet (core.analysis.
    perform_msd_analysis, unveraendert) auf eine nach Tracklaenge gefilterte
    Tracks-Teilmenge -- Endresultat je (Datei, Bin), kein Pooling ueber
    Groessen oder Bins hinweg."""
    rows = []
    for r in results.values():
        tracks_df = r.get("tracks_df")
        mpp, fps = r.get("mpp"), r.get("fps")
        fit = r.get("fit_results_MSD")
        size_nm = r.get("particle_size_nm")
        if tracks_df is None or tracks_df.empty or mpp is None or fps is None or fit is None or size_nm is None:
            continue
        fit_points = fit.get("fit_points", DEFAULT_MSD_FIT_POINTS)
        lengths = tracks_df.groupby("particle").size()
        for (lo, hi), label in zip(LENGTH_BINS, LENGTH_BIN_LABELS):
            particle_ids = lengths[(lengths >= lo) & (lengths < hi)].index
            if len(particle_ids) == 0:
                continue
            sub_tracks = tracks_df[tracks_df["particle"].isin(particle_ids)]
            temp_result = {"tracks_df": sub_tracks, "mpp": mpp, "fps": fps}
            perform_msd_analysis(temp_result, fit_points=fit_points)
            bin_fit = temp_result.get(_fit_key(fit_points))
            if bin_fit is None or bin_fit.get("D_um2_per_s") is None:
                continue
            rows.append({
                "xml_path": r.get("xml_path"), "base_name": r.get("base_name"),
                "particle_size_nm": float(size_nm), "condition": condition, "length_bin": label,
                "n_tracks_in_bin": int(len(particle_ids)),
                "D_um2_per_s": float(bin_fit["D_um2_per_s"]), "exponent": float(bin_fit["exponent"]),
            })
    return pd.DataFrame(rows)


def plot_length_bins(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), constrained_layout=True, sharex=True)
    rng = np.random.default_rng(0)
    positions = np.arange(len(LENGTH_BIN_LABELS))
    for col, condition in enumerate(("water", "hydrogel")):
        sub_cond = df[df["condition"] == condition]
        for row, (value_col, ylabel, log_y) in enumerate((("D_um2_per_s", r"$D$ (µm²/s)", True),
                                                            ("exponent", r"$n$", False))):
            ax = axes[row][col]
            box_data = [sub_cond.loc[sub_cond["length_bin"] == label, value_col].dropna().to_numpy()
                        for label in LENGTH_BIN_LABELS]
            bp = ax.boxplot(box_data, positions=positions, widths=0.5, showfliers=False, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#dddddd")
                patch.set_alpha(0.5)
            for i, label in enumerate(LENGTH_BIN_LABELS):
                for size_nm in sorted(sub_cond["particle_size_nm"].dropna().unique()):
                    vals = sub_cond.loc[(sub_cond["length_bin"] == label)
                                         & (sub_cond["particle_size_nm"] == size_nm), value_col].to_numpy()
                    if vals.size == 0:
                        continue
                    jitter = (rng.random(vals.size) - 0.5) * 0.3
                    face, edge = SIZE_COLORS.get(size_nm, ("#999999", "#555555"))
                    ax.scatter(np.full(vals.size, i) + jitter, vals, s=16, alpha=0.75,
                               facecolor=face, edgecolor=edge, linewidth=0.4, zorder=3)
            if log_y:
                ax.set_yscale("log")
            ax.set_xticks(positions)
            ax.set_xticklabels(LENGTH_BIN_LABELS)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(STYLE_WATER["label"] if condition == "water" else STYLE_HYDROGEL["label"], fontsize=10)
            if row == 1:
                ax.set_xlabel("Tracklängen-Bin (Frames)")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=face, markeredgecolor=edge,
                      markersize=7, label=f"{s:.0f} nm")
               for s, (face, edge) in sorted(SIZE_COLORS.items()) if s in df["particle_size_nm"].unique()]
    fig.legend(handles=handles, loc="outside lower center", ncol=max(len(handles), 1), frameon=False, fontsize=8)
    fig.suptitle("D/n je Tracklängen-Bin -- Dateien im Vergleich (Wasser vs. Hydrogel)",
                 fontsize=11, fontweight="semibold")
    return fig


# ── Plot C: D vs. n, pro Datei (Endresultat, eMSD) ──────────────────────────────

def build_file_standard_table(d0_results: dict, hydrogel_results: dict) -> pd.DataFrame:
    rows = []
    for results, condition in ((d0_results, "water"), (hydrogel_results, "hydrogel")):
        for r in results.values():
            size_nm = r.get("particle_size_nm")
            fit = r.get("fit_results_MSD")
            if size_nm is None or fit is None:
                continue
            D, n = fit.get("D_um2_per_s"), fit.get("exponent")
            if D is None or n is None:
                continue
            rows.append({"xml_path": r.get("xml_path"), "base_name": r.get("base_name"),
                         "particle_size_nm": float(size_nm), "condition": condition,
                         "D_um2_per_s": float(D), "exponent": float(n)})
    return pd.DataFrame(rows)


def plot_d_vs_n(file_standard: pd.DataFrame) -> plt.Figure:
    """Grid: eine eigene Box (Panel) je Partikelgroesse, Wasser/Hydrogel
    farblich unterschieden. Keine loc_err-korrigierte Variante mehr (nur
    Endresultate direkt aus fit_results_MSD, siehe Plot A fuer die
    Korrektur-Auswertung selbst)."""
    sizes = sorted(file_standard["particle_size_nm"].dropna().unique())
    n = len(sizes)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.15 / 3, nrows * 5.00 / 2.2),
                             constrained_layout=True, squeeze=False)
    ax_flat = axes.flatten()
    for ax in ax_flat[n:]:
        ax.set_visible(False)

    for ax, size_nm in zip(ax_flat, sizes):
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle=_DASH_THEORY, zorder=2)
        sub = file_standard[file_standard["particle_size_nm"] == size_nm]
        for condition, style in (("water", STYLE_WATER), ("hydrogel", STYLE_HYDROGEL)):
            g = sub[sub["condition"] == condition]
            if g.empty:
                continue
            ax.scatter(g["D_um2_per_s"], g["exponent"], s=30, alpha=0.7, facecolor=style["face"],
                      edgecolor=style["edge"], linewidth=0.6, zorder=3)
        ax.set_xscale("log")
        # Ein enger D-Bereich (z.B. nur eine Bedingung, wenige Dateien, unter
        # einer Dekade) laesst matplotlibs Standard-Log-Ticker sonst viele eng
        # stehende, sich ueberlappende wissenschaftliche Notation zeichnen
        # (unlesbar beobachtet, auch mit LogLocator(subs="auto")). Stattdessen
        # feste Anzahl "schoener" Ticks mit einfacher Dezimalbeschriftung.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{size_nm:.0f} nm", fontsize=10)
        ax.set_xlabel(r"$D$ (µm²/s)")
        ax.set_ylabel(r"Anomalie-Exponent $n$")

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.0, linestyle=_DASH_THEORY, label="n = 1"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STYLE_WATER["face"],
               markeredgecolor=STYLE_WATER["edge"], markersize=7, label="Wasser"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STYLE_HYDROGEL["face"],
               markeredgecolor=STYLE_HYDROGEL["edge"], markersize=7, label="Hydrogel"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    return fig


# ── Plot D: Qualitaets-Korrelationen (unveraendert, ohne TrackMate-Settings) ────

def _mean_brightness(tif_path: str | None, max_frames: int = BRIGHTNESS_MAX_FRAMES) -> float:
    if not tif_path:
        return float("nan")
    path = Path(tif_path)
    if not path.exists():
        return float("nan")
    try:
        with tifffile.TiffFile(path) as tif:
            n_pages = len(tif.pages)
            if n_pages == 0:
                return float("nan")
            idx = np.linspace(0, n_pages - 1, num=min(max_frames, n_pages), dtype=int)
            frames = [tif.pages[i].asarray() for i in np.unique(idx)]
        return float(np.mean([f.mean() for f in frames]))
    except Exception as exc:
        print(f"  [WARN] Helligkeit konnte nicht gelesen werden: {path.name} ({exc})")
        return float("nan")


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def build_quality_table_d0(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        if size_nm not in SIZE_COLORS:
            continue
        fit = r.get("fit_results_MSD")
        D = fit.get("D_um2_per_s") if fit else None
        if D is None:
            continue
        tracks_df = r.get("tracks_df")
        if tracks_df is not None and not tracks_df.empty:
            num_points = len(tracks_df)
            mean_track_length = float(tracks_df.groupby("particle").size().mean())
        else:
            num_points, mean_track_length = 0, float("nan")
        rows.append({
            "base_name": r.get("base_name"), "size_nm": float(size_nm), "D": float(D),
            "D_error": float(fit.get("D_error", np.nan)), "num_tracks": r.get("num_tracks", np.nan),
            "num_points": num_points, "edge_removed": r.get("edge_filter_removed", np.nan),
            "mean_track_length": mean_track_length, "mean_brightness": _mean_brightness(r.get("tif_path")),
            "sigma_loc_nm": float(fit.get("sigma_loc_nm", np.nan)),
        })
    return pd.DataFrame(rows)


def build_quality_table_20mg(results: dict) -> pd.DataFrame:
    rows = []
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        if size_nm not in TARGET_NOMINAL_SIZES_20MG | {50.0}:
            continue
        fit = r.get("fit_results_MSD")
        D = fit.get("D_um2_per_s") if fit else None
        if D is None:
            continue
        condition = condition_label_from_filename(r.get("base_name", ""))
        if condition not in CONDITION_STYLE_20MG:
            continue
        tracks_df = r.get("tracks_df")
        if tracks_df is not None and not tracks_df.empty:
            num_points = len(tracks_df)
            mean_track_length = float(tracks_df.groupby("particle").size().mean())
        else:
            num_points, mean_track_length = 0, float("nan")
        rows.append({
            "base_name": r.get("base_name", ""), "size_nm": float(size_nm), "condition": condition, "D": float(D),
            "D_error": float(fit.get("D_error", np.nan)), "num_tracks": r.get("num_tracks", np.nan),
            "num_points": num_points, "edge_removed": r.get("edge_filter_removed", np.nan),
            "mean_track_length": mean_track_length, "mean_brightness": _mean_brightness(r.get("tif_path")),
            "sigma_loc_nm": float(fit.get("sigma_loc_nm", np.nan)),
        })
    return pd.DataFrame(rows)


def plot_correlation_grid_by_size(table: pd.DataFrame, metrics: list[tuple[str, str]], title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.2 * len(metrics), 7.5), constrained_layout=True)
    for col, (key, label) in enumerate(metrics):
        x_all = table[key].to_numpy(dtype=float)
        for row, (y_key, y_label) in enumerate([("D", r"$D$ (µm²/s)"), ("D_error", r"$D_{err}$ (µm²/s)")]):
            ax = axes[row, col]
            y_all = table[y_key].to_numpy(dtype=float)
            r_val = _pearson_r(x_all, y_all)
            for size_nm, (face, edge) in SIZE_COLORS.items():
                sub = table[table["size_nm"] == size_nm]
                if sub.empty:
                    continue
                ax.scatter(sub[key], sub[y_key], s=32, marker="o", facecolor=face, edgecolor=edge,
                          linewidth=0.6, alpha=POINT_ALPHA, zorder=3)
            ax.set_xlabel(label if row == 1 else "", fontsize=8)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=9)
            ax.set_title(label if row == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.text(0.03, 0.93, f"r = {r_val:.2f}" if np.isfinite(r_val) else "r = n/a",
                   transform=ax.transAxes, fontsize=7.5, va="top",
                   bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.8))
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=face, markeredgecolor=edge, markersize=7,
              label=f"{size_nm:.0f} nm")
        for size_nm, (face, edge) in SIZE_COLORS.items() if size_nm in table["size_nm"].unique()
    ]
    fig.suptitle(f"{title}\nr = Pearson-Korrelationskoeffizient (pro Panel, x vs. y)", fontsize=10)
    fig.legend(handles=legend_elements, loc="outside lower center", ncol=max(len(legend_elements), 1),
              frameon=False, fontsize=9)
    return fig


def plot_correlation_grid_by_condition(table: pd.DataFrame, metrics: list[tuple[str, str]], title: str) -> plt.Figure:
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.2 * len(metrics), 7.5), constrained_layout=True)
    for col, (key, label) in enumerate(metrics):
        x_all = table[key].to_numpy(dtype=float)
        for row, (y_key, y_label) in enumerate([("D", r"$D$ (µm²/s)"), ("D_error", r"$D_{err}$ (µm²/s)")]):
            ax = axes[row, col]
            y_all = table[y_key].to_numpy(dtype=float)
            r_val = _pearson_r(x_all, y_all)
            for condition, cond_style in CONDITION_STYLE_20MG.items():
                for size_nm, marker in SIZE_MARKER_20MG.items():
                    sub = table[(table["size_nm"] == size_nm) & (table["condition"] == condition)]
                    if sub.empty:
                        continue
                    ax.scatter(sub[key], sub[y_key], s=32, marker=marker, facecolor=cond_style["face"],
                              edgecolor=cond_style["edge"], linewidth=0.6, alpha=POINT_ALPHA, zorder=3)
            ax.set_xlabel(label if row == 1 else "", fontsize=8)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=9)
            ax.set_title(label if row == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.text(0.03, 0.93, f"r = {r_val:.2f}" if np.isfinite(r_val) else "r = n/a",
                   transform=ax.transAxes, fontsize=7.5, va="top",
                   bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.8))
    present_sizes = set(table["size_nm"].unique())
    present_conditions = set(table["condition"].unique())
    legend_elements = [
        Line2D([0], [0], marker=marker, color="w", markerfacecolor="#888888", markeredgecolor="#444444",
              markersize=7, label=SIZE_LABEL_20MG[size_nm])
        for size_nm, marker in SIZE_MARKER_20MG.items() if size_nm in present_sizes
    ] + [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=cond_style["face"],
              markeredgecolor=cond_style["edge"], markersize=7, label=condition)
        for condition, cond_style in CONDITION_STYLE_20MG.items() if condition in present_conditions
    ]
    fig.suptitle(f"{title}\nr = Pearson-Korrelationskoeffizient (pro Panel, x vs. y)", fontsize=10)
    fig.legend(handles=legend_elements, loc="outside lower center", ncol=max(len(legend_elements), 1),
              frameon=False, fontsize=9)
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    d0_results = _load_pickle(CACHE_D0, "MSD_FromTrackmate_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    hydrogel_results = _load_pickle(CACHE_20MG, "MSD_FromTrackmate_20mg.py")
    print(f"Cache geladen: {CACHE_20MG} ({len(hydrogel_results)} Dateien)")
    locerr_df = _load_pickle(CACHE_LOCERR, "Loc_Error_Analyse_immob_particle.py")
    print(f"Cache geladen: {CACHE_LOCERR} ({len(locerr_df)} Tracks)")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    file_correction = build_file_correction_table(d0_results, hydrogel_results, locerr_df)
    print(f"Datei-Korrektur (20/50 nm, Endresultate): {len(file_correction)} Dateien.")
    file_standard = build_file_standard_table(d0_results, hydrogel_results)
    print(f"Datei-Standardwerte (alle Groessen): {len(file_standard)} Dateien.")

    print("Tracklängen-Bins: Ensemble-D/n je Datei x Bin wird neu gefittet (kann etwas dauern)...")
    bin_water = build_length_bin_table(d0_results, "water")
    bin_gel = build_length_bin_table(hydrogel_results, "hydrogel")
    length_bin_table = pd.concat([bin_water, bin_gel], ignore_index=True)
    print(f"Tracklängen-Bins: {len(length_bin_table)} (Datei x Bin)-Kombinationen erfolgreich gefittet.")

    with plt.rc_context(_RC):
        for fig, filename in (
            (plot_correction_deviation(file_correction), "A_correction_deviation_per_file.png"),
            (plot_length_bins(length_bin_table), "B_length_bin_comparison.png"),
            (plot_d_vs_n(file_standard), "C_D_vs_n_scatter.png"),
        ):
            safe_savefig(fig, SAVE_PATH / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot gespeichert: {SAVE_PATH / filename}")

        table_d0 = build_quality_table_d0(d0_results)
        table_20mg = build_quality_table_20mg(hydrogel_results)
        for fig, filename in (
            (plot_correlation_grid_by_size(table_d0, METRICS,
             "D_MSD/D_error vs. Datei-Metriken (D0, Wasser)"), "D_correlations_quality_d0.png"),
            (plot_correlation_grid_by_condition(table_20mg, METRICS,
             "D_MSD/D_error vs. Datei-Metriken (20 mg/mL Hydrogel)"), "D_correlations_quality_20mg.png"),
        ):
            safe_savefig(fig, SAVE_PATH / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot gespeichert: {SAVE_PATH / filename}")

    file_correction.to_csv(SAVE_PATH / "msd_correction_comparison.csv", index=False)
    length_bin_table.to_csv(SAVE_PATH / "length_bin_comparison.csv", index=False)
    print(f"Tabellen gespeichert unter: {SAVE_PATH}")


if __name__ == "__main__":
    main()
