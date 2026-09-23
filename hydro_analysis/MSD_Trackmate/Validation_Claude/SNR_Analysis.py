"""
SNR-Uebersicht pro Partikelgroesse -- wie gut/schlecht sind die Partikel
detektierbar (Wasser/Hydrogel/immobilisiert), je Partikelgroesse separat.

Neues Skript, ersetzt keine bestehenden Skripte. Reine Aggregation: ruft
snr_signal_profile.py::analyze_file() unveraendert mehrfach auf (gleiche
Session-Suche find_session_and_tif(), Isolationspruefung _is_isolated(),
Annulus-Hintergrundstatistik und Peak/Hintergrund-SNR-Definition
SNR = (Peak - mu_bg) / sigma_bg -- kein separater DoG-Filter-Response, siehe
dortiger Docstring). snr_signal_profile.py zeigt pro (Bedingung, Groesse)
nur bis zu N_PARTICLES_PER_CONDITION=5 Partikel aus EINEM zufaelligen Frame;
hier wird analyze_file() stattdessen wiederholt aufgerufen (es nutzt den
globalen, SEED=42-initialisierten random-Zustand, ein erneuter Aufruf zieht
daher jeweils einen neuen Frame), bis zu MAX_SPOTS_PER_GROUP=200 Partikel
gesammelt sind -- fuer eine echte SNR-Verteilung statt nur ein paar
Beispielwerte. Nur die in snr_signal_profile.py::FILES hinterlegten
(Bedingung, Groesse)-Referenzdateien werden verwendet (ein Movie pro
Kombination; fehlende Kombinationen siehe dortige Kommentare).

snr_signal_profile.py musste dafuer minimal angepasst werden (Code in
main() + if __name__ == "__main__" verschoben, Verhalten beim direkten
Ausfuehren unveraendert), da es zuvor seine gesamte Pipeline beim Import
ausgefuehrt haette.

Ausgabe (Auswertungsbilder\\SNR_Analysis\\): snr_overview_by_size.png
(Boxplot SNR je Partikelgroesse, nach Bedingung gruppiert, nie gepoolt),
snr_values_by_size.csv (jeder einzelne gesammelte Partikel),
snr_summary_by_size.csv (Median/IQR/N je (Groesse, Bedingung) --
fuettert die Detectability-Zeile in Validation_Summary.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import (
    FILES, analyze_file, CONDITION_COLORS, _RC,
)
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig

# ── Configuration ──────────────────────────────────────────────────────────────
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "SNR_Analysis"

MAX_SPOTS_PER_GROUP = 200
MAX_CALLS_PER_GROUP = 100   # analyze_file() liefert bis zu 5 Partikel je Aufruf; Sicherheitsobergrenze


def collect_snr_distribution(tracks_xml: Path, max_spots: int) -> list[dict]:
    """Ruft analyze_file() wiederholt auf (jeweils ein neuer zufaelliger Frame,
    siehe Modul-Docstring), bis max_spots Partikel gesammelt sind oder das
    Aufruf-Limit erreicht ist."""
    results: list[dict] = []
    calls = 0
    while len(results) < max_spots and calls < MAX_CALLS_PER_GROUP:
        batch = analyze_file(tracks_xml, n_particles=5)
        calls += 1
        results.extend(batch)
    return results[:max_spots]


def plot_snr_overview(df: pd.DataFrame) -> plt.Figure:
    sizes = sorted(df["particle_size_nm"].unique())
    conditions = sorted(df["condition"].unique())
    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    positions = np.arange(len(sizes))
    width = 0.8 / max(len(conditions), 1)
    for i, condition in enumerate(conditions):
        color = CONDITION_COLORS.get(condition, "#808080")
        offsets = positions + (i - (len(conditions) - 1) / 2) * width
        box_data = [df.loc[(df["particle_size_nm"] == s) & (df["condition"] == condition), "snr"].to_numpy()
                    for s in sizes]
        bp = ax.boxplot(box_data, positions=offsets, widths=width * 0.85, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor(color)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(color)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{s:.0f} nm" for s in sizes])
    ax.set_xlabel("Nominelle Partikelgröße")
    ax.set_ylabel(r"SNR $= (I_{peak} - \mu_{bg}) / \sigma_{bg}$")
    handles = [Patch(facecolor=CONDITION_COLORS.get(c, "#808080"), alpha=0.5, label=c) for c in conditions]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=False)
    return fig


def main() -> None:
    sizes = sorted({s for (_, s) in FILES})
    rows = []
    for size_nm in sizes:
        conditions = sorted(c for (c, s) in FILES if s == size_nm)
        for condition in conditions:
            tracks_xml = Path(FILES[(condition, size_nm)])
            if not tracks_xml.exists():
                print(f"  [SKIP] {condition} {size_nm:.0f} nm: {tracks_xml} nicht gefunden")
                continue
            spots = collect_snr_distribution(tracks_xml, MAX_SPOTS_PER_GROUP)
            print(f"  {size_nm:>6.0f} nm  {condition:<12}: {len(spots)} Partikel gesammelt")
            for r in spots:
                rows.append({"particle_size_nm": size_nm, "condition": condition, "snr": r["snr"],
                             "frame": r["frame"], "spot_id": r["spot_id"]})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Keine SNR-Werte gesammelt -- siehe [SKIP]/[UEBERSPRUNGEN]-Meldungen oben.")

    summary = df.groupby(["particle_size_nm", "condition"])["snr"].agg(
        n="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
    ).reset_index()
    print(summary.to_string(index=False))

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_PATH / "snr_values_by_size.csv", index=False)
    summary.to_csv(SAVE_PATH / "snr_summary_by_size.csv", index=False)

    with plt.rc_context(_RC):
        fig = plot_snr_overview(df)
        fig_path = SAVE_PATH / "snr_overview_by_size.png"
        safe_savefig(fig, fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {fig_path}")
    print(f"Tabellen gespeichert unter: {SAVE_PATH}")


if __name__ == "__main__":
    main()
