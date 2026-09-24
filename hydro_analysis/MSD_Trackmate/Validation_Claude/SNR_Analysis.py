"""
SNR-Uebersicht pro Partikelgroesse -- wie gut/schlecht sind die Partikel
detektierbar, je Bedingung (Wasser/Hydrogel) und Partikelgroesse separat.

Consumer-Skript: liest ausschliesslich cache/particle_size_snr.pkl
(geschrieben von Particle_Size_SNR_Compute.py --mode all; vorher ausfuehren).
Dort ist jede getrackte Detektion aller Dateien gemessen, keine Stichprobe
aus einer einzelnen Referenzdatei pro Groesse wie im frueheren
snr_signal_profile.analyze_file()-Ansatz -- daher sind jetzt auch alle
50 nm-Dateien enthalten.

SNR-Definition unveraendert: SNR = (Peak innerhalb R - mu_bg) / sigma_bg,
Annulus 1.5 R bis 3 R, andere Spots ausmaskiert, R = TrackMate-Radius. Jeder
Track zaehlt einmal (Median seiner Detektionen, Tabelle "tracks"), damit
lange Tracks die Verteilung nicht dominieren.

Ausgabe (Auswertungsbilder\\SNR_Analysis\\): snr_overview_by_size.png
(Boxplot SNR je Partikelgroesse, nach Bedingung gruppiert, nie gepoolt),
snr_values_by_size.csv (ein Wert je Track), snr_summary_by_size.csv
(Median/IQR/N je (Groesse, Bedingung) -- fuettert die Detectability-Zeile in
Validation_Summary.py und Validation_Summary_Refined.py).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import CONDITION_COLORS, _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude._plot_utils import safe_savefig

# ── Configuration ──────────────────────────────────────────────────────────────
PSS_CACHE = Path(__file__).resolve().parent.parent / "cache" / "particle_size_snr.pkl"
SAVE_PATH = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
) / "SNR_Analysis"


def load_track_snr(path: Path) -> pd.DataFrame:
    """One SNR value per track from particle_size_snr.pkl."""
    if not path.exists():
        raise FileNotFoundError(f"Nicht gefunden: {path}\nBitte zuerst Particle_Size_SNR_Compute.py --mode all ausfuehren.")
    with open(path, "rb") as f:
        tracks = pickle.load(f)["tracks"]
    df = tracks[["particle_size_nm", "condition", "file", "particle", "snr", "dog_snr", "tm_snr"]].copy()
    return df[np.isfinite(df["snr"])]


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


def main(pss_cache: Path = PSS_CACHE, output_dir: Path = SAVE_PATH) -> None:
    df = load_track_snr(pss_cache)
    summary = df.groupby(["particle_size_nm", "condition"])["snr"].agg(
        n="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
    ).reset_index()
    print(summary.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "snr_values_by_size.csv", index=False)
    summary.to_csv(output_dir / "snr_summary_by_size.csv", index=False)

    with plt.rc_context(_RC):
        fig = plot_snr_overview(df)
        fig_path = output_dir / "snr_overview_by_size.png"
        safe_savefig(fig, fig_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot gespeichert: {fig_path}")
    print(f"Tabellen gespeichert unter: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PSS_CACHE)
    parser.add_argument("--output-dir", type=Path, default=SAVE_PATH)
    args = parser.parse_args()
    main(args.input, args.output_dir)
