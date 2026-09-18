"""
Correlation of diffusion coefficient D and anomalous exponent n, individual
per-file points, colored by particle size -- one figure for D_eff (20 mg/mL
hydrogel, restricted to the 35 nm and 50 nm DLS sizes, the only sizes
currently populated in msd_20mg_files.pkl) and one for D0 (water, all
particle sizes present in msd_d0_results.pkl).

Pure consumer: reads msd_d0_results.pkl (MSD_FromTrackmate_D0.py) and
msd_20mg_files.pkl (MSD_FromTrackmate_20mg.py). For each file, D and n come
straight from that file's own fit_results_MSD["D_um2_per_s"]/["exponent"] --
no refitting, no aggregation. File-wise plot: no error bars on the
individual points (per project convention). Particle size is always shown
as the real DLS label (core.io.get_dls_labels()), never the nominal folder
name.

Colors follow Style_guide.txt §9's discrete 8-accent palette, one color per
particle size, shared between both figures.

Run MSD_FromTrackmate_D0.py and MSD_FromTrackmate_20mg.py first (or after
any raw-data change) to refresh both pickles.

Shows each figure first (plt.show(), blocking); only after the window is
closed does it save into the shared Auswertungsbilder\\
DiffusionExponent_Correlation\\ subfolder.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hydro_analysis.core.io import get_dls_labels
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC, _DASH_THEORY

# ── Configuration ──────────────────────────────────────────────────────────────
CACHE_D0   = Path(__file__).parent.parent / "cache" / "msd_d0_results.pkl"
CACHE_DEFF = Path(__file__).parent.parent / "cache" / "msd_20mg_files.pkl"
SAVE_PATH_BASE = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data\Auswertungsbilder"
)
SAVE_PATH: Path | None = SAVE_PATH_BASE / "DiffusionExponent_Correlation"

# D_eff is restricted to the sizes that actually have data (nominal 20/50 nm -> DLS 35/50 nm).
DEFF_ALLOWED_SIZES = {20.0, 50.0}

Y_MIN, Y_MAX = 0.0, 1.5
COLOR_THEORY = "black"
POINT_ALPHA = 0.6

# Style_guide.txt §9 -- discrete 8-accent palette, one entry per nominal size (nm).
SIZE_COLORS: dict[float, tuple[str, str]] = {
    20.0:   ("#0000da", "#000099"),
    50.0:   ("#004cff", "#0035b2"),
    100.0:  ("#00c4ff", "#0089b2"),
    200.0:  ("#49ffad", "#33b279"),
    500.0:  ("#adff49", "#79b233"),
    1000.0: ("#ffd700", "#b29600"),
}


def _load_cache(path: Path, compute_script: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache gefunden: {path}\nBitte zuerst {compute_script} ausführen.")
    with open(path, "rb") as f:
        return pickle.load(f)


def _collect_D_n(results: dict, allowed_sizes: set[float] | None) -> list[tuple[float, float, float]]:
    """[(particle_size_nm, D_um2_per_s, exponent), ...], one row per file with valid D and n."""
    rows: list[tuple[float, float, float]] = []
    for r in results.values():
        size_nm = r.get("particle_size_nm")
        fit = r.get("fit_results_MSD")
        if size_nm is None or fit is None:
            continue
        size_nm = float(size_nm)
        if allowed_sizes is not None and size_nm not in allowed_sizes:
            continue
        D = fit.get("D_um2_per_s")
        n_exp = fit.get("exponent")
        if D is None or n_exp is None:
            continue
        rows.append((size_nm, float(D), float(n_exp)))
    return rows


def _plot_correlation(rows: list[tuple[float, float, float]], dls_labels: dict[float, int],
                       window_title: str) -> plt.Figure:
    """No ax.set_title() -- matches Style_guide.txt §12 (axis labels + legend suffice) like the
    sibling *_vs_Size.py scatter plots. window_title only labels the interactive plot window."""
    if not rows:
        raise ValueError(f"No D/n data to plot for: {window_title}")

    fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
    fig.canvas.manager.set_window_title(window_title)

    ax.axhline(1.0, color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY, zorder=3)

    sizes_present = sorted({size for size, D, n in rows})
    legend_elements = [
        Line2D([0], [0], color=COLOR_THEORY, linewidth=1.2, linestyle=_DASH_THEORY,
               label="Free diffusion (n = 1)"),
    ]
    for size_nm in sizes_present:
        face, edge = SIZE_COLORS.get(size_nm, ("#999999", "#555555"))
        group = [(D, n) for s, D, n in rows if s == size_nm]
        xs = [D for D, n in group]
        ys = [n for D, n in group]
        ax.scatter(xs, ys, s=36, alpha=POINT_ALPHA, marker="o",
                   facecolor=face, edgecolor=edge, linewidth=0.6, zorder=5)
        label_nm = dls_labels.get(size_nm, int(size_nm))
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=face, markeredgecolor=edge,
                   markersize=7, markeredgewidth=0.6, label=f"{label_nm} nm", linestyle="None")
        )

    ax.set_xscale("log")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel(r"Diffusion coefficient $D$ (µm²/s)")
    ax.set_ylabel(r"Anomalous exponent $n$  (MSD $\propto \tau^{n}$)")
    ax.legend(handles=legend_elements, loc="best", frameon=False)

    return fig


def main() -> None:
    dls_labels = get_dls_labels()

    d0_results = _load_cache(CACHE_D0, "MSD_FromTrackmate_D0.py")
    print(f"Cache geladen: {CACHE_D0} ({len(d0_results)} Dateien)")
    deff_results = _load_cache(CACHE_DEFF, "MSD_FromTrackmate_20mg.py")
    print(f"Cache geladen: {CACHE_DEFF} ({len(deff_results)} Dateien)")

    deff_rows = _collect_D_n(deff_results, DEFF_ALLOWED_SIZES)
    d0_rows = _collect_D_n(d0_results, allowed_sizes=None)
    print(f"D_eff: {len(deff_rows)} Dateien, D0: {len(d0_rows)} Dateien.")

    with plt.rc_context(_RC):
        for rows, title, filename in (
            (deff_rows, "D vs. n correlation -- D_eff (20 mg/mL)", "deff_diffusion_exponent_correlation.png"),
            (d0_rows, "D vs. n correlation -- D0 (water)", "d0_diffusion_exponent_correlation.png"),
        ):
            fig = _plot_correlation(rows, dls_labels, title)
            plt.show()

            if SAVE_PATH is not None:
                SAVE_PATH.mkdir(parents=True, exist_ok=True)
                png_path = SAVE_PATH / filename
                fig.savefig(png_path, dpi=600, bbox_inches="tight")
                print(f"Plot gespeichert: {png_path}")


if __name__ == "__main__":
    main()
