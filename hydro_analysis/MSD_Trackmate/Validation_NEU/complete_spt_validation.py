"""Reproducible SPT validation from complete TrackMate XML sessions and raw TIFFs.

This is an intentionally standalone implementation. It neither imports nor
modifies the legacy SPT analysis, existing validation scripts, or their caches.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_laplace
from scipy.stats import gaussian_kde, linregress, spearmanr
import tifffile

try:
    from hydro_analysis.core.io import parse_rec_file
except ModuleNotFoundError:  # direct script execution without editable install
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from hydro_analysis.core.io import parse_rec_file

try:
    from .trackmate_session import TrackMateSession, read_trackmate_session
except ImportError:  # supports direct `python path/to/complete_spt_validation.py`
    from trackmate_session import TrackMateSession, read_trackmate_session


RC = {"font.family": "sans-serif", "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"],
      "axes.linewidth": .8, "xtick.direction": "in", "ytick.direction": "in",
      "xtick.top": True, "ytick.right": True, "xtick.labelsize": 9,
      "ytick.labelsize": 9, "axes.labelsize": 10, "legend.fontsize": 8}
FIT_LAGS = 4
MIN_TRACK_POINTS = 10
CROP_SIZE = 21
PAD = 10
INNER_RADIUS = 3
BACKGROUND_PER_FRAME = 3
BOOTSTRAPS = 250


def save(fig: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def histogram_with_kde(ax: plt.Axes, values: np.ndarray, *, color: str, dark: str,
                       label: str, bins: int = 40) -> None:
    """Density histogram with a KDE overlay; leave degenerate samples explicit."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ax.hist(values, bins=bins, density=True, color=color, edgecolor=dark, alpha=.55, label=label)
    if len(values) >= 3 and np.ptp(values) > 0:
        x = np.linspace(values.min(), values.max(), 400)
        ax.plot(x, gaussian_kde(values)(x), color=dark, lw=2.2, label=f"KDE: {label}")


def numeric(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_manifest(path: Path) -> list[dict]:
    table = pd.read_csv(path)
    needed = {"xml_path", "tiff_path", "condition", "particle_size_nm"}
    missing = needed - set(table.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    records: list[dict] = []
    for row in table.to_dict("records"):
        row["xml_path"] = Path(row["xml_path"])
        row["tiff_path"] = Path(row["tiff_path"])
        row["condition"] = str(row["condition"]).strip().lower()
        row["particle_size_nm"] = float(row["particle_size_nm"])
        if row["condition"] not in {"water", "hydrogel", "immobilized"}:
            raise ValueError(f"Unsupported condition {row['condition']!r} in {row['xml_path']}")
        if not row["xml_path"].is_file() or not row["tiff_path"].is_file():
            raise FileNotFoundError(f"Missing XML/TIFF pair: {row['xml_path']} | {row['tiff_path']}")
        row["movie"] = row.get("label") or row["xml_path"].stem
        row["session"] = read_trackmate_session(row["xml_path"])
        session: TrackMateSession = row["session"]
        spatial = session.spatialunits.strip().lower()
        temporal = session.timeunits.strip().lower()
        # A value of 1 with TrackMate's pixel/frame units is an uncalibrated
        # session, not a 1 µm pixel or a 1 s exposure. In this explicitly
        # documented fallback, read the matching PCO .rec metadata through
        # the repository's existing parser. XML remains the source of spots,
        # tracks, and all TrackMate features.
        if spatial in {"", "pixel", "pixels", "px"} or temporal in {"", "frame", "frames"}:
            candidates = [Path(str(row["tiff_path"]) + ".rec"), row["tiff_path"].with_suffix(".rec")]
            rec_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            rec = parse_rec_file(rec_path) if rec_path is not None else {}
            if rec.get("mpp") is None or rec.get("fps") is None:
                raise ValueError(
                    f"{row['xml_path'].name}: XML is uncalibrated and no readable matching .rec calibration was found "
                    f"for {row['tiff_path'].name}."
                )
            row.update({"calibration_source": ".rec fallback (XML pixel/frame)", "rec_path": str(rec_path),
                        "mpp_um": float(rec["mpp"]), "dt_s": 1.0 / float(rec["fps"]),
                        "exposure_ms": rec.get("exposure_ms")})
        else:
            row.update({"calibration_source": "TrackMate XML", "rec_path": "", "mpp_um": session.pixelwidth_um,
                        "dt_s": session.timeinterval_s, "exposure_ms": np.nan})
        records.append(row)
    return records


def data_card(records: list[dict], out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        s: TrackMateSession = r["session"]
        with tifffile.TiffFile(r["tiff_path"]) as tif:
            series = tif.series[0]
            tiff_shape, tiff_dtype, n_pages = tuple(series.shape), str(series.dtype), len(tif.pages)
        rows.append({"movie": r["movie"], "condition": r["condition"], "particle_size_nm": r["particle_size_nm"],
                     "xml_path": str(r["xml_path"]), "tiff_path": str(r["tiff_path"]),
                     "filtered_trajectories": s.trajectories["trajectory_id"].nunique(),
                     "spots": len(s.spots), "frames": s.spots["frame"].nunique(),
                     "pixelwidth_um": r["mpp_um"], "pixelheight_um": r["mpp_um"],
                     "timeinterval_s": r["dt_s"], "calibration_source": r["calibration_source"],
                     "rec_path": r["rec_path"], "exposure_ms": r["exposure_ms"], "spatialunits": s.spatialunits,
                     "timeunits": s.timeunits, "tiff_shape": tiff_shape, "tiff_dtype": tiff_dtype, "tiff_pages": n_pages,
                     "spot_features": "; ".join(s.spot_features),
                     "snr_available": "SNR" in s.spot_features,
                     "intensity_available": any(k.startswith(("MEAN_INTENSITY", "MAX_INTENSITY", "TOTAL_INTENSITY")) for k in s.spot_features)})
    card = pd.DataFrame(rows)
    card.to_csv(out / "data_card.csv", index=False)
    return card


def track_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        s: TrackMateSession = r["session"]
        for ident, g in s.trajectories.groupby("trajectory_id"):
            g = g.sort_values("frame").reset_index(drop=True)
            if s.spatialunits.strip().lower() in {"", "pixel", "pixels", "px"}:
                g = g.copy(); g["x"] *= r["mpp_um"]; g["y"] *= r["mpp_um"]
            if len(g) < 2:
                continue
            rows.append({"movie": r["movie"], "condition": r["condition"], "particle_size_nm": r["particle_size_nm"],
                         "trajectory_id": ident, "n_localizations": len(g),
                         "duration_s": (g.frame.iloc[-1] - g.frame.iloc[0]) * r["dt_s"],
                         "frame_span": int(g.frame.iloc[-1] - g.frame.iloc[0] + 1), "dt_s": r["dt_s"],
                         "trajectory": g[["frame", "x", "y"]].copy()})
    return pd.DataFrame(rows)


def individual_msd(track: pd.DataFrame, dt: float, max_lag: int = FIT_LAGS) -> pd.Series:
    """Overlapping exact-frame-lag MSD, retaining actual experimental frames."""
    pos = track.set_index("frame")[["x", "y"]]
    values = {}
    for lag in range(1, max_lag + 1):
        common = pos.index.intersection(pos.index - lag)
        if len(common):
            delta = pos.loc[common].to_numpy() - pos.loc[common + lag].to_numpy()
            values[lag * dt] = float(np.mean(np.sum(delta ** 2, axis=1)))
    return pd.Series(values, dtype=float).sort_index()


def ensemble_msd(tracks: pd.DataFrame, max_lag: int = FIT_LAGS) -> pd.Series:
    parts = [individual_msd(row.trajectory, row.dt_s, max_lag) for row in tracks.itertuples()]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).sort_index()


def power_fit(msd: pd.Series, n_lags: int = FIT_LAGS) -> dict[str, float]:
    window = msd.iloc[:n_lags]
    if len(window) != n_lags or (window <= 0).any() or not np.isfinite(window).all():
        return {"D_um2_s": np.nan, "D_err": np.nan, "n": np.nan, "n_err": np.nan,
                "fit_status": "nicht definiert: nicht-positive/fehlende MSD im festgelegten Fitbereich"}
    lag = np.asarray(window.index, dtype=float)
    log_lag = np.log(lag)
    slope, intercept, _, _, stderr = linregress(log_lag, np.log(window.values))
    residual = np.log(window.values) - (slope * log_lag + intercept)
    a_err = np.exp(intercept) * np.sqrt(np.sum(residual**2) / max(len(window) - 2, 1) /
                                        np.sum((log_lag - log_lag.mean())**2))
    return {"D_um2_s": float(np.exp(intercept) / 4), "D_err": float(a_err / 4),
            "n": float(slope), "n_err": float(stderr), "fit_status": "ok"}


def localization(tracks: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, dict[float, float]]:
    rows = []
    for row in tracks[tracks.condition.eq("immobilized")].itertuples():
        if row.n_localizations < 3:
            continue
        xy = row.trajectory[["x", "y"]].to_numpy(float)
        sx, sy = np.std(xy, axis=0, ddof=1) * 1000
        step = np.diff(xy, axis=0)
        f2f = np.std(step, axis=0, ddof=1) * 1000 / np.sqrt(2) if len(step) >= 2 else [np.nan, np.nan]
        rows.append({"movie": row.movie, "particle_size_nm": row.particle_size_nm, "trajectory_id": row.trajectory_id,
                     "n_localizations": row.n_localizations, "sigma_x_nm": sx, "sigma_y_nm": sy,
                     "sigma_loc_nm": np.sqrt((sx**2 + sy**2) / 2), "sigma_x_f2f_nm": f2f[0], "sigma_y_f2f_nm": f2f[1]})
    per_track = pd.DataFrame(rows)
    per_track.to_csv(out / "localization_per_track.csv", index=False)
    if per_track.empty:
        sigma = {float(size): 10.0 for size in tracks.particle_size_nm.unique()}
        summary = pd.DataFrame({"particle_size_nm": list(sigma), "sigma_loc_median_nm": list(sigma.values()),
                                "source": "Annahme: 10 nm oberer Richtwert; keine Fixiert-Partikel-XML"})
    else:
        summary = per_track.groupby("particle_size_nm").agg(n_tracks=("trajectory_id", "count"),
            sigma_x_mean_nm=("sigma_x_nm", "mean"), sigma_x_sd_nm=("sigma_x_nm", "std"),
            sigma_y_mean_nm=("sigma_y_nm", "mean"), sigma_y_sd_nm=("sigma_y_nm", "std"),
            sigma_loc_median_nm=("sigma_loc_nm", "median"), sigma_loc_mean_nm=("sigma_loc_nm", "mean"),
            sigma_loc_sd_nm=("sigma_loc_nm", "std")).reset_index()
        summary["source"] = "Fixiert-Partikel, pro-Track-Statik"
        sigma = dict(zip(summary.particle_size_nm, summary.sigma_loc_median_nm))
        fig, ax = plt.subplots(figsize=(7.15, 5), constrained_layout=True)
        histogram_with_kde(ax, per_track.sigma_x_nm, color="#0000da", dark="#000099", label=r"$\sigma_x$")
        histogram_with_kde(ax, per_track.sigma_y_nm, color="#da0000", dark="#990000", label=r"$\sigma_y$")
        ax.set(xlabel="Lokalisationspräzision (nm)", ylabel="Dichte")
        ax.legend(); save(fig, out, "lokalisationsfehler_verteilung")
    summary.to_csv(out / "localization_summary.csv", index=False)
    return per_track, {float(k): float(v) / 1000 for k, v in sigma.items()}


def task_a(tracks: pd.DataFrame, out: Path) -> pd.DataFrame:
    _, sigma = localization(tracks, out)
    rows = []
    for (condition, size), group in tracks[tracks.condition.isin(["water", "hydrogel"])].groupby(["condition", "particle_size_nm"]):
        raw = ensemble_msd(group)
        s = sigma.get(float(size), .010)
        corrected = raw - 4 * s**2
        before, after = power_fit(raw), power_fit(corrected)
        intercept = np.polyfit(raw.index[:FIT_LAGS], raw.values[:FIT_LAGS], 1)[1] if len(raw) >= 2 else np.nan
        rows.append({"condition": condition, "particle_size_nm": size, "n_trajectories": len(group),
                     "fit_lags": FIT_LAGS, "sigma_loc_um": s, "offset_4sigma2_um2": 4*s**2,
                     "D_before": before["D_um2_s"], "D_before_err": before["D_err"], "n_before": before["n"],
                     "n_before_err": before["n_err"], "D_after": after["D_um2_s"], "D_after_err": after["D_err"],
                     "n_after": after["n"], "n_after_err": after["n_err"], "corrected_fit_status": after["fit_status"],
                     "intercept_b_um2": intercept, "D_relative_change": (after["D_um2_s"]-before["D_um2_s"])/before["D_um2_s"],
                     "n_relative_change": (after["n"]-before["n"])/before["n"]})
        fig, ax = plt.subplots(figsize=(7.15, 5), constrained_layout=True)
        ax.loglog(raw.index, raw.values, "o-", label="gemessene eMSD")
        positive = corrected > 0
        ax.loglog(corrected.index[positive], corrected[positive], "s-", label="korrigierte eMSD")
        ax.axvspan(raw.index[0], raw.index[min(FIT_LAGS, len(raw))-1], color=".8", alpha=.5, label="Fitbereich")
        ax.set(xlabel="Zeitverzögerung (s)", ylabel="MSD (µm²)")
        ax.legend(); save(fig, out, f"msd_korrektur_{condition}_{int(size)}nm")
    table = pd.DataFrame(rows)
    table.to_csv(out / "msd_correction_summary.csv", index=False)
    return table


def task_b(tracks: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    candidate_mobile = tracks[tracks.condition.isin(["water", "hydrogel"])]
    mobile = candidate_mobile[candidate_mobile.n_localizations >= MIN_TRACK_POINTS].copy()
    candidate_mobile.assign(included=candidate_mobile.n_localizations >= MIN_TRACK_POINTS).groupby(
        ["condition", "particle_size_nm", "included"], as_index=False
    ).size().rename(columns={"size": "n_trajectories"}).to_csv(out / "track_length_inclusion_summary.csv", index=False)
    for row in mobile.itertuples():
        msd = individual_msd(row.trajectory, row.dt_s)
        fit = power_fit(msd)
        rows.append({"movie": row.movie, "condition": row.condition, "particle_size_nm": row.particle_size_nm,
                     "trajectory_id": row.trajectory_id, "n_localizations": row.n_localizations,
                     "duration_s": row.duration_s, "fit_lags": FIT_LAGS, **fit})
    individual = pd.DataFrame(rows)
    individual.to_csv(out / "track_length_individual.csv", index=False)
    corr_rows, binned_rows, d0_summary_rows = [], [], []
    for key, g in individual[individual.fit_status.eq("ok")].groupby(["condition", "particle_size_nm"]):
        for value in ("D_um2_s", "n"):
            rho, p = spearmanr(g.n_localizations, g[value]) if len(g) >= 3 else (np.nan, np.nan)
            corr_rows.append({"condition": key[0], "particle_size_nm": key[1], "value": value, "n_tracks": len(g),
                              "spearman_rho": rho, "spearman_p": p})
            if key[0] == "water":
                d0_summary_rows.append({"particle_size_nm": key[1], "value": value, "n_tracks": len(g),
                    "spearman_rho": rho, "spearman_p": p,
                    "direction": "positiv" if rho > 0 else "negativ" if rho < 0 else "null"})
            edges = np.unique(np.geomspace(max(2, g.n_localizations.min()), g.n_localizations.max() + 1, 7).astype(int))
            for lo, hi in zip(edges[:-1], edges[1:]):
                values = g.loc[g.n_localizations.between(lo, hi, inclusive="left"), value]
                if len(values): binned_rows.append({"condition": key[0], "particle_size_nm": key[1], "value": value,
                    "length_min": lo, "length_max": hi, "n_tracks": len(values), "mean": values.mean(),
                    "median": values.median(), "sd": values.std()})
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, value, label in zip(axes, ("D_um2_s", "n"), ("D (µm²/s)", "n")):
            ax.scatter(g.n_localizations, g[value], s=14, alpha=.4)
            ordered = g.sort_values("n_localizations")
            median = ordered[value].rolling(max(5, min(31, len(ordered)//8*2+1)), center=True).median()
            ax.plot(ordered.n_localizations, median, color="#990000", lw=1.8, label="gleitender Median")
            ax.set_xscale("log"); ax.set(xlabel="Tracklänge (Lokalisierungen)", ylabel=label)
        save(fig, out, f"tracklaenge_{key[0]}_{int(key[1])}nm")
        fig, ax = plt.subplots(figsize=(7.15, 5), constrained_layout=True)
        histogram_with_kde(ax, g.n_localizations.to_numpy(), color="#3B6E8C", dark="#2A4F66",
                           label="Tracklänge")
        ax.set(xlabel="Tracklänge (Lokalisierungen)", ylabel="Dichte")
        ax.legend()
        save(fig, out, f"tracklaengen_histogramm_{key[0]}_{int(key[1])}nm")
    correlation = pd.DataFrame(corr_rows); correlation.to_csv(out / "track_length_correlation.csv", index=False)
    pd.DataFrame(binned_rows).to_csv(out / "track_length_binned.csv", index=False)
    d0_summary = pd.DataFrame(d0_summary_rows)
    d0_summary.to_csv(out / "d0_track_length_bias_summary.csv", index=False)
    water = individual[(individual.condition == "water") & individual.fit_status.eq("ok")]
    if not water.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, value, ylabel in zip(axes, ("D_um2_s", "n"), ("D₀ (µm²/s)", "n₀")):
            for size, group in water.groupby("particle_size_nm"):
                ax.scatter(group.n_localizations, group[value], s=26, alpha=.55, label=f"{size:.0f} nm")
            ax.set_xscale("log")
            ax.set(xlabel="Tracklänge (Lokalisierungen)", ylabel=ylabel)
            ax.legend(title="Wasser")
        fig.suptitle("D₀: Tracklänge und individuelle Fitparameter")
        save(fig, out, "d0_tracklaengen_bias")
    sensitivity_rows = []
    rng = np.random.default_rng(20260917)
    for key, g in mobile.groupby(["condition", "particle_size_nm"]):
        for threshold in (5, 10, 15, 20, 30, 50):
            keep = g[g.n_localizations >= threshold]
            fit = power_fit(ensemble_msd(keep)) if not keep.empty else power_fit(pd.Series(dtype=float))
            per_track = individual[(individual.condition == key[0]) & (individual.particle_size_nm == key[1]) &
                                   (individual.n_localizations >= threshold) & individual.fit_status.eq("ok")]
            movies = keep.movie.unique()
            boot = []
            if len(movies) >= 2:
                for _ in range(BOOTSTRAPS):
                    selected = rng.choice(movies, len(movies), replace=True)
                    sample = pd.concat([keep[keep.movie.eq(m)] for m in selected], ignore_index=True)
                    boot.append(power_fit(ensemble_msd(sample)))
            sensitivity_rows.append({"condition": key[0], "particle_size_nm": key[1], "minimum_track_length": threshold,
                "n_trajectories": len(keep), "n_movies": len(movies), **fit,
                "D_track_mean": per_track.D_um2_s.mean(), "D_track_median": per_track.D_um2_s.median(),
                "n_track_mean": per_track.n.mean(), "n_track_median": per_track.n.median(),
                "D_bootstrap_sd": np.nanstd([x["D_um2_s"] for x in boot]) if boot else np.nan,
                "n_bootstrap_sd": np.nanstd([x["n"] for x in boot]) if boot else np.nan})
    pd.DataFrame(sensitivity_rows).to_csv(out / "min_track_length_sensitivity.csv", index=False)
    return individual


def to_pixel(value: float, spatial_unit: str, pixel_size_um: float) -> float:
    unit = spatial_unit.lower()
    if "pixel" in unit: return value
    if unit in {"nm", "nanometer", "nanometers"}: return value / (pixel_size_um * 1000)
    return value / pixel_size_um  # TrackMate sessions normally report µm


def crop_metrics(image: np.ndarray, x: int, y: int, radius: float) -> dict | None:
    half = CROP_SIZE // 2; margin = half + PAD
    if x-margin < 0 or y-margin < 0 or x+margin >= image.shape[1] or y+margin >= image.shape[0]: return None
    padded = image[y-margin:y+margin+1, x-margin:x+margin+1].astype(float)
    core = padded[PAD:PAD+CROP_SIZE, PAD:PAD+CROP_SIZE]
    yy, xx = np.mgrid[:CROP_SIZE, :CROP_SIZE]; distance = np.hypot(xx-half, yy-half)
    inner, outer = distance <= INNER_RADIUS, (distance > INNER_RADIUS) & (distance <= half)
    def measure(array: np.ndarray) -> tuple[float, float, float, float]:
        peak, bg, sd = float(array[inner].max()), float(np.median(array[outer])), float(np.std(array[outer], ddof=1))
        return peak, bg, sd, (peak-bg)/sd if sd > 0 else np.nan
    peak, bg, sd, snr = measure(core)
    log = -gaussian_laplace(padded, sigma=max(radius, .1)/np.sqrt(2))[PAD:PAD+CROP_SIZE, PAD:PAD+CROP_SIZE]
    lpeak, lbg, lsd, lsnr = measure(log)
    return {"local_max": peak, "background": bg, "background_sd": sd, "contrast": (peak-bg)/bg if bg else np.nan,
            "snr": snr, "log_peak": lpeak, "log_background": lbg, "log_background_sd": lsd, "log_snr": lsnr}


def task_c(records: list[dict], out: Path) -> pd.DataFrame:
    rows = []
    for r in records:
        if r["condition"] not in {"water", "hydrogel"} or r["particle_size_nm"] != 35: continue
        s: TrackMateSession = r["session"]; spots = s.trajectories.copy()
        spots["x_px"] = spots.x.map(lambda x: to_pixel(x, s.spatialunits, s.pixelwidth_um))
        spots["y_px"] = spots.y.map(lambda y: to_pixel(y, s.spatialunits, s.pixelheight_um))
        radius_raw = np.nanmedian(pd.to_numeric(spots.get("RADIUS", pd.Series([3.])), errors="coerce"))
        radius = to_pixel(radius_raw, s.spatialunits, s.pixelwidth_um) if np.isfinite(radius_raw) else 3.
        with tifffile.TiffFile(r["tiff_path"]) as tif:
            n_pages = len(tif.pages)
            for frame, g in spots.groupby("frame"):
                if frame >= n_pages: continue
                image = tif.asarray(key=int(frame))
                if image.ndim != 2: raise ValueError(f"{r['movie']}: TIFF must be a 2-D time stack.")
                xy = g[["x_px", "y_px"]].to_numpy(float)
                for point in g.itertuples():
                    x, y = round(point.x_px), round(point.y_px); metrics = crop_metrics(image, x, y, radius)
                    if metrics: rows.append({"movie": r["movie"], "condition": r["condition"], "frame": frame,
                        "particle_id": point.trajectory_id, "x_px": x, "y_px": y, "is_background": False,
                        "xml_snr": numeric(getattr(point, "SNR", np.nan)), "quality": numeric(getattr(point, "QUALITY", np.nan)),
                        "nearest_neighbour_px": np.min(np.hypot(xy[:,0]-x, xy[:,1]-y)[np.hypot(xy[:,0]-x, xy[:,1]-y)>0]) if len(xy)>1 else np.inf,
                        "radius_px": radius, "xml_path": str(r["xml_path"]), "tiff_path": str(r["tiff_path"]), **metrics})
                seed = int.from_bytes(hashlib.sha256(f"{r['xml_path']}:{frame}".encode()).digest()[:8], "little")
                rng = np.random.default_rng(seed); margin = CROP_SIZE//2+PAD; placed = 0
                for _ in range(100):
                    if placed == BACKGROUND_PER_FRAME: break
                    x, y = int(rng.integers(margin, image.shape[1]-margin)), int(rng.integers(margin, image.shape[0]-margin))
                    if len(xy) and np.min(np.hypot(xy[:,0]-x, xy[:,1]-y)) < CROP_SIZE: continue
                    metrics = crop_metrics(image, x, y, radius)
                    if metrics: rows.append({"movie": r["movie"], "condition": r["condition"], "frame": frame,
                        "particle_id": "", "x_px": x, "y_px": y, "is_background": True, "xml_snr": np.nan,
                        "quality": np.nan, "nearest_neighbour_px": np.nan, "radius_px": radius,
                        "xml_path": str(r["xml_path"]), "tiff_path": str(r["tiff_path"]), **metrics}); placed += 1
    quality = pd.DataFrame(rows); quality.to_csv(out / "detection_quality.csv", index=False); return quality


def select_and_plot(quality: pd.DataFrame, out: Path) -> None:
    detections = quality[~quality.is_background].copy(); bg = quality[quality.is_background].copy()
    eligible = detections[(detections.snr > 0) & (detections.log_snr > 0) & (detections.nearest_neighbour_px >= CROP_SIZE)]
    if eligible.empty or bg.empty: raise ValueError("No eligible 35-nm detections/background crops for automatic selection.")
    challenging = eligible.iloc[(eligible.snr - eligible.snr.quantile(.15)).abs().argsort().iloc[0]]
    # All three visual examples must share an acquisition context. This avoids
    # comparing camera backgrounds or gain settings across water and hydrogel.
    context = eligible[eligible.movie.eq(challenging.movie)]
    if context.empty:
        context = eligible[eligible.condition.eq(challenging.condition)]
    good = context.iloc[(context.snr - context.snr.quantile(.90)).abs().argsort().iloc[0]]
    candidates = bg[(bg.movie == challenging.movie) & (bg.frame == challenging.frame)]
    no_particle = candidates.iloc[(candidates.background_sd-challenging.background_sd).abs().argsort().iloc[0]] if not candidates.empty else bg.iloc[(bg.background_sd-challenging.background_sd).abs().argsort().iloc[0]]
    selected = pd.DataFrame([good, challenging, no_particle], index=["good", "challenging", "no_particle"])
    selected["selection_criteria"] = ["SNR-90. Perzentil; positiver LoG-SNR; isoliert", "SNR-15. Perzentil; positiver LoG-SNR; isoliert", "TrackMate-negativ; Hintergrund-SD passend"]
    selected.to_csv(out / "selection_manifest.csv")
    fig, ax = plt.subplots(figsize=(7.15, 5), constrained_layout=True)
    ax.scatter(detections.snr, detections.log_snr, s=10, alpha=.25, label="Detektionen")
    ax.scatter(bg.snr, bg.log_snr, s=10, alpha=.2, label="TrackMate-negative Bereiche")
    for label, row in selected.iterrows(): ax.scatter(row.snr, row.log_snr, s=70, label=label)
    ax.set(xlabel="Rohbild-SNR", ylabel="LoG-SNR"); ax.legend(); save(fig, out, "log_snr_gegen_roh_snr")
    # Re-open selected raw crops so the CSV remains the only persistent selection state.
    fig = plt.figure(figsize=(10.5, 10), constrained_layout=True)
    raw = [] ; log = []
    for _, row in selected.iterrows():
        image = tifffile.imread(row.tiff_path, key=int(row.frame)); x, y = int(row.x_px), int(row.y_px)
        half=CROP_SIZE//2; raw.append(image[y-half:y+half+1, x-half:x+half+1].astype(float))
        log.append(-gaussian_laplace(raw[-1], sigma=float(row.radius_px)/np.sqrt(2)))
    rv=(min(x.min() for x in raw), max(x.max() for x in raw)); lv=(min(x.min() for x in log), max(x.max() for x in log))
    yy, xx=np.mgrid[:CROP_SIZE,:CROP_SIZE]
    for i, (label, rimg, limg) in enumerate(zip(selected.index, raw, log)):
        a=fig.add_subplot(4,3,i+1); a.imshow(rimg,cmap="gray",vmin=rv[0],vmax=rv[1]); a.set_title(label); a.axis("off")
        a=fig.add_subplot(4,3,3+i+1,projection="3d"); a.plot_surface(xx,yy,rimg,cmap="gray",vmin=rv[0],vmax=rv[1]); a.set_xticks([]);a.set_yticks([])
        a=fig.add_subplot(4,3,6+i+1); a.imshow(limg,cmap="magma",vmin=lv[0],vmax=lv[1]); a.axis("off")
        a=fig.add_subplot(4,3,9+i+1,projection="3d"); a.plot_surface(xx,yy,limg,cmap="magma",vmin=lv[0],vmax=lv[1]);a.set_xticks([]);a.set_yticks([])
    fig.suptitle("Rohfluoreszenz → Intensitätslandschaft → LoG-Antwort\n(gemeinsame Farbskalen je Zeile; fester ROI, keine Partikelgröße)")
    save(fig, out, "detectability_composite")


def report(card: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, root: Path) -> None:
    def csv_block(frame: pd.DataFrame) -> str:
        return "```csv\n" + frame.to_csv(index=False) + "```"
    detect = c.groupby("is_background")[["snr", "log_snr", "background_sd"]].median().reset_index()
    lines = ["# Befunde", "", "## Datenkarte", "", csv_block(card), "", "## Lokalisation und MSD-Korrektur", "", csv_block(a), "", "## Tracklänge", "", csv_block(b), "", "## Detektierbarkeit", "", csv_block(detect)]
    (root / "befunde.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True); records = load_manifest(args.manifest)
    b_out, c_out = args.output / "out_NEU-B", args.output / "out_NEU-C"
    with plt.rc_context(RC):
        card = data_card(records, args.output); tracks = track_table(records)
        b_out.mkdir(parents=True, exist_ok=True); c_out.mkdir(parents=True, exist_ok=True)
        card.to_csv(b_out / "data_card.csv", index=False); card.to_csv(c_out / "data_card.csv", index=False)
        a = task_a(tracks, b_out); b = task_b(tracks, b_out); c = task_c(records, c_out); select_and_plot(c, c_out)
        report(card, a, b, c, b_out); report(card, a, b, c, c_out)


if __name__ == "__main__":
    main()
