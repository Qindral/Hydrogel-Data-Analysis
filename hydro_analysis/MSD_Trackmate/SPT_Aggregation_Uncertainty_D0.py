"""
SPT aggregation-scheme & uncertainty comparison — water (D0) data.

Pure statistics exercise on top of the EXISTING, UNCHANGED MSD computation and
EXISTING, UNCHANGED fit (core.analysis.fit_powerlaw_with_errors, the log-log
power-law regression already used everywhere else in this repo). This module
does not alter the pipeline; it only reads outputs already produced by it
(msd_d0_results.pkl: per-file tracks_df + per-trajectory iMSD) and
recombines / refits / bootstraps them in different ways. See
spt_aggregation_uncertainty_brief.md for the full specification.

Terminology mapping (brief -> this codebase)
---------------------------------------------
  trajectory i  = one TrackMate particle track (tracks_df['particle'])
  movie j       = one TrackMate XML file / recording
  D_i, alpha_i, s_i = fit_powerlaw_with_errors() applied to trajectory i's OWN
                       msd curve (same function used everywhere else, just
                       called at per-trajectory instead of per-movie/pooled
                       granularity)
  n_i           = trackpy's own effective per-lag pair count ('N' column of
                  tp.motion.msd(..., detail=True)), summed over the lags used
                  in the fit -- this is what tp.emsd already weights by
                  internally, not a hand-rolled displacement count
  n_j, m_j, M   = per movie: total n_i / trajectory count; number of movies

Fast-but-exact pooling
-----------------------
tp.emsd's ensemble average at each lag is a count-weighted mean of the
per-particle msd values: emsd(tau) = sum_i N_i(tau)*msd_i(tau) / sum_i N_i(tau)
(verified empirically against tp.emsd's actual output to 1e-15). Every
aggregation/bootstrap scheme below that needs a pooled eMSD is therefore
built by precomputing each trajectory's own (msd_i, N_i) ONCE via trackpy's
own tp.motion.msd(detail=True) (same underlying computation tp.imsd/tp.emsd
already use), then recombining sums of msd_i*N_i and N_i over whatever subset
or resample is required. This reproduces "recompute the full aggregation
(including the fit) inside each bootstrap replicate" exactly, without paying
trackpy's per-particle Python-loop cost 2000+ times over.

Run separately per particle size: movies of different particle sizes are not
comparable replicates of the same D, so pooling across sizes is out of scope.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp

from hydro_analysis.core.analysis import fit_powerlaw_with_errors, DEFAULT_MSD_FIT_POINTS, MIN_TRACK_LENGTH

# ── Style (publication design, matches MSD_per_file_publication.py) ────────
_RC = {
    "text.usetex":       False,
    "mathtext.fontset":  "dejavusans",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Open Sans", "Arial", "DejaVu Sans"],
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.size":  5.0,
    "ytick.major.size":  5.0,
    "xtick.minor.size":  3.0,
    "ytick.minor.size":  3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "legend.fontsize":   8,
}

# ── Configuration ───────────────────────────────────────────────────────────
CACHE_FILES = Path(__file__).parent / "cache" / "msd_d0_results.pkl"
RESULT_CACHE = Path(__file__).parent / "cache" / "spt_aggregation_uncertainty_d0.pkl"
SAVE_PATH = Path(
    rf"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung"
    rf"\AggregationUncertainty_{pd.Timestamp.now().strftime('%Y%m%d')}"
)

NEUBERECHNEN = False  # Set True to force recomputation instead of loading RESULT_CACHE

FIT_POINTS = DEFAULT_MSD_FIT_POINTS
FPS_SMALL_TARGET = 60.0
FPS_LARGE_TARGET = 20.0
FPS_TOLERANCE = 3.0
FPS_SIZE_EXACT: dict[float, float] = {50.0: 60.0}
DRIFT_DENSITY_THRESHOLD = 50.0  # tracks-per-frame ratio; same condition perform_msd_analysis uses

N_BOOT = 2000
N_PERM = 2000
RNG_SEED = 12345

MIN_VALID_LAGS_FOR_TRAJ_FIT = 4  # need >=4 of the FIT_POINTS lags populated for a meaningful D_i


# ── Step 0: per-trajectory precompute (msd_i, N_i) ──────────────────────────

def _target_fps(size_nm: float) -> tuple[float, float]:
    if size_nm in FPS_SIZE_EXACT:
        return FPS_SIZE_EXACT[size_nm], 0.5
    return (FPS_SMALL_TARGET if size_nm < 200 else FPS_LARGE_TARGET), FPS_TOLERANCE


def precompute_trajectories(files_results: dict, size_nm: float, fit_points: int = FIT_POINTS) -> dict:
    """
    Pool every trajectory of `size_nm` across all movies (same drift-subtraction
    / stub-filtering condition as perform_msd_analysis, unchanged), then compute
    each surviving trajectory's own msd(lag) and trackpy's own effective
    per-lag pair count N(lag) once via tp.motion.msd(detail=True) -- the same
    underlying call tp.imsd/tp.emsd already make internally.

    Returns a dict with:
      msd_mat, N_mat   (n_traj, fit_points) arrays
      movie_idx        (n_traj,) int array -> index into `movies`
      movies           list of movie base_names, in pooling order
      mpp, fps          calibration used for the pool
      lag_times         (fit_points,) array, seconds
    """
    target_fps, tol = _target_fps(size_nm)
    entries = []
    for r in files_results.values():
        if r.get("particle_size_nm") != size_nm:
            continue
        fps = r.get("fps")
        tracks_df = r.get("tracks_df")
        if fps is None or tracks_df is None or tracks_df.empty:
            continue
        if abs(float(fps) - target_fps) > tol:
            continue
        entries.append(r)
    entries.sort(key=lambda r: r["xml_path"])
    if not entries:
        raise ValueError(f"No files found for particle_size_nm={size_nm}")

    mpp_vals = {round(float(e["mpp"]), 6) for e in entries}
    if len(mpp_vals) > 1:
        print(f"  WARNING: mixed mpp for {size_nm:.0f} nm: {mpp_vals} -- using the first file's mpp")
    mpp = float(entries[0]["mpp"])

    combined_frames = []
    movie_of_row = []
    offset = 0
    for movie_idx, e in enumerate(entries):
        df = e["tracks_df"][["frame", "particle", "x", "y"]].copy()
        df["particle"] = df["particle"] + offset
        offset = int(df["particle"].max()) + 1
        combined_frames.append(df)
        movie_of_row.append(np.full(len(df), movie_idx))
    combined = pd.concat(combined_frames, ignore_index=True)
    movie_of_row = np.concatenate(movie_of_row)
    combined["_movie_idx"] = movie_of_row

    n_particles = combined["particle"].nunique()
    n_frames = combined["frame"].nunique()
    if n_particles / max(n_frames, 1) > DRIFT_DENSITY_THRESHOLD:
        pos = tp.subtract_drift(combined)
    else:
        pos = combined

    pos = tp.filter_stubs(pos, threshold=MIN_TRACK_LENGTH)
    # particle -> movie_idx (first row per particle; a particle never spans movies)
    particle_movie = combined.drop_duplicates("particle").set_index("particle")["_movie_idx"]

    msd_rows, N_rows, pids, midx = [], [], [], []
    lag_index = None
    for pid, g in pos.groupby("particle"):
        out = tp.motion.msd(g[["frame", "x", "y", "particle"]], mpp=mpp, fps=target_fps,
                            max_lagtime=fit_points, detail=True)
        if lag_index is None:
            lag_index = out["lagt"].reindex(range(1, fit_points + 1)).to_numpy()
        m = out["msd"].reindex(range(1, fit_points + 1)).to_numpy()
        n = out["N"].reindex(range(1, fit_points + 1)).to_numpy()
        msd_rows.append(m)
        N_rows.append(n)
        pids.append(pid)
        midx.append(int(particle_movie.loc[pid]))

    return {
        "size_nm": size_nm,
        "msd_mat": np.asarray(msd_rows),
        "N_mat": np.asarray(N_rows),
        "movie_idx": np.asarray(midx),
        "movies": [e["base_name"] for e in entries],
        "mpp": mpp,
        "fps": target_fps,
        "lag_times": lag_index,
        "particle_ids": pids,
    }


# ── Pooled-eMSD reconstruction + fit (the "existing fit", unchanged) ───────

def _pooled_fit(msd_mat: np.ndarray, N_mat: np.ndarray, row_mask: np.ndarray,
                lag_times: np.ndarray, fit_points: int = FIT_POINTS) -> dict | None:
    """Count-weighted pooled eMSD over the rows selected by `row_mask` (a
    boolean or integer-index array, may contain repeats for bootstrap
    resampling-with-replacement), then fit with fit_powerlaw_with_errors
    (unchanged). Returns None if fewer than 2 lags have data.
    """
    m = msd_mat[row_mask]
    n = N_mat[row_mask]
    valid = ~np.isnan(m)
    sqdisp = np.where(valid, m * n, 0.0)
    nsafe = np.where(valid, n, 0.0)
    denom = nsafe.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = sqdisp.sum(axis=0) / denom
    ok = np.isfinite(pooled) & (pooled > 0) & (denom > 0)
    if ok.sum() < 2:
        return None
    series = pd.Series(pooled[ok], index=lag_times[ok])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = fit_powerlaw_with_errors(series, points=int(ok.sum()))
    return {
        "D": float(fr["A"][0]) / 4.0, "D_err": float(fr["A_err"][0]) / 4.0,
        "alpha": float(fr["n"][0]), "alpha_err": float(fr["n_err"][0]),
        "r_squared": float(fr["r_squared"][0]),
    }


def _simple_curve_average_fit(msd_mat: np.ndarray, row_mask: np.ndarray,
                              lag_times: np.ndarray) -> dict | None:
    """Equal weight per trajectory (row), ignoring N -- scheme A2."""
    m = msd_mat[row_mask]
    pooled = np.nanmean(m, axis=0)
    ok = np.isfinite(pooled) & (pooled > 0)
    if ok.sum() < 2:
        return None
    series = pd.Series(pooled[ok], index=lag_times[ok])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = fit_powerlaw_with_errors(series, points=int(ok.sum()))
    return {
        "D": float(fr["A"][0]) / 4.0, "D_err": float(fr["A_err"][0]) / 4.0,
        "alpha": float(fr["n"][0]), "alpha_err": float(fr["n_err"][0]),
        "r_squared": float(fr["r_squared"][0]),
    }


def per_trajectory_fits(pre: dict, min_valid_lags: int = MIN_VALID_LAGS_FOR_TRAJ_FIT) -> pd.DataFrame:
    """D_i, alpha_i, s_i (fit SE), alpha_err_i, n_i for every trajectory,
    using fit_powerlaw_with_errors (unchanged) on that trajectory's own
    msd curve. n_i = trackpy's own effective pair count, summed over the
    lags actually used.
    """
    msd_mat, N_mat, lag_times = pre["msd_mat"], pre["N_mat"], pre["lag_times"]
    rows = []
    for i in range(msd_mat.shape[0]):
        m, n = msd_mat[i], N_mat[i]
        ok = np.isfinite(m) & (m > 0)
        if ok.sum() < min_valid_lags:
            continue
        series = pd.Series(m[ok], index=lag_times[ok])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fr = fit_powerlaw_with_errors(series, points=int(ok.sum()))
        rows.append({
            "movie_idx": int(pre["movie_idx"][i]),
            "particle": pre["particle_ids"][i],
            "D_i": float(fr["A"][0]) / 4.0,
            "s_i": float(fr["A_err"][0]) / 4.0,
            "alpha_i": float(fr["n"][0]),
            "alpha_err_i": float(fr["n_err"][0]),
            "n_i": float(np.nansum(n[ok])),
        })
    return pd.DataFrame(rows)


# ── Part A: six aggregation schemes ─────────────────────────────────────────

def scheme_A1_pooled(pre: dict) -> dict:
    all_rows = np.arange(pre["msd_mat"].shape[0])
    return _pooled_fit(pre["msd_mat"], pre["N_mat"], all_rows, pre["lag_times"])


def scheme_A2_curve_average(pre: dict) -> dict:
    all_rows = np.arange(pre["msd_mat"].shape[0])
    return _simple_curve_average_fit(pre["msd_mat"], all_rows, pre["lag_times"])


def scheme_A3_A4_A5_A6(pre: dict, movie_D: pd.DataFrame, sigma2_within: float,
                       sigma2_between: float) -> dict:
    """Per-movie D_j (already computed via A1-within-movie), combined three
    (four, with A6) different ways."""
    D_j = movie_D["D_j"].to_numpy()
    n_j = movie_D["n_j"].to_numpy()
    m_j = movie_D["m_j"].to_numpy()

    A3 = float(np.mean(D_j))  # unweighted mean over movies
    A4 = float(np.average(D_j, weights=n_j))  # weight by displacements
    A5 = float(np.average(D_j, weights=m_j))  # weight by trajectory count

    w6 = 1.0 / (sigma2_within / np.maximum(m_j, 1) + max(sigma2_between, 0.0))
    A6 = float(np.average(D_j, weights=w6))

    return {"A3": A3, "A4": A4, "A5": A5, "A6": A6, "w6": w6}


# ── Part B: five uncertainty estimators ─────────────────────────────────────

def bootstrap_trajectories(pre: dict, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> np.ndarray:
    """B2: resample trajectories with replacement, ignoring movie labels;
    recompute the FULL pooled-eMSD + fit each time."""
    rng = np.random.default_rng(seed)
    n = pre["msd_mat"].shape[0]
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fr = _pooled_fit(pre["msd_mat"], pre["N_mat"], idx, pre["lag_times"])
        if fr is not None:
            out[b] = fr["D"]
    return out


def bootstrap_movies(pre: dict, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> np.ndarray:
    """B3: resample movies with replacement; every trajectory of a drawn
    movie is included (with the movie's own multiplicity)."""
    rng = np.random.default_rng(seed)
    M = len(pre["movies"])
    row_movie = pre["movie_idx"]
    rows_by_movie = [np.where(row_movie == j)[0] for j in range(M)]
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        movies_draw = rng.integers(0, M, size=M)
        idx = np.concatenate([rows_by_movie[j] for j in movies_draw]) if M else np.array([], dtype=int)
        if idx.size == 0:
            continue
        fr = _pooled_fit(pre["msd_mat"], pre["N_mat"], idx, pre["lag_times"])
        if fr is not None:
            out[b] = fr["D"]
    return out


def bootstrap_hierarchical(pre: dict, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> np.ndarray:
    """B4: resample movies with replacement, then resample trajectories
    WITHIN each resampled movie with replacement (nested)."""
    rng = np.random.default_rng(seed)
    M = len(pre["movies"])
    row_movie = pre["movie_idx"]
    rows_by_movie = [np.where(row_movie == j)[0] for j in range(M)]
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        movies_draw = rng.integers(0, M, size=M)
        idx_parts = []
        for j in movies_draw:
            rows_j = rows_by_movie[j]
            if rows_j.size == 0:
                continue
            idx_parts.append(rng.choice(rows_j, size=rows_j.size, replace=True))
        if not idx_parts:
            continue
        idx = np.concatenate(idx_parts)
        fr = _pooled_fit(pre["msd_mat"], pre["N_mat"], idx, pre["lag_times"])
        if fr is not None:
            out[b] = fr["D"]
    return out


def between_movie_sem(movie_D: pd.DataFrame) -> float:
    """B5: SD(D_j) / sqrt(M)."""
    D_j = movie_D["D_j"].to_numpy()
    M = len(D_j)
    if M < 2:
        return float("nan")
    return float(np.std(D_j, ddof=1) / np.sqrt(M))


# ── Part C: variance decomposition (unbalanced one-way random effects) ─────

def anova_variance_components(D_i: np.ndarray, movie_id: np.ndarray) -> dict:
    """Unbalanced one-way random-effects ANOVA (Type II, method-of-moments).

    MS_between = sigma2_within + n0 * sigma2_between, with
    n0 = (1/(M-1)) * (sum(m_j) - sum(m_j**2)/sum(m_j))  -- NOT mean(m_j).
    """
    df = pd.DataFrame({"D_i": D_i, "movie": movie_id})
    groups = df.groupby("movie")["D_i"]
    m_j = groups.count().to_numpy()
    means_j = groups.mean().to_numpy()
    M = len(m_j)
    N = int(m_j.sum())
    grand_mean = float(df["D_i"].mean())

    ss_between = float(np.sum(m_j * (means_j - grand_mean) ** 2))
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for _, g in groups))
    df_between = M - 1
    df_within = N - M

    ms_between = ss_between / df_between if df_between > 0 else float("nan")
    ms_within = ss_within / df_within if df_within > 0 else float("nan")

    if M > 1:
        n0 = (1.0 / (M - 1)) * (m_j.sum() - np.sum(m_j ** 2) / m_j.sum())
    else:
        n0 = float("nan")

    sigma2_within = ms_within if np.isfinite(ms_within) else float(np.var(D_i, ddof=1))
    if np.isfinite(ms_between) and np.isfinite(n0) and n0 > 0:
        sigma2_between = max(0.0, (ms_between - sigma2_within) / n0)
    else:
        sigma2_between = 0.0

    rho = sigma2_between / (sigma2_between + sigma2_within) if (sigma2_between + sigma2_within) > 0 else 0.0
    deff = 1.0 + (n0 - 1.0) * rho if np.isfinite(n0) else float("nan")
    n_eff = N / deff if np.isfinite(deff) and deff > 0 else float(N)

    return {
        "M": M, "N": N, "m_j": m_j, "n0": float(n0),
        "ms_between": ms_between, "ms_within": ms_within,
        "sigma2_within": float(sigma2_within), "sigma2_between": float(sigma2_between),
        "rho": float(rho), "deff": float(deff), "n_eff": float(n_eff),
        "grand_mean": grand_mean,
    }


def anova_ci_bootstrap(D_i: np.ndarray, movie_id: np.ndarray, n_boot: int = N_BOOT,
                       seed: int = RNG_SEED) -> dict:
    """Bootstrap CI (resample movies with replacement, recompute the whole
    ANOVA each time) for sigma2_within/between, rho, deff, n_eff."""
    rng = np.random.default_rng(seed)
    movies = np.unique(movie_id)
    M = len(movies)
    idx_by_movie = {mv: np.where(movie_id == mv)[0] for mv in movies}

    keys = ["sigma2_within", "sigma2_between", "rho", "deff", "n_eff"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        draw_movies = rng.choice(movies, size=M, replace=True)
        idx = np.concatenate([idx_by_movie[mv] for mv in draw_movies])
        new_movie_id = np.concatenate([
            np.full(len(idx_by_movie[mv]), k) for k, mv in enumerate(draw_movies)
        ])
        res = anova_variance_components(D_i[idx], new_movie_id)
        for k in keys:
            v = res[k]
            if np.isfinite(v):
                draws[k].append(v)

    ci = {}
    for k in keys:
        arr = np.array(draws[k])
        if arr.size >= 20:
            ci[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
        else:
            ci[k] = (float("nan"), float("nan"))
    return ci


def heterogeneity_variance(traj_df: pd.DataFrame, sigma2_within_anova: float) -> dict:
    """Compare observed within-movie spread of D_i (sigma2_within from the
    ANOVA) against the spread predicted by each fit's own s_i. Excess is
    genuine particle-to-particle heterogeneity (sigma2_het); if the mean
    fit variance already exceeds sigma2_within_anova, report an upper bound
    instead of a negative "excess"."""
    mean_s2 = float(np.mean(traj_df["s_i"].to_numpy() ** 2))
    excess = sigma2_within_anova - mean_s2
    if excess > 0:
        return {"sigma2_het": excess, "resolvable": True, "mean_fit_var": mean_s2}
    return {"sigma2_het": 0.0, "resolvable": False, "mean_fit_var": mean_s2,
            "upper_bound": sigma2_within_anova}


# ── Part D: validation ───────────────────────────────────────────────────────

def permutation_test_rho(D_i: np.ndarray, movie_id: np.ndarray, n_perm: int = N_PERM,
                         seed: int = RNG_SEED) -> dict:
    """D1: shuffle trajectory->movie labels (preserving m_j), recompute rho,
    build a null distribution, compare the observed rho against it."""
    observed = anova_variance_components(D_i, movie_id)["rho"]
    rng = np.random.default_rng(seed)
    n = len(D_i)
    labels = movie_id.copy()
    null = np.empty(n_perm)
    for p in range(n_perm):
        shuffled = rng.permutation(labels)
        null[p] = anova_variance_components(D_i, shuffled)["rho"]
    p_value = float(np.mean(null >= observed))
    return {"observed_rho": observed, "null": null, "p_value": p_value}


def validate_bootstrap_machinery(n_movies: int = 15, mean_m: int = 80, n_boot: int = 1000,
                                 seed: int = RNG_SEED) -> dict:
    """D2: synthetic D_i ~ N(mu, sigma), NO movie effect, random unbalanced
    m_j. Verify B2/B3/B4/B5-style resampling (mean-based, since there is no
    curve to refit here -- this checks the resampling/ANOVA code, not the
    physics) agree, and that the ANOVA recovers sigma2_between ~ 0.
    """
    rng = np.random.default_rng(seed)
    mu, sigma = 1.0, 0.2
    m_j = rng.integers(max(2, mean_m // 4), mean_m * 2, size=n_movies)
    movie_id = np.concatenate([np.full(m, j) for j, m in enumerate(m_j)])
    D_i = rng.normal(mu, sigma, size=movie_id.size)

    anova = anova_variance_components(D_i, movie_id)

    def _boot_traj(nb):
        out = np.empty(nb)
        n = len(D_i)
        for b in range(nb):
            idx = rng.integers(0, n, size=n)
            out[b] = D_i[idx].mean()
        return out

    def _boot_movie(nb):
        movies = np.unique(movie_id)
        idx_by = {mv: np.where(movie_id == mv)[0] for mv in movies}
        out = np.empty(nb)
        for b in range(nb):
            draw = rng.choice(movies, size=len(movies), replace=True)
            idx = np.concatenate([idx_by[mv] for mv in draw])
            out[b] = D_i[idx].mean()
        return out

    b2 = _boot_traj(n_boot)
    b3 = _boot_movie(n_boot)
    means_j = pd.Series(D_i).groupby(movie_id).mean().to_numpy()
    b5_sem = float(np.std(means_j, ddof=1) / np.sqrt(len(means_j)))

    return {
        "sigma2_between_recovered": anova["sigma2_between"],
        "sigma2_within_recovered": anova["sigma2_within"],
        "sigma2_within_true": sigma ** 2,
        "b2_sd": float(np.std(b2, ddof=1)),
        "b3_sd": float(np.std(b3, ddof=1)),
        "b5_sem": b5_sem,
        "pass": bool(
            anova["sigma2_between"] < 0.3 * sigma ** 2
            and abs(np.std(b2, ddof=1) - np.std(b3, ddof=1)) < 0.5 * np.std(b2, ddof=1)
        ),
    }


# ── Per-movie pooled D (A1-within-movie, needed by A3-A6) ──────────────────

def compute_per_movie_D(pre: dict) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(pre["movies"]):
        mask = np.where(pre["movie_idx"] == j)[0]
        if mask.size == 0:
            continue
        fr = _pooled_fit(pre["msd_mat"], pre["N_mat"], mask, pre["lag_times"])
        if fr is None:
            continue
        rows.append({
            "movie_idx": j, "movie": name,
            "D_j": fr["D"], "D_j_err": fr["D_err"],
            "alpha_j": fr["alpha"], "alpha_j_err": fr["alpha_err"],
            "m_j": mask.size,
        })
    out = pd.DataFrame(rows)
    return out


# ── Orchestration: run the full brief for one particle size ────────────────

def run_full_analysis(pre: dict) -> dict:
    traj_df = per_trajectory_fits(pre)
    movie_D = compute_per_movie_D(pre)
    n_by_movie = traj_df.groupby("movie_idx")["n_i"].sum()
    movie_D["n_j"] = movie_D["movie_idx"].map(n_by_movie).fillna(0.0)

    A1 = scheme_A1_pooled(pre)
    A2 = scheme_A2_curve_average(pre)

    anova = anova_variance_components(traj_df["D_i"].to_numpy(), traj_df["movie_idx"].to_numpy())
    anova_ci = anova_ci_bootstrap(traj_df["D_i"].to_numpy(), traj_df["movie_idx"].to_numpy())
    het = heterogeneity_variance(traj_df, anova["sigma2_within"])

    A346 = scheme_A3_A4_A5_A6(pre, movie_D, anova["sigma2_within"], anova["sigma2_between"])

    D_points = {
        "A1": A1["D"], "A2": A2["D"],
        "A3": A346["A3"], "A4": A346["A4"], "A5": A346["A5"], "A6": A346["A6"],
    }
    alpha_points = {
        "A1": A1["alpha"], "A2": A2["alpha"],
        "A3": float(np.mean(movie_D["alpha_j"])),
        "A4": float(np.average(movie_D["alpha_j"], weights=movie_D["n_j"])),
        "A5": float(np.average(movie_D["alpha_j"], weights=movie_D["m_j"])),
        "A6": float(np.average(movie_D["alpha_j"], weights=A346["w6"])),
    }

    weight_table = pd.DataFrame({
        "movie": movie_D["movie"],
        "A1_A4": movie_D["n_j"] / movie_D["n_j"].sum(),
        "A2_A5": movie_D["m_j"] / movie_D["m_j"].sum(),
        "A3": 1.0 / len(movie_D),
        "A6": A346["w6"] / A346["w6"].sum(),
    })

    B1 = A1["D_err"]
    B2 = bootstrap_trajectories(pre)
    B3 = bootstrap_movies(pre)
    B4 = bootstrap_hierarchical(pre)
    B5 = between_movie_sem(movie_D)

    uncertainties = {
        "B1_fit_covariance": B1,
        "B2_bootstrap_trajectories": float(np.nanstd(B2, ddof=1)),
        "B3_bootstrap_movies": float(np.nanstd(B3, ddof=1)),
        "B4_bootstrap_hierarchical": float(np.nanstd(B4, ddof=1)),
        "B5_between_movie_sem": B5,
    }

    perm = permutation_test_rho(traj_df["D_i"].to_numpy(), traj_df["movie_idx"].to_numpy())

    return {
        "size_nm": pre["size_nm"],
        "movies": pre["movies"],
        "traj_df": traj_df,
        "movie_D": movie_D,
        "D_points": D_points,
        "alpha_points": alpha_points,
        "weight_table": weight_table,
        "uncertainties": uncertainties,
        "B2_draws": B2, "B3_draws": B3, "B4_draws": B4,
        "anova": anova, "anova_ci": anova_ci, "heterogeneity": het,
        "permutation": perm,
        "D_existing_scheme": "A1",
        "D_existing": A1["D"],
    }


# ── Cache I/O ────────────────────────────────────────────────────────────────

def load_files_cache() -> dict:
    with open(CACHE_FILES, "rb") as f:
        return pickle.load(f)


def load_result_cache() -> dict | None:
    if NEUBERECHNEN or not RESULT_CACHE.exists():
        return None
    try:
        with open(RESULT_CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"Result cache geladen: {RESULT_CACHE} ({len(data)} Größen)")
        return data
    except Exception as e:
        print(f"Result cache load failed: {e}")
        return None


def save_result_cache(data: dict) -> None:
    RESULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CACHE, "wb") as f:
        pickle.dump(data, f)
    print(f"Result cache gespeichert: {RESULT_CACHE} ({len(data)} Größen)")


# ── Plotting ─────────────────────────────────────────────────────────────────

_SCHEME_COLORS = {
    "A1": "#0000da", "A2": "#004cff", "A3": "#00c4ff",
    "A4": "#49ffad", "A5": "#ffd700", "A6": "#ff6800",
}
_SCHEME_LABELS = {
    "A1": "A1 pooled (all disp.)", "A2": "A2 curve-average",
    "A3": "A3 movie mean", "A4": "A4 movie, w∝disp.",
    "A5": "A5 movie, w∝traj.", "A6": "A6 inverse-variance",
}
_UNC_LABELS = {
    "B1_fit_covariance": "B1 fit covariance",
    "B2_bootstrap_trajectories": "B2 bootstrap (trajectories)",
    "B3_bootstrap_movies": "B3 bootstrap (movies)",
    "B4_bootstrap_hierarchical": "B4 bootstrap (hierarchical)",
    "B5_between_movie_sem": "B5 between-movie SEM",
}


def plot_six_schemes(results_by_size: dict[float, dict], save_path: Path | None) -> None:
    sizes = sorted(results_by_size.keys())
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
        x_base = np.arange(len(sizes), dtype=float)
        n_schemes = len(_SCHEME_COLORS)
        offsets = np.linspace(-0.30, 0.30, n_schemes)

        for k, (scheme, off) in enumerate(zip(_SCHEME_COLORS, offsets)):
            xs, ys, yerr = [], [], []
            for si, size_nm in enumerate(sizes):
                res = results_by_size[size_nm]
                xs.append(x_base[si] + off)
                ys.append(res["D_points"][scheme])
                yerr.append(res["uncertainties"]["B1_fit_covariance"] if scheme == "A1" else 0.0)
            ax.errorbar(xs, ys, yerr=yerr, fmt="o", markersize=5,
                       markerfacecolor=_SCHEME_COLORS[scheme], markeredgecolor="black",
                       markeredgewidth=0.4, ecolor="black", elinewidth=1.0, capsize=2.0,
                       linestyle="None", label=_SCHEME_LABELS[scheme], zorder=3)

        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{int(s)}" for s in sizes])
        ax.set_xlabel("Nominal particle size (nm)")
        ax.set_ylabel(r"$D$  (µm²/s)")
        ax.set_yscale("log")
        ax.set_title("Water (D0): D under six aggregation schemes", fontsize=11)
        ax.legend(loc="best", fontsize=7, ncol=2)

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "six_schemes_D.png", dpi=600, bbox_inches="tight")
            fig.savefig(save_path / "six_schemes_D.pdf")
            print(f"Saved: {save_path / 'six_schemes_D.png'}")
        plt.show()


def plot_five_uncertainties(results_by_size: dict[float, dict], save_path: Path | None) -> None:
    sizes = sorted(results_by_size.keys())
    unc_keys = list(_UNC_LABELS.keys())
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7.15, 5.00), constrained_layout=True)
        x_base = np.arange(len(sizes), dtype=float)
        offsets = np.linspace(-0.28, 0.28, len(unc_keys))
        colors = ["#000099", "#0035b2", "#0089b2", "#33b279", "#b29600"]

        for off, key, col in zip(offsets, unc_keys, colors):
            xs, ys, yerr = [], [], []
            for si, size_nm in enumerate(sizes):
                res = results_by_size[size_nm]
                xs.append(x_base[si] + off)
                ys.append(res["D_existing"])
                yerr.append(res["uncertainties"][key])
            ax.errorbar(xs, ys, yerr=yerr, fmt="o", markersize=5,
                       markerfacecolor=col, markeredgecolor="black", markeredgewidth=0.4,
                       ecolor=col, elinewidth=1.4, capsize=3.0,
                       linestyle="None", label=_UNC_LABELS[key], zorder=3)

        ax.set_xticks(x_base)
        ax.set_xticklabels([f"{int(s)}" for s in sizes])
        ax.set_xlabel("Nominal particle size (nm)")
        ax.set_ylabel(r"$D_{A1}$  (µm²/s)  —  five different error bars")
        ax.set_yscale("log")
        ax.set_title("Water (D0): same point estimate, five uncertainty estimators", fontsize=11)
        ax.legend(loc="best", fontsize=7)

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "five_uncertainties.png", dpi=600, bbox_inches="tight")
            fig.savefig(save_path / "five_uncertainties.pdf")
            print(f"Saved: {save_path / 'five_uncertainties.png'}")
        plt.show()


def plot_variance_decomposition(results_by_size: dict[float, dict], save_path: Path | None) -> None:
    sizes = sorted(results_by_size.keys())
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)

        rho = [results_by_size[s]["anova"]["rho"] for s in sizes]
        rho_ci = [results_by_size[s]["anova_ci"]["rho"] for s in sizes]
        rho_lo = [rho[i] - rho_ci[i][0] for i in range(len(sizes))]
        rho_hi = [rho_ci[i][1] - rho[i] for i in range(len(sizes))]
        axes[0].errorbar(sizes, rho, yerr=[rho_lo, rho_hi], fmt="o", markersize=6,
                        markerfacecolor="#0000da", markeredgecolor="#000099",
                        ecolor="#000099", elinewidth=1.2, capsize=3.0)
        axes[0].set_xscale("log")
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].set_xlabel("Particle size (nm)")
        axes[0].set_ylabel(r"Intraclass correlation $\rho$")
        axes[0].set_title("Between-movie vs. within-movie variance", fontsize=10)

        n_eff = [results_by_size[s]["anova"]["n_eff"] for s in sizes]
        n_raw = [results_by_size[s]["anova"]["N"] for s in sizes]
        axes[1].plot(sizes, n_raw, "o--", color="#999999", label="raw trajectory count N")
        axes[1].plot(sizes, n_eff, "o-", color="#da0000", label=r"effective $n_{eff}$")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Particle size (nm)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Pseudoreplication: N vs. n_eff", fontsize=10)
        axes[1].legend(loc="best", fontsize=8)

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "variance_decomposition.png", dpi=600, bbox_inches="tight")
            fig.savefig(save_path / "variance_decomposition.pdf")
            print(f"Saved: {save_path / 'variance_decomposition.png'}")
        plt.show()


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(res: dict) -> None:
    size_nm = res["size_nm"]
    print(f"\n{'='*78}\n Particle size: {size_nm:.0f} nm  ({len(res['movies'])} movies, "
          f"{len(res['traj_df'])} trajectories)\n{'='*78}")

    print("\n-- Part A: D under six aggregation schemes --")
    for k in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        marker = "  <- D_existing" if k == res["D_existing_scheme"] else ""
        print(f"  {k}: D = {res['D_points'][k]:.4f} µm²/s   alpha = {res['alpha_points'][k]:.4f}{marker}")

    print("\n-- Part A: realised weight per movie --")
    print(res["weight_table"].to_string(index=False))

    print("\n-- Part B: five uncertainty estimates for D_existing (A1) --")
    for k, v in res["uncertainties"].items():
        print(f"  {_UNC_LABELS[k]:<32s}: {v:.4f} µm²/s")

    a = res["anova"]
    print("\n-- Part C: variance decomposition (unbalanced one-way random effects) --")
    print(f"  n0 (Satterthwaite coeff.)        = {a['n0']:.2f}")
    print(f"  sigma2_within                    = {a['sigma2_within']:.5g}  "
          f"CI95 {res['anova_ci']['sigma2_within']}")
    print(f"  sigma2_between                   = {a['sigma2_between']:.5g}  "
          f"CI95 {res['anova_ci']['sigma2_between']}")
    print(f"  rho (ICC)                        = {a['rho']:.3f}  CI95 {res['anova_ci']['rho']}")
    print(f"  DEFF                             = {a['deff']:.2f}  CI95 {res['anova_ci']['deff']}")
    print(f"  n_eff / N                        = {a['n_eff']:.1f} / {a['N']}")
    het = res["heterogeneity"]
    if het["resolvable"]:
        print(f"  sigma2_het (particle heterogeneity beyond fit noise) = {het['sigma2_het']:.5g}")
    else:
        print(f"  sigma2_het: not resolvable (mean fit variance {het['mean_fit_var']:.5g} "
              f">= observed within-movie variance); upper bound = {het['upper_bound']:.5g}")

    perm = res["permutation"]
    print("\n-- Part D1: permutation test for rho --")
    print(f"  observed rho = {perm['observed_rho']:.3f}, p-value = {perm['p_value']:.4f} "
          f"({'movies are NOT interchangeable -> movie is the experimental unit' if perm['p_value'] < 0.05 else 'movies statistically interchangeable -> pooling defensible'})")


def print_verdict(res: dict) -> None:
    unc = res["uncertainties"]
    widest_key = max(unc, key=unc.get)
    print(f"\n  VERDICT ({res['size_nm']:.0f} nm): D_existing (scheme {res['D_existing_scheme']}) "
          f"= {res['D_existing']:.3g} µm²/s. Quoted uncertainty should be "
          f"{_UNC_LABELS[widest_key]} = {unc[widest_key]:.3g} µm²/s "
          f"({unc[widest_key] / unc['B1_fit_covariance']:.1f}x the fit-covariance error bar), "
          f"which generalises to reproducibility across independent movies/preparations, "
          f"not curve-fit precision. rho={res['anova']['rho']:.2f} -> "
          f"{'substantial' if res['anova']['rho'] > 0.2 else 'modest'} movie-to-movie structure.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    results_by_size = load_result_cache()

    if results_by_size is None:
        files_results = load_files_cache()
        sizes_present = sorted({r.get("particle_size_nm") for r in files_results.values()
                                if r.get("particle_size_nm") is not None})
        print(f"Particle sizes found in {CACHE_FILES.name}: {sizes_present}")

        results_by_size = {}
        for size_nm in sizes_present:
            print(f"\nProcessing {size_nm:.0f} nm ...")
            pre = precompute_trajectories(files_results, size_nm)
            res = run_full_analysis(pre)
            results_by_size[size_nm] = res

        print("\nValidating bootstrap/ANOVA machinery on synthetic no-movie-effect data ...")
        val = validate_bootstrap_machinery()
        print(f"  {val}  -> {'PASS' if val['pass'] else 'FAIL'}")
        results_by_size["_validation"] = val

        save_result_cache(results_by_size)

    for size_nm, res in sorted((k, v) for k, v in results_by_size.items() if k != "_validation"):
        print_report(res)
        print_verdict(res)

    if "_validation" in results_by_size:
        v = results_by_size["_validation"]
        print(f"\nSynthetic validation (Part D2): {'PASS' if v['pass'] else 'FAIL'} -- "
              f"sigma2_between recovered = {v['sigma2_between_recovered']:.4g} "
              f"(should be ~0); B2 SD={v['b2_sd']:.4g}, B3 SD={v['b3_sd']:.4g}, "
              f"B5 SEM={v['b5_sem']:.4g} (should agree, no movie effect).")

    plot_results = {k: v for k, v in results_by_size.items() if k != "_validation"}
    plot_six_schemes(plot_results, SAVE_PATH)
    plot_five_uncertainties(plot_results, SAVE_PATH)
    plot_variance_decomposition(plot_results, SAVE_PATH)


if __name__ == "__main__":
    main()
