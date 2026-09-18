"""
Shared example-selection helper for Task C (particle detectability).

Not runnable standalone. Imported by TaskC_Detectability_Figure.py and
TaskC_Detectability_Distribution.py so both scripts always mark the exact
same three examples ("very good" / "challenging" / "no particle") from
cache/detectability_35nm.pkl.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _nearest_rank_row(df: pd.DataFrame, col: str, pct: float) -> pd.Series:
    """Row whose `col` value is at the given percentile, using an ACTUAL sampled
    row (nearest-rank), not an interpolated value -- so the "example" is a real
    detection, not a synthetic point."""
    sorted_df = df.sort_values(col).reset_index(drop=True)
    idx = int(round((pct / 100.0) * (len(sorted_df) - 1)))
    idx = max(0, min(idx, len(sorted_df) - 1))
    return sorted_df.iloc[idx]


def select_examples(df: pd.DataFrame, snr_col: str = "snr",
                     good_pct: float = 90.0, challenging_pct: float = 12.0) -> dict:
    """
    Selects "very good" and "challenging" from non-background rows by nearest-rank
    percentile on `snr_col` alone (not a blended composite score -- avoids an
    opaque combined metric). bandpass/LoG-response SNR is reported alongside as a
    printed corroboration check only, never used to re-rank.

    "No particle" is chosen among background rows (is_background=True) from the
    SAME movie and frame as the "challenging" row, picking whichever background
    row's background_sd is closest to the challenging row's background_sd --
    matches noise characteristics rather than just picking any empty patch.
    Falls back to same-movie/any-frame, then to any background row, if no
    same-frame background candidate exists.

    Returns {"good": Series, "challenging": Series, "no_particle": Series}.
    """
    detections = df[~df["is_background"]].dropna(subset=[snr_col])
    background = df[df["is_background"]]

    good_row = _nearest_rank_row(detections, snr_col, good_pct)
    challenging_row = _nearest_rank_row(detections, snr_col, challenging_pct)

    same_frame = background[
        (background["movie"] == challenging_row["movie"]) & (background["frame"] == challenging_row["frame"])
    ]
    candidates = same_frame
    if candidates.empty:
        candidates = background[background["movie"] == challenging_row["movie"]]
    if candidates.empty:
        candidates = background

    target_bg_sd = challenging_row["background_sd"]
    no_particle_row = candidates.iloc[(candidates["background_sd"] - target_bg_sd).abs().argsort().iloc[0]]

    print(f"[selection] good: {good_row['movie']} frame={good_row['frame']} "
          f"snr={good_row[snr_col]:.3g} (LoG-SNR={good_row.get('log_snr', float('nan')):.3g})")
    print(f"[selection] challenging: {challenging_row['movie']} frame={challenging_row['frame']} "
          f"snr={challenging_row[snr_col]:.3g} (LoG-SNR={challenging_row.get('log_snr', float('nan')):.3g})")
    print(f"[selection] no_particle: {no_particle_row['movie']} frame={no_particle_row['frame']} "
          f"background_sd={no_particle_row['background_sd']:.3g} "
          f"(challenging background_sd={target_bg_sd:.3g})")

    return {"good": good_row, "challenging": challenging_row, "no_particle": no_particle_row}
