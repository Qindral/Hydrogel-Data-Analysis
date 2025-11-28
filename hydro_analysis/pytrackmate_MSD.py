#!/usr/bin/env python3
"""
calc_trackmate_stats.py

Parse a TrackMate XML file (TrackMate 3–7 variants) and compute per-track and ensemble
MSD plus common track statistics. Attempts to use pytrackmate if installed; otherwise
falls back to robust XML parsing.

Usage:
    python calc_trackmate_stats.py tracks.xml --dt 0.5 --min-len 5 --fit-lags 3 --out-prefix results

Outputs:
    - {out_prefix}_per_track_stats.csv
    - {out_prefix}_per_track_msd.npy   (dict of arrays saved with numpy.save)
    - {out_prefix}_ensemble_msd.csv
    - optionally plots if matplotlib is available
"""

import argparse
import sys
import math
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import xml.etree.ElementTree as ET

def get_attr_any(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    # case-insensitive
    for k in d.keys():
        if k.lower() in [x.lower() for x in keys]:
            return d[k]
    return default

def parse_spots_from_xml(root):
    # Find all Spot elements and gather their attributes.
    spots = {}
    # Element tag may have namespace; match by suffix
    for el in root.iter():
        tag = el.tag
        if isinstance(tag, str) and tag.lower().endswith('spot'):
            attrs = el.attrib
            # possible attribute names
            id_attr = get_attr_any(attrs, ['ID', 'Id', 'id', 'SPOT_ID', 'SPOTID'])
            if id_attr is None:
                continue
            spot_id = int(id_attr)
            x = get_attr_any(attrs, ['POSITION_X', 'POSITIONx', 'X', 'x'])
            y = get_attr_any(attrs, ['POSITION_Y', 'POSITIONy', 'Y', 'y'])
            z = get_attr_any(attrs, ['POSITION_Z', 'POSITIONz', 'Z', 'z'])
            frame = get_attr_any(attrs, ['FRAME', 'frame', 'FRAME_INDEX', 'T', 't'])
            try:
                pos = (float(x), float(y), float(z)) if (z is not None) else (float(x), float(y))
            except Exception:
                # skip if coordinates not parseable
                continue
            try:
                frame_val = int(float(frame)) if frame is not None else None
            except Exception:
                frame_val = None
            spots[spot_id] = {'pos': np.array(pos, dtype=float), 'frame': frame_val}
    return spots

def parse_tracks_from_xml(root, spots):
    # Try several strategies to get track -> ordered spot id list
    tracks = {}
    # 1) Some TrackMate versions store track id on Spot: TRACK_ID attribute
    track_groups = defaultdict(list)
    for sid, s in spots.items():
        # No consistent attribute reading here because we parsed spots only; but if track id exists in XML as separate attribute,
        # the above parse_spots_from_xml would not have stored it. So we need to scan Spot elements again for TRACK_ID.
        pass

    # Re-scan Spot elements to get TRACK_ID if present
    for el in root.iter():
        tag = getattr(el, 'tag', '')
        if isinstance(tag, str) and tag.lower().endswith('spot'):
            attrs = el.attrib
            sid_attr = get_attr_any(attrs, ['ID', 'Id', 'id', 'SPOT_ID'])
            if sid_attr is None:
                continue
            sid = int(sid_attr)
            track_attr = get_attr_any(attrs, ['TRACK_ID', 'TRACKID', 'track_id', 'track'])
            if track_attr is not None:
                try:
                    tid = int(track_attr)
                except Exception:
                    continue
                track_groups[tid].append(sid)
    if track_groups:
        # sort spots in each track by frame
        for tid, sids in track_groups.items():
            sids_sorted = sorted(sids, key=lambda s: spots[s]['frame'] if spots[s]['frame'] is not None else 0)
            tracks[tid] = sids_sorted
        return tracks

    # 2) Parse <Track> elements and collect referenced spot ids via <Edge> or <SpotRef>
    found_any = False
    for el in root.iter():
        tag = getattr(el, 'tag', '')
        if isinstance(tag, str) and tag.lower().endswith('track'):
            attrs = el.attrib
            tid = get_attr_any(attrs, ['TRACK_ID', 'TRACKID', 'id', 'ID', 'Id'])
            try:
                tid = int(tid) if tid is not None else id(el)
            except Exception:
                tid = id(el)
            sids_set = set()
            # edge children
            for child in el:
                ctag = getattr(child, 'tag', '')
                if isinstance(ctag, str) and ctag.lower().endswith('edge'):
                    a = child.attrib
                    sid1 = get_attr_any(a, ['sourceID', 'source', 'SPOT_SOURCE_ID', 'SPOT_SOURCE'])
                    sid2 = get_attr_any(a, ['targetID', 'target', 'SPOT_TARGET_ID', 'SPOT_TARGET'])
                    try:
                        if sid1 is not None:
                            sids_set.add(int(sid1))
                        if sid2 is not None:
                            sids_set.add(int(sid2))
                    except Exception:
                        pass
                # SpotRef style
                if isinstance(ctag, str) and ctag.lower().endswith('spotref'):
                    a = child.attrib
                    sidr = get_attr_any(a, ['SPOT_ID', 'spotID', 'ID', 'id', 'SpotID'])
                    try:
                        if sidr is not None:
                            sids_set.add(int(sidr))
                    except Exception:
                        pass
            if sids_set:
                # order by frame
                sids_sorted = sorted(list(sids_set), key=lambda s: spots[s]['frame'] if spots[s]['frame'] is not None else 0)
                tracks[int(tid)] = sids_sorted
                found_any = True
    if found_any:
        return tracks

    # 3) Fallback: if there is an <AllTracks> with edges listed separately
    # collect all Edge elements that may have trackID attribute
    edge_map = defaultdict(set)  # trackid -> set of sids
    for el in root.iter():
        tag = getattr(el, 'tag', '')
        if isinstance(tag, str) and tag.lower().endswith('edge'):
            a = el.attrib
            tid = get_attr_any(a, ['TRACK_ID', 'track_id', 'TRACKID'])
            sid1 = get_attr_any(a, ['sourceID', 'source', 'SPOT_SOURCE_ID', 'SPOT_SOURCE'])
            sid2 = get_attr_any(a, ['targetID', 'target', 'SPOT_TARGET_ID', 'SPOT_TARGET'])
            try:
                if sid1 is not None:
                    sid1 = int(sid1)
                if sid2 is not None:
                    sid2 = int(sid2)
            except Exception:
                continue
            if tid is not None:
                try:
                    tid = int(tid)
                except Exception:
                    tid = str(tid)
                if sid1 is not None:
                    edge_map[tid].add(sid1)
                if sid2 is not None:
                    edge_map[tid].add(sid2)
    if edge_map:
        for tid, sids in edge_map.items():
            sids_sorted = sorted(list(sids), key=lambda s: spots[s]['frame'] if spots[s]['frame'] is not None else 0)
            tracks[int(tid)] = sids_sorted
        return tracks

    # 4) Last resort: no tracks defined => treat each spot as its own single-spot "track"
    # (not helpful for MSD but consistent)
    for sid in spots.keys():
        tracks[sid] = [sid]

    return tracks

def compute_msd_for_track(positions, max_lag=None):
    # positions: (N, dim)
    N = positions.shape[0]
    if max_lag is None:
        max_lag = N - 1
    else:
        max_lag = min(max_lag, N - 1)
    msd = np.zeros(max_lag, dtype=float)
    counts = np.zeros(max_lag, dtype=int)
    for lag in range(1, max_lag+1):
        diffs = positions[lag:] - positions[:-lag]  # shape (N-lag, dim)
        sqd = np.sum(diffs**2, axis=1)
        msd[lag-1] = np.mean(sqd) if sqd.size>0 else np.nan
        counts[lag-1] = sqd.size
    return msd, counts

def linear_fit(x, y):
    # simple linear regression returning slope and intercept
    # ignore nan
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan
    a, b = np.polyfit(x[mask], y[mask], 1)
    return a, b

def analyze_tracks(tracks, spots, dt=1.0, min_length=2, fit_lags=3, max_msd_lag=None):
    per_track_stats = []
    per_track_msd = {}
    # for ensemble: collect all msd arrays up to common max_lag
    ensemble_accumulator = []
    for tid, sids in tracks.items():
        if len(sids) < min_length:
            continue
        # build ordered positions and frames
        pts = []
        frames = []
        for sid in sids:
            sp = spots.get(sid)
            if sp is None:
                continue
            pts.append(sp['pos'])
            frames.append(sp['frame'] if sp['frame'] is not None else np.nan)
        if len(pts) < min_length:
            continue
        positions = np.vstack(pts)  # (N, dim)
        frames = np.array(frames, dtype=float)
        N, dim = positions.shape
        # compute step displacements and speeds
        deltas = positions[1:] - positions[:-1]
        step_lengths = np.sqrt(np.sum(deltas**2, axis=1))
        if np.all(np.isnan(frames)):
            # use uniform dt steps
            durations = (np.arange(N)-np.arange(N)[0]) * dt
            track_duration = (N-1) * dt
            speeds = step_lengths / dt
        else:
            # derive dt per step from frame differences if available
            frame_diffs = frames[1:] - frames[:-1]
            # avoid zero or nan:
            frame_diffs = np.where(np.isfinite(frame_diffs) & (frame_diffs>0), frame_diffs, 1.0)
            times_per_step = frame_diffs * dt
            speeds = step_lengths / times_per_step
            track_duration = (frames[-1] - frames[0]) * dt if np.isfinite(frames[-1]) and np.isfinite(frames[0]) else (N-1)*dt
        total_path = np.nansum(step_lengths)
        net_disp = math.dist(positions[0], positions[-1])
        straightness = net_disp/total_path if total_path>0 else np.nan
        mean_speed = np.nanmean(speeds) if speeds.size>0 else np.nan
        median_speed = np.nanmedian(speeds) if speeds.size>0 else np.nan
        max_speed = np.nanmax(speeds) if speeds.size>0 else np.nan
        # MSD
        msd, counts = compute_msd_for_track(positions, max_lag=max_msd_lag)
        per_track_msd[tid] = {'msd': msd, 'counts': counts, 'dt': dt, 'dim': dim}
        # diffusion fit on first fit_lags points
        n_fit = min(fit_lags, len(msd))
        if n_fit >= 2:
            times = np.arange(1, n_fit+1) * dt
            slope, intercept = linear_fit(times, msd[:n_fit])
            # for d-dimensional Brownian motion MSD = 2*d*D*t  (or 2*D*t in 1D). Many use MSD = 2nDt. For 2D: 4Dt.
            D = slope / (2.0 * dim) if np.isfinite(slope) else np.nan
        else:
            slope = intercept = D = np.nan
        per_track_stats.append({
            'track_id': tid,
            'n_spots': N,
            'dim': dim,
            'duration_s': track_duration,
            'total_path': total_path,
            'net_displacement': net_disp,
            'straightness': straightness,
            'mean_speed': mean_speed,
            'median_speed': median_speed,
            'max_speed': max_speed,
            'D_estimate': D,
            'msd_points': len(msd)
        })
        ensemble_accumulator.append(msd)
    # build ensemble MSD by averaging across tracks where counts exist
    if ensemble_accumulator:
        # determine common length
        max_len = max(len(m) for m in ensemble_accumulator)
        ensemble = np.full(max_len, np.nan)
        counts = np.zeros(max_len, dtype=int)
        sums = np.zeros(max_len, dtype=float)
        for m in ensemble_accumulator:
            L = len(m)
            sums[:L] += np.nan_to_num(m, nan=0.0)
            counts[:L] += (~np.isnan(m)).astype(int)
        mask = counts>0
        ensemble[mask] = sums[mask] / counts[mask]
    else:
        ensemble = np.array([])
        counts = np.array([])
    return per_track_stats, per_track_msd, ensemble, counts

def try_load_with_pytrackmate(path):
    try:
        import pytrackmate
    except Exception:
        return None
    # Unknown pytrackmate API across versions; attempt common entrypoints
    try:
        # Some versions have a load function
        tm = pytrackmate.load(path)
        # Expect tm.model.spots or similar; attempt to extract spots and tracks
        spots = {}
        tracks = {}
        # try to adapt common attributes
        model = getattr(tm, 'model', None) or getattr(tm, 'trackmodel', None) or tm
        # collect spots
        for s in getattr(model, 'spots', []):
            sid = int(getattr(s, 'id', getattr(s, 'ID', None)))
            x = float(getattr(s, 'x', getattr(s, 'POSITION_X', None)))
            y = float(getattr(s, 'y', getattr(s, 'POSITION_Y', None)))
            z = getattr(s, 'z', None)
            if z is not None:
                pos = np.array([x, y, float(z)])
            else:
                pos = np.array([x, y])
            frame = getattr(s, 'frame', getattr(s, 't', None))
            spots[sid] = {'pos': pos, 'frame': int(frame) if frame is not None else None}
        # tracks
        for tr in getattr(model, 'tracks', []):
            tid = int(getattr(tr, 'id', None))
            sids = [int(getattr(sp, 'id', getattr(sp, 'ID', None))) for sp in getattr(tr, 'spots', [])]
            tracks[tid] = sids
        if spots and tracks:
            return spots, tracks
    except Exception:
        return None
    return None

def main():
    # p = argparse.ArgumentParser(description='Compute MSD and track statistics from TrackMate XML.')
    # p.add_argument('xml', help='TrackMate XML file')
    # p.add_argument('--dt', type=float, default=1.0, help='time between frames in seconds (default 1.0)')
    # p.add_argument('--min-len', type=int, default=2, help='minimum spots per track to include')
    # p.add_argument('--fit-lags', type=int, default=3, help='number of initial lags to fit linear MSD for D estimate')
    # p.add_argument('--max-msd-lag', type=int, default=None, help='maximum lag (in frames) for MSD calculation (default use track length-1)')
    # p.add_argument('--out-prefix', default='trackmate_stats', help='output file prefix')
    # p.add_argument('--plot', action='store_true', help='show/save MSD plots if matplotlib available')
    # args = p.parse_args()
    file_path = r"Z:\Diffusion in Hydrogel Data\20mg_20nm\Trajektorien\ResultofB1_20nm_20mg_1d_nichtzentral_1_ohne_Tracks.xml"
    file_path = r"E:\PhD Data Analysis\SPT 2025 II\2025.11.27\tracks\20 nm_2_Tracks_2.xml"
    dx = 0.150  # µm per pixel
    class Args:
        dt = 0.05  # seconds
        min_len = 2
        dt = 0.05  # 50ms per frame
        fit_lags = 3
        max_msd_lag = None
        out_prefix = 'trackmate_stats_20nm_2_2'
        plot = True
    # Try pytrackmate
    loaded = try_load_with_pytrackmate(file_path)
    if loaded is not None:
        spots, tracks = loaded
    else:
        # parse XML using ElementTree
        tree = ET.parse(file_path)
        root = tree.getroot()
        spots = parse_spots_from_xml(root)
        if not spots:
            print('No spots found in XML. Exiting.', file=sys.stderr)
            sys.exit(1)
        tracks = parse_tracks_from_xml(root, spots)

    per_track_stats, per_track_msd, ensemble_msd, ensemble_counts = analyze_tracks(
        tracks, spots, dt=args.dt, min_length=args.min_len, fit_lags=args.fit_lags, max_msd_lag=args.max_msd_lag)

    # Save per-track stats
    df = pd.DataFrame(per_track_stats)
    df.sort_values('track_id', inplace=True)
    per_track_csv = f'{args.out_prefix}_per_track_stats.csv'
    df.to_csv(per_track_csv, index=False)
    # Save per-track msd arrays using numpy
    out_msd = {tid: {'msd': v['msd'], 'counts': v['counts'], 'dt': v['dt'], 'dim': v['dim']} for tid, v in per_track_msd.items()}
    np.save(f'{args.out_prefix}_per_track_msd.npy', out_msd, allow_pickle=True)
    # Save ensemble MSD
    if ensemble_msd.size > 0:
        times = np.arange(1, len(ensemble_msd)+1) * args.dt
        ens_df = pd.DataFrame({'lag_index': np.arange(1, len(ensemble_msd)+1), 'time_s': times, 'msd': ensemble_msd, 'counts': ensemble_counts})
        ens_df.to_csv(f'{args.out_prefix}_ensemble_msd.csv', index=False)
    # Optional plotting
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            # ensemble plot
            if ensemble_msd.size>0:
                plt.figure()
                times = np.arange(1, len(ensemble_msd)+1) * args.dt
                plt.plot(times, ensemble_msd, marker='o')
                plt.xlabel('Time (s)')
                plt.ylabel('MSD')
                plt.title('Ensemble MSD')
                plt.grid(True)
                plt.savefig(f'{args.out_prefix}_ensemble_msd.png', dpi=150)
            # sample some per-track MSDs
            plt.figure()
            plotted = 0
            for tid, d in list(per_track_msd.items()):
                msd = d['msd']
                if len(msd) < 2:
                    continue
                times = np.arange(1, len(msd)+1) * args.dt
                plt.plot(times, msd, label=f'track {tid}')
                plotted += 1
                if plotted >= 10:
                    break
            if plotted>0:
                plt.xlabel('Time (s)'); plt.ylabel('MSD'); plt.legend(); plt.grid(True)
                plt.savefig(f'{args.out_prefix}_sample_tracks_msd.png', dpi=150)
            plt.show()
        except Exception:
            print('matplotlib not available or plotting failed; plots skipped.', file=sys.stderr)

    print('Done. Outputs:')
    print(' -', per_track_csv)
    print(' -', f'{args.out_prefix}_per_track_msd.npy')
    print(' -', f'{args.out_prefix}_ensemble_msd.csv')

if __name__ == '__main__':
    main()