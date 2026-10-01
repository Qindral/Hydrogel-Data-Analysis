"""Plot raw intensity against recorded time, aligned to laser on/off events.

This implements the requested alignment stage, not control correction or fits.
Requires the raw-trace CSVs produced by autofluorescence_inspect.py.
PCO_RAW bytes 4:20 are interpreted as little-endian Windows SYSTEMTIME.
This layout is inferred from these files and checked against calendar dates;
the timestamps are Camware recording timestamps, not validated exposure triggers.
SYSTEMTIME reference: https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-systemtime
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from autofluorescence_inspect import DATA_DIR, OUTPUT_DIR as INSPECTION_DIR
from autofluorescence_inspect import STYLE_PATH, new_figure, save_figure

OUTPUT_DIR = INSPECTION_DIR.parent / "time_aligned"
PRE_SWITCH_SECONDS = 1.0
MIN_JUMP_COUNTS = 10.0
NOISE_MULTIPLIER = 10.0
SIGNAL_RANGE_FRACTION = 0.08
MERGE_GAP_FRAMES = 2
DISCARD_FRAMES = 0  # Raw preview: every frame is retained.
HYDROGEL_COLOR = "#D98C3D"
OFF_COLOR = "#555555"
MANUAL_EVENTS = {}  # filename: [(zero_based_frame, "on" or "off"), ...]


def read_frame_times(path):
    stamps = []
    with tifffile.TiffFile(path) as stack:
        for page in stack.pages:
            tag = page.tags.get("PCO_RAW")
            if tag is None or len(tag.value) < 20:
                raise ValueError(f"Missing PCO timestamp: {path.name}")
            year, month, weekday, day, hour, minute, second, millis = struct.unpack_from("<8H", tag.value, 4)
            if not 0 <= millis < 1000:
                raise ValueError("Invalid timestamp milliseconds")
            stamp = datetime(year, month, day, hour, minute, second, millis * 1000)
            if (stamp.weekday() + 1) % 7 != weekday:
                raise ValueError(f"Invalid timestamp weekday: {path.name}")
            stamps.append(stamp)
    t = np.array([(stamp - stamps[0]).total_seconds() for stamp in stamps])
    dt = np.diff(t)
    if np.any(dt < 0):
        raise ValueError(f"Decreasing timestamps: {path.name}")
    median_dt = float(np.median(dt))
    qc = {"first_timestamp": stamps[0].isoformat(), "duration_s": float(t[-1]),
          "median_interval_s": median_dt, "minimum_interval_s": float(dt.min()),
          "maximum_interval_s": float(dt.max()),
          "duplicate_timestamp_count": int(np.sum(dt == 0)),
          "irregular_interval_count": int(np.sum((dt < 0.5 * median_dt) | (dt > 1.5 * median_dt)))}
    return t, qc


def detect_events(y):
    difference = np.diff(y)
    noise = 1.4826 * np.median(np.abs(difference - np.median(difference)))
    threshold = max(MIN_JUMP_COUNTS, NOISE_MULTIPLIER * noise,
                    SIGNAL_RANGE_FRACTION * (np.quantile(y, .99) - np.quantile(y, .01)))
    candidates = np.flatnonzero(np.abs(difference) > threshold) + 1
    groups = []
    for i in candidates:
        direction = "on" if difference[i - 1] > 0 else "off"
        if groups and i - groups[-1][-1][0] <= MERGE_GAP_FRAMES and direction == groups[-1][-1][1]:
            groups[-1].append((int(i), direction))
        else:
            groups.append([(int(i), direction)])
    return [group[0] for group in groups], float(threshold)


def plot_aligned(ax, t, y, events, direction):
    count = 0
    for j, (frame, state) in enumerate(events):
        if state != direction:
            continue
        count += 1
        previous_frame = events[j - 1][0] if j else 0
        next_frame = events[j + 1][0] if j + 1 < len(events) else len(y)
        start = max(previous_frame, int(np.searchsorted(t, t[frame] - PRE_SWITCH_SECONDS)))
        index = np.arange(start, next_frame)
        ax.plot(t[index] - t[frame], y[index], color=HYDROGEL_COLOR if state == "on" else OFF_COLOR,
                ls=["-", "--", ":", "-."][(count - 1) % 4], lw=1.0,
                label=f"Switch {count}")
    ax.axvline(0, color="0.5", ls=(0, (4, 3)), lw=.8, zorder=0)
    ax.set_xlabel(f"Time from laser switch-{direction} (s)")
    ax.set_ylabel("Mean pixel intensity (counts)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if count > 1:
        ax.legend(loc="upper right")


def main():
    plt.style.use(STYLE_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inspection = json.loads((INSPECTION_DIR / "inspection.json").read_text(encoding="utf-8"))
    reports, event_rows, cached = [], [], {}
    for item in inspection["inventory"]:
        if item.get("duplicate_of"):
            continue
        filename = item["file"]
        stem = Path(filename).stem
        trace = np.loadtxt(INSPECTION_DIR / f"raw_trace_{stem}.csv", delimiter=",", skiprows=1)
        y = trace[:, 1]
        events, threshold = detect_events(y)
        if filename in MANUAL_EVENTS:
            events = MANUAL_EVENTS[filename]
        if not events:
            reports.append({"file": filename, "status": "no switch detected; cannot align",
                            "threshold_counts": threshold})
            print(filename, "NO SWITCH DETECTED", flush=True)
            continue
        if any(state not in ("on", "off") or not 0 < frame < len(y) for frame, state in events):
            raise ValueError(f"Invalid events: {filename}")
        if any(events[i][0] >= events[i+1][0] for i in range(len(events)-1)):
            raise ValueError(f"Events not strictly ordered: {filename}")
        t, timing = read_frame_times(DATA_DIR / filename)
        if len(t) != len(y):
            raise ValueError("Trace/timestamp length mismatch")
        warnings = []
        if timing["irregular_interval_count"]:
            warnings.append("Irregular recording timestamps retained without interpolation")
        if any(events[i][1] == events[i+1][1] for i in range(len(events)-1)):
            warnings.append("Non-alternating events: inspect for missed switches or artifacts")
        if events[0][1] == "off":
            warnings.append("Recording starts during presumed on-phase; initial switch-on absent")
        if events[-1][1] == "on":
            warnings.append("Last on-phase has no subsequent recorded switch-off")
        report = {"file": filename, "threshold_counts": threshold, "timing": timing,
                  "events": [], "warnings": warnings}
        aligned_rows = []
        for j, (frame, direction) in enumerate(events):
            event = {"file": filename, "event": j + 1, "direction": direction,
                     "frame_index": int(frame), "recorded_time_s": float(t[frame]),
                     "jump_counts": float(y[frame] - y[frame-1]),
                     "preceding_frame_interval_s": float(t[frame] - t[frame-1])}
            event_rows.append(event)
            report["events"].append(event)
            start = max(events[j-1][0] if j else 0, int(np.searchsorted(t, t[frame] - PRE_SWITCH_SECONDS)))
            end = events[j+1][0] if j+1 < len(events) else len(y)
            for i in range(start, end):
                aligned_rows.append({"event": j+1, "direction": direction, "frame_index": i,
                                     "time_from_switch_s": float(t[i]-t[frame]),
                                     "mean_intensity_counts": float(y[i])})
        write_csv(OUTPUT_DIR / f"aligned_{stem}.csv", aligned_rows)
        directions = list(dict.fromkeys(state for _, state in events))
        for direction in directions:
            fig, ax = new_figure("full")
            plot_aligned(ax, t, y, events, direction)
            save_figure(fig, OUTPUT_DIR / f"laser_{direction}_{stem}")
            plt.close(fig)
        fig, ax = new_figure("full")
        ax.plot(t, y, color=HYDROGEL_COLOR, lw=1)
        for frame, state in events:
            ax.axvline(t[frame], color="0.4", ls="--", lw=.8)
            ax.text(t[frame], .98, state, transform=ax.get_xaxis_transform(), va="top", fontsize=8)
        on = events[0][1] == "off"
        left = 0.0
        for frame, state in events:
            if on:
                ax.axvspan(left, t[frame], color="0.5", alpha=.12, lw=0)
            on, left = state == "on", t[frame]
        if on:
            ax.axvspan(left, t[-1], color="0.5", alpha=.12, lw=0)
        ax.set_xlabel("Recorded time (s)")
        ax.set_ylabel("Mean pixel intensity (counts)")
        save_figure(fig, OUTPUT_DIR / f"qc_switches_{stem}")
        plt.close(fig)
        reports.append(report)
        cached[filename] = (t, y, events)
        print(filename, events, "timing warnings:", timing["irregular_interval_count"], flush=True)

    for group in ["20mg", "40mg", "60mg", "switch_off"]:
        selected = [(name, values) for name, values in cached.items()
                    if (any(s == "off" for _, s in values[2]) if group == "switch_off" else name.startswith(group))]
        direction = "off" if group == "switch_off" else "on"
        fig, axes = new_figure("full", nrows=2, ncols=2, height=5.2)
        for ax, (name, values) in zip(axes.flat, selected):
            plot_aligned(ax, *values, direction)
            short = name.replace("20mg_ohnePartikel_", "20mg_").replace(".tif", "")
            ax.text(.97, .95 if direction == "off" else .05, short,
                    transform=ax.transAxes, va="top" if direction == "off" else "bottom",
                    ha="right", fontsize=7)
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        save_figure(fig, OUTPUT_DIR / f"overview_{group}")
        plt.close(fig)
    write_csv(OUTPUT_DIR / "switch_events.csv", event_rows)
    sidecar = {"parameters": {"pre_switch_seconds": PRE_SWITCH_SECONDS,
               "minimum_jump_counts": MIN_JUMP_COUNTS, "noise_multiplier": NOISE_MULTIPLIER,
               "signal_range_fraction": SIGNAL_RANGE_FRACTION, "merge_gap_frames": MERGE_GAP_FRAMES,
               "discard_frames": DISCARD_FRAMES, "manual_events": MANUAL_EVENTS,
               "roi": "full recorded image", "baseline_subtraction": False,
               "smoothing": False, "t0": "first frame of detected transition",
               "time_source": "PCO_RAW inferred SYSTEMTIME at byte offset 4; recording timestamps",
               "phase_window": "1 s before switch, truncated at adjacent switches",
               "timestamp_correction": "none", "style": str(STYLE_PATH)},
               "limitations": ["Laser state inferred from intensity; QC plots require inspection",
                               "Recording timestamps are not validated exposure-trigger timestamps",
                               "Alignment uncertainty includes one frame and recording-time jitter",
                               "No sample/control assignments or bleaching fits performed"],
               "files": reports}
    (OUTPUT_DIR / "time_alignment.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print("Output:", OUTPUT_DIR)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
