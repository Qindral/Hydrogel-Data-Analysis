"""Autofluorescence decays after laser switch-on: normalised intensity and half-time t1/2.

Requires the raw-trace CSVs and inspection.json produced by autofluorescence_inspect.py.
The intensity is the mean of all recorded pixels per frame (full image, no ROI).
Laser switches are detected from intensity jumps (detect_events); the QC figure shows them.
PCO_RAW bytes 4:20 are interpreted as little-endian Windows SYSTEMTIME.
This layout is inferred from these files and checked against calendar dates;
the timestamps are Camware recording timestamps, not validated exposure triggers.
SYSTEMTIME reference: https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-systemtime

Decay analysis (user, 2026-10-07; ENTSCHEIDUNGEN.md E011):
  every detected switch-on is one curve, including repeated switch-ons after an off-phase;
  I_norm = (I - I_dark) / (I_0 - I_dark), with I_dark the mean of the off-phase frames before
  the switch-on and I_0 the first frame after the transition frame (the transition frame is
  partly exposed and is not used); t = 0 at the I_0 frame;
  t1/2 = time of the first frame with I_norm <= 0.5 (no fit, no smoothing).
  The curve ends before the next detected switch or at the end of the recording.
  Legend (user, 2026-10-07; E012): gel concentration from the file name prefix (20/40/60 mg/mL),
  wavelength and laser power from the .rec comment; 20mg_ohnePartikel_B1 states no wavelength,
  514 nm is set in WAVELENGTH_NM_NOT_IN_REC.
  Pauses (switch-off followed by switch-on), arrow in the QC figure (user, 2026-10-07; E012):
  pause = time between the detected switch frames; rise = I_0 after the pause / I_before - 1 on raw
  counts (no dark subtraction), I_before = frame two before the detected switch-off.

Writes to OUTPUT_DIR (no pickle; no other script reads these outputs):
  autofluorescence_decays_normalised.pdf   all normalised decays in one plot
  autofluorescence_switch_qc.pdf           raw traces with the detected switches; pauses marked by an arrow;
                                           panel title = gel concentration, wavelength, laser power
                                           (user, 2026-10-14) instead of the file name
  autofluorescence_half_times.csv          t1/2 per switch-on (the table)
  autofluorescence_pause_rise.csv          pause duration and rise per pause
  autofluorescence_decays_normalised.csv   the plotted curves, long format
  autofluorescence_exclusions.csv          every file, frame range or file not used, with reason
  aligned_<stem>.csv, switch_events.csv, time_alignment.json  raw alignment, as before
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
import re
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from hydro_analysis.Auto_fluorecence.autofluorescence_inspect import DATA_DIR, OUTPUT_DIR as INSPECTION_DIR
from hydro_analysis.Auto_fluorecence.autofluorescence_inspect import new_figure, read_record

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR = INSPECTION_DIR.parent
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
PRE_SWITCH_SECONDS = 1.0      # aligned CSVs only; PROPOSED, not approved
MIN_JUMP_COUNTS = 10.0        # switch detection; PROPOSED, not approved
NOISE_MULTIPLIER = 10.0       # switch detection; PROPOSED, not approved
SIGNAL_RANGE_FRACTION = 0.08  # switch detection; PROPOSED, not approved
MERGE_GAP_FRAMES = 2          # switch detection; PROPOSED, not approved
DISCARD_FRAMES = 0            # every frame is retained
MANUAL_EVENTS = {}            # filename: [(zero_based_frame, "on" or "off"), ...]
I0_OFFSET_FRAMES = 1          # I_0 = first frame after the transition frame; decided by user, 2026-10-07
HALF_LEVEL = 0.5              # t1/2 at I_norm <= 0.5, first crossing; decided by user, 2026-10-07
WAVELENGTH_NM_NOT_IN_REC = {"20mg_ohnePartikel_B1.tif": 514}   # .rec states none; decided by user, 2026-10-07
PRE_PAUSE_OFFSET_FRAMES = 2   # I_before = frame (detected switch-off - 2); decided by user, 2026-10-07
# (gel concentration mg/mL, wavelength nm, power mW) -> colour; style guide §11, palette 2/3/7/4 dark
GROUP_COLORS = {(20, 514, 0.5): "#0035b2", (20, 488, 0.5): "#0089b2",
                (40, 488, 1.2): "#b24900", (60, 514, 0.6): "#33b279"}
TRACE_COLOR = "#2A4F66"       # style guide §11, single series dark
ARROW_COLOR = "#990000"       # style guide §11, palette 8 dark
QC_FIG_HEIGHT_IN = 6.0


# ── Data ───────────────────────────────────────────────────────────────────────
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


def laser_power_mw(tif_path):
    """The single laser power stated in the comment of the .rec file; error if absent or ambiguous."""
    record_name, text = read_record(tif_path)
    if text is None:
        raise FileNotFoundError(f"No .rec file for {tif_path.name}")
    comment = text.split("Comment:", 1)[-1]
    values = {float(v.replace(",", ".")) for v in re.findall(r"(\d+(?:[.,]\d+)?)\s*mW", comment)}
    if len(values) != 1:
        raise ValueError(f"{record_name}: expected one laser power in the comment, found {sorted(values)}")
    return values.pop(), " ".join(comment.split())


def wavelength_nm(tif_path, comment):
    """488 or 514 nm as stated in the .rec comment, else WAVELENGTH_NM_NOT_IN_REC; error if neither."""
    values = {int(v) for v in re.findall(r"(?<!\d)(488|514)(?!\d)", comment)}
    if len(values) > 1:
        raise ValueError(f"{tif_path.name}: several wavelengths in the .rec comment: {sorted(values)}")
    if values and tif_path.name in WAVELENGTH_NM_NOT_IN_REC:
        raise ValueError(f"{tif_path.name}: wavelength stated in the .rec and in WAVELENGTH_NM_NOT_IN_REC")
    if values:
        return values.pop(), "rec comment"
    if tif_path.name in WAVELENGTH_NM_NOT_IN_REC:
        return WAVELENGTH_NM_NOT_IN_REC[tif_path.name], "user, 2026-10-07 (not in .rec)"
    raise ValueError(f"{tif_path.name}: no wavelength in the .rec comment")


def gel_concentration_mg_ml(filename):
    """Gel concentration from the file name prefix (e.g. "40mg_..." -> 40 mg/mL)."""
    match = re.match(r"(\d+)mg_", filename)
    if match is None:
        raise ValueError(f"{filename}: no gel concentration in the file name")
    return int(match.group(1))


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


# ── Analysis ───────────────────────────────────────────────────────────────────
def normalise_switch_on(t, y, events, j):
    """Normalised decay and t1/2 of the switch-on events[j] (rules in the module docstring).

    Dark frames: from the frame after the preceding switch-off (or frame 0) to the frame before
    the transition frame; transition frames of both switches are excluded.
    """
    frame, direction = events[j]
    if direction != "on":
        raise ValueError("normalise_switch_on needs a switch-on event")
    if j and events[j - 1][1] != "off":
        raise ValueError(f"Switch-on {j + 1} is not preceded by a switch-off")
    dark_start = events[j - 1][0] + 1 if j else 0
    if frame - dark_start < 1:
        raise ValueError(f"No dark frame before switch-on {j + 1}")
    i0 = frame + I0_OFFSET_FRAMES
    end = events[j + 1][0] if j + 1 < len(events) else len(y)
    if end - i0 < 2:
        raise ValueError(f"Switch-on {j + 1} has fewer than two frames after I_0")
    dark = float(np.mean(y[dark_start:frame]))
    amplitude = float(y[i0]) - dark
    if amplitude <= 0:
        raise ValueError(f"I_0 does not exceed the dark level at switch-on {j + 1}")
    norm = (y[i0:end] - dark) / amplitude
    time = t[i0:end] - t[i0]
    below = np.flatnonzero(norm <= HALF_LEVEL)
    k = int(below[0]) if below.size else None
    return {"time_s": time, "norm": norm, "frames": np.arange(i0, end), "dark_counts": dark,
            "dark_frames": (dark_start, frame - 1), "i0_frame": i0, "i0_counts": float(y[i0]),
            "end_frame": end - 1, "t_half_s": float(time[k]) if k is not None else np.nan,
            "t_half_frame": i0 + k if k is not None else None,
            "t_half_resolution_s": float(time[k] - time[k - 1]) if k else np.nan}


def pause_rise(t, y, events, j):
    """Pause between the switch-off events[j - 1] and the switch-on events[j], and the rise of the raw
    intensity: I_0 after the pause relative to the frame PRE_PAUSE_OFFSET_FRAMES before the switch-off,
    without dark subtraction (user, 2026-10-07). Pause = time between the two detected switch frames."""
    (off, off_state), (on, on_state) = events[j - 1], events[j]
    if (off_state, on_state) != ("off", "on"):
        raise ValueError("pause_rise needs a switch-off followed by a switch-on")
    before, after = off - PRE_PAUSE_OFFSET_FRAMES, on + I0_OFFSET_FRAMES
    return {"before_frame": before, "I_before_counts": float(y[before]), "t_before_s": float(t[before]),
            "I0_frame": after, "I0_counts": float(y[after]), "t_I0_s": float(t[after]),
            "pause_s": float(t[on] - t[off]), "rise_percent": 100 * (float(y[after]) / float(y[before]) - 1)}


# ── Plotting ───────────────────────────────────────────────────────────────────
def group_label(group):
    conc, wavelength, power = group
    return f"{conc} mg/mL, {wavelength} nm, {power:g} mW"


def plot_decays(curves):
    fig, ax = new_figure("full")
    ax.axhline(HALF_LEVEL, color="0.65", lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.axhline(0, color="0.65", lw=0.6, zorder=0)
    for c in sorted(curves, key=lambda c: c["group"]):
        ax.plot(c["time_s"], c["norm"], color=GROUP_COLORS[c["group"]], lw=1.0, alpha=0.8,
                label=group_label(c["group"]))
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=False)
    ax.set_xlim(0, None)
    ax.set_xlabel("Time after switch-on t (s)")
    ax.set_ylabel(r"Normalised intensity $I_\mathrm{norm}$")
    return fig


def qc_title(group):
    conc, wavelength, power = group
    return f"{conc} mg mL$^{{-1}}$, {wavelength} nm, {power:g} mW"


def plot_qc(cached, pauses, groups):
    n = len(cached)
    ncols = 4
    nrows = -(-n // ncols)
    fig, axes = new_figure("full", nrows=nrows, ncols=ncols, height=QC_FIG_HEIGHT_IN)
    for ax, (name, (t, y, events)) in zip(axes.flat, cached.items()):
        ax.plot(t, y, color=TRACE_COLOR, lw=0.6)
        for frame, state in events:
            ax.axvline(t[frame], color="#b24900" if state == "on" else "0.4", ls=(0, (4, 3)), lw=0.6)
        for p in (p for p in pauses if p["file"] == name):
            ax.annotate("", xy=(p["t_I0_s"], p["I0_counts"]), xytext=(p["t_before_s"], p["I_before_counts"]),
                        arrowprops={"arrowstyle": "->", "color": ARROW_COLOR, "lw": 0.8,
                                    "shrinkA": 0, "shrinkB": 0})
            low, high = ax.get_ylim()
            ax.text((p["t_before_s"] + p["t_I0_s"]) / 2, p["I_before_counts"] + 0.04 * (high - low),
                    f"{p['pause_s']:.1f} s\n{p['rise_percent']:+.1f} %", ha="center", va="bottom",
                    fontsize=6, color=ARROW_COLOR, zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
        ax.set_title(qc_title(groups[name]), fontsize=7, fontweight="normal", pad=2)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.supxlabel("Recorded time (s)", fontsize=9)
    fig.supylabel("Mean pixel intensity (counts)", fontsize=9)
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inspection = json.loads((INSPECTION_DIR / "inspection.json").read_text(encoding="utf-8"))
    reports, event_rows, cached, curves, exclusions, pauses = [], [], {}, [], [], []
    groups = {}
    for name in inspection["orphan_record_files"]:
        exclusions.append({"file": name, "step": "inventory", "parameter": "", "n_before": 1, "n_after": 0,
                           "reason": ".rec file without TIFF recording"})
    for item in inspection["inventory"]:
        filename = item["file"]
        if item.get("duplicate_of"):
            exclusions.append({"file": filename, "step": "inventory", "parameter": "SHA-256",
                               "n_before": 1, "n_after": 0,
                               "reason": f"exact copy of {item['duplicate_of']}, counted once"})
            continue
        stem = Path(filename).stem
        trace = np.loadtxt(INSPECTION_DIR / f"raw_trace_{stem}.csv", delimiter=",", skiprows=1)
        y = trace[:, 1]
        events, threshold = detect_events(y)
        if filename in MANUAL_EVENTS:
            events = MANUAL_EVENTS[filename]
        if not events:
            reports.append({"file": filename, "status": "no switch detected; cannot align",
                            "threshold_counts": threshold})
            exclusions.append({"file": filename, "step": "switch detection", "parameter": f"{threshold:.1f} counts",
                               "n_before": len(y), "n_after": 0, "reason": "no laser switch detected; no decay"})
            print(filename, "NO SWITCH DETECTED", flush=True)
            continue
        if any(state not in ("on", "off") or not 0 < frame < len(y) for frame, state in events):
            raise ValueError(f"Invalid events: {filename}")
        if any(events[i][0] >= events[i+1][0] for i in range(len(events)-1)):
            raise ValueError(f"Events not strictly ordered: {filename}")
        t, timing = read_frame_times(DATA_DIR / filename)
        if len(t) != len(y):
            raise ValueError("Trace/timestamp length mismatch")
        power, comment = laser_power_mw(DATA_DIR / filename)
        wavelength, wavelength_source = wavelength_nm(DATA_DIR / filename, comment)
        group = (gel_concentration_mg_ml(filename), wavelength, power)
        if group not in GROUP_COLORS:
            raise ValueError(f"{filename}: no colour configured for {group_label(group)}")
        warnings = []
        if timing["irregular_interval_count"]:
            warnings.append("Irregular recording timestamps retained without interpolation")
        if any(events[i][1] == events[i+1][1] for i in range(len(events)-1)):
            warnings.append("Non-alternating events: inspect for missed switches or artifacts")
        if events[0][1] == "off":
            warnings.append("Recording starts during presumed on-phase; initial switch-on absent")
            exclusions.append({"file": filename, "step": "decay", "parameter": f"frames 0-{events[0][0] - 1}",
                               "n_before": events[0][0], "n_after": 0,
                               "reason": "on-phase without recorded switch-on: no I_dark, no I_0"})
        if events[-1][1] == "on":
            warnings.append("Last on-phase has no subsequent recorded switch-off")
        report = {"file": filename, "threshold_counts": threshold, "laser_power_mW": power,
                  "wavelength_nm": wavelength, "wavelength_source": wavelength_source,
                  "gel_concentration_mg_mL": group[0], "rec_comment": comment, "timing": timing, "events": [], "warnings": warnings}
        aligned_rows = []
        n_on = 0
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
            if direction != "on":
                exclusions.append({"file": filename, "step": "transition frame", "parameter": f"frame {frame}",
                                   "n_before": 1, "n_after": 0,
                                   "reason": f"switch-off {j + 1}: partly exposed, excluded from I_dark"})
                continue
            n_on += 1
            c = normalise_switch_on(t, y, events, j)
            c.update(file=filename, event=j + 1, switch_on=n_on, power_mW=power, group=group,
                     median_interval_s=timing["median_interval_s"])
            if j:
                pauses.append({"file": filename, "switch_off_event": j, "switch_on_event": j + 1,
                               **pause_rise(t, y, events, j)})
            curves.append(c)
            exclusions.append({"file": filename, "step": "transition frame", "parameter": f"frame {frame}",
                               "n_before": 1, "n_after": 0,
                               "reason": f"switch-on {j + 1}: partly exposed, I_0 taken at frame {c['i0_frame']}"})
        write_csv(OUTPUT_DIR / f"aligned_{stem}.csv", aligned_rows)
        reports.append(report)
        cached[filename] = (t, y, events)
        groups[filename] = group
        print(filename, events, f"{power:g} mW", "timing warnings:", timing["irregular_interval_count"], flush=True)

    table = [{"file": c["file"], "event": c["event"], "switch_on": c["switch_on"],
              "gel_concentration_mg_mL": c["group"][0], "wavelength_nm": c["group"][1],
              "laser_power_mW": c["power_mW"], "dark_frames": f"{c['dark_frames'][0]}-{c['dark_frames'][1]}",
              "I_dark_counts": round(c["dark_counts"], 2), "I0_frame": c["i0_frame"],
              "I0_counts": round(c["i0_counts"], 2), "last_frame": c["end_frame"],
              "curve_duration_s": round(float(c["time_s"][-1]), 3),
              "median_frame_interval_s": round(c["median_interval_s"], 4),
              "t_half_s": round(c["t_half_s"], 3), "t_half_frame": c["t_half_frame"],
              "t_half_resolution_s": round(c["t_half_resolution_s"], 4),
              "t_half_status": "reached" if c["t_half_frame"] is not None else "not reached within curve"}
             for c in curves]
    write_csv(OUTPUT_DIR / "autofluorescence_half_times.csv", table)
    write_csv(OUTPUT_DIR / "autofluorescence_decays_normalised.csv",
              [{"file": c["file"], "event": c["event"], "laser_power_mW": c["power_mW"], "frame_index": int(f),
                "time_after_I0_s": float(s), "I_norm": float(v)}
               for c in curves for f, s, v in zip(c["frames"], c["time_s"], c["norm"])])
    write_csv(OUTPUT_DIR / "autofluorescence_pause_rise.csv",
              [{k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()} for p in pauses])
    write_csv(OUTPUT_DIR / "autofluorescence_exclusions.csv", exclusions)
    write_csv(OUTPUT_DIR / "switch_events.csv", event_rows)

    with plt.style.context(STYLE_PATH):
        for stem, fig in [("autofluorescence_decays_normalised", plot_decays(curves)),
                          ("autofluorescence_switch_qc", plot_qc(cached, pauses, groups))]:
            fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
            plt.close(fig)

    sidecar = {"parameters": {"pre_switch_seconds": PRE_SWITCH_SECONDS,
               "minimum_jump_counts": MIN_JUMP_COUNTS, "noise_multiplier": NOISE_MULTIPLIER,
               "signal_range_fraction": SIGNAL_RANGE_FRACTION, "merge_gap_frames": MERGE_GAP_FRAMES,
               "discard_frames": DISCARD_FRAMES, "manual_events": MANUAL_EVENTS,
               "roi": "full recorded image", "smoothing": False,
               "normalisation": "(I - I_dark) / (I_0 - I_dark); I_dark = mean of the off-phase frames "
                                "before the switch-on; I_0 = frame after the transition frame",
               "t_half": f"first frame with I_norm <= {HALF_LEVEL}, time from the I_0 frame",
               "curves": "every detected switch-on",
               "pause_rise": "I_0 after the pause / frame (switch-off - 2) - 1, raw counts; "
                             "pause = time between the detected switch frames",
               "legend": "gel concentration (file name), wavelength (.rec), laser power (.rec)",
               "wavelength_not_in_rec": WAVELENGTH_NM_NOT_IN_REC,
               "time_source": "PCO_RAW inferred SYSTEMTIME at byte offset 4; recording timestamps",
               "timestamp_correction": "none", "style": str(STYLE_PATH)},
               "limitations": ["Laser state inferred from intensity; the QC figure requires inspection",
                               "Recording timestamps are not validated exposure-trigger timestamps",
                               "t1/2 resolution is one frame interval; no fit performed"],
               "files": reports}
    (OUTPUT_DIR / "time_alignment.json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    print(f"\n{len(curves)} switch-on curves; t1/2 table, 2 figures and exclusions written to {OUTPUT_DIR}")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
