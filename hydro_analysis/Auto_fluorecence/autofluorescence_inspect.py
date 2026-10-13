"""Stage 1: inspect TIFF measurements before defining the scientific analysis.

No laser states, sample identities, timing, or controls are inferred. Frame means
are an explicitly provisional full-image preview, not an approved analysis ROI.
Run with the project's Python environment: python autofluorescence_inspect.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

DATA_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Autofluoreszenz")
OUTPUT_DIR = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
    r"\Experiments and Results - Data\autofluorescence\inspection"
)
STYLE_PATH = Path(__file__).resolve().parent / "hydro_analysis" / "thesis.mplstyle"
WIDTH_IN = {"full": 6.30, "narrow": 4.72, "half": 3.07, "third": 2.01}
RATIO = 1.42
PREVIEW_ROI = None  # None = entire recorded image; not yet approved for analysis.
HYDROGEL_COLOR = "#D98C3D"


def new_figure(kind="full", nrows=1, ncols=1, height=None, **kwargs):
    width = WIDTH_IN[kind]
    return plt.subplots(nrows, ncols, figsize=(width, height or width / RATIO),
                        layout="constrained", **kwargs)


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", dpi=600)
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=600)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(path):
    candidates = [Path(str(path) + ".rec"), path.with_suffix(".rec")]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        return None, None
    raw = source.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "cp1252"
    return source.name, raw.decode(encoding)


def inspect_stack(path):
    record_name, record_text = read_record(path)
    with tifffile.TiffFile(path) as stack:
        first = stack.pages[0]
        if len(first.shape) != 2 or not np.issubdtype(first.dtype, np.integer):
            raise ValueError(f"Expected integer grayscale frames: {path}")
        limit = np.iinfo(first.dtype).max
        values = []
        for frame, page in enumerate(stack.pages):
            pixels = page.asarray()
            if pixels.shape != first.shape or pixels.dtype != first.dtype:
                raise ValueError(f"Frame geometry or dtype changes: {path}, frame {frame}")
            if PREVIEW_ROI is not None:
                y0, y1, x0, x1 = PREVIEW_ROI
                if not (0 <= y0 < y1 <= pixels.shape[0] and 0 <= x0 < x1 <= pixels.shape[1]):
                    raise ValueError("Invalid preview ROI")
                pixels = pixels[y0:y1, x0:x1]
            values.append((frame, float(pixels.mean()), int(pixels.min()),
                           int(pixels.max()), float(np.mean(pixels == limit))))
        trace = np.asarray(values)
        info = {
            "file": path.name, "bytes": path.stat().st_size,
            "frames": len(stack.pages), "height_px": first.shape[0],
            "width_px": first.shape[1], "dtype": str(first.dtype),
            "series": [{"shape": list(s.shape), "axes": s.axes} for s in stack.series],
            "compression": first.compression.name,
            "description": first.description,
            "has_pco_raw_tag": "PCO_RAW" in first.tags,
            "record_file": record_name, "record_text": record_text,
            "frame_mean_min": float(trace[:, 1].min()),
            "frame_mean_max": float(trace[:, 1].max()),
            "pixel_min": int(trace[:, 2].min()), "pixel_max": int(trace[:, 3].max()),
            "maximum_dtype_ceiling_fraction": float(trace[:, 4].max()),
            "saturation_status": "unknown: camera ADC ceiling and alignment unconfirmed",
            "time_status": "frame index only; exposure is not assumed to equal frame interval",
        }
    return info, trace


def plot_trace(ax, trace):
    ax.plot(trace[:, 0], trace[:, 1], color=HYDROGEL_COLOR, lw=1.0)
    ax.set_xlabel("Frame number (zero-based)")
    ax.set_ylabel("Mean pixel intensity (counts)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def main():
    plt.style.use(STYLE_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.tif"), key=lambda p: (" - Kopie" in p.stem, p.name))
    if not files:
        raise FileNotFoundError(f"No TIFF files in {DATA_DIR}")
    seen, inventory, traces = {}, [], {}
    for path in files:
        digest = sha256_file(path)
        if digest in seen:
            inventory.append({"file": path.name, "sha256": digest,
                              "duplicate_of": seen[digest], "bytes": path.stat().st_size})
            print(f"Exact duplicate: {path.name} -> {seen[digest]}", flush=True)
            continue
        seen[digest] = path.name
        info, trace = inspect_stack(path)
        info.update(sha256=digest, duplicate_of=None)
        inventory.append(info)
        traces[path.name] = trace
        np.savetxt(OUTPUT_DIR / f"raw_trace_{path.stem}.csv", trace, delimiter=",",
                   header="frame_index,mean_intensity_counts,pixel_min,pixel_max,dtype_ceiling_fraction",
                   comments="", fmt=["%d", "%.10g", "%d", "%d", "%.10g"])
        fig, ax = new_figure("full")
        plot_trace(ax, trace)
        save_figure(fig, OUTPUT_DIR / f"raw_trace_{path.stem}")
        plt.close(fig)
        print(json.dumps({k: info[k] for k in ["file", "frames", "height_px", "width_px",
              "frame_mean_min", "frame_mean_max", "pixel_min", "pixel_max"]}), flush=True)

    representatives = ["20mg_ohnePartikel_B1_488nmLaser.tif", "40mg_autof.tif", "60mg_Auto_f_00.tif"]
    fig, axes = new_figure("full", nrows=3, height=7.0)
    for ax, name in zip(axes, representatives):
        if name in traces:
            plot_trace(ax, traces[name])
            ax.text(0.98, 0.05, name, transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    save_figure(fig, OUTPUT_DIR / "raw_preview_representatives")
    plt.close(fig)

    mapped_records = {item.get("record_file") for item in inventory}
    orphan_records = [p.name for p in sorted(DATA_DIR.glob("*.rec")) if p.name not in mapped_records]
    report = {
        "stage": "inspection only; analysis pending clarification",
        "parameters": {"data_dir": str(DATA_DIR), "output_dir": str(OUTPUT_DIR),
                       "style_path": str(STYLE_PATH), "preview_roi": PREVIEW_ROI,
                       "intensity_statistic": "arithmetic mean of all recorded pixels",
                       "time_axis": "zero-based frame index", "frame_discard": 0,
                       "duplicate_policy": "SHA-256-identical copies listed, not counted twice"},
        "inventory": inventory, "orphan_record_files": orphan_records,
        "unresolved": ["sample/position mapping", "controls", "analysis ROI", "switching frames",
                       "discard frames", "frame interval", "filter", "gain", "ADC saturation ceiling",
                       "SPT/FRAP comparability", "relevance threshold"],
        "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__,
                     "tifffile": tifffile.__version__},
    }
    (OUTPUT_DIR / "inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["file", "duplicate_of", "frames", "height_px", "width_px", "dtype",
               "frame_mean_min", "frame_mean_max", "pixel_min", "pixel_max", "sha256"]
    with (OUTPUT_DIR / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(inventory)
    print("Orphan records:", orphan_records)
    print("Output:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
