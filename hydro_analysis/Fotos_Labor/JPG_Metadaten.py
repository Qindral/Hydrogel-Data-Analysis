"""
JPG Metadaten.

Recursively finds all JPGs under ROOT_DIR, reads each one's EXIF capture
timestamp, and reports how many days after the first photo (t0) each
later photo was taken -- e.g. for a shrinking/drying series photographed
at irregular intervals.

Timestamp source, in order of preference:
  1. EXIF DateTimeOriginal (Exif IFD, tag 36867) -- when the shutter opened
  2. EXIF DateTime (top-level, tag 306) -- camera clock, same on this
     phone but kept as a fallback for cameras that only write this one
  3. The "IMG<YYYYMMDDHHMMSS>" pattern encoded in the filename (Android
     camera convention) -- used only if there is no EXIF at all
  4. Filesystem modification time, as a last resort -- flagged, since
     OneDrive sync can rewrite this independently of capture time

t0 = the earliest timestamp found across all photos; elapsed_days for
every photo is (timestamp - t0) in days (fractional, so same-day repeat
shots stay distinguishable).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from PIL.ExifTags import Base as ExifBase

# ── Configuration ───────────────────────────────────────────────────────────
ROOT_DIR = Path(
    r"C:\Users\Jonas\OneDrive - Universität Bayreuth\Promotion\Fotos_Labor\Shrinking"
)
OUT_CSV = Path(__file__).parent / "jpg_metadaten_shrinking.csv"
SAVE_PATH = Path(__file__).parent

EXIF_IFD_TAG = 0x8769           # ExifOffset / Exif IFD pointer
DATETIME_ORIGINAL_TAG = ExifBase.DateTimeOriginal.value   # 36867
DATETIME_TAG = ExifBase.DateTime.value                    # 306
EXIF_DT_FORMAT = "%Y:%m:%d %H:%M:%S"
FILENAME_PATTERN = re.compile(r"IMG(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

# ── Style (Style_guide.txt) ─────────────────────────────────────────────────
_RC = {
    "figure.figsize": (7.15, 5.00),
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Open Sans", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
}
COL_POINT = "#0000da"
COL_POINT_DARK = "#000099"


# ── Timestamp extraction ─────────────────────────────────────────────────────

def _parse_exif_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), EXIF_DT_FORMAT)
    except ValueError:
        return None


def _parse_filename_datetime(name: str) -> datetime | None:
    m = FILENAME_PATTERN.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


def read_jpg_timestamp(jpg_path: Path) -> dict:
    """Return {'timestamp': datetime, 'source': str} for one JPG."""
    with Image.open(jpg_path) as img:
        exif = img.getexif()
        exif_ifd = {}
        try:
            exif_ifd = exif.get_ifd(EXIF_IFD_TAG)
        except Exception:
            pass

    ts = _parse_exif_datetime(exif_ifd.get(DATETIME_ORIGINAL_TAG))
    if ts is not None:
        return {"timestamp": ts, "source": "EXIF DateTimeOriginal"}

    ts = _parse_exif_datetime(exif.get(DATETIME_TAG))
    if ts is not None:
        return {"timestamp": ts, "source": "EXIF DateTime"}

    ts = _parse_filename_datetime(jpg_path.name)
    if ts is not None:
        return {"timestamp": ts, "source": "filename pattern"}

    return {"timestamp": datetime.fromtimestamp(jpg_path.stat().st_mtime),
            "source": "filesystem mtime (UNRELIABLE -- no EXIF/filename timestamp found)"}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_timeline(df: pd.DataFrame, save_path: Path | None) -> None:
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(constrained_layout=True)

        ax.vlines(df["elapsed_days"], 0, 1, color=COL_POINT, linewidth=1.4, zorder=2)
        ax.plot(df["elapsed_days"], [1.0] * len(df), "o", markersize=6,
               markerfacecolor=COL_POINT, markeredgecolor=COL_POINT_DARK,
               markeredgewidth=0.8, zorder=3)

        for _, row in df.iterrows():
            ax.annotate(f"{row['elapsed_days']:.2f} d", xy=(row["elapsed_days"], 1.0),
                       xytext=(0, 6), textcoords="offset points", ha="center",
                       fontsize=7, rotation=90, va="bottom")

        ax.set_ylim(0, 1.6)
        ax.set_yticks([])
        ax.set_xlabel(r"Time since $t_0$ (days)")
        ax.set_title("Shrinking series — imaging timepoints", fontsize=11)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(left=False)

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "jpg_metadaten_timeline.png", dpi=600, bbox_inches="tight")
            fig.savefig(save_path / "jpg_metadaten_timeline.pdf")
            print(f"Saved: {save_path / 'jpg_metadaten_timeline.png'}")

        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not ROOT_DIR.exists():
        print(f"ERROR: folder not found: {ROOT_DIR}")
        return

    jpg_paths = sorted(set(ROOT_DIR.rglob("*.jpg")) | set(ROOT_DIR.rglob("*.jpeg")))
    print(f"Found {len(jpg_paths)} JPG file(s) under {ROOT_DIR}")
    if not jpg_paths:
        return

    rows = []
    for p in jpg_paths:
        info = read_jpg_timestamp(p)
        rows.append({
            "file": str(p),
            "folder": str(p.parent.relative_to(ROOT_DIR)) or ".",
            "filename": p.name,
            "timestamp": info["timestamp"],
            "source": info["source"],
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    unreliable = df[df["source"].str.contains("UNRELIABLE")]
    if not unreliable.empty:
        print(f"\nWARNING: {len(unreliable)} file(s) had no EXIF/filename timestamp; "
              f"using filesystem mtime, which OneDrive sync can change:")
        for _, row in unreliable.iterrows():
            print(f"  {row['filename']}")

    t0 = df["timestamp"].min()
    elapsed_days_exact = (df["timestamp"] - t0).dt.total_seconds() / 86400.0
    df["elapsed_days"] = elapsed_days_exact.round().astype(int)

    print(f"\nt0 = {t0}")
    table = df[["filename", "folder", "timestamp", "elapsed_days", "source"]]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(table.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
