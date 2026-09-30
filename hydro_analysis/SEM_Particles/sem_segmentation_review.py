"""
Interactive review of the Cellpose-SAM segmentation of all SEM particle images,
to remove mis-segmented particles, add missed ones and set per-image
thresholds before the Feret sizing.

Every annotated raw image of sem_feret_measure.DATA_DIR (same annotation choice
as sem_feret_measure.find_annotations) is shown in turn with the outlines of
its particles:
  magenta   particle kept (enters the size statistics)
  yellow    particle removed by click
  orange    particle below a threshold (see sliders)
  green     ellipse added by hand
  grey      excluded automatically by sem_feret_measure.py (touches the
            scanned-area border / FEI databar, or smaller than MIN_AREA_PX);
            these cannot be clicked
Controls (with the zoom/pan tool of the toolbar switched off):
  left click    remove the particle under the cursor (an added ellipse is deleted)
  right click   restore it; on a particle below a threshold it keeps the
                particle despite the threshold
  e             switch the ellipse tool on/off. Three left clicks add an
                ellipse: 1st and 2nd click = ends of the long axis, 3rd click =
                a point on the particle edge, which sets the width
  Escape        discard the points of an unfinished ellipse
  Enter         confirm this image, save, go to the next one
  b             save, go back one image
  q             save the current state and quit (resume later)
Sliders below the image (per image, saved with it):
  Brightness    minimum mean intensity of a particle, as fraction of the
                displayed grey range (CONTRAST_PERCENTILES of the scanned
                area); removes dark gaps between particles segmented as particles
  Roundness     minimum Feret_min / Feret_max
  Size          allowed range of the mean Feret diameter (Feret_min + Feret_max) / 2 in nm
Threshold values are computed on the displayed outlines (convex hull of the
contour); the final Feret sizing is done by sem_feret_measure.py.

Decisions are saved after every image into
  hydro_analysis/SEM_Particles/cache/sem_segmentation_review.json
as {"<size folder>/<raw image>": {"annotation", "removed_ids", "manual_removed",
"manual_kept", "thresholds", "added_ovals", "reviewed"}}. "removed_ids" is the
final set (clicked + below threshold - kept by hand) that sem_feret_measure.py
flags as removed_manually; particle ids are the label values of the mask (or
the 1-based position in an ImageJ ROI set), as in its "particle_id" column.
"added_ovals" are [cx, cy, a, b, theta] in pixels (ImageJ pixel-corner
coordinates, theta in rad); sem_feret_measure.py measures them as additional
particles. Entries written by the first version of this script (only
"removed_ids") are read as clicked removals.
In addition, the kept particles (incl. added ellipses) of every confirmed image
are written as an ImageJ ROI set for Fiji into
  hydro_analysis/SEM_Particles/cache/reviewed_rois/<size folder>/<raw stem>_reviewed_rois.zip
These zips are deliberately not stored next to the raw data, where
find_annotations() would pick them up as annotation.

The review starts at the first image not yet confirmed; if all are confirmed,
it starts again at the first image. When the last image has been confirmed and
RERUN_AFTER_REVIEW is set, all downstream results are recomputed in order:
sem_feret_measure.py, sem_feret_size_histogram.py,
sem_vs_dls_size_distribution.py, sem_example_cutout.py and
sem_review_before_after.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import roifile
import tifffile
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.widgets import RangeSlider, Slider
from skimage.draw import polygon as fill_polygon
from skimage.measure import find_contours, regionprops

from hydro_analysis.SEM_Particles.sem_feret_measure import (
    DATA_DIR, MIN_AREA_PX, REVIEW_PATH, SIZE_FOLDERS, ellipse_polygon, fei_scan_geometry, feret_min_max,
    find_annotations, load_review, review_key, save_review,
)

# ── Configuration ──────────────────────────────────────────────────────────────
ROI_EXPORT_DIR = Path(__file__).resolve().parent / "cache" / "reviewed_rois"
RERUN_AFTER_REVIEW = True

FIG_SIZE_IN = (13.0, 9.0)
CONTRAST_PERCENTILES = (0.5, 99.9)
COLOR_KEPT = "#da00bd"
COLOR_REMOVED = "#ffd400"
COLOR_THRESHOLD = "#ff7a00"
COLOR_ADDED = "#00e676"
COLOR_AUTO = "#9e9e9e"
OUTLINE_LW = 0.9


@dataclass
class ImageItem:
    folder_name: str
    raw_name: str
    ann_name: str

    @property
    def key(self) -> str:
        return review_key(self.folder_name, self.raw_name)


@dataclass
class Segmentation:
    """Outlines (x, y in ImageJ pixel-corner coordinates), flags and metrics of all particles of one image."""
    ids: list[int] = field(default_factory=list)
    outlines: list[np.ndarray] = field(default_factory=list)
    auto_excluded: list[bool] = field(default_factory=list)
    brightness: np.ndarray | None = None      # mean intensity, fraction of the displayed grey range
    roundness: np.ndarray | None = None       # Feret_min / Feret_max
    size_nm: np.ndarray | None = None         # mean Feret diameter
    labels: np.ndarray | None = None          # label mask, for hit testing (None for ROI sets)
    rois: dict | None = None                  # original ImageJ ROIs by id (ROI sets only)

    def particle_at(self, x: float, y: float) -> int | None:
        """Id of the particle under (x, y), or None."""
        if self.labels is not None:
            r, c = int(np.floor(y)), int(np.floor(x))
            if 0 <= r < self.labels.shape[0] and 0 <= c < self.labels.shape[1] and self.labels[r, c] > 0:
                return int(self.labels[r, c])
            return None
        hits = []
        for pid, xy in zip(self.ids, self.outlines):
            if xy[:, 0].min() <= x <= xy[:, 0].max() and xy[:, 1].min() <= y <= xy[:, 1].max():
                if MplPath(xy).contains_point((x, y)):
                    hits.append((abs(_polygon_area(xy)), pid))
        return min(hits)[1] if hits else None


def _polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ellipse_from_points(p1, p2, p3) -> list[float] | None:
    """[cx, cy, a, b, theta] from the long-axis ends p1, p2 and an edge point p3; None if p3 is not usable."""
    p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p1, p2, p3))
    centre = 0.5 * (p1 + p2)
    a = 0.5 * np.linalg.norm(p2 - p1)
    if a <= 0:
        return None
    theta = float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    d = p3 - centre
    u = d[0] * np.cos(theta) + d[1] * np.sin(theta)
    v = -d[0] * np.sin(theta) + d[1] * np.cos(theta)
    if abs(u) >= a:
        return None
    b = abs(v) / np.sqrt(1.0 - (u / a) ** 2)
    return None if b <= 0 else [float(centre[0]), float(centre[1]), float(a), float(b), theta]


def list_images() -> list[ImageItem]:
    items = []
    for folder_name in SIZE_FOLDERS.values():
        annotations, _ = find_annotations(DATA_DIR / folder_name)
        items += [ImageItem(folder_name, raw, ann) for raw, ann in annotations.items()]
    return items


def _metrics(xy: np.ndarray, mean_intensity: float, lo: float, hi: float, px_nm: float) -> tuple[float, float, float]:
    fmin, fmax = feret_min_max(xy)
    return (float(np.clip((mean_intensity - lo) / (hi - lo), 0.0, 1.0)), fmin / fmax if fmax > 0 else 0.0,
            0.5 * (fmin + fmax) * px_nm)


def load_segmentation(item: ImageItem, image: np.ndarray, width: int, height: int, px_nm: float,
                      lo: float, hi: float) -> Segmentation:
    """Particles of one annotation with the automatic exclusions of sem_feret_measure.py and threshold metrics."""
    path = DATA_DIR / item.folder_name / item.ann_name
    seg = Segmentation()
    metrics = []
    if path.suffix.lower() == ".zip":
        seg.rois = {}
        for i, roi in enumerate(roifile.roiread(path), start=1):
            xy = np.asarray(roi.coordinates(), dtype=float)
            if len(xy) < 3:
                continue
            touches = xy[:, 0].min() <= 0 or xy[:, 1].min() <= 0 or xy[:, 0].max() >= width or xy[:, 1].max() >= height
            rr, cc = fill_polygon(xy[:, 1] - 0.5, xy[:, 0] - 0.5, shape=image.shape)
            mean_int = image[rr, cc].mean() if rr.size else lo
            seg.ids.append(i)
            seg.outlines.append(np.vstack([xy, xy[:1]]))
            seg.auto_excluded.append(bool(touches or abs(_polygon_area(xy)) < MIN_AREA_PX))
            seg.rois[i] = roi
            metrics.append(_metrics(xy, mean_int, lo, hi, px_nm))
    else:
        seg.labels = tifffile.imread(path)
        for region in regionprops(seg.labels, intensity_image=image):
            rmin, cmin, rmax, cmax = region.bbox
            contours = find_contours(np.pad(region.image, 1).astype(float), 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)
            # Padded index p is array index p - 1, whose pixel centre lies at p - 0.5 in corner coordinates.
            xy = np.column_stack([contour[:, 1] - 0.5 + cmin, contour[:, 0] - 0.5 + rmin])
            touches = rmin == 0 or cmin == 0 or rmax >= height or cmax >= width
            seg.ids.append(region.label)
            seg.outlines.append(xy)
            seg.auto_excluded.append(bool(touches or region.area < MIN_AREA_PX))
            metrics.append(_metrics(xy, region.intensity_mean, lo, hi, px_nm))
    m = np.array(metrics, dtype=float).reshape(-1, 3)
    seg.brightness, seg.roundness, seg.size_nm = m[:, 0], m[:, 1], m[:, 2]
    return seg


def export_reviewed_rois(item: ImageItem, seg: Segmentation, removed: set[int], ovals: list[list[float]]) -> Path:
    """ImageJ ROI set of the kept particles (not removed, not automatically excluded) and the added ellipses."""
    out = ROI_EXPORT_DIR / item.folder_name / f"{Path(item.raw_name).stem}_reviewed_rois.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    rois = []
    for pid, xy, auto in zip(seg.ids, seg.outlines, seg.auto_excluded):
        if auto or pid in removed:
            continue
        if seg.rois is not None:
            rois.append(seg.rois[pid])
        else:
            rois.append(roifile.ImagejRoi.frompoints(xy.astype(np.float32), name=f"particle_{pid:05d}"))
    for k, oval in enumerate(ovals, start=1):
        rois.append(roifile.ImagejRoi.frompoints(ellipse_polygon(*oval).astype(np.float32), name=f"added_{k:04d}"))
    if out.exists():
        out.unlink()
    if rois:
        roifile.roiwrite(out, rois)
    return out


class Reviewer:
    """One figure that steps through all images; state is written to REVIEW_PATH after every image."""

    def __init__(self, items: list[ImageItem], review: dict, start: int):
        self.items, self.review, self.index = items, review, start
        self.finished = False
        self.oval_mode = False
        self.oval_points: list[tuple[float, float]] = []
        plt.rcParams["keymap.quit"] = []           # q is handled here (save before quitting)
        plt.rcParams["keymap.back"] = [k for k in plt.rcParams["keymap.back"] if k != "b"]
        self.fig = plt.figure(figsize=FIG_SIZE_IN)
        self.ax = self.fig.add_axes([0.01, 0.13, 0.98, 0.80])
        self.ax_bright = self.fig.add_axes([0.12, 0.085, 0.76, 0.022])
        self.ax_round = self.fig.add_axes([0.12, 0.055, 0.76, 0.022])
        self.ax_size = self.fig.add_axes([0.12, 0.025, 0.76, 0.022])
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.s_bright = Slider(self.ax_bright, "Brightness", 0.0, 1.0, valinit=0.0)
        self.s_round = Slider(self.ax_round, "Roundness", 0.0, 1.0, valinit=0.0)
        self.s_size = RangeSlider(self.ax_size, "Size (nm)", 0.0, 1.0, valinit=(0.0, 1.0))
        for slider in (self.s_bright, self.s_round, self.s_size):
            slider.on_changed(lambda _v: self.refresh())
        self.load()

    # -- image handling ---------------------------------------------------------
    def load(self) -> None:
        item = self.items[self.index]
        self.ax.clear()
        self.ax.set_title(f"loading {item.key} ...", fontsize=9)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        folder = DATA_DIR / item.folder_name
        px_nm, width, height = fei_scan_geometry(folder / item.raw_name)
        image = tifffile.imread(folder / item.raw_name).astype(float)
        image = image.mean(axis=2) if image.ndim == 3 else image
        lo, hi = np.percentile(image[:height, :width], CONTRAST_PERCENTILES)
        self.seg = load_segmentation(item, image, width, height, px_nm, lo, hi)
        self.id_to_index = {pid: i for i, pid in enumerate(self.seg.ids)}

        entry = self.review.get(item.key)
        entry = entry if entry and entry["annotation"] == item.ann_name else {}
        self.manual_removed = set(entry.get("manual_removed", entry.get("removed_ids", [])))
        self.manual_kept = set(entry.get("manual_kept", []))
        self.ovals = [list(o) for o in entry.get("added_ovals", [])]
        self.oval_points = []

        self.ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, extent=(0, image.shape[1], image.shape[0], 0))
        self.lines = LineCollection(self.seg.outlines, linewidths=OUTLINE_LW)
        self.ax.add_collection(self.lines)
        self.oval_lines = LineCollection([], colors=COLOR_ADDED, linewidths=OUTLINE_LW + 0.3)
        self.ax.add_collection(self.oval_lines)
        (self.point_marks,) = self.ax.plot([], [], "+", color=COLOR_ADDED, markersize=10, markeredgewidth=1.5)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.set_axis_off()
        self.set_sliders(entry.get("thresholds", {}))
        self.refresh()

    def set_sliders(self, saved: dict) -> None:
        """Size range of this image and the saved thresholds (defaults: nothing excluded)."""
        free = ~np.array(self.seg.auto_excluded, dtype=bool)
        size_max = float(np.ceil(1.05 * self.seg.size_nm[free].max())) if free.any() else 1.0
        self.s_size.valmax = size_max
        self.ax_size.set_xlim(0.0, size_max)
        for slider in (self.s_bright, self.s_round, self.s_size):
            slider.eventson = False
        self.s_bright.set_val(saved.get("brightness_min", 0.0))
        self.s_round.set_val(saved.get("roundness_min", 0.0))
        self.s_size.set_val((min(saved.get("size_min_nm", 0.0), size_max),
                             min(saved.get("size_max_nm", size_max), size_max)))
        for slider in (self.s_bright, self.s_round, self.s_size):
            slider.eventson = True

    def thresholds(self) -> dict:
        lo, hi = self.s_size.val
        return {"brightness_min": float(self.s_bright.val), "roundness_min": float(self.s_round.val),
                "size_min_nm": float(lo), "size_max_nm": float(hi)}

    def removed_ids(self) -> set[int]:
        """Clicked removals plus particles below a threshold, minus particles kept by hand."""
        t = self.thresholds()
        below = ((self.seg.brightness < t["brightness_min"]) | (self.seg.roundness < t["roundness_min"])
                 | (self.seg.size_nm < t["size_min_nm"]) | (self.seg.size_nm > t["size_max_nm"]))
        by_threshold = {pid for pid, b, auto in zip(self.seg.ids, below, self.seg.auto_excluded) if b and not auto}
        self.below_threshold = by_threshold
        return (self.manual_removed | by_threshold) - self.manual_kept

    def refresh(self) -> None:
        removed = self.removed_ids()
        colors = [COLOR_AUTO if auto else
                  COLOR_REMOVED if pid in self.manual_removed else
                  COLOR_THRESHOLD if pid in removed else COLOR_KEPT
                  for pid, auto in zip(self.seg.ids, self.seg.auto_excluded)]
        self.lines.set_colors(colors)
        self.oval_lines.set_segments([ellipse_polygon(*o) for o in self.ovals])
        pts = np.array(self.oval_points).reshape(-1, 2)
        self.point_marks.set_data(pts[:, 0], pts[:, 1])

        item = self.items[self.index]
        n_auto = sum(self.seg.auto_excluded)
        n_kept = len(self.seg.ids) - n_auto - len(removed) + len(self.ovals)
        status = "confirmed" if self.review.get(item.key, {}).get("reviewed") else "not confirmed"
        mode = (f"ELLIPSE TOOL: click {'long-axis end 1' if not self.oval_points else 'long-axis end 2' if len(self.oval_points) == 1 else 'a point on the edge'}"
                "   (e: tool off, Escape: discard points)" if self.oval_mode else
                "left: remove   right: restore/keep   e: ellipse tool   Enter: confirm + next   b: back   q: save + quit")
        self.ax.set_title(
            f"[{self.index + 1}/{len(self.items)}] {item.key}  ({item.ann_name}, {status})\n"
            f"kept {n_kept}   clicked away {len(self.manual_removed - self.manual_kept)}   "
            f"below threshold {len(removed - self.manual_removed)}   added {len(self.ovals)}   auto-excluded {n_auto}\n"
            f"{mode}", fontsize=9)
        self.fig.canvas.draw_idle()

    def store(self, confirmed: bool) -> None:
        item = self.items[self.index]
        previous = self.review.get(item.key, {})
        same_annotation = previous.get("annotation") == item.ann_name
        now = datetime.now().isoformat(timespec="seconds")
        reviewed = now if confirmed else (previous.get("reviewed") if same_annotation else None)
        removed = self.removed_ids()
        self.review[item.key] = {
            "annotation": item.ann_name,
            "removed_ids": sorted(int(i) for i in removed),
            "manual_removed": sorted(int(i) for i in self.manual_removed),
            "manual_kept": sorted(int(i) for i in self.manual_kept),
            "thresholds": self.thresholds(),
            "added_ovals": self.ovals,
            "reviewed": reviewed,
        }
        save_review(self.review)
        if confirmed:
            export_reviewed_rois(item, self.seg, removed, self.ovals)

    # -- events -------------------------------------------------------------------
    def _oval_at(self, x: float, y: float) -> int | None:
        for k, oval in enumerate(self.ovals):
            if MplPath(ellipse_polygon(*oval)).contains_point((x, y)):
                return k
        return None

    def on_click(self, event) -> None:
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if event.inaxes is not self.ax or event.xdata is None or (toolbar is not None and toolbar.mode):
            return
        x, y = event.xdata, event.ydata
        if self.oval_mode:
            if event.button != 1:
                return
            self.oval_points.append((x, y))
            if len(self.oval_points) == 3:
                oval = ellipse_from_points(*self.oval_points)
                if oval is None:
                    print("Ellipse not possible: the edge point must lie between the two long-axis ends "
                          "(measured along the axis). Points discarded.")
                else:
                    self.ovals.append(oval)
                self.oval_points = []
            self.refresh()
            return

        k = self._oval_at(x, y)
        if k is not None:
            if event.button == 1:
                del self.ovals[k]
                self.refresh()
            return
        pid = self.seg.particle_at(x, y)
        if pid is None or self.seg.auto_excluded[self.id_to_index[pid]]:
            return
        if event.button == 1:
            self.manual_removed.add(pid)
            self.manual_kept.discard(pid)
        elif event.button == 3:
            self.manual_removed.discard(pid)
            self.removed_ids()
            if pid in self.below_threshold:
                self.manual_kept.add(pid)
        self.refresh()

    def on_key(self, event) -> None:
        if event.key == "e":
            self.oval_mode = not self.oval_mode
            self.oval_points = []
            self.refresh()
        elif event.key == "escape":
            self.oval_points = []
            self.refresh()
        elif event.key == "enter":
            self.store(confirmed=True)
            if self.index + 1 < len(self.items):
                self.index += 1
                self.load()
            else:
                self.finished = True
                plt.close(self.fig)
        elif event.key == "b" and self.index > 0:
            self.store(confirmed=False)
            self.index -= 1
            self.load()
        elif event.key == "q":
            plt.close(self.fig)

    def on_close(self, _event) -> None:
        if not self.finished:
            self.store(confirmed=False)
            print(f"Review saved at image {self.index + 1}/{len(self.items)}: {REVIEW_PATH}")


def rerun_pipeline() -> None:
    """Recompute the measurement and every SEM figure from the reviewed segmentation."""
    from hydro_analysis.SEM_Particles import (
        sem_example_cutout, sem_feret_measure, sem_feret_size_histogram, sem_review_before_after,
        sem_vs_dls_size_distribution,
    )
    sem_feret_measure.main()
    sem_feret_size_histogram.main(show=False)
    sem_vs_dls_size_distribution.main(show=False)
    sem_example_cutout.main(show=False)
    sem_review_before_after.main(show=False)
    plt.close("all")


def main():
    """Run the review; recompute all results when every image has been confirmed."""
    items = list_images()
    review = load_review()
    pending = [i for i, it in enumerate(items)
               if not (review.get(it.key, {}).get("reviewed") and review[it.key]["annotation"] == it.ann_name)]
    start = pending[0] if pending else 0
    if not pending:
        print("All images are already confirmed; starting again at the first image.")
    print(f"{len(items)} images, starting at {start + 1}: {items[start].key}")

    reviewer = Reviewer(items, review, start)
    plt.show()

    confirmed = sum(1 for it in items if review.get(it.key, {}).get("reviewed")
                    and review[it.key]["annotation"] == it.ann_name)
    removed = sum(len(e["removed_ids"]) for e in review.values())
    added = sum(len(e.get("added_ovals", [])) for e in review.values())
    print(f"{confirmed}/{len(items)} images confirmed, {removed} particles removed, {added} ellipses added in total.")
    if reviewer.finished and confirmed == len(items) and RERUN_AFTER_REVIEW:
        print("Recomputing measurement and figures ...")
        rerun_pipeline()


if __name__ == "__main__":
    main()
