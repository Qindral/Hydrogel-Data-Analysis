"""
SEM particle sizing from Cellpose-SAM annotations -- compute stage.

Measures the Feret diameters of every annotated particle in all size
subfolders of DATA_DIR ("20 nm", "50 nm", ..., "1000 nm") and stores one
per-particle table. The size measure used downstream is the mean Feret
diameter d_F = (Feret_max + Feret_min) / 2.

Input are the annotations produced in Fiji (Cellpose-SAM plugin) next to the
raw FEI SEM images, e.g.
  20 nm\\20nm_T1_02_Cellpose.tif                         (16-bit label mask)
  20 nm\\Latex particle_T2_20nm_004_ROIs_Corrected.tif.zip   (ImageJ ROI set)
A raw image is every TIFF carrying the FEI metadata block (tag 34682). Its
annotations are the .tif/.zip files whose name starts with the raw file stem;
exactly one is used per raw image, chosen by priority: ImageJ ROI set
(manually corrected) > file name containing "best" (optimised Cellpose
parameters) > any other Cellpose label mask. Raw images without annotation
are listed and skipped. Files not named after a raw image (RoiSet.zip at 1/4
scale, Test_Makro_006.zip with the reference ROIs of the parameter sweep) and
the overlays stored in "_best" TIFFs are never read.

Feret values follow the ImageJ definition: Feret_max is the largest and
Feret_min the smallest caliper width of the convex hull of the particle
outline in pixel-corner coordinates (masks: corners of the boundary pixels;
ROIs: polygon vertices). Pixel size is the FEI Scan/PixelWidth of the raw
image (identical to the ImageJ calibration of the label masks).

Exclusions (flagged in the table, not dropped):
  touches_border  particle touches the edge of the scanned area. The scanned
                  height is the FEI Image/ResolutionY; rows below it are the
                  FEI databar, whose text Cellpose partly segments as
                  particles. Those labels are excluded with this flag too.
  too_small       area < MIN_AREA_PX (the Cellpose min_size); removes the
                  1-2 px fragments left in the manually corrected ROI set of
                  20 nm image 004.
  removed_manually  particle rejected by hand in sem_segmentation_review.py
                  (mis-segmentations such as gaps between particles, merged
                  or partly hidden particles, debris). The decisions are read
                  from REVIEW_PATH (cache/sem_segmentation_review.json); an
                  entry is applied only if it was made on the same annotation
                  file that is used now, otherwise it is ignored with a warning.
                  "removed_ids" there is the final set, i.e. clicked particles
                  plus those below the brightness / roundness / size
                  thresholds of that image, minus particles kept by hand.

Particles added by hand in the review (ellipses from the 3-point tool, stored
as "added_ovals" [cx, cy, a, b, theta] in pixels) are appended to their image
with negative particle ids (-1, -2, ...) and added_manually = True; they are
measured like ROIs on an ELLIPSE_VERTICES-gon of the ellipse and pass the
same border and size checks.

Always recomputes from the annotation files (and the review file, if present)
and overwrites:
  hydro_analysis/SEM_Particles/cache/sem_feret_particles.pkl
      {"particles": per-particle DataFrame, "images": per-image summary DataFrame}
  <SAVE_PATH>/sem_feret_particles.xlsx   (same two tables as sheets)
Consumers (read the pickle only):
  sem_feret_size_histogram.py        per-size frequency histograms
  sem_vs_dls_size_distribution.py    comparison with the LiteSizer (DLS) distributions
  sem_example_cutout.py              example cutouts with outlines of the used particles
find_annotations(), fei_scan_geometry(), feret_min_max(), ellipse_polygon(),
load_review(), save_review() and added_ovals_for() are also used by
sem_segmentation_review.py and sem_example_cutout.py.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import roifile
import tifffile
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
from skimage.measure import regionprops

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = Path(r"E:\Daten Promotion Sicherung\Diffusion in Hydrogel Data\SEM Particles")
SIZE_FOLDERS = {20: "20 nm", 50: "50 nm", 100: "100 nm", 200: "200 nm", 500: "500 nm", 1000: "1000 nm"}

CACHE_PATH = Path(__file__).resolve().parent / "cache" / "sem_feret_particles.pkl"
REVIEW_PATH = Path(__file__).resolve().parent / "cache" / "sem_segmentation_review.json"
SAVE_PATH: Path | None = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "SEM_particle_size"

MIN_AREA_PX = 11                       # Cellpose min_size of run_parameters.ijm
ELLIPSE_VERTICES = 72                  # polygon resolution of hand-drawn ellipses
FEI_TAG = 34682


# ── Feret measurement ──────────────────────────────────────────────────────────

def feret_min_max(points: np.ndarray) -> tuple[float, float]:
    """Smallest and largest caliper width (px) of a 2-D point set.

    Rotating calipers on the convex hull: the maximum is the largest distance
    between two hull vertices, the minimum the smallest over all hull edges of
    the largest vertex distance to that edge's supporting line.
    """
    hull = points[ConvexHull(points).vertices]
    diff = hull[:, None, :] - hull[None, :, :]
    feret_max = float(np.sqrt((diff ** 2).sum(-1)).max())

    edges = np.roll(hull, -1, axis=0) - hull
    normals = np.column_stack([-edges[:, 1], edges[:, 0]]) / np.linalg.norm(edges, axis=1)[:, None]
    widths = np.abs(((hull[None, :, :] - hull[:, None, :]) * normals[:, None, :]).sum(-1)).max(axis=1)
    return float(widths.min()), feret_max


_PIXEL_CORNERS = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)


def fei_scan_geometry(raw_path: Path) -> tuple[float, int, int]:
    """Pixel size (nm) and scanned width/height (px, databar excluded) of a raw FEI TIFF."""
    with tifffile.TiffFile(raw_path) as tif:
        page = tif.pages[0]
        fei = page.tags[FEI_TAG].value
    px_nm = float(fei["Scan"]["PixelWidth"]) * 1e9
    height, width = page.shape[:2]
    scan_height = int(fei.get("Image", {}).get("ResolutionY", height))
    return px_nm, width, min(height, scan_height)


def measure_label_mask(path: Path, width: int, height: int) -> pd.DataFrame:
    """Feret values (px) of every label in a Cellpose label TIFF; scanned area width x height."""
    labels = tifffile.imread(path)
    rows = []
    for region in regionprops(labels):
        mask = region.image
        r, c = np.nonzero(mask & ~binary_erosion(mask))
        r0, c0 = region.bbox[:2]
        xy = np.column_stack([c + c0, r + r0]).astype(float)
        corners = (xy[:, None, :] + _PIXEL_CORNERS[None, :, :]).reshape(-1, 2)
        fmin, fmax = feret_min_max(corners)
        rmin, cmin, rmax, cmax = region.bbox                  # rmax, cmax exclusive
        touches = rmin == 0 or cmin == 0 or rmax >= height or cmax >= width
        rows.append({"particle_id": region.label, "feret_min_px": fmin, "feret_max_px": fmax,
                     "area_px": float(region.area), "touches_border": bool(touches)})
    return pd.DataFrame(rows)


def measure_roi_set(path: Path, width: int, height: int) -> pd.DataFrame:
    """Feret values (px) of every ROI in an ImageJ ROI zip; scanned area width x height."""
    rows = []
    for i, roi in enumerate(roifile.roiread(path), start=1):
        xy = np.asarray(roi.coordinates(), dtype=float)
        if len(xy) < 3:
            continue
        fmin, fmax = feret_min_max(xy)
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        touches = x.min() <= 0 or y.min() <= 0 or x.max() >= width or y.max() >= height
        rows.append({"particle_id": i, "feret_min_px": fmin, "feret_max_px": fmax,
                     "area_px": area, "touches_border": bool(touches)})
    return pd.DataFrame(rows)


# ── Annotation discovery ───────────────────────────────────────────────────────

def _is_fei_raw(path: Path) -> bool:
    with tifffile.TiffFile(path) as tif:
        return FEI_TAG in tif.pages[0].tags


def _annotation_priority(name: str) -> int:
    if name.lower().endswith(".zip"):
        return 0
    return 1 if "best" in name.lower() else 2


def find_annotations(folder: Path) -> tuple[dict[str, str], list[str]]:
    """{raw image name: chosen annotation name} of one folder, and the raw images without annotation."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".zip"))
    raws = [p for p in files if p.suffix.lower() == ".tif" and _is_fei_raw(p)]
    chosen, missing = {}, []
    for raw in raws:
        stem = raw.stem
        candidates = [p.name for p in files if p not in raws and p.name.startswith(stem)
                      and p.name[len(stem):len(stem) + 1] in (".", "_", " ")]
        if candidates:
            chosen[raw.name] = min(candidates, key=_annotation_priority)
        else:
            missing.append(raw.name)
    return chosen, missing


# ── Manual review ──────────────────────────────────────────────────────────────

def review_key(folder_name: str, raw_name: str) -> str:
    return f"{folder_name}/{raw_name}"


def load_review(path: Path = REVIEW_PATH) -> dict:
    """{"<size folder>/<raw image>": {"annotation", "removed_ids", "reviewed"}}; empty if no review exists."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_review(review: dict, path: Path = REVIEW_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(review, fh, indent=1, sort_keys=True)
    tmp.replace(path)


def _review_entry(review: dict, folder_name: str, raw_name: str, ann_name: str) -> dict | None:
    entry = review.get(review_key(folder_name, raw_name))
    if entry is None:
        return None
    if entry["annotation"] != ann_name:
        print(f"  [WARNING] review of {raw_name} was made on {entry['annotation']}, now {ann_name} is used; "
              "manual review ignored")
        return None
    return entry


def ellipse_polygon(cx: float, cy: float, a: float, b: float, theta: float,
                    n: int = ELLIPSE_VERTICES) -> np.ndarray:
    """Closed-ring vertices (n x 2, x/y in px) of an ellipse with semi-axes a, b rotated by theta (rad)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    return np.column_stack([cx + x * np.cos(theta) - y * np.sin(theta),
                            cy + x * np.sin(theta) + y * np.cos(theta)])


def added_ovals_for(review: dict, folder_name: str, raw_name: str, ann_name: str) -> list[list[float]]:
    """Hand-drawn ellipses [cx, cy, a, b, theta] of one image (empty if none or annotation changed)."""
    entry = _review_entry(review, folder_name, raw_name, ann_name)
    return [] if entry is None else entry.get("added_ovals", [])


def measure_ovals(ovals: list[list[float]], width: int, height: int) -> pd.DataFrame:
    """Feret values (px) of hand-drawn ellipses, ids -1, -2, ... in drawing order."""
    rows = []
    for k, (cx, cy, a, b, theta) in enumerate(ovals, start=1):
        xy = ellipse_polygon(cx, cy, a, b, theta)
        fmin, fmax = feret_min_max(xy)
        touches = xy[:, 0].min() <= 0 or xy[:, 1].min() <= 0 or xy[:, 0].max() >= width or xy[:, 1].max() >= height
        rows.append({"particle_id": -k, "feret_min_px": fmin, "feret_max_px": fmax,
                     "area_px": float(np.pi * a * b), "touches_border": bool(touches)})
    return pd.DataFrame(rows, columns=["particle_id", "feret_min_px", "feret_max_px", "area_px", "touches_border"])


def removed_ids_for(review: dict, folder_name: str, raw_name: str, ann_name: str) -> set[int]:
    """Removed particle ids of one image, if its review was made on the annotation used now."""
    entry = review.get(review_key(folder_name, raw_name))
    if entry is None or entry["annotation"] != ann_name:
        return set()
    return set(entry["removed_ids"])


# ── Pipeline ───────────────────────────────────────────────────────────────────

def measure_all(data_dir: Path, size_folders: dict[int, str]) -> pd.DataFrame:
    """Per-particle table over all size folders, sizes in nm; exclusions are flagged, not dropped."""
    tables = []
    review = load_review()
    for nominal, folder_name in size_folders.items():
        folder = data_dir / folder_name
        annotations, missing = find_annotations(folder)
        print(f"\n{folder_name}: {len(annotations)} annotated image(s)")
        for raw_name, ann_name in annotations.items():
            print(f"  {raw_name}  <-  {ann_name}")
        for raw_name in missing:
            print(f"  {raw_name}  (no annotation, skipped)")

        for raw_name, ann_name in annotations.items():
            px_nm, width, height = fei_scan_geometry(folder / raw_name)
            ann_path = folder / ann_name
            if ann_path.suffix.lower() == ".zip":
                df = measure_roi_set(ann_path, width, height)
            else:
                df = measure_label_mask(ann_path, width, height)
            df["added_manually"] = False
            ovals = measure_ovals(added_ovals_for(review, folder_name, raw_name, ann_name), width, height)
            if len(ovals):
                ovals["added_manually"] = True
                df = pd.concat([df, ovals], ignore_index=True)
            df.insert(0, "nominal_nm", nominal)
            df.insert(1, "image", raw_name)
            df.insert(2, "annotation", ann_name)
            df["pixel_size_nm"] = px_nm
            df["feret_min_nm"] = df["feret_min_px"] * px_nm
            df["feret_max_nm"] = df["feret_max_px"] * px_nm
            df["feret_mean_nm"] = 0.5 * (df["feret_min_nm"] + df["feret_max_nm"])
            df["area_nm2"] = df["area_px"] * px_nm ** 2
            df["too_small"] = df["area_px"] < MIN_AREA_PX
            df["removed_manually"] = df["particle_id"].isin(removed_ids_for(review, folder_name, raw_name, ann_name))
            df["used"] = ~(df["touches_border"] | df["too_small"] | df["removed_manually"])
            tables.append(df)
    return pd.concat(tables, ignore_index=True)


def summarize_images(particles: pd.DataFrame) -> pd.DataFrame:
    """Per-image count, exclusions and mean Feret statistics of the used particles."""
    rows = []
    for (nominal, image, ann), grp in particles.groupby(["nominal_nm", "image", "annotation"], sort=False):
        kept = grp.loc[grp["used"], "feret_mean_nm"]
        rows.append({"nominal_nm": nominal, "image": image, "annotation": ann,
                     "pixel_size_nm": grp["pixel_size_nm"].iloc[0],
                     "n_annotated": len(grp), "n_border_excluded": int(grp["touches_border"].sum()),
                     "n_too_small": int((grp["too_small"] & ~grp["touches_border"]).sum()),
                     "n_removed_manually": int((grp["removed_manually"] & ~grp["touches_border"]
                                                & ~grp["too_small"]).sum()),
                     "n_added_manually": int(grp["added_manually"].sum()),
                     "n_used": len(kept), "feret_mean_mean_nm": kept.mean(),
                     "feret_mean_sd_nm": kept.std(ddof=1), "feret_mean_median_nm": kept.median()})
    return pd.DataFrame(rows)


def main():
    """Measure all annotations and write the cache pickle and the per-particle workbook."""
    particles = measure_all(DATA_DIR, SIZE_FOLDERS)
    images = summarize_images(particles)

    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 2):
        print()
        print(images.drop(columns="annotation").to_string(index=False))

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump({"particles": particles, "images": images}, fh)
    print(f"\nCache written: {CACHE_PATH}")

    if SAVE_PATH is not None:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(SAVE_PATH / "sem_feret_particles.xlsx") as writer:
            particles.to_excel(writer, sheet_name="particles", index=False)
            images.to_excel(writer, sheet_name="images", index=False)
        print(f"Workbook written: {SAVE_PATH / 'sem_feret_particles.xlsx'}")
    return particles, images


if __name__ == "__main__":
    main()
