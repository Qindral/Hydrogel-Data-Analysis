"""Select an illustrative weak tracked 20 nm hydrogel particle and show raw/DoG.
Selection is intentionally for improvement, not a representative SNR estimate.
Filtering and radius conventions are shared with snr_signal_profile.py.

Figure (Abb16, Styleguide v2, width class full, PNG 600 dpi): A full raw frame,
as tall as B (raw cutout) and C (DoG cutout) together; only panel labels and
scalebars (lower right, grey box). The local SNR values are printed, not drawn.
Writes the figure, the candidate table and the selection JSON to OUT; the raw
and DoG frames as float TIFF to SCRATCH. No other script reads these outputs.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import (
    FILES, find_session_and_tif, dog_filter, MPP_BY_IMAGE_WIDTH_PX)

# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data")
OUT = RESULTS / "SPT"
SCRATCH = RESULTS / "_scratch" / "snr_hydrogel_20nm_example"
FIG_NAME = "hydrogel_20nm_weak_particle_dog"
STYLE = Path(__file__).resolve().parents[3] / "thesis.mplstyle"
FIG_WIDTH_IN = 6.30          # Styleguide width class "full"
PANEL_GAP_IN = 0.06
SCALEBAR_FRAME_UM = 5.0
SCALEBAR_CUTOUT_UM = 1.0
COLOR_MARK = "#e69f00"       # cutout box in A and centre brackets in B/C (unchanged)


def _scalebar(ax, mpp, length_um):
    """Scalebar lower right, white bar and 8 pt text on a grey box (as Abb01/Abb02)."""
    ax.add_artist(ScaleBar(
        mpp, units="um", dimension="si-length", fixed_value=length_um, fixed_units="um",
        location="lower right", width_fraction=0.02, sep=2, pad=0.4, border_pad=0.5,
        frameon=True, color="white", box_color="#555555", box_alpha=0.85, scale_loc="top",
        font_properties={"size": 8, "family": "Open Sans"},
        scale_formatter=lambda value, unit: f"{value:g} {unit}"))


def _panel_label(ax, letter):
    ax.text(0.03, 0.97, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="white")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    session, tif = find_session_and_tif(Path(FILES[("hydrogel", 20.0)]))
    root = ET.parse(session).getroot()
    detector = root.find("Settings/DetectorSettings").attrib
    image_info = root.find("Settings/ImageData").attrib
    calibrated = "pixel" not in root.find("Model").get("spatialunits", "pixel").lower()
    px, py = (float(image_info["pixelwidth"]), float(image_info["pixelheight"])) if calibrated else (1., 1.)
    radius = float(detector["RADIUS"])
    keep_tracks = {int(n.get("TRACK_ID")) for n in root.findall("Model/FilteredTracks/TrackID")}
    tracked = set()
    for track in root.findall("Model/AllTracks/Track"):
        if int(track.get("TRACK_ID")) in keep_tracks:
            for edge in track.findall("Edge"):
                tracked.update((int(edge.get("SPOT_SOURCE_ID")), int(edge.get("SPOT_TARGET_ID"))))
    spots = pd.DataFrame([dict(id=int(n.get("ID")), frame=int(float(n.get("FRAME"))),
        x=float(n.get("POSITION_X"))/px, y=float(n.get("POSITION_Y"))/py,
        r=float(n.get("RADIUS", radius))/px, visible=int(n.get("VISIBILITY", "0")))
        for n in root.findall("Model/AllSpots/SpotsInFrame/Spot")])
    candidates = spots[spots.id.isin(tracked) & (spots.visible == 1)]
    frames = np.unique(candidates.frame)
    frames = frames[np.unique(np.linspace(0, len(frames)-1, min(200,len(frames))).astype(int))]
    rows = []
    with tifffile.TiffFile(tif) as stack:
        for frame in frames:
            image = stack.pages[int(frame)].asarray().astype(float)
            filtered, small, large = dog_filter(image, (radius/py, radius/px))
            h,w = image.shape
            all_frame = spots[spots.frame == frame]
            for spot in candidates[candidates.frame == frame].itertuples():
                xi, yi = int(round(spot.x)), int(round(spot.y))
                half = max(15, int(np.ceil(3*spot.r))+2)
                if min(xi,yi,w-1-xi,h-1-yi) < half:
                    continue
                others = all_frame[all_frame.id != spot.id]
                if len(others) and np.hypot(others.x-spot.x,others.y-spot.y).min() < 12:
                    continue
                yy,xx = np.mgrid[yi-half:yi+half+1,xi-half:xi+half+1]
                dist = np.hypot(xx-spot.x, yy-spot.y)
                bg = (dist >= 1.5*spot.r) & (dist <= 3*spot.r)
                for other in others.itertuples():
                    bg &= np.hypot(xx-other.x,yy-other.y) >= other.r
                core = dist <= spot.r
                if bg.sum()<20 or not core.any():
                    continue
                raw_crop = image[yi-half:yi+half+1,xi-half:xi+half+1]
                dog_crop = filtered[yi-half:yi+half+1,xi-half:xi+half+1]
                raw_mu,raw_sd = raw_crop[bg].mean(),raw_crop[bg].std(ddof=1)
                dog_mu,dog_sd = dog_crop[bg].mean(),dog_crop[bg].std(ddof=1)
                if min(raw_sd,dog_sd)<=0:
                    continue
                raw_snr = (raw_crop[core].max()-raw_mu)/raw_sd
                dog_snr = (dog_crop[core].max()-dog_mu)/dog_sd
                rows.append(dict(frame=int(frame),spot_id=int(spot.id),x=spot.x,y=spot.y,r=spot.r,
                    raw_snr=raw_snr,dog_snr=dog_snr,gain=dog_snr/raw_snr if raw_snr>0 else np.nan,
                    raw_mu=raw_mu,raw_sd=raw_sd,dog_mu=dog_mu,dog_sd=dog_sd,n_bg=int(bg.sum())))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"{FIG_NAME}_candidates.csv",index=False)
    weak = table[(table.raw_snr >= 2) & (table.raw_snr <= 5) & (table.dog_snr >= 5) & (table.gain >= 1.5)]
    if weak.empty:
        raise RuntimeError("No weak-to-clear example met the stated selection criteria; inspect candidate_metrics.csv")
    selected = weak.sort_values(["gain","dog_snr"],ascending=False).iloc[0].to_dict()
    selected.update(source_tiff=str(tif),settings_xml=str(session),sampled_frames=len(frames),
        eligible_candidates=len(table),selection="raw local SNR 2-5, DoG >=5, gain >=1.5; highest gain",
        metric="Peak in spot-radius disk, background annulus 1.5R-3R, sample SD; same regions before/after",
        original_detector=detector.get("DETECTOR_NAME"),median_prefilter=False)
    image = tifffile.imread(tif,key=int(selected["frame"])).astype(float)
    filtered,small,large=dog_filter(image,(radius/py,radius/px))
    mpp = MPP_BY_IMAGE_WIDTH_PX[image.shape[1]]
    selected.update(mpp=mpp,sigma_small_yx=small.tolist(),sigma_large_yx=large.tolist())
    (OUT/f"{FIG_NAME}_selected.json").write_text(json.dumps(selected,indent=2))
    xi,yi=int(round(selected["x"])),int(round(selected["y"]))
    half=15
    raw=image[yi-half:yi+half+1,xi-half:xi+half+1]
    dog=filtered[yi-half:yi+half+1,xi-half:xi+half+1]
    tifffile.imwrite(SCRATCH/"raw_frame.tif",image.astype(np.float32))
    tifffile.imwrite(SCRATCH/"dog_frame.tif",filtered.astype(np.float32))
    # Same mapping in background-SD units for both cutouts: no separate auto contrast.
    zraw=(raw-selected["raw_mu"])/selected["raw_sd"]
    zdog=(dog-selected["dog_mu"])/selected["dog_sd"]
    vmax=max(8.,float(np.ceil(selected["dog_snr"])))
    # Geometry: A has the frame aspect and the height of B + gap + C (square cutouts).
    h_img, w_img = image.shape
    aspect = w_img / h_img
    side = (FIG_WIDTH_IN - PANEL_GAP_IN - aspect * PANEL_GAP_IN) / (2 * aspect + 1)
    height = 2 * side + PANEL_GAP_IN
    width_a = aspect * height
    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(FIG_WIDTH_IN, height), layout="none")
        W, H = FIG_WIDTH_IN, height
        ax = fig.add_axes((0, 0, width_a / W, 1))
        x_bc = (width_a + PANEL_GAP_IN) / W
        ar = fig.add_axes((x_bc, (side + PANEL_GAP_IN) / H, side / W, side / H))
        ad = fig.add_axes((x_bc, 0, side / W, side / H))
        ax.imshow(image,cmap="gray",vmin=np.percentile(image,1),vmax=np.percentile(image,99.8),interpolation="nearest")
        ax.add_patch(Rectangle((xi-half-.5,yi-half-.5),2*half+1,2*half+1,fill=False,edgecolor=COLOR_MARK,lw=1.0))
        extent=(-half-.5,half+.5,half+.5,-half-.5)
        ar.imshow(zraw,cmap="gray",vmin=-2,vmax=vmax,extent=extent,interpolation="nearest")
        ad.imshow(zdog,cmap="gray",vmin=-2,vmax=vmax,extent=extent,interpolation="nearest")
        for a in (ar,ad):
            # Small brackets mark the tracked center without obscuring the signal.
            a.plot([-5,-3],[0,0],color=COLOR_MARK,lw=1)
            a.plot([0,0],[-5,-3],color=COLOR_MARK,lw=1)
            _scalebar(a, mpp, SCALEBAR_CUTOUT_UM)
        _scalebar(ax, mpp, SCALEBAR_FRAME_UM)
        for a, letter in ((ax, "A"), (ar, "B"), (ad, "C")):
            a.set_axis_off()
            _panel_label(a, letter)
        fig.savefig(OUT/f"{FIG_NAME}.png",dpi=600)
        plt.close(fig)
    print(f"Frame {int(selected['frame'])+1} (index {int(selected['frame'])}), cutout {2*half+1} x {2*half+1} px; "
          f"local SNR raw {selected['raw_snr']:.2f}, DoG {selected['dog_snr']:.2f}, factor {selected['gain']:.2f}; "
          f"display: A percentiles 1-99.8 of the frame, B/C -2 to {vmax:g} background sigma")
    print(json.dumps(selected,indent=2))

if __name__ == "__main__":
    main()
