"""Select an illustrative weak tracked 20 nm hydrogel particle and show raw/DoG.
Selection is intentionally for improvement, not a representative SNR estimate.
Filtering and radius conventions are shared with snr_signal_profile.py.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import (
    FILES, find_session_and_tif, dog_filter, MPP_BY_IMAGE_WIDTH_PX, _RC)

OUT = Path(__file__).resolve().parents[4] / "analysis_results" / "snr_hydrogel_20nm_example"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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
    table.to_csv(OUT / "candidate_metrics.csv",index=False)
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
    (OUT/"selected_example.json").write_text(json.dumps(selected,indent=2))
    xi,yi=int(round(selected["x"])),int(round(selected["y"]))
    half=15
    raw=image[yi-half:yi+half+1,xi-half:xi+half+1]
    dog=filtered[yi-half:yi+half+1,xi-half:xi+half+1]
    tifffile.imwrite(OUT/"raw_frame.tif",image.astype(np.float32))
    tifffile.imwrite(OUT/"dog_frame.tif",filtered.astype(np.float32))
    # Same mapping in background-SD units for both cutouts: no separate auto contrast.
    zraw=(raw-selected["raw_mu"])/selected["raw_sd"]
    zdog=(dog-selected["dog_mu"])/selected["dog_sd"]
    vmax=max(8.,float(np.ceil(selected["dog_snr"])))
    with plt.rc_context(_RC):
        fig=plt.figure(figsize=(10,7),layout="constrained")
        grid=fig.add_gridspec(2,2,width_ratios=[1.3,1])
        ax=fig.add_subplot(grid[:,0]); ar=fig.add_subplot(grid[0,1]); ad=fig.add_subplot(grid[1,1])
        ax.imshow(image,cmap="gray",vmin=np.percentile(image,1),vmax=np.percentile(image,99.8),interpolation="nearest")
        ax.add_patch(Rectangle((xi-half-.5,yi-half-.5),2*half+1,2*half+1,fill=False,edgecolor="#e69f00",lw=1.3))
        ax.set_title(f"A  Raw frame {int(selected['frame'])+1}",loc="left",fontsize=11)
        extent=(-half-.5,half+.5,half+.5,-half-.5)
        ar.imshow(zraw,cmap="gray",vmin=-2,vmax=vmax,extent=extent,interpolation="nearest")
        im=ad.imshow(zdog,cmap="gray",vmin=-2,vmax=vmax,extent=extent,interpolation="nearest")
        ar.set_title(f"B  Raw cutout | local SNR {selected['raw_snr']:.1f}",loc="left",fontsize=11)
        ad.set_title(f"C  DoG cutout | local SNR {selected['dog_snr']:.1f}",loc="left",fontsize=11)
        for a in (ar,ad):
            # Small brackets mark the tracked center without obscuring the signal.
            a.plot([-5,-3],[0,0],color="#e69f00",lw=1)
            a.plot([0,0],[-5,-3],color="#e69f00",lw=1)
            a.plot([5-1/mpp,5],[12,12],color="white",lw=2)
            a.text(5-.5/mpp,10.8,"1 µm",color="white",ha="center",fontsize=8)
        ax.plot([image.shape[1]*.08,image.shape[1]*.08+5/mpp],[image.shape[0]*.92]*2,color="white",lw=2)
        ax.text(image.shape[1]*.08+2.5/mpp,image.shape[0]*.89,"5 µm",color="white",ha="center",fontsize=8)
        for a in (ax,ar,ad):
            a.set_xticks([]);a.set_yticks([])
        cb=fig.colorbar(im,ax=[ar,ad],shrink=.8,pad=.02)
        cb.set_label("Local background-normalized intensity (σ)",fontsize=9)
        fig.suptitle("20 nm particles in hydrogel",fontsize=13)
        fig.supxlabel(f"Same 31 × 31 pixel cutout | local SNR {selected['gain']:.2f}× after DoG\n"
                      "Cutouts share the same scale in background σ units; full frame uses raw contrast.",fontsize=9)
        fig.savefig(OUT/"hydrogel_20nm_weak_particle_dog.png",dpi=600,bbox_inches="tight")
        fig.savefig(OUT/"preview.png",dpi=150,bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(selected,indent=2))

if __name__ == "__main__":
    main()
