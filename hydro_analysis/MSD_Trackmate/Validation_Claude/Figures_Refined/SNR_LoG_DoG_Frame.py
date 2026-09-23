"""Fixed-frame raw/LoG/DoG comparison, without reselecting the particle."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import convolve
from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import (
    FILES, find_session_and_tif, dog_filter, MPP_BY_IMAGE_WIDTH_PX, _RC)

OUT=Path(__file__).resolve().parents[4]/"analysis_results"/"snr_log_dog_frame678"
FRAME=677
SPOT_ID=12914


def log_filter(image,radius):
    # TrackMate DetectionUtils.createLoGKernel, isotropic 2D pixel coordinates.
    sigma=radius/np.sqrt(2)
    half=1+max(2,int(3*sigma+.5)+1)
    y,x=np.mgrid[-half:half+1,-half:half+1]
    q=(x*x+y*y)/(sigma*sigma)
    kernel=(2-q)*np.exp(-q/2)/(np.pi*sigma**4)
    return convolve(np.asarray(image,dtype=float),kernel,mode="mirror"),sigma


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    session,tif=find_session_and_tif(Path(FILES[("hydrogel",20.)]))
    tree=ET.parse(session).getroot()
    detector=tree.find("Settings/DetectorSettings").attrib
    radius=float(detector["RADIUS"])
    assert "pixel" in tree.find("Model").get("spatialunits","pixel").lower(), "This comparison expects pixel-coordinate sessions"
    spots=[s.attrib for s in tree.findall("Model/AllSpots/SpotsInFrame/Spot") if int(float(s.get("FRAME")))==FRAME]
    selected=next(s for s in spots if int(s["ID"])==SPOT_ID)
    x,y=float(selected["POSITION_X"]),float(selected["POSITION_Y"])
    r=float(selected["RADIUS"])
    xi,yi=int(round(x)),int(round(y))
    raw=tifffile.imread(tif,key=FRAME).astype(float)
    log,sigma=log_filter(raw,radius)
    dog,small,large=dog_filter(raw,(radius,radius))
    images={"Raw":raw,"LoG":log,"DoG":dog}
    yy,xx=np.indices(raw.shape)
    dist=np.hypot(xx-x,yy-y)
    core=dist<=r
    backgrounds={}
    for name,(inner,outer) in {"primary":(1.5,3.),"wider_annulus":(2.,4.)}.items():
        bg=(dist>=inner*r)&(dist<=outer*r)
        for s in spots:
            if int(s["ID"])!=SPOT_ID:
                bg &= np.hypot(xx-float(s["POSITION_X"]),yy-float(s["POSITION_Y"]))>=float(s.get("RADIUS",radius))
        backgrounds[name]=bg
    rows=[]
    for region,bg in backgrounds.items():
        for method,image in images.items():
            mean=float(image[bg].mean());sd=float(image[bg].std(ddof=1))
            peak=float(image[core].max())
            rows.append(dict(background=region,method=method,peak=peak,background_mean=mean,background_sd=sd,
                snr=(peak-mean)/sd,center_snr=(image[yi,xi]-mean)/sd,mean_core_snr=(image[core].mean()-mean)/sd,
                background_pixels=int(bg.sum())))
    metrics=pd.DataFrame(rows)
    primary=metrics[metrics.background=="primary"].set_index("method")
    metrics.to_csv(OUT/"filter_comparison.csv",index=False)
    meta=dict(frame_zero_based=FRAME,frame_one_based=FRAME+1,spot_id=SPOT_ID,x=x,y=y,radius_px=radius,
              source_tiff=str(tif),session_xml=str(session),log_sigma_px=sigma,
              dog_sigmas_px=[small.tolist(),large.tolist()],median_prefilter=False,
              implementation="SciPy convolution using TrackMate LoG kernel formula; shared SciPy DoG approximation; mirror boundaries")
    (OUT/"metadata.json").write_text(json.dumps(meta,indent=2))
    half=15;mpp=MPP_BY_IMAGE_WIDTH_PX[raw.shape[1]]
    normalized={k:(im-primary.loc[k,"background_mean"])/primary.loc[k,"background_sd"] for k,im in images.items()}
    for k,im in images.items():
        tifffile.imwrite(OUT/(k.lower()+"_frame.tif"),im.astype(np.float32))
    colors={"Raw":"#666666","LoG":"#0072b2","DoG":"#e69f00"}
    vmax=max(8.,float(np.ceil(primary.snr.max())))
    with plt.rc_context(_RC):
        fig=plt.figure(figsize=(10.5,7),layout="constrained")
        grid=fig.add_gridspec(2,3,height_ratios=[1,1])
        cut_axes=[]
        for col,(method,image) in enumerate(normalized.items()):
            ax=fig.add_subplot(grid[0,col]);cut_axes.append(ax)
            crop=image[yi-half:yi+half+1,xi-half:xi+half+1]
            im=ax.imshow(crop,cmap="gray",vmin=-2,vmax=vmax,interpolation="nearest",extent=[-15.5,15.5,15.5,-15.5])
            ax.plot([-5,-3],[0,0],color="#e69f00",lw=1)
            ax.plot([0,0],[-5,-3],color="#e69f00",lw=1)
            ax.plot([12-1/mpp,12],[12,12],color="white",lw=2)
            ax.text(12-.5/mpp,10.5,"1 µm",ha="center",color="white",fontsize=8)
            ax.set_title(f"{chr(65+col)}  {method} | local SNR {primary.loc[method,'snr']:.2f}",loc="left",fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(im,ax=cut_axes,shrink=.85,pad=.01,label="Above local background (σ)")
        ax=fig.add_subplot(grid[1,0])
        ax.imshow(raw,cmap="gray",vmin=np.percentile(raw,1),vmax=np.percentile(raw,99.8),interpolation="nearest")
        ax.add_patch(Rectangle((xi-half-.5,yi-half-.5),31,31,fill=False,ec="#e69f00",lw=1))
        ax.plot([12,12+5/mpp],[raw.shape[0]-12]*2,color="white",lw=2)
        ax.text(12+2.5/mpp,raw.shape[0]-17,"5 µm",color="white",ha="center",fontsize=8)
        ax.set_title("D  Raw frame 678",loc="left",fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        ax=fig.add_subplot(grid[1,1])
        offsets=np.arange(-half,half+1)
        profiles=[]
        for method,image in normalized.items():
            values=image[yi,xi+offsets]
            ax.plot(offsets*mpp,values,color=colors[method],label=method,lw=1.3)
            profiles.extend(dict(method=method,offset_um=o*mpp,normalized_intensity=v) for o,v in zip(offsets,values))
        pd.DataFrame(profiles).to_csv(OUT/"center_line_profiles.csv",index=False)
        ax.axhline(0,color="0.6",lw=.7,ls=":")
        ax.set_title("E  Center-line profiles",loc="left",fontsize=10)
        ax.set_xlabel("Offset (µm)");ax.set_ylabel("Above local background (σ)")
        ax.legend(frameon=False,fontsize=8)
        ax=fig.add_subplot(grid[1,2])
        for i,method in enumerate(images):
            snr=primary.loc[method,"snr"]
            ax.bar(i,snr,color=colors[method],width=.6)
            ax.text(i,snr+.15,f"{snr:.2f}",ha="center",fontsize=9)
        ax.set_xticks(range(3),list(images));ax.set_ylabel("Local peak SNR")
        ax.set_ylim(0,vmax);ax.set_title("F  Same particle and background",loc="left",fontsize=10)
        fig.suptitle("20 nm hydrogel particle: raw vs LoG vs DoG",fontsize=13)
        fig.supxlabel("Same cutout and background mask; shared contrast scale in background σ units. No median prefilter.",fontsize=8)
        fig.savefig(OUT/"raw_log_dog_comparison.png",dpi=600,bbox_inches="tight")
        fig.savefig(OUT/"preview.png",dpi=160,bbox_inches="tight")
        plt.close(fig)
    # Check the calculations against the earlier fixed example and basic filter properties.
    assert np.isclose(primary.loc["Raw","snr"],2.766752001524084)
    assert np.isclose(primary.loc["DoG","snr"],6.09733822172232)
    assert np.allclose(log_filter(2*raw+100,radius)[0]-log_filter(np.full_like(raw,100),radius)[0],2*log)
    pd.testing.assert_series_equal(primary.snr,(primary.peak-primary.background_mean)/primary.background_sd,check_names=False)
    log_snr,dog_snr=primary.loc["LoG","snr"],primary.loc["DoG","snr"]
    (OUT/"README.md").write_text(
        "# Raw / LoG / DoG comparison on frame 678\n\n"
        f"Same preselected particle 12914: raw SNR {primary.loc['Raw','snr']:.4f}, LoG {log_snr:.4f}, DoG {dog_snr:.4f}. "
        f"LoG versus DoG difference: {(log_snr/dog_snr-1)*100:+.2f}% in this local metric.\n\n"
        "Signal is the maximum in the same spot-radius disk. Background is the same 1.5R–3R annulus, masking other recorded spots; "
        "each image uses its own background mean and sample SD. Center-pixel and mean-core metrics plus a 2R–4R annulus are exported as sensitivity checks. "
        "Images use the same normalized grayscale range. SNR bars are descriptive; no independent-observation error bars are implied.\n\n"
        "LoG uses the sampled analytical kernel in TrackMate DetectionUtils.createLoGKernel, sigma=R/sqrt(2), convolved in SciPy. "
        "DoG reuses the previous implementation and TrackMate/ImgLib2 scale convention. Both use R=2.5 px and mirror boundaries, without median prefiltering. "
        "This is a numerical filter comparison, not an exact Fiji detector replay or a rerun of tracking.\n\n"
        "This frame and particle were previously selected for a strong DoG improvement, so the result cannot establish which filter performs best across the dataset. "
        "The measured SNR includes background structure and filtering correlations; it is not photon-count SNR or detection significance. "
        "A different filter radius or background definition can alter the ranking.\n\n"
        "Source: https://github.com/trackmate-sc/TrackMate/blob/master/src/main/java/fiji/plugin/trackmate/detection/DetectionUtils.java\n",encoding="utf-8")
    print(metrics.to_string(index=False))

if __name__=="__main__":
    main()
