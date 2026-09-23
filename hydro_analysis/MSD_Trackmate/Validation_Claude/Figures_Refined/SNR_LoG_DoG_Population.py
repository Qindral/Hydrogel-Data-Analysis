"""Paired raw/LoG/DoG local SNR for 100 distinct tracks per hydrogel size.
Deterministic, movie-balanced sampling; no selection on intensity or gain.
"""
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from hydro_analysis.MSD_Trackmate.MSD_FromTrackmate_20mg import XML_FOLDERS
from hydro_analysis.MSD_Trackmate.Validation_Claude.snr_signal_profile import dog_filter, _RC
from hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.SNR_LoG_DoG_Frame import log_filter

OUT=Path(__file__).resolve().parents[4]/"analysis_results"/"snr_log_dog_100_per_size"
N=100
SEED=42


def norm(s):
    return re.sub(r"[\s_]","",s).lower()


def sources(size):
    found=[];excluded=[]
    for folder in XML_FOLDERS[size]:
        movie_dir=folder.parent
        for tracks in sorted(folder.glob("*.xml")):
            stem=re.sub(r"_tracks$","",tracks.stem,flags=re.I)
            if any(k in stem.lower() for k in ("processed","resultof","filtered_")):
                excluded.append(dict(size_nm=size,file=str(tracks),reason="derived input excluded to avoid reusing same recording"));continue
            matches=[p for d in movie_dir.iterdir() if d.is_dir() and "analys" in d.name.lower()
                     for p in d.glob("*.xml") if norm(p.stem)==norm(stem)]
            if "new" in folder.name.lower():
                preferred=[p for p in matches if "new" in p.parent.name.lower()]
                if preferred:matches=preferred
            exact=movie_dir/(stem+".xml")
            if not matches and exact.exists():matches=[exact]
            tifs=[p for p in movie_dir.glob("*.tif") if norm(p.stem)==norm(stem)]
            if len(matches)!=1 or len(tifs)!=1:
                excluded.append(dict(size_nm=size,file=str(tracks),reason=f"unresolved source: {len(matches)} sessions, {len(tifs)} TIFFs"));continue
            found.append((matches[0],tifs[0]))
    return sorted(set(found)),excluded


def candidates(session,tif,rng):
    root=ET.parse(session).getroot()
    if "pixel" not in root.find("Model").get("spatialunits","pixel").lower():
        raise ValueError("Expected pixel-coordinate source")
    radius=float(root.find("Settings/DetectorSettings").get("RADIUS"))
    spots={int(s.get("ID")):dict(id=int(s.get("ID")),frame=int(float(s.get("FRAME"))),
        x=float(s.get("POSITION_X")),y=float(s.get("POSITION_Y")),r=float(s.get("RADIUS",radius)),
        visible=int(float(s.get("VISIBILITY","0")))) for s in root.findall("Model/AllSpots/SpotsInFrame/Spot")}
    by_frame=defaultdict(list)
    for s in spots.values():by_frame[s["frame"]].append(s)
    with tifffile.TiffFile(tif) as stack:
        h,w=stack.pages[0].shape
        n_frames=len(stack.pages)
    retained={int(t.get("TRACK_ID")) for t in root.findall("Model/FilteredTracks/TrackID")}
    result=[]
    for track in root.findall("Model/AllTracks/Track"):
        tid=int(track.get("TRACK_ID"))
        if tid not in retained:continue
        ids=sorted({int(e.get(k)) for e in track.findall("Edge") for k in ("SPOT_SOURCE_ID","SPOT_TARGET_ID")})
        rng.shuffle(ids)
        for sid in ids:
            s=spots[sid];r=s["r"];xi,yi=int(round(s["x"])),int(round(s["y"]))
            if not s["visible"] or not 0<=s["frame"]<n_frames or r<=0:continue
            half=int(np.ceil(4*r))+2
            if min(xi,yi,w-1-xi,h-1-yi)<half:continue
            others=[o for o in by_frame[s["frame"]] if o["id"]!=sid]
            if any(np.hypot(o["x"]-s["x"],o["y"]-s["y"])<2*r+o["r"] for o in others):continue
            yy,xx=np.mgrid[yi-half:yi+half+1,xi-half:xi+half+1]
            dist=np.hypot(xx-s["x"],yy-s["y"])
            bg_masks=[]
            for lo,hi in ((1.5,3),(2,4)):
                bg=(dist>=lo*r)&(dist<=hi*r)
                for o in others:bg &= np.hypot(xx-o["x"],yy-o["y"])>=o["r"]
                bg_masks.append(bg)
            if min(m.sum() for m in bg_masks)<20:continue
            result.append(dict(**s,track_id=tid,session=str(session),tif=str(tif),radius=radius,
                half=half,core=dist<=r,masks=bg_masks))
            break
    rng.shuffle(result)
    return result


def measure(c):
    raw=tifffile.imread(c["tif"],key=c["frame"]).astype(float)
    images={"Raw":raw,"LoG":log_filter(raw,c["radius"])[0],"DoG":dog_filter(raw,(c["radius"],c["radius"]))[0]}
    xi,yi=int(round(c["x"])),int(round(c["y"]));half=c["half"]
    rows=[]
    for region,bg in zip(("primary","wider_annulus"),c["masks"]):
        row={k:v for k,v in c.items() if k not in ("core","masks")}
        row.update(background=region,n_bg=int(bg.sum()))
        for name,image in images.items():
            crop=image[yi-half:yi+half+1,xi-half:xi+half+1]
            mean,sd=float(crop[bg].mean()),float(crop[bg].std(ddof=1))
            if sd<=0:raise ValueError("Zero background variance")
            peak=float(crop[c["core"]].max())
            row.update({name+"_snr":(peak-mean)/sd,name+"_mean_core_snr":(crop[c["core"]].mean()-mean)/sd,
                        name+"_center_snr":(image[yi,xi]-mean)/sd,name+"_peak":peak,name+"_bg_mean":mean,name+"_bg_sd":sd})
        for name in ("LoG","DoG"):
            row[name+"_gain"]=row[name+"_snr"]/row["Raw_snr"] if row["Raw_snr"]>0 else np.nan
            row[name+"_increase_pct"]=100*(row[name+"_gain"]-1)
        row["LoG_minus_DoG_snr"]=row["LoG_snr"]-row["DoG_snr"]
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    measurements=[];inventory=[];excluded=[]
    for size in (20.,50.):
        source_list,skip=sources(size);excluded.extend(skip)
        pools=[]
        for session,tif in source_list:
            try:
                pool=candidates(session,tif,rng)
                inventory.append(dict(size_nm=size,session=str(session),tif=str(tif),eligible_tracks=len(pool)))
                if pool:pools.append(pool)
            except Exception as exc:excluded.append(dict(size_nm=size,file=str(session),reason=str(exc)))
        if sum(map(len,pools))<N:raise RuntimeError(f"Only {sum(map(len,pools))} eligible distinct tracks for {size} nm")
        rng.shuffle(pools)
        count=0
        while count<N:
            for pool in pools:
                if not pool or count>=N:continue
                c=pool.pop()
                try:
                    rows=measure(c)
                    for row in rows:row["size_nm"]=size
                    measurements.extend(rows);count+=1
                    if count%20==0:print(f"{size:.0f} nm: {count}/{N}",flush=True)
                except Exception as exc:excluded.append(dict(size_nm=size,file=c["tif"],reason=f"track {c['track_id']}: {exc}"))
            if not any(pools) and count<N:raise RuntimeError("Not enough measurable tracks")
    data=pd.DataFrame(measurements)
    data.to_csv(OUT/"particle_measurements.csv",index=False)
    pd.DataFrame(inventory).to_csv(OUT/"source_inventory.csv",index=False)
    pd.DataFrame(excluded).to_csv(OUT/"excluded_sources_or_measurements.csv",index=False)
    summary=[]
    for (size,region),g in data.groupby(["size_nm","background"]):
        for method in ("Raw","LoG","DoG"):
            snr=g[method+"_snr"]
            gain=g[method+"_increase_pct"] if method!="Raw" else pd.Series(np.zeros(len(g)))
            summary.append(dict(size_nm=size,background=region,method=method,n=len(g),movies=g.tif.nunique(),
                median_snr=snr.median(),q25_snr=snr.quantile(.25),q75_snr=snr.quantile(.75),
                median_increase_pct=gain.median(),q25_increase_pct=gain.quantile(.25),q75_increase_pct=gain.quantile(.75),
                increased=int((snr>g.Raw_snr).sum()) if method!="Raw" else 0,
                valid_ratios=int(gain.notna().sum())))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/"summary.csv",index=False)
    primary=data[data.background=="primary"]
    primary.groupby(["size_nm","tif"])[["Raw_snr","LoG_snr","DoG_snr","LoG_increase_pct","DoG_increase_pct","LoG_minus_DoG_snr"]].median().to_csv(OUT/"per_movie_medians.csv")
    assert primary.groupby("size_nm").size().eq(N).all()
    assert not primary.duplicated(["size_nm","tif","track_id"]).any()
    for name in ("Raw","LoG","DoG"):
        assert np.allclose(data[name+"_snr"],(data[name+"_peak"]-data[name+"_bg_mean"])/data[name+"_bg_sd"])
    colors=["#777777","#0072b2","#e69f00"]
    with plt.rc_context(_RC):
        fig,axes=plt.subplots(2,3,figsize=(10.5,7),layout="constrained")
        for row,size in enumerate((20.,50.)):
            g=primary[primary.size_nm==size]
            for col,(columns,labels,ylabel) in enumerate([
                (["Raw_snr","LoG_snr","DoG_snr"],["Raw","LoG","DoG"],"Local peak SNR"),
                (["LoG_increase_pct","DoG_increase_pct"],["LoG","DoG"],"Paired SNR increase (%)")]):
                ax=axes[row,col]
                for j,(column,label) in enumerate(zip(columns,labels)):
                    vals=g[column].dropna().to_numpy()
                    color=colors[j] if col==0 else colors[j+1]
                    ax.scatter(j+rng.uniform(-.12,.12,len(vals)),vals,s=7,color=color,alpha=.35)
                    bp=ax.boxplot([vals],positions=[j],widths=.45,showfliers=False,patch_artist=True)
                    bp["boxes"][0].set(facecolor="white",edgecolor=color,alpha=.8)
                    bp["medians"][0].set(color=color,linewidth=1.5)
                ax.set_xticks(range(len(labels)),labels);ax.set_ylabel(ylabel)
                ax.set_title(f"{size:.0f} nm | 100 tracks, {g.tif.nunique()} movies",fontsize=9)
                if col==1:ax.axhline(0,color="0.5",ls=":",lw=.8)
            ax=axes[row,2]
            ax.scatter(g.DoG_snr,g.LoG_snr,s=15,color="#555555",alpha=.65)
            lo=min(g.DoG_snr.min(),g.LoG_snr.min(),0);hi=max(g.DoG_snr.max(),g.LoG_snr.max())*1.08
            ax.plot([lo,hi],[lo,hi],ls=":",color="black",lw=.8)
            ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
            ax.set_xlabel("DoG SNR");ax.set_ylabel("LoG SNR")
            ax.set_title(f"LoG higher: {(g.LoG_snr>g.DoG_snr).sum()}/100",fontsize=9)
        fig.suptitle("Hydrogel: paired raw, LoG and DoG comparison",fontsize=12)
        fig.supxlabel("One observation per distinct retained track; no selection on SNR gain. Same local masks; no median prefilter.",fontsize=8)
        fig.savefig(OUT/"snr_100_particles_per_size.png",dpi=600,bbox_inches="tight")
        fig.savefig(OUT/"preview.png",dpi=150,bbox_inches="tight")
        plt.close(fig)
    lines=["# LoG versus DoG: 100 tracks per size in hydrogel", "", "100 observations per nominal size (20 nm and 50 nm), each from a distinct retained TrackMate track. One geometrically eligible frame is randomly selected per track (seed 42); movies contribute in round-robin order so one large movie does not dominate. This is not a claim of 100 independently identified physical particles: tracks can fragment or share a movie.", "", "No selection on raw brightness, SNR, or improvement. Sources are the active hydrogel input folders, with uniquely matched full sessions and original TIFFs. Derived/processed inputs are excluded to avoid duplicate recordings. All eligible tracks are considered, not only weak spots. Existing track selection can favor the original detector. No detection or tracking is rerun.", "", "| Size | Filter | Median SNR | Median paired increase | Increased |", "|---|---|---:|---:|---:|"]
    for r in summary[summary.background=="primary"].itertuples():
        lines.append(f"| {r.size_nm:.0f} nm | {r.method} | {r.median_snr:.3f} | {r.median_increase_pct:+.1f}% | {r.increased}/100 |")
    for size,g in primary.groupby("size_nm"):
        lines += ["",f"{size:.0f} nm: {g.tif.nunique()} movies; LoG higher for {(g.LoG_snr>g.DoG_snr).sum()}/100, DoG higher for {(g.DoG_snr>g.LoG_snr).sum()}/100; median paired LoG minus DoG SNR = {g.LoG_minus_DoG_snr.median():+.4f}."]
    lines += ["", "## Method and limits", "", "Local peak SNR = (maximum in the spot-radius disk - background mean) / background sample SD. The same 1.5R–3R annulus and neighboring-spot mask are used for all three images. A 2R–4R annulus, center-pixel SNR and mean-core SNR are also exported as sensitivity checks. Peaks are independently maximized inside the same disk. Ratios are undefined for nonpositive raw SNR and retained as missing, not silently excluded from absolute comparisons.", "", "LoG uses the sampled TrackMate kernel formula (sigma=detector radius/sqrt(2)); DoG uses the prior TrackMate/ImgLib2 scale convention, computed in SciPy. Both operate on the full frame with the same session detector radius and mirror boundaries, with no median prefilter. This is a numerical filter comparison, not an exact Fiji replay. Filtered pixels are correlated; local SNR is descriptive contrast, not photon SNR or detection significance. No significance test assumes that all tracks are independent.", "", "Median paired percent change is computed per track before taking the median; it is not the percentage change between the two SNR medians. See summary.csv for IQRs and wider-annulus results; per_movie_medians.csv for recording-level variation; particle_measurements.csv for exact frames, track/spot IDs and all measurements; source_inventory.csv and excluded_sources_or_measurements.csv for source accounting.", "", "Reproduce: .\\.venv\\Scripts\\python.exe -m hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.SNR_LoG_DoG_Population"]
    lines += ["", "## Background sensitivity", "", "| Size | LoG median gain (wider annulus) | DoG median gain (wider annulus) |", "|---|---:|---:|"]
    for size in (20.,50.):
        wide=summary[(summary.size_nm==size)&(summary.background=="wider_annulus")].set_index("method")
        lines.append(f"| {size:.0f} nm | {wide.loc['LoG','median_increase_pct']:+.1f}% | {wide.loc['DoG','median_increase_pct']:+.1f}% |")
    lines += ["", "Both filters usually improve SNR. Small differences depend on the background definition; these results do not establish a robust overall winner. The earlier selected weak-particle example is not a population-average improvement."]
    (OUT/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(summary.to_string(index=False))

if __name__=="__main__":main()
