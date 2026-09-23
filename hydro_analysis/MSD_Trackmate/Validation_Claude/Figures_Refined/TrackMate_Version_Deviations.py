"""Plot the confirmed chronological reanalysis pair for the same TIFF.
Consumes compare_trackmate_versions.py outputs; does not refit or alter caches.
"""
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hydro_analysis.MSD_Trackmate.MSD_per_file_publication import _RC

OUT = Path(__file__).resolve().parents[4] / "analysis_results" / "trackmate_version_comparison"


def main():
    df = pd.read_csv(OUT / "comparison.csv")
    names = ("20 nm_2_Tracks.xml", "20 nm_2_var_Tracks.xml")
    rows=[]
    for name in names:
        selected=df[(df.folder=="Tracks_old") & (df.xml==name)]
        if len(selected)!=1:
            raise ValueError(f"Expected exactly one record for {name}")
        r=selected.iloc[0].copy()
        stamp=ET.parse(r.xml_path).getroot().get("generationDateTime")
        r["export_timestamp"]=datetime.strptime(stamp.split(", ",1)[1],"%d %b %Y %H:%M:%S")
        rows.append(r)
    pair=pd.DataFrame(rows).sort_values("export_timestamp")
    if pair.movie.nunique()!=1 or not (pair.status=="ok").all():
        raise ValueError("Pair must represent the same movie with two successful fits")
    if not np.allclose(pair.fps,pair.fps.iloc[0]) or not np.allclose(pair.mpp,pair.mpp.iloc[0]):
        raise ValueError("Calibration differs between analyses")
    old,new=pair.iloc[0],pair.iloc[1]
    d_change=100*(new.D/old.D-1)
    n_change=new.n-old.n
    pair["version"]=["Older","Newer"]
    pair.to_csv(OUT/"older_newer_figure_data.csv",index=False)
    with plt.rc_context(_RC):
        fig,axes=plt.subplots(1,2,figsize=(7.15,5),layout="constrained")
        for ax,metric,error,label,letter in zip(axes,["D","n"],["D_error","n_error"],
                [r"Fitted $D$ ($µm^2/s^n$)",r"Anomalous exponent $n$"],["A","B"]):
            values=pair[metric].to_numpy()
            ax.plot([0,1],values,color="0.6",lw=1.1,zorder=1)
            for i,(value,err,color) in enumerate(zip(values,pair[error],["#6b6b6b","#0072b2"])):
                ax.errorbar(i,value,yerr=err,fmt="o",color=color,ms=6,capsize=4,lw=1.2,zorder=3)
                ax.annotate(f"{value:.3f}",(i,value),xytext=(0,17),textcoords="offset points",ha="center",fontsize=10)
            ax.set_xticks([0,1],["Older\n27 Nov, 23:11","Newer\n28 Nov, 00:00"])
            ax.set_xlim(-.4,1.4)
            ax.set_ylabel(label)
            ax.set_title(f"{letter}  "+("Diffusion coefficient" if metric=="D" else "Diffusion exponent"),loc="left",fontsize=11)
            if metric=="D":
                ax.set_ylim(0,3.6)
                change=f"ΔD = {new.D-old.D:+.3f}\n({d_change:+.1f}%)"
            else:
                ax.axhline(1,color="0.65",lw=.8,ls=":")
                ax.set_ylim(.80,1.18)
                change=f"Δn = {n_change:+.3f}\n({100*n_change/old.n:+.1f}%)"
            ax.text(.05,.94,change,transform=ax.transAxes,ha="left",va="top",fontsize=10)
        fig.suptitle("Older vs newer analysis of the same movie\n20 nm_2.tif",fontsize=12)
        fig.supxlabel("12 → 9 usable tracks | identical calibration and MSD fitting settings\n"
                      "Error bars: individual fit errors; one paired movie (November 2025)",fontsize=8)
        fig.savefig(OUT/"older_vs_newer_D_n.png",dpi=600,bbox_inches="tight")
        fig.savefig(OUT/"older_vs_newer_D_n_preview.png",dpi=160,bbox_inches="tight")
        plt.close(fig)
    (OUT/"older_newer_figure_notes.md").write_text(
        "# Older versus newer analysis figure\n\n"
        "The older export is 20 nm_2_Tracks.xml (27 Nov 2025 23:11:27); the newer export is "
        "20 nm_2_var_Tracks.xml (28 Nov 2025 00:00:10). Both refer to 20 nm_2.tif, "
        "with 20 Hz and 0.15 µm/pixel calibration. Both associated sessions use DoG; "
        "this is a reanalysis comparison, not a LoG-versus-DoG experiment.\n\n"
        f"D: {old.D:.6f} → {new.D:.6f}, change {d_change:+.2f}%. "
        f"n: {old.n:.6f} → {new.n:.6f}, change {n_change:+.6f}.\n\n"
        "D is the fitted coefficient in MSD = 4 D t^n; its time exponent in the units changes with n. "
        "The numerical D comparison corresponds to the model MSD/4 at t = 1 s. "
        "Error bars are the existing per-fit errors. Errors of paired differences are not inferred "
        "because the two fits use related data with unknown covariance.\n\n"
        "Only one confirmed chronological pair of distinct usable analyses was found for an identical TIFF. "
        "Other session/export pairs reproduce the same analysis and are not additional independent pairs. "
        "The current Tracks folder contains different movies, so its results cannot be presented as "
        "paired reanalysis changes.\n",encoding="utf-8")
    print(f"D change: {d_change:+.2f}%; n change: {n_change:+.6f}")


if __name__=="__main__":
    main()
