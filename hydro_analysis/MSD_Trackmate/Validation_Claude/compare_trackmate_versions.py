"""Compare XML analyses of the same movie; never modify source data/caches.
Run from the repository root with -m hydro_analysis.MSD_Trackmate.Validation_Claude.compare_trackmate_versions.
"""
from pathlib import Path
import argparse
import re
import xml.etree.ElementTree as ET
import itertools
import hashlib
import logging
import io
import tifffile
import numpy as np
import pandas as pd
from hydro_analysis.core.io import read_trackmate_xml, parse_rec_file, remove_edge_artifacts
from hydro_analysis.core.analysis import perform_msd_analysis, MIN_TRACK_LENGTH, DEFAULT_MSD_FIT_POINTS

BASE = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung\20 nm")
OUT = Path(__file__).resolve().parents[3] / "analysis_results" / "trackmate_version_comparison"
FOLDERS = ("Tracks_old", "Tracks", "Trackmate_Analyses")


def clean(stem):
    stem = re.sub(r"_tracks$", "", stem, flags=re.I)
    return re.sub(r"^(?:Resultof|filtered_)+|_(?:processed|var)$", "", stem, flags=re.I)


def resolve_movie(name, movies):
    stem = Path(name).stem if Path(name).suffix.lower() in (".tif", ".tiff", ".xml") else name
    stem = re.sub(r"_tracks$", "", stem, flags=re.I)
    exact = [p for p in movies if p.stem.casefold() == stem.casefold()]
    if len(exact) == 1:
        return exact[0], "exact_name"
    parent = [p for p in movies if p.stem.casefold() == clean(stem).casefold()]
    if len(parent) == 1:
        return parent[0], "derived_name_inferred"
    key = lambda s: re.sub(r"[\s_]", "", s).casefold()
    normalized = [p for p in movies if key(p.stem) == key(clean(stem))]
    if len(normalized) == 1:
        return normalized[0], "normalized_name"
    return None, "ambiguous_or_missing_movie"


def read_tracks(path):
    root = ET.parse(path).getroot()
    image = root.find("Settings/ImageData")
    detector = root.find("Settings/DetectorSettings")
    if root.tag == "Tracks":
        return read_trackmate_xml(path), dict(kind="track_export", detector="unknown", movie_hint=path.stem)
    if root.find("Model/AllSpots") is None:
        raise ValueError("Unsupported XML format")
    spots = {int(s.get("ID")): s.attrib for s in root.findall("Model/AllSpots/SpotsInFrame/Spot")}
    retained = {int(t.get("TRACK_ID")) for t in root.findall("Model/FilteredTracks/TrackID")}
    rows = []
    units = root.find("Model").get("spatialunits", "pixel").lower()
    sx = float(image.get("pixelwidth", 1)) if "pixel" not in units else 1
    sy = float(image.get("pixelheight", 1)) if "pixel" not in units else 1
    for t in root.findall("Model/AllTracks/Track"):
        tid = int(t.get("TRACK_ID"))
        if tid not in retained:
            continue
        ids = {int(e.get(k)) for e in t.findall("Edge") for k in ("SPOT_SOURCE_ID", "SPOT_TARGET_ID")}
        for sid in ids:
            s = spots[sid]
            rows.append(dict(particle=tid, frame=int(float(s["FRAME"])), x=float(s["POSITION_X"])/sx, y=float(s["POSITION_Y"])/sy))
    tracks = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    return tracks.sort_values(["particle", "frame"]), dict(kind="full_session",detector=detector.get("DETECTOR_NAME", "unknown") if detector is not None else "unknown",
        movie_hint=image.get("filename",path.stem) if image is not None else path.stem,
        all_detected_spots=len(spots),retained_session_tracks=len(retained))


def fingerprint(df):
    return {(int(r.frame),round(r.x,4),round(r.y,4)) for r in df.itertuples()}


def main(base=BASE, output=OUT):
    output.mkdir(parents=True,exist_ok=True)
    movies = list(base.glob("*.tif")) + list(base.glob("preprocess/*.tif"))
    # Detect renamed copies by decoded pixels, independent of TIFF metadata.
    movie_hashes=[]
    for movie in movies:
        digest=hashlib.sha256()
        messages=io.StringIO()
        handler=logging.StreamHandler(messages)
        logger=logging.getLogger("tifffile")
        logger.addHandler(handler)
        error=""
        try:
            with tifffile.TiffFile(movie) as stack:
                for page in stack.pages:
                    array=page.asarray()
                    digest.update(str((array.shape,str(array.dtype))).encode())
                    digest.update(array.tobytes())
        except Exception as exc:
            error=str(exc)
        finally:
            logger.removeHandler(handler)
        issue=(messages.getvalue()+error).strip()
        movie_hashes.append(dict(movie=str(movie),pixel_sha256=digest.hexdigest(),
                                 identity_verified=not bool(issue),tiff_warning=issue))
    identity=pd.DataFrame(movie_hashes)
    identity.to_csv(output/"movie_identity.csv",index=False)
    canonical={str(p):str(p) for p in movies}
    for _,group in identity[identity.identity_verified].groupby("pixel_sha256"):
        first=sorted(group.movie)[0]
        canonical.update({p:first for p in group.movie})
    records=[]; failures=[]
    for folder in FOLDERS:
        for path in sorted((base/folder).glob("*.xml")):
            try:
                tracks,meta=read_tracks(path)
                movie,method=resolve_movie(meta["movie_hint"],movies)
                records.append(dict(path=path,tracks=tracks,points=fingerprint(tracks),movie=movie,match=method,**meta))
            except Exception as e:
                failures.append(dict(xml=str(path),error=str(e)))
    # Recover export provenance from matching detections when a full session exists.
    sessions=[r for r in records if r["kind"]=="full_session"]
    for r in records:
        if r["kind"]!="track_export" or not r["points"]:
            continue
        matched=[s for s in sessions if r["points"] <= s["points"] and len(r["points"]) >= 10]
        candidates={s["movie"] for s in matched if s["movie"] is not None}
        if len(candidates)==1:
            r["movie"]=next(iter(candidates));r["match"]="session_coordinates"
            r["matched_sessions"]=" | ".join(str(s["path"]) for s in matched)
            detectors={s["detector"] for s in matched}
            r["detector"]=";".join(sorted(detectors))
    summaries=[];length_rows=[]
    for r in records:
        p=r["path"];df=r["tracks"];movie=r["movie"]
        row=dict(folder=p.parent.name,xml=p.name,xml_path=str(p),movie=canonical[str(movie)] if movie else "",source_movie=str(movie) if movie else "",match=r["match"],
                 kind=r["kind"],detector=r["detector"],matched_sessions=r.get("matched_sessions",""),
                 tracks_before_filter=df.particle.nunique(),detections_before_filter=len(df),all_detected_spots=r.get("all_detected_spots",np.nan))
        try:
            if movie is None:
                raise ValueError("Movie unresolved: no calibration assumed")
            # Processed TIFFs inherit acquisition calibration from the exact raw parent.
            raw_stem=clean(movie.stem)
            recs=[movie.with_suffix(movie.suffix+".rec"),movie.with_suffix(".rec"),base/(raw_stem+".tif.rec"),base/(raw_stem+".rec")]
            rec=next((p for p in recs if p.exists()),None)
            if rec is None:
                raise ValueError("Missing REC calibration")
            cal=parse_rec_file(rec)
            if not cal["fps"] or not cal["mpp"]:
                raise ValueError("Incomplete REC calibration")
            fps=cal["fps"]
            row.update(rec=str(rec),fps=fps,mpp=cal["mpp"])
            if df.duplicated(["particle","frame"]).any():
                raise ValueError("Branched track: multiple positions per track/frame; cannot fit as a single trajectory")
            filtered,stats=remove_edge_artifacts(df,cal["size_x"],cal["size_y"])
            lengths=filtered.groupby("particle").agg(detections=("frame","size"),first_frame=("frame","min"),last_frame=("frame","max"))
            lengths["duration_s"]=(lengths.last_frame-lengths.first_frame)/fps
            used=lengths[lengths.detections>=MIN_TRACK_LENGTH]
            for stage,data in (("exported",df),("edge_filtered",filtered)):
                for particle,g in data.groupby("particle"):
                    length_rows.append(dict(xml_path=str(p),movie=canonical[str(movie)],stage=stage,particle=particle,
                        detections=len(g),duration_s=(g.frame.max()-g.frame.min())/fps,used_for_fit=stage=="edge_filtered" and len(g)>=MIN_TRACK_LENGTH))
            row.update(tracks_after_edge_filter=len(lengths),tracks_used_for_fit=len(used),
                detections_used_for_fit=int(used.detections.sum()),track_length_median=used.detections.median(),
                track_length_mean=used.detections.mean(),track_length_q25=used.detections.quantile(.25),
                track_length_q75=used.detections.quantile(.75),duration_median_s=used.duration_s.median(),
                edge_detections_removed=stats["n_removed"],fit_points=DEFAULT_MSD_FIT_POINTS)
            result=dict(tracks_df=filtered,mpp=cal["mpp"],fps=fps)
            perform_msd_analysis(result)
            fit=result.get("fit_results_MSD")
            if fit is None:
                raise ValueError("No fit: insufficient retained tracks")
            row.update(D=fit["D_um2_per_s"],D_error=fit["D_error"],n=fit["exponent"],n_error=fit["exponent_error"],r_squared=fit["r_squared"],status="ok")
        except Exception as e:
            row.update(status="review",note=str(e))
        summaries.append(row)
        print(p.name,row["status"],flush=True)
    table=pd.DataFrame(summaries).sort_values(["movie","folder","xml"])
    table.to_csv(output/"comparison.csv",index=False)
    pd.DataFrame(length_rows).to_csv(output/"track_lengths.csv",index=False)
    pd.DataFrame(failures,columns=["xml","error"]).to_csv(output/"parse_failures.csv",index=False)
    pairs=[]
    for movie,group in table[table.movie!=""].groupby("movie"):
        for (_,a),(_,b) in itertools.combinations(group.iterrows(),2):
            pair=dict(movie=movie,a=a.xml_path,b=b.xml_path,match_a=a.match,match_b=b.match)
            for metric in ("tracks_used_for_fit","track_length_median","D","n"):
                av,bv=a.get(metric,np.nan),b.get(metric,np.nan)
                pair[metric+"_a"]=av;pair[metric+"_b"]=bv
                pair[metric+"_difference_b_minus_a"]=bv-av
            pairs.append(pair)
    pd.DataFrame(pairs).to_csv(output/"paired_comparisons.csv",index=False)
    lines=["# TrackMate version comparison", "", "Same-movie groups use exact TIFF names, full-session ImageData, or matching detection coordinates. Whitespace is normalized only if unique. Derived Resultof/processed/var mappings are explicitly marked as inferred; these do not prove identical processing. Different TIFF names are grouped only when their complete decoded pixel hashes match; original paths remain in source_movie and movie_identity.csv.", "", "All D/n fits use the existing core pipeline: 3% edge exclusion with splitting, minimum 10 detections per track, first 6 MSD lag times; its existing conditional drift rule is preserved. No 40 Hz exclusion is applied: all versions are compared and fps is exported. D = A/4 in MSD = A t^n (effective coefficient, units µm²/s^n for n != 1). Track lengths below are detections per retained trajectory; duration is (last-first frame)/fps. Particle count means trajectories, not unique physical particles. Full-session inputs use FilteredTracks only. Counts of all detected spots are available only for full sessions.", "", "No source XML, TIFF, or canonical MSD cache is modified.", ""]
    for movie,group in table.groupby("movie"):
        lines += ["## "+(Path(movie).name if movie else "Unresolved movies"),"", "| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |", "|---|---|---:|---:|---:|---:|---|"]
        for _,r in group.iterrows():
            fmt=lambda x: f"{x:.4g}" if pd.notna(x) else "—"
            lines.append(f"| {r.folder}/{r.xml} | {r.match} | {r.tracks_before_filter} / {fmt(r.get('tracks_used_for_fit',np.nan))} | {fmt(r.get('track_length_median',np.nan))} | {fmt(r.get('D',np.nan))} | {fmt(r.get('n',np.nan))} | {r.status} |")
        lines.append("")
    issues=identity[~identity.identity_verified]
    if len(issues):
        lines += ["## TIFF integrity notes", "", "Some TIFFs emitted read warnings; their partial pixel hashes are not used to merge movies. XML-based fits remain available. See movie_identity.csv for affected filenames and warnings.", ""]
    lines += ["## Files", "", "comparison.csv contains calibration, fit errors and review notes. track_lengths.csv contains individual track lengths before/after edge filtering. paired_comparisons.csv contains within-movie differences. parse_failures.csv records any XML parsing failures."]
    (output/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(table[["folder","xml","movie","match","tracks_used_for_fit","D","n","status"]].to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=BASE)
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    main(args.base,args.output_dir)
