# TrackMate version comparison

Same-movie groups use exact TIFF names, full-session ImageData, or matching detection coordinates. Whitespace is normalized only if unique. Derived Resultof/processed/var mappings are explicitly marked as inferred; these do not prove identical processing. Different TIFF names are grouped only when their complete decoded pixel hashes match; original paths remain in source_movie and movie_identity.csv.

All D/n fits use the existing core pipeline: 3% edge exclusion with splitting, minimum 10 detections per track, first 6 MSD lag times; its existing conditional drift rule is preserved. No 40 Hz exclusion is applied: all versions are compared and fps is exported. D = A/4 in MSD = A t^n (effective coefficient, units µm²/s^n for n != 1). Track lengths below are detections per retained trajectory; duration is (last-first frame)/fps. Particle count means trajectories, not unique physical particles. Full-session inputs use FilteredTracks only. Counts of all detected spots are available only for full sessions.

No source XML, TIFF, or canonical MSD cache is modified.

## 20 nm water_4.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Tracks/20 nm water_4_Tracks.xml | exact_name | 354 / 323 | 16 | 13.06 | 1.024 | ok |

## 20 nm.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Tracks_old/20 nm_Tracks.xml | exact_name | 98 / 97 | 17 | 1.501 | 0.6401 | ok |

## 20 nm_2.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Trackmate_Analyses/20nm_2.xml | exact_name | 9 / 9 | 23 | 2.748 | 1.054 | ok |
| Tracks_old/20 nm_2_Tracks.xml | session_coordinates | 23 / 12 | 19 | 1.36 | 0.9382 | ok |
| Tracks_old/20 nm_2_var_Tracks.xml | session_coordinates | 9 / 9 | 23 | 2.748 | 1.054 | ok |
| Tracks_old/20nm_2_tracks.xml | exact_name | 23 / 12 | 19 | 1.36 | 0.9382 | ok |

## 20 nm_3.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Trackmate_Analyses/Resultof20nm_3.xml | normalized_name | 2 / 2 | 11.5 | 7.443 | 1.186 | ok |
| Tracks_old/Resultof20 nm_3_Tracks.xml | session_coordinates | 2 / 2 | 11.5 | 7.443 | 1.186 | ok |

## 20 nm_5.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Trackmate_Analyses/Resultof20nm_5.xml | normalized_name | 9 / 9 | 16 | 2.385 | 1.087 | ok |
| Tracks_old/Resultof20 nm_5_Tracks.xml | session_coordinates | 9 / 9 | 16 | 2.385 | 1.087 | ok |

## 20 nm_water_02.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Tracks/20 nm_water_02_Tracks.xml | exact_name | 198 / 181 | 17 | 13.15 | 1.024 | ok |

## 20 nm_water_03.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Tracks/20 nm_water_03_Tracks.xml | exact_name | 182 / 166 | 16.5 | 12.58 | 0.9632 | ok |

## 20nm.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Trackmate_Analyses/20nm.xml | exact_name | 6 / 6 | 18.5 | 1.403 | 0.9064 | ok |
| Tracks_old/20nm_Tracks.xml | session_coordinates | 6 / 6 | 18.5 | 1.403 | 0.9064 | ok |

## 20nm_2.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Trackmate_Analyses/Resultof20nm_2.xml | derived_name_inferred | 7 / 0 | — | — | — | review |
| Tracks_old/Resultof20nm_2_Tracks.xml | session_coordinates | 7 / 0 | — | — | — | review |
| Tracks_old/ResultofResultof20nm_2_Tracks.xml | derived_name_inferred | 12 / 0 | — | — | — | review |

## 20nm_4.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Trackmate_Analyses/Resultof20nm_4.xml | derived_name_inferred | 18 / 0 | — | — | — | review |
| Tracks_old/Resultof20nm_4_Tracks.xml | session_coordinates | 18 / 0 | — | — | — | review |

## 20nm_60fps_2_processed.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Trackmate_Analyses/20nm_60fps_2_processed.xml | exact_name | 10 / 10 | 41.5 | 0.7065 | 0.6715 | ok |
| Tracks_old/20nm_60fps_2_processed_Tracks.xml | session_coordinates | 10 / 10 | 41.5 | 0.7065 | 0.6715 | ok |

## 20nm_processed.tif

| Folder / XML | Mapping | Tracks before / used | Median length | D | n | Status |
|---|---|---:|---:|---:|---:|---|
| Trackmate_Analyses/20nm_processed.xml | exact_name | 6 / 6 | 19.5 | 1.395 | 0.7999 | ok |
| Tracks_old/20nm_processed_Tracks.xml | session_coordinates | 6 / 6 | 19.5 | 1.395 | 0.7999 | ok |

## TIFF integrity notes

Some TIFFs emitted read warnings; their partial pixel hashes are not used to merge movies. XML-based fits remain available. See movie_identity.csv for affected filenames and warnings.

## Files

comparison.csv contains calibration, fit errors and review notes. track_lengths.csv contains individual track lengths before/after edge filtering. paired_comparisons.csv contains within-movie differences. parse_failures.csv records any XML parsing failures.