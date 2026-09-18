# Standalone SPT validation pipeline

`complete_spt_validation.py` is independent of the existing `Validation/`
scripts, shared `core/` code, and existing pickle caches. It reads complete
TrackMate session XML directly and uses the calibration contained in every XML.

## Run

1. Copy `datasets.example.csv` to a location outside version control and fill
   in one row per XML/TIFF pair. `condition` must be `water`, `hydrogel`, or
   `immobilized`; `particle_size_nm` is the reported particle diameter.
2. Run from the repository root:

   ```powershell
   python hydro_analysis/MSD_Trackmate/Validation_NEU/complete_spt_validation.py `
       --manifest C:/path/to/datasets.csv --output C:/path/to/out_validation
   ```

The output contains `out_NEU-B/` and `out_NEU-C/`, each with PNG figures at
600 dpi, numerical CSV tables, and a German `befunde.md`. The scripts do not
write into any existing cache or output folder.

Important operational rule: a full TrackMate session must contain
`Model/AllSpots`, `Model/AllTracks`, `Model/FilteredTracks`, and
`Settings/ImageData`. The program writes a data card before analysing and
stops for a data set lacking the calibration or a readable TIFF stack.
