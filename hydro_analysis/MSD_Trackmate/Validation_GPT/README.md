# SPT validation pipeline

Tasks A/B in `complete_spt_validation.py` read complete TrackMate sessions.
Task C now delegates to the shared DoG-only SNR workflow in
[`Validation_Claude/README.md`](../Validation_Claude/README.md). LoG sessions
are reported and excluded from SNR; conditions, particle sizes and mpp remain
separate. Blurry frames and aggregate candidates are excluded. The previous
LoG scatter and 3D surface figures have been removed.

## Run

1. Copy `datasets.example.csv` to a location outside version control and fill
   in one row per XML/TIFF pair. `condition` must be `water`, `hydrogel`, or
   `immobilized`; `particle_size_nm` is the reported particle diameter.
2. Run from the repository root:

   ```powershell
   python hydro_analysis/MSD_Trackmate/Validation_GPT/complete_spt_validation.py `
       --manifest C:/path/to/datasets.csv --output C:/path/to/out_validation
   ```

The output contains `out_NEU-B/` and `out_NEU-C/`, each with PNG figures at
600 dpi, numerical CSV tables, and a German `befunde.md`. The scripts do not
write into any existing cache or output folder.

Important operational rule: a full TrackMate session must contain
`Model/AllSpots`, `Model/AllTracks`, `Model/FilteredTracks`, and
`Settings/ImageData`. The program writes a data card before analysing and
stops for a data set lacking the calibration or a readable TIFF stack.
