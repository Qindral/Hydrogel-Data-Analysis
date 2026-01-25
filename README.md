Hydrogel-Data-Analysis
======================

Standardisierte Ergebnisstruktur
-------------------------------
Alle dateispezifischen Analysen arbeiten mit einem `result_dict` pro Datei.
Die Standard-Keys sind:

- `xml_path` (str)
- `tif_path` (str | None)
- `rec_path` (str | None)
- `base_name` (str, `xml_path.name`)
- `tracks_df` (DataFrame)
- `mpp` (float)
- `fps` (float)
- `particle_size_nm` (float | None)
- `num_tracks` (int)
- `num_frames` (int)
- `D_MSD` (float | None)
- `D_step` (float | None)
- `fit_results_MSD` (dict | None)
- `fit_results_step` (dict | None)

Standardparameter
-----------------
- MSD-Fit: `fit_points = 6` -> Ergebnisse in `D_MSD` + `fit_results_MSD`
- Stepsize: `step_interval = 1` -> Ergebnisse in `D_step` + `fit_results_step`

Nicht-Standard Ergebnisse
-------------------------
Falls andere Parameter genutzt werden, werden die Ergebnisse **zusätzlich**
unter eindeutigen Keys abgelegt:

- MSD: `fit_MSD_fp_<fit_points>` (z.B. `fit_MSD_fp_12`)
- Stepsize: `fit_step_si_<step_interval>` (z.B. `fit_step_si_2`)
