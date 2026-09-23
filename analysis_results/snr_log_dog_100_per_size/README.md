# LoG versus DoG: 100 tracks per size in hydrogel

100 observations per nominal size (20 nm and 50 nm), each from a distinct retained TrackMate track. One geometrically eligible frame is randomly selected per track (seed 42); movies contribute in round-robin order so one large movie does not dominate. This is not a claim of 100 independently identified physical particles: tracks can fragment or share a movie.

No selection on raw brightness, SNR, or improvement. Sources are the active hydrogel input folders, with uniquely matched full sessions and original TIFFs. Derived/processed inputs are excluded to avoid duplicate recordings. All eligible tracks are considered, not only weak spots. Existing track selection can favor the original detector. No detection or tracking is rerun.

| Size | Filter | Median SNR | Median paired increase | Increased |
|---|---|---:|---:|---:|
| 20 nm | Raw | 4.388 | +0.0% | 0/100 |
| 20 nm | LoG | 5.999 | +31.9% | 91/100 |
| 20 nm | DoG | 5.899 | +30.8% | 91/100 |
| 50 nm | Raw | 3.998 | +0.0% | 0/100 |
| 50 nm | LoG | 5.482 | +30.4% | 83/100 |
| 50 nm | DoG | 5.362 | +27.0% | 85/100 |

20 nm: 16 movies; LoG higher for 67/100, DoG higher for 33/100; median paired LoG minus DoG SNR = +0.0532.

50 nm: 12 movies; LoG higher for 66/100, DoG higher for 34/100; median paired LoG minus DoG SNR = +0.0655.

## Background sensitivity and interpretation

| Size | LoG median gain, wider annulus | DoG median gain, wider annulus |
|---|---:|---:|
| 20 nm | +25.9% | +27.1% |
| 50 nm | +22.0% | +22.8% |

Both filters usually improve local SNR. LoG has a small advantage under the original background definition, but DoG has a slightly higher median percentage gain with the wider annulus. The results therefore do not establish a robust overall winner. The earlier approximately 120% gain was a deliberately selected example, not the typical improvement in this broader sample.

## Method and limits

Local peak SNR = (maximum in the spot-radius disk - background mean) / background sample SD. The same 1.5R–3R annulus and neighboring-spot mask are used for all three images. A 2R–4R annulus, center-pixel SNR and mean-core SNR are also exported as sensitivity checks. Peaks are independently maximized inside the same disk. Ratios are undefined for nonpositive raw SNR and retained as missing, not silently excluded from absolute comparisons.

LoG uses the sampled TrackMate kernel formula (sigma=detector radius/sqrt(2)); DoG uses the prior TrackMate/ImgLib2 scale convention, computed in SciPy. Both operate on the full frame with the same session detector radius and mirror boundaries, with no median prefilter. This is a numerical filter comparison, not an exact Fiji replay. Filtered pixels are correlated; local SNR is descriptive contrast, not photon SNR or detection significance. No significance test assumes that all tracks are independent.

Median paired percent change is computed per track before taking the median; it is not the percentage change between the two SNR medians. See summary.csv for IQRs and wider-annulus results; per_movie_medians.csv for recording-level variation; particle_measurements.csv for exact frames, track/spot IDs and all measurements; source_inventory.csv and excluded_sources_or_measurements.csv for source accounting.

Reproduce: .\.venv\Scripts\python.exe -m hydro_analysis.MSD_Trackmate.Validation_Claude.Figures_Refined.SNR_LoG_DoG_Population