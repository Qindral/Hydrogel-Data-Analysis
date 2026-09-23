# Raw / LoG / DoG comparison on frame 678

Same preselected particle 12914: raw SNR 2.7668, LoG 6.2760, DoG 6.0973. LoG versus DoG difference: +2.93% in this local metric.

Signal is the maximum in the same spot-radius disk. Background is the same 1.5R–3R annulus, masking other recorded spots; each image uses its own background mean and sample SD. Center-pixel and mean-core metrics plus a 2R–4R annulus are exported as sensitivity checks. Images use the same normalized grayscale range. SNR bars are descriptive; no independent-observation error bars are implied.

LoG uses the sampled analytical kernel in TrackMate DetectionUtils.createLoGKernel, sigma=R/sqrt(2), convolved in SciPy. DoG reuses the previous implementation and TrackMate/ImgLib2 scale convention. Both use R=2.5 px and mirror boundaries, without median prefiltering. This is a numerical filter comparison, not an exact Fiji detector replay or a rerun of tracking.

This frame and particle were previously selected for a strong DoG improvement, so the result cannot establish which filter performs best across the dataset. The measured SNR includes background structure and filtering correlations; it is not photon-count SNR or detection significance. A different filter radius or background definition can alter the ranking.

Source: https://github.com/trackmate-sc/TrackMate/blob/master/src/main/java/fiji/plugin/trackmate/detection/DetectionUtils.java
