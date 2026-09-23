"""Regression checks for per-frame PSF fits; no external data needed."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from hydro_analysis.MSD_Trackmate.Validation_Claude import PSF_Analysis as psf


class FrameFitTests(unittest.TestCase):
    def test_drift_does_not_broaden_individual_fits(self):
        yy, xx = np.indices((25, 25))
        widths = []
        for x0 in (9.2, 11.3, 14.4):
            image = psf._gaussian_2d((xx, yy), 100, x0, 12.2, 1.2, 1.5, 10)
            fit = psf.fit_frame_psf(image, x0, 12.2, 2.4, 3.0)
            self.assertEqual(fit['status'], 'accepted')
            widths.append([fit['sigma_x_px'], fit['sigma_y_px']])
        np.testing.assert_allclose(widths, [[1.2, 1.5]] * 3, atol=1e-5)

    def test_cannot_move_to_bright_neighbor(self):
        yy, xx = np.indices((25, 25))
        image = psf._gaussian_2d((xx, yy), 1000, 16, 12, 1, 1, 10)
        fit = psf.fit_frame_psf(image, 10, 12, 2, 2)
        self.assertNotEqual(fit['status'], 'accepted')
        if 'x0_px' in fit:
            self.assertLessEqual(abs(fit['x0_px'] - 10), psf.MAX_CENTER_SHIFT_PX)

    def test_zero_mad_and_small_samples(self):
        np.testing.assert_array_equal(psf.width_outliers([[1, 1]] * 6 + [[4, 3]]),
                                      [False] * 6 + [True])
        self.assertFalse(psf.width_outliers([[1, 1], [4, 3]]).any())
        np.testing.assert_array_equal(psf.width_outliers([[1, 1]] * 6 + [[.2, .2]], upper_only=True),
                                      [False] * 7)

    def test_frame_filtering_and_mean_of_fits(self):
        # XML coordinates/radii are in calibrated units, not pixels.
        spots = pd.DataFrame([dict(spot_id=f, frame=f, x=10 + f*.1, y=10,
                                   RADIUS='1') for f in range(9)])
        # An untracked neighbor must still exclude the overlapping target.
        spots = pd.concat([spots, pd.DataFrame([dict(spot_id=99, frame=1,
                                                   x=10.2, y=10, RADIUS='1')])], ignore_index=True)
        tracks = spots.iloc[:9].assign(trajectory_id='a')
        session = SimpleNamespace(spots=spots, trajectories=tracks, pixelwidth_um=.5,
                                  pixelheight_um=.5, path=Path('session.xml'))
        settings = dict(detector_radius='1', analysis_xml='session.xml')
        blurry = np.array([True] + [False]*8)
        results = [dict(status='accepted', fit_attempted=True, sigma_x_px=s, sigma_y_px=s,
                        amplitude=100, background=10, r2=.99)
                   for s in [1, 1, 1, 1, 1, 1, 4]]
        records = []
        with patch.object(psf, 'select_session', return_value=(session, settings)), \
             patch.object(psf, 'find_rec_tif_files', return_value={'tiff_file': Path('image.tif'), 'mpp': .1}), \
             patch.object(psf, 'extract_particle_size_from_path', return_value=50), \
             patch.object(psf, 'compute_sharpness_series', return_value=np.ones(9)), \
             patch.object(psf, 'flag_blurry_frames', return_value=(blurry, 0)), \
             patch.object(psf.tifffile, 'imread', return_value=np.ones((50, 50))), \
             patch.object(psf, 'fit_frame_psf', side_effect=results) as fitter:
            rows, detector_row, records = psf.process_movie('test', Path('50nm_Tracks.xml'), Path('session.xml'),
                                                             fit_counts={}, rng=np.random.default_rng(0))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['n_frames_fitted'], 6)
        self.assertEqual(rows[0]['sigma_x_psf_px'], 1)
        by_frame = {r['frame']: r for r in records}
        self.assertEqual(by_frame[0]['status'], 'blurry_frame')
        self.assertEqual(by_frame[1]['status'], 'overlapping_particles')
        self.assertEqual(sum(r['status'] == 'frame_width_outlier' for r in records), 1)
        self.assertEqual(fitter.call_args_list[0].args[3:5], (2., 2.))
        self.assertAlmostEqual(by_frame[2]['xml_x_px'], 20.4)

    def test_shared_track_budget_and_frame_cap(self):
        # Group is (Bedingung=dataset_label, Partikelgroesse). The budget counts
        # whole tracks ("Spots" the way the task phrased it), not individual
        # frame-fit calls, so a violin plot gets up to 200 distinct tracks
        # rather than a handful of very long ones. MAX_FRAMES_PER_TRACK bounds
        # the compute cost (and the group-budget share) of any single long
        # track -- real bug seen on immobilized data: without this cap, one
        # track spanning hundreds of frames could exhaust a whole group's
        # budget by itself, leaving only 1-9 tracks instead of up to 200.
        def make_session(n_tracks, frames_per_track):
            rows = [
                dict(spot_id=f"{t}_{f}", frame=t * frames_per_track + f, x=20, y=20, RADIUS='2',
                     trajectory_id=str(t))
                for t in range(n_tracks) for f in range(frames_per_track)
            ]
            spots = pd.DataFrame(rows)
            return SimpleNamespace(spots=spots, trajectories=spots, pixelwidth_um=1.,
                                   pixelheight_um=1., path=Path('session.xml'))

        counts = {}
        n_tracks, frames_per_track = 3, 15
        n_frames = n_tracks * frames_per_track
        with patch.object(psf, 'select_session',
                          return_value=(make_session(n_tracks, frames_per_track), {'detector_radius': '2'})) as select, \
             patch.object(psf, 'find_rec_tif_files', return_value={'tiff_file': Path('image.tif'), 'mpp': .1}), \
             patch.object(psf, 'extract_particle_size_from_path', return_value=50) as size, \
             patch.object(psf, 'compute_sharpness_series', return_value=np.ones(n_frames)), \
             patch.object(psf, 'flag_blurry_frames', return_value=(np.zeros(n_frames, dtype=bool), 0)), \
             patch.object(psf.tifffile, 'imread', return_value=np.ones((50, 50))), \
             patch.object(psf, 'MAX_SPOTS_PER_GROUP', 2), \
             patch.object(psf, 'MAX_FRAMES_PER_TRACK', 4), \
             patch.object(psf, 'fit_frame_psf', return_value={'status': 'fit_failed', 'fit_attempted': True}):
            rng = np.random.default_rng(42)

            rows_a, det_a, records_a = psf.process_movie('test', Path('a.xml'), Path('session.xml'), counts, rng)
            self.assertEqual(counts[('test', 50.)], 2)   # 2 tracks selected, not frames
            attempted = [r for r in records_a if r.get('fit_attempted')]
            by_track: dict[str, set] = {}
            for r in attempted:
                by_track.setdefault(r['trajectory_id'], set()).add(r['spot_id'])
            self.assertEqual(len(by_track), 2)
            for spot_ids in by_track.values():
                self.assertEqual(len(spot_ids), 4)   # capped per track, not all 15 frames

            # Group already at budget: the next movie is skipped before select_session runs.
            before = select.call_count
            rows_b, det_b, records_b = psf.process_movie('test', Path('b.xml'), Path('session.xml'), counts, rng)
            self.assertEqual(select.call_count, before)
            self.assertEqual(rows_b, [])
            self.assertIsNone(det_b)

            # A different particle size is a separate group, independent of the first budget.
            size.return_value = 100
            select.return_value = (make_session(n_tracks, frames_per_track), {'detector_radius': '2'})
            psf.process_movie('test', Path('c.xml'), Path('session.xml'), counts, rng)
            self.assertEqual(counts[('test', 100.)], 2)

            # A different dataset label (Bedingung) is likewise a separate group.
            size.return_value = 50
            select.return_value = (make_session(n_tracks, frames_per_track), {'detector_radius': '2'})
            psf.process_movie('other', Path('d.xml'), Path('session.xml'), counts, rng)
            self.assertEqual(counts[('other', 50.)], 2)

    def test_only_optimizer_calls_count(self):
        flat = np.ones((17, 17))
        fit = psf.fit_frame_psf(flat, 8, 8, 2, 2)
        self.assertFalse(fit.get('fit_attempted', False))
        yy, xx = np.indices(flat.shape)
        image = psf._gaussian_2d((xx, yy), 100, 8, 8, 1, 1, 10)
        with patch.object(psf, 'curve_fit', side_effect=RuntimeError('no convergence')):
            self.assertTrue(psf.fit_frame_psf(image, 8, 8, 2, 2)['fit_attempted'])


if __name__ == '__main__':
    unittest.main()
