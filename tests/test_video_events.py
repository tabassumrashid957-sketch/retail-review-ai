import unittest
import numpy as np
from shoplifting.events import review_intervals, smooth_scores
from shoplifting.video import window_bounds
from shoplifting.prepare import infer_label
from pathlib import Path


class TemporalTests(unittest.TestCase):
    def test_short_video_still_has_one_complete_window(self):
        self.assertEqual(window_bounds(0.3, 2, 2), [(0.0, 0.3)])

    def test_tail_is_included_without_invalid_window(self):
        bounds = window_bounds(5, 2, 2)
        self.assertEqual(bounds, [(0.0, 2.0), (2.0, 4.0), (3.0, 5.0)])

    def test_sampling_keeps_beginning_and_end(self):
        bounds = window_bounds(100, 2, 1, 4)
        self.assertEqual(len(bounds), 4)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 100)

    def test_negative_windows_break_an_alert(self):
        events = review_intervals([0, 1, 2, 3], [2, 3, 4, 5], [.8, .9, .2, .7], .6)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["positive_windows"], 2)
        self.assertEqual(events[1]["start_seconds"], 3)

    def test_unsampled_gaps_do_not_merge(self):
        events = review_intervals([0, 10], [2, 12], [.9, .9], .5)
        self.assertEqual(len(events), 2)

    def test_minimum_duration_rejects_isolated_hit(self):
        self.assertEqual(review_intervals([0], [2], [.9], .5, 2), [])

    def test_smoothing_preserves_constant_edges_and_length(self):
        values = smooth_scores([.8, .8], 5)
        np.testing.assert_allclose(values, [.8, .8])

    def test_invalid_scores_fail(self):
        with self.assertRaises(ValueError):
            review_intervals([0], [1], [float("nan")], .5)
        with self.assertRaises(ValueError):
            window_bounds(10, stride_seconds=float("nan"))

    def test_non_shoplifting_is_negative(self):
        self.assertEqual(infer_label(Path("data/Non-Shoplifting/a.mp4"), Path("data")), 0)
        self.assertEqual(infer_label(Path("data/Shoplifting/a.mp4"), Path("data")), 1)
        self.assertEqual(infer_label(Path("data/Abnormal/a.mp4"), Path("data")), 1)


if __name__ == "__main__":
    unittest.main()
