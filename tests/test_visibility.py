import unittest
import numpy as np
from shoplifting.visibility import VisibilityAssessment


class VisibilityTests(unittest.TestCase):
    def info(self, duration=4):
        return {"frames": int(duration * 30), "width": 320, "height": 240, "duration": duration}

    def test_blank_footage_is_rejected_even_with_sufficient_duration(self):
        assessment = VisibilityAssessment()
        assessment.update([np.zeros((16, 128, 171, 3), dtype=np.uint8)])
        result = assessment.summarize(self.info(), [(0, 2), (2, 4)])
        self.assertIn("mostly_blank_or_extreme_exposure", result["issues"])
        self.assertIn("very_little_image_detail", result["issues"])

    def test_temporal_gaps_are_reported_but_overlap_is_not(self):
        rng = np.random.default_rng(4)
        assessment = VisibilityAssessment()
        assessment.update([rng.integers(0, 255, (16, 128, 171, 3), dtype=np.uint8)])
        continuous = assessment.summarize(self.info(), [(0, 2), (1, 3), (2, 4)])
        self.assertEqual(continuous["issues"], [])
        gaps = assessment.summarize(self.info(), [(0, 1), (3, 4)])
        self.assertEqual(gaps["unsampled_gaps"], [[1, 3]])
        self.assertIn("unsampled_temporal_gaps", gaps["issues"])

    def test_technical_checks_do_not_assert_an_action_is_visible(self):
        assessment = VisibilityAssessment()
        info = self.info();info.update(frames=8, width=80, height=60)
        result = assessment.summarize(info, [(0, 4)])
        self.assertIn("too_few_source_frames", result["issues"])
        self.assertIn("very_low_resolution", result["issues"])
        self.assertIn("does not verify", result["scope"])
