"""Review locations must come from valid model evidence, never example times."""
import copy
import math
import unittest

from shoplifting.review_suggestions import build_suggestions


def prediction(duration=30, scores=(.1, .7, .1, .6, .1, .8, .1)):
    return {
        "duration_seconds": duration,
        "status": "insufficient_evidence",
        "decision": {"reason_codes": ["uncertain_model_signal"]},
        "visibility": {"issues": []},
        "review_intervals": [],
        "windows": [
            {"start_seconds": index * 4, "end_seconds": index * 4 + 2,
             "raw_score": score, "review_score": score}
            for index, score in enumerate(scores)
        ],
    }


class ReviewSuggestionsTests(unittest.TestCase):
    def test_ranked_locations_follow_actual_varying_scores(self):
        result = prediction()
        suggestions = build_suggestions(result)
        self.assertEqual([item["start_seconds"] for item in suggestions], [19, 3, 11])
        self.assertEqual([item["peak_score"] for item in suggestions], [.8, .7, .6])
        self.assertTrue(all(item["kind"] == "unconfirmed" for item in suggestions))
        result["windows"][1]["review_score"] = .9
        self.assertEqual(build_suggestions(result)[0]["start_seconds"], 3)

    def test_flagged_intervals_take_priority_and_merge(self):
        result = prediction()
        result["review_intervals"] = [
            {"start_seconds": 12, "end_seconds": 17, "peak_score": .65},
            {"start_seconds": 15, "end_seconds": 19, "peak_score": .75},
            {"start_seconds": 2, "end_seconds": 4, "peak_score": .6},
        ]
        suggestions = build_suggestions(result)
        self.assertEqual(len(suggestions), 2)
        self.assertEqual((suggestions[0]["start_seconds"], suggestions[0]["end_seconds"]), (12, 19))
        self.assertEqual(suggestions[0]["peak_score"], .75)
        self.assertTrue(all(item["kind"] == "flagged" for item in suggestions))

    def test_clamp_bounds_and_reject_nan_infinite_and_invalid_scores(self):
        result = prediction(duration=10)
        result["review_intervals"] = [
            {"start_seconds": -3, "end_seconds": 50, "peak_score": .8},
            {"start_seconds": math.nan, "end_seconds": 5, "peak_score": .9},
            {"start_seconds": 2, "end_seconds": math.inf, "peak_score": .9},
            {"start_seconds": 2, "end_seconds": 4, "peak_score": math.nan},
            {"start_seconds": 2, "end_seconds": 4, "peak_score": 1.1},
        ]
        suggestions = build_suggestions(result)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual((suggestions[0]["start_seconds"], suggestions[0]["end_seconds"]), (0, 10))

    def test_padded_overlapping_peaks_keep_highest_without_duplicate_action(self):
        result = prediction(duration=20, scores=())
        result["windows"] = [
            {"start_seconds": start, "end_seconds": start + 3, "review_score": score}
            for start, score in [(0, .1), (2, .8), (4, .1), (5, .7), (10, .1)]
        ]
        suggestions = build_suggestions(result)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual((suggestions[0]["start_seconds"], suggestions[0]["end_seconds"]), (1, 6))

    def test_nearby_oscillating_peaks_cannot_chain_into_whole_video(self):
        result = prediction(duration=40, scores=())
        result["windows"] = [
            {"start_seconds": start, "end_seconds": start + 2,
             "review_score": (.8 - start * .001) if start % 2 == 0 else .1}
            for start in range(39)
        ]
        suggestions = build_suggestions(result)
        self.assertEqual(len(suggestions), 3)
        self.assertTrue(all(item["end_seconds"] - item["start_seconds"] <= 4 for item in suggestions))
        self.assertLess(sum(item["end_seconds"] - item["start_seconds"] for item in suggestions), 40)
        chronological = sorted(suggestions, key=lambda item: item["start_seconds"])
        for left, right in zip(chronological, chronological[1:]):
            self.assertLess(left["end_seconds"], right["start_seconds"])

    def test_short_healthy_nonflat_clip_is_offered_in_full(self):
        result = prediction(duration=5, scores=())
        result["windows"] = [
            {"start_seconds": 0, "end_seconds": 2, "review_score": .2},
            {"start_seconds": 2, "end_seconds": 5, "review_score": .6},
        ]
        suggestion, = build_suggestions(result)
        self.assertEqual(suggestion["kind"], "whole_clip")
        self.assertEqual((suggestion["start_seconds"], suggestion["end_seconds"]), (0, 5))

    def test_flat_near_flat_empty_and_invalid_windows_have_no_fallback(self):
        for scores in [(0, 0, 0), (.8, .8, .8), (.2, .200001, .2), ()]:
            self.assertEqual(build_suggestions(prediction(scores=scores)), [])
        result = prediction()
        result["windows"] = [{"start_seconds": 0, "end_seconds": 5, "review_score": math.nan}]
        self.assertEqual(build_suggestions(result), [])

    def test_blank_blurred_or_incomplete_coverage_is_not_localized(self):
        for issue in ["mostly_blank_or_extreme_exposure", "very_little_image_detail", "unsampled_temporal_gaps", "occluded"]:
            result = prediction()
            result["visibility"]["issues"] = [issue]
            result["review_intervals"] = [{"start_seconds": 4, "end_seconds": 6, "peak_score": .9}]
            self.assertEqual(build_suggestions(result), [])
        result = prediction()
        result["decision"]["reason_codes"] = ["very_low_resolution"]
        self.assertEqual(build_suggestions(result), [])

    def test_suggestions_do_not_change_decision_or_claim_confirmed_theft(self):
        result = prediction()
        original = copy.deepcopy(result)
        suggestions = build_suggestions(result)
        self.assertEqual(result, original)
        self.assertTrue(all("unconfirmed" in item["reason"] for item in suggestions))
        result["status"] = "no_review_flag"
        self.assertEqual(build_suggestions(result), [])

    def test_limits_and_padding_are_validated(self):
        self.assertEqual(len(build_suggestions(prediction(), max_suggestions=1)), 1)
        self.assertEqual(build_suggestions(prediction(), max_suggestions=0), [])
        for kwargs in [{"max_suggestions": -1}, {"max_suggestions": 1.5}, {"padding_seconds": math.nan}, {"padding_seconds": -1}]:
            with self.assertRaises(ValueError):
                build_suggestions(prediction(), **kwargs)

    def test_fallback_padding_stays_within_video_duration(self):
        result = prediction(duration=10, scores=())
        result["windows"] = [
            {"start_seconds": -1, "end_seconds": 2, "review_score": .7},
            {"start_seconds": 3, "end_seconds": 4, "review_score": .1},
            {"start_seconds": 8, "end_seconds": 12, "review_score": .8},
        ]
        suggestions = build_suggestions(result, padding_seconds=2)
        self.assertEqual([(item["start_seconds"], item["end_seconds"]) for item in suggestions], [(6, 10), (0, 4)])


if __name__ == "__main__":
    unittest.main()
