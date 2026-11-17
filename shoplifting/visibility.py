"""Conservative image-detail checks, not a detector of action completeness.

These technical heuristics can reject unusable footage. Passing them does not
establish that a hand, item, shelf, or pocket is visible.
"""
from __future__ import annotations

import cv2
import numpy as np


class VisibilityAssessment:
    def __init__(self):
        self.detail = []
        self.blank = []

    def update(self, clips):
        for clip in clips:
            for frame in clip:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                self.detail.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
                self.blank.append(bool(np.mean(gray <= 4) > .98 or np.mean(gray >= 251) > .98))

    def summarize(self, info, bounds):
        issues = []
        if info["frames"] < 16:
            issues.append("too_few_source_frames")
        if info["width"] < 160 or info["height"] < 120:
            issues.append("very_low_resolution")
        detail = float(np.median(self.detail)) if self.detail else 0.0
        blank = float(np.mean(self.blank)) if self.blank else 1.0
        if blank >= .9:
            issues.append("mostly_blank_or_extreme_exposure")
        if detail < 2.0:
            issues.append("very_little_image_detail")
        covered_end = 0.0
        gaps = []
        for start, end in bounds:
            if start > covered_end + 1e-5:
                gaps.append([covered_end, start])
            covered_end = max(covered_end, end)
        if covered_end < info["duration"] - 1e-5:
            gaps.append([covered_end, info["duration"]])
        if gaps:
            issues.append("unsampled_temporal_gaps")
        return {
            "issues": issues,
            "sampled_frame_count": len(self.detail),
            "median_laplacian_variance": detail,
            "blank_frame_fraction": blank,
            "unsampled_gaps": gaps,
            "scope": "Technical image-detail heuristics only; passing does not verify shelf-to-pocket visibility or completeness.",
        }
