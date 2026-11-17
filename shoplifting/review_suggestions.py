"""Locate model-scored segments for staff review without changing a decision.

Unconfirmed suggestions are navigation aids, not detections of theft or of a
complete physical action. Near-flat scores provide no useful localization.
The score-contrast tolerance below only suppresses near-flat navigation hints;
it is not an alert threshold or a calibrated measure of certainty.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


_MIN_SCORE_CONTRAST = 0.01
_TECHNICAL_REASONS = frozenset({
    "too_few_source_frames", "very_low_resolution",
    "mostly_blank_or_extreme_exposure", "very_little_image_detail",
    "unsampled_temporal_gaps", "blur", "blurred", "occlusion", "occluded",
})


def _finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _interval(row, duration, score_key):
    if not isinstance(row, Mapping):
        return None
    start = _finite_number(row.get("start_seconds"))
    end = _finite_number(row.get("end_seconds"))
    score = _finite_number(row.get(score_key))
    if start is None or end is None or score is None or not 0 <= score <= 1:
        return None
    start, end = max(0.0, min(start, duration)), max(0.0, min(end, duration))
    if end <= start:
        return None
    return {"start_seconds": start, "end_seconds": end, "peak_score": score}


def _merge(intervals):
    """Merge overlapping/touching intervals, keeping their actual peak score."""
    merged = []
    for interval in sorted(intervals, key=lambda item: (item["start_seconds"], item["end_seconds"])):
        if merged and interval["start_seconds"] <= merged[-1]["end_seconds"]:
            merged[-1]["end_seconds"] = max(merged[-1]["end_seconds"], interval["end_seconds"])
            merged[-1]["peak_score"] = max(merged[-1]["peak_score"], interval["peak_score"])
        else:
            merged.append(dict(interval))
    return sorted(merged, key=lambda item: (-item["peak_score"], item["start_seconds"], item["end_seconds"]))


def _score_peaks(windows):
    """Find temporal score peaks so low-scored windows cannot bridge actions."""
    plateaus = []
    for window in sorted(windows, key=lambda item: (item["start_seconds"], item["end_seconds"])):
        if (plateaus and abs(window["peak_score"] - plateaus[-1]["peak_score"]) <= 1e-9
                and window["start_seconds"] <= plateaus[-1]["end_seconds"]):
            plateaus[-1]["end_seconds"] = max(plateaus[-1]["end_seconds"], window["end_seconds"])
        else:
            plateaus.append(dict(window))
    floor = min(window["peak_score"] for window in windows)
    peaks = []
    for index, interval in enumerate(plateaus):
        score = interval["peak_score"]
        previous = plateaus[index - 1]["peak_score"] if index else -1.0
        following = plateaus[index + 1]["peak_score"] if index + 1 < len(plateaus) else -1.0
        if score >= previous and score >= following and score - floor >= _MIN_SCORE_CONTRAST:
            peaks.append(interval)
    return peaks


def build_suggestions(result, max_suggestions=3, padding_seconds=1.0):
    """Return ranked, bounded suggestions derived only from prediction data.

    Existing review intervals take priority. For an inconclusive decision with
    usable footage, non-flat score peaks may be offered as explicitly
    unconfirmed review locations. A short clip is offered in full, since a
    tighter crop could hide needed context. Technical visibility failures and
    missing/invalid windows yield no suggestion. The input is never modified.
    """
    if isinstance(max_suggestions, bool) or not isinstance(max_suggestions, int) or max_suggestions < 0:
        raise ValueError("max_suggestions must be a nonnegative integer")
    padding = _finite_number(padding_seconds)
    if padding is None or padding < 0:
        raise ValueError("padding_seconds must be finite and nonnegative")
    if not isinstance(result, Mapping) or max_suggestions == 0:
        return []
    duration = _finite_number(result.get("duration_seconds"))
    if duration is None or duration <= 0:
        return []
    visibility = result.get("visibility", {})
    decision = result.get("decision", {})
    if isinstance(visibility, Mapping) and visibility.get("issues"):
        return []
    reasons = decision.get("reason_codes", ()) if isinstance(decision, Mapping) else ()
    if isinstance(reasons, (list, tuple, set)) and _TECHNICAL_REASONS.intersection(str(reason) for reason in reasons):
        return []
    rows = result.get("windows", ())
    if not isinstance(rows, (list, tuple)):
        return []
    windows = []
    for row in rows:
        # review_score is the score used to form the real review intervals.
        key = "review_score" if isinstance(row, Mapping) and "review_score" in row else "raw_score"
        interval = _interval(row, duration, key)
        if interval is not None:
            windows.append(interval)
    if not windows:
        return []

    flagged_rows = result.get("review_intervals", ())
    flagged = []
    if isinstance(flagged_rows, (list, tuple)):
        for row in flagged_rows:
            interval = _interval(row, duration, "peak_score")
            if interval is not None:
                flagged.append(interval)
    if flagged:
        candidates = _merge(flagged)
        kind = "flagged"
        reason = "Model-flagged interval for staff review. The action and theft are not confirmed."
    else:
        status = result.get("status", decision.get("status") if isinstance(decision, Mapping) else None)
        scores = [window["peak_score"] for window in windows]
        if status != "insufficient_evidence" or max(scores) - min(scores) < _MIN_SCORE_CONTRAST:
            return []
        if duration <= 6.0:
            candidates = [{"start_seconds": 0.0, "end_seconds": duration, "peak_score": max(scores)}]
            kind = "whole_clip"
            reason = "Review the whole short clip for context. The model signal and action remain unconfirmed."
        else:
            peaks = sorted(_score_peaks(windows), key=lambda item: (-item["peak_score"], item["start_seconds"]))
            candidates = []
            for interval in peaks:
                interval["start_seconds"] = max(0.0, interval["start_seconds"] - padding)
                interval["end_seconds"] = min(duration, interval["end_seconds"] + padding)
                # Suppress a lower-ranked nearby peak before any merging can
                # chain several movements into an unhelpful whole-video hint.
                if any(interval["start_seconds"] <= chosen["end_seconds"]
                       and interval["end_seconds"] >= chosen["start_seconds"]
                       for chosen in candidates):
                    continue
                candidates.append(interval)
                if len(candidates) == max_suggestions:
                    break
            kind = "unconfirmed"
            reason = "Higher-scored segment offered for manual review. The model signal and action remain unconfirmed."
    return [
        {"id": f"suggestion-{index}", **interval, "kind": kind, "reason": reason}
        for index, interval in enumerate(candidates[:max_suggestions], 1)
    ]
