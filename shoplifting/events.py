"""Convert model window scores into contiguous suggestions for human review."""
import numpy as np


def smooth_scores(scores, width=1):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional array.")
    if width < 1 or width % 2 == 0:
        raise ValueError("Smoothing width must be a positive odd integer.")
    if not len(values) or width == 1:
        return values.copy()
    padding = width // 2
    padded = np.pad(values, (padding, padding), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def review_intervals(starts, ends, scores, threshold, min_positive_windows=1):
    starts, ends, scores = (np.asarray(v, dtype=np.float64) for v in (starts, ends, scores))
    if not (starts.ndim == ends.ndim == scores.ndim == 1 and len(starts) == len(ends) == len(scores)):
        raise ValueError("Starts, ends and scores must have matching one-dimensional shapes.")
    if not all(np.isfinite(v).all() for v in (starts, ends, scores)):
        raise ValueError("Window data must be finite.")
    if not 0 <= threshold <= 1 or min_positive_windows < 1:
        raise ValueError("Threshold must be between 0 and 1; minimum windows must be positive.")
    if len(starts) and (np.any(starts < 0) or np.any(ends <= starts) or np.any(np.diff(starts) < 0)):
        raise ValueError("Windows must be ordered with positive durations.")
    result, current = [], None

    def finish():
        if current is not None and current["positive_windows"] >= min_positive_windows:
            result.append({**current, "start_seconds": round(current["start_seconds"], 3), "end_seconds": round(current["end_seconds"], 3), "peak_score": round(current["peak_score"], 6), "status": "needs_review"})

    for start, end, score in zip(starts, ends, scores):
        if score < threshold:
            finish()
            current = None
            continue
        if current is None or start > current["end_seconds"] + 1e-5:
            finish()
            current = {"start_seconds": float(start), "end_seconds": float(end), "peak_score": float(score), "positive_windows": 1}
        else:
            current["end_seconds"] = max(current["end_seconds"], float(end))
            current["peak_score"] = max(current["peak_score"], float(score))
            current["positive_windows"] += 1
    finish()
    return result
