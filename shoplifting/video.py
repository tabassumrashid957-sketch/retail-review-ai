"""Deterministic temporal windows and RGB decoding; no facial analysis."""
from pathlib import Path
import cv2
import numpy as np


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError(f"Video has invalid frame count or frame rate: {path}")
        return {"fps": fps, "frames": frames, "width": width, "height": height, "duration": frames / fps}
    finally:
        capture.release()


def window_bounds(duration, window_seconds=2.0, stride_seconds=2.0, max_windows=0):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be positive and finite.")
    if not np.isfinite(window_seconds) or not np.isfinite(stride_seconds) or window_seconds <= 0 or stride_seconds <= 0 or max_windows < 0:
        raise ValueError("Window/stride must be positive; max_windows must be nonnegative.")
    last = max(0.0, duration - window_seconds)
    starts = list(np.arange(0.0, last + 1e-7, stride_seconds))
    if not starts or last - starts[-1] > 1e-5:
        starts.append(last)
    if max_windows and len(starts) > max_windows:
        selected = np.linspace(0, len(starts) - 1, max_windows, dtype=int)
        starts = [starts[i] for i in selected]
    return [(float(start), float(min(duration, start + window_seconds))) for start in starts]


def read_window(capture, start, end, fps, total_frames, num_frames=16):
    """Uniformly sample a short window; repeat frames only for very short videos."""
    if num_frames < 2:
        raise ValueError("At least two sampled frames are required.")
    final = min(total_frames - 1, max(0, int(np.ceil(end * fps)) - 1))
    first = min(final, max(0, int(start * fps)))
    indices = np.linspace(first, final, num_frames).astype(int)
    decoded = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Failed to decode frame {index}; the video may be corrupt.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize before stacking so unusually large inputs cannot exhaust memory.
        decoded.append(cv2.resize(rgb, (171, 128), interpolation=cv2.INTER_AREA))
    return np.stack(decoded)
