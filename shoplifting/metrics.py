"""Video-level binary metrics with explicit handling of undefined quantities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _validated_arrays(labels, scores):
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if len(labels_array) == 0 or len(labels_array) != len(scores_array):
        raise ValueError("labels and scores must be non-empty and have equal length")
    if not np.isin(labels_array, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(scores_array).all():
        raise ValueError("scores must be finite")
    if ((scores_array < 0) | (scores_array > 1)).any():
        raise ValueError("scores must be probabilities in [0, 1]")
    return labels_array.astype(np.int64), scores_array


def select_threshold(labels, scores) -> float:
    """Select a probability threshold using validation labels only.

    Maximise F1 for the positive class; break ties by proximity to 0.5,
    then prefer the higher threshold. Predictions use score >= threshold.
    """
    y, probabilities = _validated_arrays(labels, scores)
    if np.unique(y).size != 2:
        raise ValueError("Threshold selection requires both classes in validation data")
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1], denominator,
        out=np.zeros_like(denominator), where=denominator != 0,
    )
    # Include 0.5 to prefer the conventional threshold when it gives the same F1.
    default_prediction = probabilities >= 0.5
    tp = int(np.sum((y == 1) & default_prediction))
    fp = int(np.sum((y == 0) & default_prediction))
    fn = int(np.sum((y == 1) & ~default_prediction))
    default_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    candidates = list(zip(thresholds.tolist(), f1.tolist())) + [(0.5, default_f1)]
    best_f1 = max(value for _, value in candidates)
    tied = [threshold for threshold, value in candidates if np.isclose(value, best_f1, rtol=0, atol=1e-12)]
    return float(min(tied, key=lambda threshold: (abs(threshold - 0.5), -threshold)))


def binary_metrics(labels, scores, threshold: float = 0.5) -> dict:
    """Return JSON-safe metrics. PR-AUC here means average precision.

    Single-class data has no defined discrimination AUC, represented by null
    rather than NaN or a misleading score. Training requires both classes in
    each split, but this helper also supports small standalone evaluations.
    """
    y, probabilities = _validated_arrays(labels, scores)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    predictions = (probabilities >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", zero_division=0,
    )
    both_classes = np.unique(y).size == 2
    negatives, positives = int(np.sum(y == 0)), int(np.sum(y == 1))
    return {
        "sample_count": int(len(y)),
        "negative_count": negatives,
        "positive_count": positives,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)) if both_classes else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y, probabilities)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "confusion_matrix_order": [["true_negative", "false_positive"], ["false_negative", "true_positive"]],
        "undefined_metrics_reason": None if both_classes else "Only one class is present; discrimination AUCs and balanced accuracy are undefined.",
    }
