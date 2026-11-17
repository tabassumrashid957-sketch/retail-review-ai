"""Validation-derived abstention bounds, never a judgement of theft.

Target precision is an empirical validation criterion, not a calibrated
probability or a guarantee on new footage. The context-duration rule is an
engineering safeguard; it does not detect whether physical stages are visible.
"""
import math
import numpy as np


def calibrate_policy(labels, scores, reference_threshold, min_support=3, target_precision=.9):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Need matching nonempty binary labels and scores")
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("Scores must be finite within [0,1]")
    if not math.isfinite(reference_threshold) or not 0 <= reference_threshold <= 1 or min_support < 1 or not 0 < target_precision <= 1:
        raise ValueError("Invalid policy calibration settings")
    positive, negative = None, None
    support = {}
    for bound in sorted(set(s[s >= reference_threshold])):
        mask = s >= bound
        if mask.sum() >= min_support and float(y[mask].mean()) >= target_precision:
            positive = float(bound)
            support['positive'] = {'count': int(mask.sum()), 'correct': int(y[mask].sum()), 'empirical_precision': float(y[mask].mean())}
            break
    for bound in sorted(set(s[s < reference_threshold]), reverse=True):
        mask = s <= bound
        if mask.sum() >= min_support and float((y[mask] == 0).mean()) >= target_precision:
            negative = float(bound)
            support['negative'] = {'count': int(mask.sum()), 'correct': int((y[mask] == 0).sum()), 'empirical_negative_predictive_value': float((y[mask] == 0).mean())}
            break
    return {'schema_version': 1, 'positive_threshold': positive, 'negative_threshold': negative,
            'min_context_seconds': 8.0, 'target_empirical_precision': target_precision,
            'minimum_validation_support': min_support, 'validation_support': support,
            'source': 'validation_only_empirical_score_bounds',
            'caveat': 'Uncalibrated model scores. Small validation support does not guarantee precision in new footage. Duration does not verify sequence completeness.'}


def classify_decision(score, duration, policy, quality_issues=()):
    if not math.isfinite(score) or not 0 <= score <= 1 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Need a finite score in [0,1] and positive finite duration')
    if not isinstance(policy, dict):
        raise ValueError('Need an explicit decision policy')
    pos, neg = policy.get('positive_threshold'), policy.get('negative_threshold')
    context = policy.get('min_context_seconds', 8.0)
    if not math.isfinite(context) or context < 0:
        raise ValueError('Invalid context duration')
    for bound in (pos, neg):
        if bound is not None and (not math.isfinite(bound) or not 0 <= bound <= 1):
            raise ValueError('Invalid decision bound')
    if pos is not None and neg is not None and neg >= pos:
        raise ValueError('Positive and negative decision bounds must not overlap')
    if quality_issues:
        status, reasons, message = 'insufficient_evidence', list(quality_issues), 'Cannot determine: the footage has insufficient image detail or temporal coverage. Use clearer, continuous footage.'
    elif pos is not None and score >= pos:
        status, reasons, message = 'needs_review', ['strong_model_review_signal'], 'Possible concealment: review the full action. Shelf-to-pocket stages and theft are not confirmed by this model.'
    elif duration < context:
        status, reasons, message = 'insufficient_evidence', ['limited_context_not_verified'], 'Cannot determine: this short clip has no strong review signal and may omit part of the action. Provide footage before and after the movement.'
    elif neg is not None and score <= neg:
        status, reasons, message = 'no_review_flag', ['low_model_review_signal'], 'No suspicious action was flagged. This does not verify that theft did not occur or that the full action is visible.'
    else:
        status, reasons, message = 'insufficient_evidence', ['uncertain_model_signal'], 'Cannot determine: the model signal is inconclusive. Review clearer footage showing the complete action.'
    return {'status': status, 'reason_codes': reasons, 'message': message,
            'sequence_completeness_verified': False, 'shelf_to_pocket_verified': False,
            'theft_confirmed': False, 'requires_staff_confirmation': True}
