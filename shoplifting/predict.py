"""Score recorded retail actions, abstaining when evidence is insufficient."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .backbone import VideoEncoder, configure_device
from .decision import classify_decision
from .events import smooth_scores, review_intervals
from .mil import aggregate_scores, head_from_checkpoint
from .video import read_window, video_info, window_bounds
from .visibility import VisibilityAssessment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_TEXT = {
    "needs_review": "POSSIBLE CONCEALMENT - REVIEW REQUIRED",
    "insufficient_evidence": "CANNOT DETERMINE - INSUFFICIENT EVIDENCE",
    "no_review_flag": "NO SUSPICIOUS ACTION FLAGGED",
}


def decision_for_score(score, duration, checkpoint, quality_issues, threshold_override=None):
    policy = checkpoint.get("decision_policy")
    if policy is None:
        policy = {"positive_threshold": float(checkpoint["threshold"]), "negative_threshold": None,
                  "min_context_seconds": 8.0, "source": "legacy_review_threshold_no_negative_calibration"}
    if threshold_override is not None:
        policy = {"positive_threshold": float(threshold_override), "negative_threshold": None,
                  "min_context_seconds": 8.0, "source": "manual_uncalibrated_review_threshold"}
    return classify_decision(score, duration, policy, quality_issues=quality_issues), policy


def write_annotated(source, destination, starts, ends, scores, threshold, video_alert, *, decision=None):
    info = video_info(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["width"], info["height"]))
    if not writer.isOpened():
        capture.release()
        raise ValueError("The video writer could not open. Use an .mp4 output path.")
    status = decision["status"] if decision else "needs_review" if video_alert else "no_review_flag"
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_number / info["fps"]
            active = [score for start, end, score in zip(starts, ends, scores) if start <= timestamp < end]
            score = max(active, default=0.0)
            color = {"needs_review": (50, 100, 240), "insufficient_evidence": (30, 190, 240), "no_review_flag": (70, 180, 100)}[status]
            title = STATUS_TEXT[status]
            detail = "Need more context or a clearer view." if status == "insufficient_evidence" else "Human review needed. Theft is not confirmed."
            footer = f"Model score {score:.2f} | {detail}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = min(.65, max(.15, (info["width"] - 16) / max(1, cv2.getTextSize(title, font, 1, 1)[0][0])))
            footer_scale = min(scale, max(.15, (info["width"] - 16) / max(1, cv2.getTextSize(footer, font, 1, 1)[0][0])))
            cv2.rectangle(frame, (0, 0), (info["width"], min(60, info["height"])), (20, 20, 20), -1)
            cv2.putText(frame, title, (8, 22), font, scale, color, 1, cv2.LINE_AA)
            cv2.putText(frame, footer, (8, 45), font, footer_scale, (230, 230, 230), 1, cv2.LINE_AA)
            writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        writer.release()
    if frame_number == 0:
        raise ValueError("No frames could be written.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "artifacts/model/model.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/predictions/prediction.json")
    parser.add_argument("--annotated-video", type=Path)
    parser.add_argument("--threshold", type=float, help="Uncalibrated manual review threshold override; negative conclusions remain disabled.")
    parser.add_argument("--smooth-windows", type=int, default=1)
    parser.add_argument("--min-positive-windows", type=int, default=1)
    parser.add_argument("--stride-seconds", type=float, help="Defaults to the training stride.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts/torch")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error("trained model not found; restore artifacts/model or use BUILD_MODEL.command")
    if not args.video.is_file():
        parser.error("input video not found")
    if args.batch_size < 1 or args.min_positive_windows < 1:
        parser.error("batch size and minimum positive windows must be positive")
    if args.smooth_windows < 1 or args.smooth_windows % 2 == 0:
        parser.error("--smooth-windows must be positive and odd")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json")
    if args.annotated_video and args.annotated_video.suffix.lower() != ".mp4":
        parser.error("--annotated-video must end in .mp4")
    protected = {args.video.resolve(), args.model.resolve()}
    outputs = [args.output.resolve(), args.output.with_suffix(".csv").resolve()]
    if args.annotated_video:
        outputs.append(args.annotated_video.resolve())
    if any(path in protected for path in outputs) or len(set(outputs)) != len(outputs):
        parser.error("output paths must be distinct and must not overwrite the input or model")
    device = configure_device(args.device, args.threads)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    config = checkpoint["feature_config"]
    if config.get("backbone") != "r3d_18" or config.get("weights") != "KINETICS400_V1":
        parser.error("unsupported feature configuration in checkpoint")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    head = head_from_checkpoint(checkpoint).eval().to(device)
    mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(checkpoint["feature_scale"], dtype=torch.float32, device=device).clamp_min(1e-6)
    encoder = VideoEncoder(str(device), args.cache_dir, args.threads)
    info = video_info(args.video)
    stride = config["stride_seconds"] if args.stride_seconds is None else args.stride_seconds
    bounds = window_bounds(info["duration"], config["window_seconds"], stride, config.get("max_windows", 0))
    capture = cv2.VideoCapture(str(args.video))
    features, visibility = [], VisibilityAssessment()
    try:
        for begin in range(0, len(bounds), args.batch_size):
            clips = [read_window(capture, start, end, info["fps"], info["frames"], config["num_frames"]) for start, end in bounds[begin:begin + args.batch_size]]
            visibility.update(clips)
            features.append(encoder.encode(clips))
            print(f"Scored {min(begin+args.batch_size,len(bounds))}/{len(bounds)} video windows", flush=True)
    finally:
        capture.release()
    with torch.inference_mode():
        values = torch.from_numpy(np.concatenate(features)).to(device)
        logits = head((values - mean) / scale)
        raw_scores = logits.sigmoid().cpu().numpy()
        video_score = float(aggregate_scores(logits, checkpoint["topk_fraction"]).sigmoid().cpu())
    scores = smooth_scores(raw_scores, args.smooth_windows)
    starts, ends = [v[0] for v in bounds], [v[1] for v in bounds]
    quality = visibility.summarize(info, bounds)
    decision, policy = decision_for_score(video_score, info["duration"], checkpoint, quality["issues"], args.threshold)
    review_threshold = policy.get("positive_threshold")
    intervals = review_intervals(starts, ends, scores, review_threshold, args.min_positive_windows) if decision["status"] == "needs_review" and review_threshold is not None else []
    if decision["status"] == "needs_review" and not intervals:
        decision = {"status": "insufficient_evidence", "reason_codes": ["insufficient_temporal_support"], "message": "Cannot determine: the requested temporal support is missing."}
    windows = [{"start_seconds": round(start, 6), "end_seconds": round(end, 6), "raw_score": round(float(raw), 6), "review_score": round(float(score), 6)} for (start, end), raw, score in zip(bounds, raw_scores, scores)]
    result = {
        "schema_version": 2, "task": "possible_concealment_action_review", "video": str(args.video.resolve()), "model": str(args.model.resolve()),
        "duration_seconds": info["duration"], "video_score": video_score, "review_threshold": review_threshold,
        "status": decision["status"], "result_text": STATUS_TEXT[decision["status"]], "decision": decision, "decision_policy": policy,
        "visibility": quality, "shelf_to_pocket_sequence_verified": False, "theft_confirmed": False,
        "alert_count": len(intervals), "review_intervals": intervals, "windows": windows,
        "feature_config": config, "inference_stride_seconds": stride, "smoothing_windows": args.smooth_windows,
        "minimum_positive_windows": args.min_positive_windows,
        "limitations": [
            "Uncalibrated model scores are not certainty, proof of theft, or proof of payment status.",
            "Dataset annotations identify incident intervals, not shelf-pickup and pocket-placement stages. This model cannot verify those physical stages.",
            "Short weak-signal clips and technically unusable footage are inconclusive; duration alone never confirms an action.",
            "The image-detail and minimum-context checks are conservative engineering heuristics, not validated action-completeness detectors.",
            "Suggested review intervals are not independently validated physical-action boundaries.",
            "Performance on a small source-grouped subset does not establish performance in new stores.",
            "No face recognition, identity matching or external notifications are performed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["start_seconds", "end_seconds", "raw_score", "review_score"])
        writer.writeheader(); writer.writerows(windows)
    if args.annotated_video:
        write_annotated(args.video, args.annotated_video, starts, ends, scores, review_threshold, decision["status"] == "needs_review", decision=decision)
    print(f"Result: {result['result_text']}")
    print(decision["message"])
    print(f"Model score: {video_score:.3f}; review bound: {review_threshold}; {result['status']}")
    print(f"Review data: {args.output}; window scores: {csv_path}")
    return result


if __name__ == "__main__":
    main()
