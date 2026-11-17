"""Train and evaluate video-level baselines and a weakly supervised MIL head.

The visual encoder is frozen before this stage. No person identification,
face matching, action timestamps, or window-level ground truth is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .metrics import binary_metrics, select_threshold
from .decision import calibrate_policy, classify_decision
from .mil import MILHead, LinearHead, TemporalGRUHead, aggregate_scores

SPLITS = ("train", "val", "test")


def split_records(records: list[dict], seed: int = 42) -> list[dict]:
    """Assign complete groups to reproducible 60/20/20 splits.

    Coverage of both classes takes precedence over approximate split sizes.
    Groups may contain normal and positive videos. Explicit splits are honoured
    only when every record supplies a valid split and no group crosses splits.
    """
    if not records:
        raise ValueError("Feature index has no records. Extract features from labelled videos first.")
    copied = [dict(record) for record in records]
    ids = set()
    groups = {}
    for record in copied:
        for key in ("id", "path", "group", "feature_file"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"Each record needs a non-empty string {key!r}")
        if record["id"] in ids:
            raise ValueError(f"Duplicate video id: {record['id']}")
        ids.add(record["id"])
        if type(record.get("label")) is not int or record["label"] not in (0, 1):
            raise ValueError(f"Video {record['id']} must have integer label 0 or 1")
        groups.setdefault(record["group"], []).append(record)

    explicit = ["split" in record for record in copied]
    if any(explicit) and not all(explicit):
        raise ValueError("Partial explicit splits are unsafe. Supply train/val/test for every video, or remove all split fields.")
    if all(explicit):
        for group, members in groups.items():
            if any(not isinstance(record["split"], str) for record in members):
                raise ValueError(f"Group {group!r} has an invalid split; use train, val, or test")
            values = {record["split"] for record in members}
            if not values.issubset(SPLITS):
                raise ValueError(f"Group {group!r} has an invalid split; use train, val, or test")
            if len(values) != 1:
                raise ValueError(f"Group leakage: group {group!r} appears in multiple splits. Keep related clips in the same split.")
    else:
        signatures = {group: frozenset(record["label"] for record in members) for group, members in groups.items()}
        for label in (0, 1):
            count = sum(label in signature for signature in signatures.values())
            if count < 3:
                raise ValueError(
                    f"Label {label} occurs in only {count} independent groups; at least 3 groups per class are needed "
                    "for train/val/test coverage. Add independent source videos and group related clips together."
                )
        rng = np.random.default_rng(seed)
        buckets = {signature: sorted(group for group, value in signatures.items() if value == signature)
                   for signature in (frozenset({0}), frozenset({1}), frozenset({0, 1}))}
        for bucket in buckets.values():
            rng.shuffle(bucket)
        assignments = {}
        mixed = buckets[frozenset({0, 1})]
        # Two mixed groups cover val/test, leaving pure-class pairs for train.
        destinations = SPLITS if len(mixed) >= 3 else ("test", "val")
        for group, split in zip(mixed[:3], destinations):
            assignments[group] = split
        for label in (0, 1):
            pure = iter(buckets[frozenset({label})])
            for split in SPLITS:
                if not any(value == split and label in signatures[group] for group, value in assignments.items()):
                    assignments[next(pure)] = split
        target = dict(zip(SPLITS, np.array([0.6, 0.2, 0.2]) * len(copied)))
        sizes = {split: sum(len(groups[group]) for group, value in assignments.items() if value == split) for split in SPLITS}
        remaining = sorted(set(groups) - set(assignments))
        rng.shuffle(remaining)
        # Place larger groups first, retaining seeded ordering for size ties.
        remaining.sort(key=lambda group: -len(groups[group]))
        for group in remaining:
            split = max(SPLITS, key=lambda value: target[value] - sizes[value])
            assignments[group] = split
            sizes[split] += len(groups[group])
        for record in copied:
            record["split"] = assignments[record["group"]]

    for split in SPLITS:
        labels = {record["label"] for record in copied if record["split"] == split}
        if labels != {0, 1}:
            raise ValueError(
                f"The {split} split must contain both labels 0 and 1. "
                "Add independent groups of the missing class or correct the explicit splits."
            )
    return copied


def load_feature_index(features_dir: str | Path, seed: int = 42) -> tuple[list[dict], dict]:
    directory = Path(features_dir).resolve()
    index_path = directory / "index.json"
    if not index_path.is_file():
        raise ValueError(f"Missing {index_path}. Extract labelled video features before training.")
    with index_path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    if index.get("schema_version") != 1 or not isinstance(index.get("feature_config"), dict):
        raise ValueError("Feature index must have schema_version 1 and a feature_config object")
    if not isinstance(index.get("records"), list):
        raise ValueError("Feature index must contain a records list")
    if index.get("complete") is False:
        raise ValueError("Feature extraction is incomplete. Finish or resume extraction before training.")
    if "expected_video_count" in index and index["expected_video_count"] != len(index["records"]):
        raise ValueError(
            f"Feature index contains {len(index['records'])} videos but expected {index['expected_video_count']}. "
            "Finish or resume extraction before training."
        )
    return split_records(index["records"], seed=seed), index["feature_config"]


def load_bags(features_dir: str | Path, records: list[dict]) -> dict[str, torch.Tensor]:
    """Validate NPZ contents and load frozen [windows, 512] feature bags."""
    directory = Path(features_dir).resolve()
    bags = {}
    for record in records:
        relative = Path(record["feature_file"])
        feature_path = (directory / relative).resolve()
        if relative.is_absolute() or not feature_path.is_relative_to(directory):
            raise ValueError(f"feature_file must be relative to the feature directory: {relative}")
        if not feature_path.is_file():
            raise ValueError(f"Missing features for {record['id']}: {feature_path}")
        with np.load(feature_path, allow_pickle=False) as archive:
            if not {"features", "starts", "ends"}.issubset(archive.files):
                raise ValueError(f"{feature_path.name} needs features, starts, and ends arrays")
            features = np.asarray(archive["features"])
            starts, ends = np.asarray(archive["starts"]), np.asarray(archive["ends"])
        if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 512 or features.shape[0] < 1:
            raise ValueError(f"{feature_path.name}: features must be float32 [windows, 512] with at least one window")
        count = len(features)
        if starts.dtype != np.float32 or ends.dtype != np.float32 or starts.shape != (count,) or ends.shape != (count,):
            raise ValueError(f"{feature_path.name}: starts and ends must be float32 arrays with one entry per window")
        if not (np.isfinite(features).all() and np.isfinite(starts).all() and np.isfinite(ends).all()):
            raise ValueError(f"{feature_path.name} contains non-finite features or timestamps")
        if (starts < 0).any() or (ends <= starts).any() or (np.diff(starts) < 0).any():
            raise ValueError(f"{feature_path.name} has invalid or unordered window timestamps")
        bags[record["id"]] = torch.from_numpy(features.copy())
    return bags


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _mil_scores(model, records, bags, topk_fraction, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for record in records:
            logit = aggregate_scores(model(bags[record["id"]].to(device)), topk_fraction)
            scores.append(float(torch.sigmoid(logit).cpu()))
    return np.asarray(scores, dtype=np.float64)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


DEPLOYMENT_RULE = "Highest validation average precision; ties by validation F1, then mean-logistic simplicity. Test metrics are not used."


def select_deployment(model_metrics: dict) -> str:
    """Choose between learned models using validation metrics only."""
    return max(("mil", "mean_logistic"), key=lambda name: (
        model_metrics[name]["val"]["pr_auc"],
        model_metrics[name]["val"]["f1"],
        name == "mean_logistic",
    ))


def train_models(
    features_dir: str | Path,
    output_dir: str | Path,
    *, seed: int = 42, epochs: int = 100, device: str = "cpu",
    hidden_dim: int = 128, dropout: float = 0.2, topk_fraction: float = 0.25,
    learning_rate: float = 1e-3, weight_decay: float = 1e-4,
    patience: int = 15, smoothness: float = 1e-3, threads: int = 4, head_type: str = "mil",
) -> dict:
    """Fit on train, select on validation, and evaluate the untouched test split."""
    if epochs < 1 or patience < 1 or threads < 1:
        raise ValueError("epochs, patience, and threads must be positive")
    if head_type not in {"mil", "temporal_gru"}:
        raise ValueError("Unsupported neural head type")
    if not 0 < topk_fraction <= 1 or learning_rate <= 0 or weight_decay < 0 or smoothness < 0:
        raise ValueError("Invalid top-k fraction, learning rate, weight decay, or smoothness")
    _seed_everything(seed)
    torch.set_num_threads(threads)
    records, feature_config = load_feature_index(features_dir, seed=seed)
    is_ucf_subset = all(record["group"].startswith("ucf:") for record in records)
    evaluation_unit = "derived_clip" if is_ucf_subset else "video"
    dataset_protocol = ("Custom source-video-group-disjoint resplit of temporally annotated original UCF-Crime TEST videos; not the official UCF benchmark." if is_ucf_subset else "Project-specific split of the supplied feature index.")
    bags = load_bags(features_dir, records)
    splits = {split: [record for record in records if record["split"] == split] for split in SPLITS}
    labels = {split: np.array([record["label"] for record in members], dtype=np.int64) for split, members in splits.items()}
    means = {split: np.stack([bags[record["id"]].mean(dim=0).numpy() for record in members]) for split, members in splits.items()}
    # Fit normalisation on training windows only. Float64 accumulation avoids
    # cancellation, and a small scale floor keeps inactive dimensions stable.
    window_count = sum(len(bags[record["id"]]) for record in splits["train"])
    feature_sum = torch.zeros(512, dtype=torch.float64)
    feature_square_sum = torch.zeros(512, dtype=torch.float64)
    for record in splits["train"]:
        values = bags[record["id"]].to(torch.float64)
        feature_sum += values.sum(dim=0)
        feature_square_sum += values.square().sum(dim=0)
    mean64 = feature_sum / window_count
    variance64 = (feature_square_sum / window_count - mean64.square()).clamp_min(0)
    feature_mean = mean64.float()
    feature_scale = variance64.sqrt().clamp_min(1e-6).float()
    bags = {video_id: (features - feature_mean) / feature_scale for video_id, features in bags.items()}
    chosen_device = torch.device(device)
    head_class = TemporalGRUHead if head_type == "temporal_gru" else MILHead
    model = head_class(input_dim=512, hidden_dim=hidden_dim, dropout=dropout).to(chosen_device)
    train_negatives = int(np.sum(labels["train"] == 0))
    train_positives = int(np.sum(labels["train"] == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_negatives / train_positives, device=chosen_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_rng = np.random.default_rng(seed)
    history, best_state = [], None
    best_ap, best_epoch, stale_epochs = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for position in epoch_rng.permutation(len(splits["train"])):
            record = splits["train"][int(position)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(bags[record["id"]].to(chosen_device))
            pooled = aggregate_scores(logits, topk_fraction)
            target = torch.tensor(float(record["label"]), device=chosen_device)
            loss = criterion(pooled, target)
            if smoothness and logits.numel() > 1:
                loss = loss + smoothness * torch.diff(torch.sigmoid(logits)).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite. Inspect feature magnitudes and reduce the learning rate.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        val_scores = _mil_scores(model, splits["val"], bags, topk_fraction, chosen_device)
        val_ap = float(average_precision_score(labels["val"], val_scores))
        history.append({"epoch": epoch, "train_loss": total_loss / len(splits["train"]), "val_pr_auc": val_ap})
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{epochs}: train loss {history[-1]['train_loss']:.4f}; "
                f"validation PR-AUC {val_ap:.4f}", flush=True,
            )
        if val_ap > best_ap + 1e-8:
            best_ap, best_epoch, stale_epochs = val_ap, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    model.load_state_dict(best_state)
    print(f"{head_type} training complete after {len(history)} epochs; restored epoch {best_epoch} with validation PR-AUC {best_ap:.4f}.", flush=True)

    # Fixed C avoids using test data or cross-validation folds with related clips.
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed))
    logistic.fit(means["train"], labels["train"])
    majority_label = int(train_positives > train_negatives)
    model_scores = {
        "majority": {split: np.full(len(members), float(majority_label)) for split, members in splits.items()},
        "mean_logistic": {split: logistic.predict_proba(means[split])[:, 1] for split in SPLITS},
        "mil": {split: _mil_scores(model, members, bags, topk_fraction, chosen_device) for split, members in splits.items()},
    }
    thresholds = {"majority": 0.5}
    thresholds.update({name: select_threshold(labels["val"], values["val"]) for name, values in model_scores.items() if name != "majority"})
    model_metrics = {name: {"threshold": thresholds[name], **{split: binary_metrics(labels[split], values[split], thresholds[name]) for split in SPLITS}} for name, values in model_scores.items()}
    selected_model = select_deployment(model_metrics)
    metrics = {
        "schema_version": 1,
        "evaluation_unit": evaluation_unit,
        "dataset_protocol": dataset_protocol,
        "grouping_unit": "original_source_video" if is_ucf_subset else "declared_group",
        "positive_label": "dataset-labelled shoplifting/concealment",
        "pr_auc_definition": "average precision (not trapezoidal PR area)",
        "seed": seed,
        "feature_config": feature_config,
        "split_method": "explicit_group_disjoint" if all("split" in record for record in json.loads((Path(features_dir) / "index.json").read_text(encoding="utf-8"))["records"]) else "seeded_group_disjoint_60_20_20_with_class_coverage",
        "split_counts": {split: {"videos": len(members), "groups": len({record["group"] for record in members}), "normal": int(np.sum(labels[split] == 0)), "positive": int(np.sum(labels[split] == 1))} for split, members in splits.items()},
        "selected_model": selected_model,
        "neural_head_type": head_type,
        "selection": {"threshold": "maximise positive-class F1 on validation only; ties closest to 0.5", "logistic_C": 1.0, "early_stopping": "validation average precision", "best_epoch": best_epoch, "epochs_run": len(history), "deployed_model": selected_model, "deployment_rule": DEPLOYMENT_RULE},
        "models": model_metrics,
        "history": history,
        "limitations": ["Metrics apply to the declared evaluation unit, not an independently validated event-localization benchmark.", "Window scores are ranking signals, not proof of shoplifting or intent.", "Performance is specific to the evaluated data and grouping; dataset bias and distribution shift remain possible.", "Source-file grouping does not verify independence of actors, cameras, stores or countries.", "This model performs no face matching or person identification."] + (["The original UCF test annotations were reused in a custom source-disjoint project split; these are not official UCF benchmark results."] if is_ucf_subset else []),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1, "head_type": head_type, "state_dict": best_state,
        "input_dim": 512, "hidden_dim": hidden_dim, "dropout": dropout,
        "topk_fraction": topk_fraction, "threshold": thresholds["mil"],
        "feature_config": feature_config, "seed": seed, "best_epoch": best_epoch,
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "label_mapping": {"0": "normal", "1": "dataset-labelled shoplifting/concealment"},
        "training_config": {"epochs": epochs, "patience": patience, "learning_rate": learning_rate, "weight_decay": weight_decay, "smoothness": smoothness, "positive_class_weight": train_negatives / train_positives, "threads": threads, "normalisation": "per-feature mean and population standard deviation across training windows only; scale floor 1e-6"},
        "evaluation_unit": evaluation_unit, "dataset_protocol": dataset_protocol,
        "grouping_unit": "original_source_video" if is_ucf_subset else "declared_group",
        "supervision": "derived_clip_labels_from_temporal_annotations" if is_ucf_subset else "weak_video_labels",
    }
    scaler, classifier = logistic.steps[0][1], logistic.steps[1][1]
    linear = LinearHead(512)
    with torch.no_grad():
        linear.linear.weight.copy_(torch.from_numpy(classifier.coef_.astype(np.float32)))
        linear.linear.bias.copy_(torch.from_numpy(classifier.intercept_.astype(np.float32)))
    logistic_checkpoint = {
        **checkpoint,
        "head_type": "mean_logistic", "state_dict": {name: value.detach().cpu().clone() for name, value in linear.state_dict().items()},
        "hidden_dim": 0, "dropout": 0.0, "topk_fraction": 1.0,
        "threshold": thresholds["mean_logistic"], "best_epoch": None,
        "feature_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
        "feature_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
        "training_config": {"method": "mean-feature logistic regression", "C": 1.0, "class_weight": "balanced", "normalisation": "StandardScaler fitted on training video means only"},
    }
    checkpoints = {"mil": checkpoint, "mean_logistic": logistic_checkpoint}
    for name, item in checkpoints.items():
        item["decision_policy"] = calibrate_policy(labels["val"], model_scores[name]["val"], thresholds[name])
        item["deployment_selection_rule"] = DEPLOYMENT_RULE
        item["selected_model"] = selected_model
        torch.save(item, output / f"{name}_model.pt")
    torch.save(checkpoints[selected_model], output / "model.pt")
    policy = checkpoints[selected_model]["decision_policy"]
    _write_json(output / "decision_policy.json", policy)
    selective = {}
    for split in SPLITS:
        decisions = [classify_decision(float(score), float(record.get("duration", 8.0)), policy)["status"]
                     for record, score in zip(splits[split], model_scores[selected_model][split])]
        decided = np.array([value != "insufficient_evidence" for value in decisions])
        flagged = np.array([value == "needs_review" for value in decisions])
        y = labels[split]
        selective[split] = {"sample_count": len(y), "coverage": float(decided.mean()),
            "insufficient_evidence_count": int((~decided).sum()), "review_count": int(flagged.sum()),
            "review_precision": float(y[flagged].mean()) if flagged.any() else None,
            "positive_review_recall": float((flagged & (y == 1)).sum() / (y == 1).sum()),
            "accuracy_among_decided": float((flagged[decided] == y[decided]).mean()) if decided.any() else None}
    metrics["selective_decision"] = {"policy_source": "validation only", "counts_before_technical_visibility_checks": selective,
        "caveat": "Empirical bounds and eight-second negative-context safeguard do not verify physical stages or guarantee precision. Report coverage alongside conditional accuracy."}
    if head_type == "temporal_gru":
        metrics["limitations"].append("Exploratory whole-incident follow-up reuses previously evaluated source groups; it is not a pristine new holdout. No shelf/pocket stage ground truth is available.")
    print(f"Deployment selected from validation only: {selected_model}", flush=True)
    with (output / "mean_logistic.pkl").open("wb") as stream:
        pickle.dump(logistic, stream)
    _write_json(output / "split_manifest.json", {"schema_version": 1, "seed": seed, "records": [{key: record[key] for key in ("id", "group", "label", "split")} for record in records]})
    _write_json(output / "metrics.json", metrics)
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["id", "group", "label", "duration", "mean_logistic_score", "mil_score"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for position, record in enumerate(splits["val"]):
            row = {key: record.get(key, "") for key in ("id", "group", "label", "duration")}
            row.update({f"{name}_score": float(model_scores[name]["val"][position]) for name in ("mean_logistic", "mil")})
            writer.writerow(row)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["id", "group", "label", "majority_score", "majority_prediction", "mean_logistic_score", "mean_logistic_prediction", "mil_score", "mil_prediction"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for position, record in enumerate(splits["test"]):
            row = {key: record[key] for key in ("id", "group", "label")}
            for name, values in model_scores.items():
                score = float(values["test"][position])
                row[f"{name}_score"] = score
                row[f"{name}_prediction"] = int(score >= thresholds[name])
            writer.writerow(row)
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Directory containing index.json and frozen feature NPZ files")
    parser.add_argument("--output", type=Path, required=True, help="Directory for checkpoints and actual evaluation results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--head-type", choices=["mil", "temporal_gru"], default="mil")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--topk-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--smoothness", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args(argv)
    try:
        metrics = train_models(arguments.features, arguments.output, **{key: value for key, value in vars(arguments).items() if key not in ("features", "output")})
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Training failed: {error}\n")
    print(f"Saved checkpoint and video-level metrics to {arguments.output}")
    selected = metrics["selected_model"]
    print(f"Deployed {selected}; test PR-AUC {metrics['models'][selected]['test']['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
