"""Synthetic, offline checks for leakage prevention and the training contract."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from shoplifting.metrics import binary_metrics, select_threshold
from shoplifting.mil import MILHead, aggregate_scores, head_from_checkpoint
from shoplifting.train import load_bags, load_feature_index, select_deployment, split_records, train_models


def records_for_groups(groups_per_class=6, clips_per_group=2):
    return [
        {"id": f"{label}-{group}-{clip}", "path": f"video-{label}-{group}-{clip}.mp4",
         "group": f"source-{label}-{group}", "label": label,
         "feature_file": f"{label}-{group}-{clip}.npz"}
        for label in (0, 1)
        for group in range(groups_per_class)
        for clip in range(clips_per_group)
    ]


class SplitTests(unittest.TestCase):
    def test_related_clips_stay_together_and_splits_cover_classes(self):
        records = records_for_groups()
        first = split_records(records, seed=7)
        self.assertEqual(first, split_records(records, seed=7))
        self.assertTrue(all("split" not in record for record in records))
        groups = {}
        for record in first:
            groups.setdefault(record["group"], set()).add(record["split"])
        self.assertTrue(all(len(values) == 1 for values in groups.values()))
        for split in ("train", "val", "test"):
            self.assertEqual({record["label"] for record in first if record["split"] == split}, {0, 1})

    def test_minimal_mixed_and_pure_groups_can_cover_splits(self):
        # Two groups each contain both classes; the last normal/positive groups
        # must share the remaining split. This is feasible with four groups.
        records = records_for_groups(groups_per_class=3, clips_per_group=1)
        for record in records:
            if record["group"].endswith("-0"):
                record["group"] = "mixed-0"
            if record["group"].endswith("-1"):
                record["group"] = "mixed-1"
        result = split_records(records)
        for split in ("train", "val", "test"):
            self.assertEqual({record["label"] for record in result if record["split"] == split}, {0, 1})

    def test_explicit_splits_honoured_and_leakage_rejected(self):
        records = records_for_groups(groups_per_class=3)
        names = ["train", "val", "test"]
        for record in records:
            record["split"] = names[int(record["group"].split("-")[-1])]
        self.assertEqual(records, split_records(records, seed=99))
        leaking = [dict(record) for record in records]
        leaking[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            split_records(leaking)
        partial = [dict(record) for record in records]
        partial[0].pop("split")
        with self.assertRaisesRegex(ValueError, "Partial"):
            split_records(partial)

    def test_insufficient_independent_groups_fails_actionably(self):
        with self.assertRaisesRegex(ValueError, "at least 3 groups per class"):
            split_records(records_for_groups(groups_per_class=2, clips_per_group=20))

    def test_interrupted_extraction_is_rejected_without_breaking_legacy_indices(self):
        records = records_for_groups(groups_per_class=3, clips_per_group=1)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "index.json"
            index = {"schema_version": 1, "feature_config": {}, "records": records}
            path.write_text(json.dumps(index))
            self.assertEqual(len(load_feature_index(directory)[0]), len(records))
            # A partially written index may already contain both classes and
            # enough groups. Coverage alone must not silently legitimise it.
            index.update(complete=False, expected_video_count=len(records))
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "extraction is incomplete"):
                load_feature_index(directory)
            index.update(complete=True, expected_video_count=len(records) + 1)
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "Finish or resume extraction"):
                load_feature_index(directory)
            index["expected_video_count"] = len(records)
            path.write_text(json.dumps(index))
            self.assertEqual(len(load_feature_index(directory)[0]), len(records))


class MetricTests(unittest.TestCase):
    def test_deployment_selection_ignores_test_metrics_and_has_a_fixed_tie_rule(self):
        values = {"mil": {"val": {"pr_auc": .8, "f1": .7}, "test": {"pr_auc": 1.0}}, "mean_logistic": {"val": {"pr_auc": .9, "f1": .6}, "test": {"pr_auc": 0.0}}}
        self.assertEqual(select_deployment(values), "mean_logistic")
        values["mil"]["val"] = {"pr_auc": .9, "f1": .7}
        self.assertEqual(select_deployment(values), "mil")
        values["mean_logistic"]["val"]["f1"] = .7
        self.assertEqual(select_deployment(values), "mean_logistic")

    def test_validation_threshold_and_auc(self):
        labels, scores = [0, 0, 1, 1], [0.1, 0.3, 0.4, 0.6]
        threshold = select_threshold(labels, scores)
        self.assertAlmostEqual(threshold, 0.4)
        metrics = binary_metrics(labels, scores, threshold)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(metrics["pr_auc"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(select_threshold([0, 1], [0.1, 0.9]), 0.5)

    def test_constant_scores_and_single_class_are_json_safe(self):
        constant = binary_metrics([0, 1, 0, 1], [0.0] * 4)
        self.assertEqual(constant["roc_auc"], 0.5)
        self.assertEqual(constant["pr_auc"], 0.5)
        self.assertEqual(constant["precision"], 0.0)
        single = binary_metrics([0, 0], [0.1, 0.2])
        self.assertIsNone(single["roc_auc"])
        self.assertIsNone(single["pr_auc"])
        json.dumps(single, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            binary_metrics([0, 1], [0.1, float("nan")])


class MILTests(unittest.TestCase):
    def test_window_dimension_and_topk_gradient(self):
        head = MILHead(hidden_dim=8, dropout=0)
        self.assertEqual(tuple(head(torch.zeros(1, 512)).shape), (1,))
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
        pooled = aggregate_scores(logits, 0.25)
        self.assertEqual(pooled.item(), 3.5)  # ceil(5 * .25) = 2
        pooled.backward()
        self.assertEqual(logits.grad.tolist(), [0, 0, 0, 0.5, 0.5])

    def test_offline_training_checkpoint_and_train_only_normalisation(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_dir, output = root / "features", root / "model"
            feature_dir.mkdir()
            records = records_for_groups(groups_per_class=6, clips_per_group=1)
            names = ["train", "train", "train", "val", "test", "test"]
            training_arrays = []
            for record in records:
                record["split"] = names[int(record["group"].split("-")[-1])]
                array = rng.normal(0, 0.1, (3, 512)).astype(np.float32)
                array[:, :8] += 1 if record["label"] else -1
                # A held-out-only change must not contaminate saved scaler.
                if record["split"] != "train":
                    array[:, 10] += 10
                else:
                    training_arrays.append(array)
                np.savez(feature_dir / record["feature_file"], features=array,
                         starts=np.array([0, 1, 2], dtype=np.float32),
                         ends=np.array([1, 2, 3], dtype=np.float32))
            config = {"encoder": "synthetic-offline-test", "feature_dim": 512}
            (feature_dir / "index.json").write_text(json.dumps({"schema_version": 1, "feature_config": config, "records": records}))
            metrics = train_models(feature_dir, output, seed=4, epochs=3, hidden_dim=8, dropout=0, patience=2)
            checkpoint = torch.load(output / "mil_model.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["feature_config"], config)
            stacked = np.concatenate(training_arrays).astype(np.float64)
            np.testing.assert_allclose(checkpoint["feature_mean"].numpy(), stacked.mean(axis=0), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(checkpoint["feature_scale"].numpy(), stacked.std(axis=0), rtol=1e-5, atol=1e-6)
            restored = MILHead(512, checkpoint["hidden_dim"], checkpoint["dropout"])
            restored.load_state_dict(checkpoint["state_dict"])
            restored.eval()
            records_test = [record for record in records if record["split"] == "test"]
            bags = load_bags(feature_dir, records_test)
            with (output / "test_predictions.csv").open() as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len(predictions), len(records_test))
            for row, record in zip(predictions, records_test):
                features = (bags[record["id"]] - checkpoint["feature_mean"]) / checkpoint["feature_scale"]
                with torch.no_grad():
                    expected = torch.sigmoid(aggregate_scores(restored(features), checkpoint["topk_fraction"])).item()
                self.assertAlmostEqual(float(row["mil_score"]), expected, places=6)
            # Both portable exports must reproduce the recorded scores. The
            # deployed checkpoint must be the predeclared validation winner.
            self.assertEqual(metrics["selected_model"], select_deployment(metrics["models"]))
            for name in ("mil", "mean_logistic", "selected"):
                filename = "model.pt" if name == "selected" else f"{name}_model.pt"
                exported = torch.load(output / filename, map_location="cpu", weights_only=True)
                head = head_from_checkpoint(exported).eval()
                score_name = metrics["selected_model"] if name == "selected" else name
                for row, record in zip(predictions, records_test):
                    features = (bags[record["id"]] - exported["feature_mean"]) / exported["feature_scale"]
                    with torch.no_grad():
                        actual = torch.sigmoid(aggregate_scores(head(features), exported["topk_fraction"])).item()
                    self.assertAlmostEqual(float(row[f"{score_name}_score"]), actual, delta=1e-5)
            self.assertEqual(set(metrics["models"]), {"majority", "mean_logistic", "mil"})
            for model_metrics in metrics["models"].values():
                for split in ("train", "val", "test"):
                    self.assertEqual(model_metrics[split]["threshold"], model_metrics["threshold"])
                    self.assertTrue(np.isfinite(model_metrics[split]["pr_auc"]))
                    self.assertTrue(np.isfinite(model_metrics[split]["roc_auc"]))
            json.loads((output / "metrics.json").read_text())


if __name__ == "__main__":
    unittest.main()
