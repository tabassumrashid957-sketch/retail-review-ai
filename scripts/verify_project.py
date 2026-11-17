"""Check the local project without downloading data or training any model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shoplifting.data_config import active_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts/model/model.pt"
ENCODER_PATH = PROJECT_ROOT / "artifacts/torch/checkpoints/r3d_18-b3b3357e.pth"
DEPENDENCIES = ("numpy", "cv2", "torch", "torchvision", "sklearn", "certifi", "flask", "imageio_ffmpeg")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv"}


def dependency_status() -> list[dict]:
    """Actually import dependencies, catching broken binary installations too."""
    checks = []
    for name in DEPENDENCIES:
        try:
            module = importlib.import_module(name)
            detail = version("Flask") if name == "flask" else str(getattr(module, "__version__", "installed"))
            checks.append({"name": name, "status": "ready", "detail": detail, "action": None})
        except Exception as error:
            checks.append({"name": name, "status": "missing", "detail": f"{type(error).__name__}: {error}", "action": "Run SETUP.command to install dependencies into this project's .venv."})
    return checks


def check_encoder(path: Path) -> dict:
    """Check the official filename's published SHA-256 prefix, offline."""
    if not path.is_file():
        return {"name": "official_encoder", "status": "missing", "detail": str(path), "action": "The official R3D-18 weights are required at this path. Restore the supplied cache before extracting features or predicting."}
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        return {"name": "official_encoder", "status": "invalid", "detail": str(error), "action": "Restore a readable official R3D-18 checkpoint."}
    if not digest.hexdigest().startswith("b3b3357e"):
        return {"name": "official_encoder", "status": "invalid", "detail": "The cached file does not match the official SHA-256 prefix b3b3357e.", "action": "Replace this corrupt or incorrect cache with the official r3d_18-b3b3357e.pth file."}
    return {"name": "official_encoder", "status": "ready", "detail": f"Official SHA-256 prefix verified: {path}", "action": None}


def check_model(path: Path, *, inspect_contents: bool = True) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        return {"name": "trained_model", "status": "missing", "detail": str(path), "action": "Restore the supplied artifacts/model folder, or run scripts/run_pipeline.py after providing labelled videos."}
    if inspect_contents and "torch" in sys.modules:
        torch = sys.modules["torch"]
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            required = {"state_dict", "input_dim", "hidden_dim", "dropout", "topk_fraction", "threshold", "feature_config", "feature_mean", "feature_scale"}
            if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
                raise ValueError("checkpoint is missing required video-model fields")
            if checkpoint["input_dim"] != 512 or not isinstance(checkpoint["state_dict"], dict):
                raise ValueError("checkpoint has an incompatible feature dimension or state dictionary")
            for key in ("feature_mean", "feature_scale"):
                value = torch.as_tensor(checkpoint[key])
                if tuple(value.shape) != (512,) or not torch.isfinite(value).all():
                    raise ValueError(f"{key} must contain 512 finite values")
            if not (torch.as_tensor(checkpoint["feature_scale"]) > 0).all():
                raise ValueError("feature_scale contains a non-positive value")
            if not 0 <= checkpoint["threshold"] <= 1 or not 0 < checkpoint["topk_fraction"] <= 1:
                raise ValueError("invalid threshold or top-k fraction")
            config = checkpoint["feature_config"]
            if config.get("backbone") != "r3d_18" or config.get("weights") != "KINETICS400_V1":
                raise ValueError("unsupported visual encoder configuration")
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from shoplifting.mil import head_from_checkpoint
            head_from_checkpoint(checkpoint)
        except Exception as error:
            return {"name": "trained_model", "status": "invalid", "detail": f"{type(error).__name__}: {error}", "action": "Restore the supplied model or train a new candidate; do not replace the working model with this file."}
    detail = f"Checkpoint fields verified: {path}" if inspect_contents and "torch" in sys.modules else f"File present; checkpoint contents were not checked: {path}"
    return {"name": "trained_model", "status": "ready", "detail": detail, "action": None}


def inspect_data(data_dir: Path) -> dict:
    counts = {"normal": 0, "positive": 0, "unlabelled": 0}
    if data_dir.is_dir():
        for path in data_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS or path.name.startswith("."):
                continue
            label = None
            for component in reversed(path.relative_to(data_dir).parts):
                text = component.lower()
                if re.search(r"(?:^|[^a-z])normal(?:$|[^a-z])|non[ _-]?(?:shoplift|theft|steal)|no[ _-]?theft", text):
                    label = "normal"
                    break
                if re.search(r"shoplift|stealing|theft|abnormal", text):
                    label = "positive"
                    break
            counts[label or "unlabelled"] += 1
    return {"directory": str(data_dir), "counts": counts, "labelled_videos_available": counts["normal"] > 0 and counts["positive"] > 0}


def inspect_project(project_root: Path | None = None, *, model_check_only: bool = False,
                    check_dependencies: bool = True, dependencies_only: bool = False) -> dict:
    """Return JSON-safe readiness information; this function performs no writes."""
    root = Path(project_root or PROJECT_ROOT).resolve()
    expected_prefix = root / ".venv"
    environment_ok = Path(sys.prefix).resolve() == expected_prefix.resolve()
    environment = {"name": "project_environment", "status": "ready" if environment_ok else "missing",
                   "detail": f"Python: {sys.executable}; environment: {sys.prefix}",
                   "action": None if environment_ok else f"Run {root / 'SETUP.command'}, or use {expected_prefix / 'bin/python'}."}
    dependencies = dependency_status() if check_dependencies else []
    runtime_ready = environment_ok and all(item["status"] == "ready" for item in dependencies)
    checks = [environment, *dependencies]
    result = {"project_root": str(root), "python": sys.executable, "ready": runtime_ready,
              "inference_ready": False, "training_ready": False, "checks": checks,
              "problems": [f"{check['name']}: {check['detail']} {check.get('action') or ''}" for check in checks if check["status"] in {"missing", "invalid"}]}
    if dependencies_only:
        return result
    model = check_model(root / "artifacts/model/model.pt", inspect_contents=check_dependencies)
    encoder = check_encoder(root / "artifacts/torch/checkpoints/r3d_18-b3b3357e.pth")
    checks.extend([model, encoder])
    result["problems"].extend(f"{check['name']}: {check['detail']} {check.get('action') or ''}" for check in (model, encoder) if check["status"] in {"missing", "invalid"})
    result["inference_ready"] = runtime_ready and model["status"] == "ready" and encoder["status"] == "ready"
    result["ready"] = result["inference_ready"]
    if not model_check_only:
        config = active_dataset(root)
        data = inspect_data(root / config["clips_dir"] if config else root / "data/raw/mnnit")
        if config:
            data["dataset_name"] = config["name"]
            data["manifest"] = str(root / config["manifest"])
            data["manifest_present"] = (root / config["manifest"]).is_file()
            data["source_video_count"] = sum(1 for _ in (root / config["source_dir"]).glob("*.mp4"))
        archive = root / "data/raw/Dataset.zip"
        data["archive"] = str(archive)
        data["archive_present"] = archive.is_file()
        result["data"] = data
        available = data["labelled_videos_available"]
        checks.append({"name": "training_data", "status": "ready" if available else "optional",
                       "detail": f"Dataset: {data.get('dataset_name', 'legacy dataset')}; normal videos: {data['counts']['normal']}; positive-labelled videos: {data['counts']['positive']}.",
                       "action": None if available else "Prediction can use the supplied model without training data. To train later, provide Normal/ and Shoplifting/ video folders or a local Dataset.zip, then run scripts/run_pipeline.py."})
        result["training_ready"] = runtime_ready and encoder["status"] == "ready" and available
    return result


def print_status(result: dict) -> None:
    print(f"Project: {result['project_root']}")
    for check in result["checks"]:
        print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        if check.get("action"):
            print(f"  Next step: {check['action']}")
    if result["inference_ready"]:
        print("Ready to run video prediction with the local trained model.")
    elif result["ready"]:
        print("The project Python environment and dependencies are ready.")
    else:
        print("Resolve the missing or invalid items listed above before running this operation.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable status")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--model-check-only", action="store_true", help="Check inference requirements; training data is not required")
    mode.add_argument("--dependencies-only", action="store_true", help="Check only the local Python environment and installed imports")
    parser.add_argument("--skip-dependencies", action="store_true", help="Skip package imports; model contents will not be checked")
    args = parser.parse_args(argv)
    result = inspect_project(model_check_only=args.model_check_only, check_dependencies=not args.skip_dependencies, dependencies_only=args.dependencies_only)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print_status(result)
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
