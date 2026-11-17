"""Prepare local labelled videos and train a candidate without replacing the current model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_data import SHA256 as DATASET_SHA256, SIZE as DATASET_SIZE
from scripts.verify_project import check_encoder, inspect_data
from shoplifting.data_config import active_dataset, rebase_manifest


def canonical_interpreter(project_root: Path) -> str:
    interpreter = project_root / ".venv/bin/python"
    if not interpreter.is_file():
        raise ValueError(f"Project Python is missing: {interpreter}. Run SETUP.command first.")
    return str(interpreter)


def verify_local_archive(path: Path) -> None:
    """Stop before invoking downloader if it could access the network."""
    if not path.is_file() or path.stat().st_size != DATASET_SIZE:
        raise ValueError(f"Local archive is missing or incomplete: {path}. Provide the complete Dataset.zip, or explicitly use --download-data.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != DATASET_SHA256:
        raise ValueError(f"Local archive checksum is incorrect: {path}. Replace it with the verified Dataset.zip before continuing.")


def build_commands(*, project_root: Path, interpreter: str, data_dir: Path,
                   manifest: Path, features: Path, candidate: Path,
                   extract_archive: bool = False, epochs: int = 100,
                   seed: int = 42, threads: int = 4, device: str = "cpu",
                   allow_download: bool = False, manifest_ready: bool = False,
                   head_type: str = "mil", feature_config: dict | None = None,
                   hidden_dim: int = 128, frozen_transfer: Path | None = None) -> list[tuple[str, list[str]]]:
    """Build argument lists; no shell expansion or commands execute here."""
    commands = []
    if extract_archive:
        archive_command = [interpreter, str(project_root / "scripts/download_data.py"), "--output", str(project_root / "data/raw")]
        if not allow_download:
            archive_command.append("--offline")
        commands.append(("Verify and extract the public dataset archive", archive_command))
    if not manifest_ready:
        commands.append(("Prepare labelled-video manifest", [interpreter, "-m", "shoplifting.prepare", "--data", str(data_dir), "--output", str(manifest)]))
    commands.extend([
        ("Extract frozen video features", [interpreter, "-m", "shoplifting.extract", "--manifest", str(manifest), "--output", str(features), "--device", device, "--threads", str(threads), "--cache-dir", str(project_root / "artifacts/torch")]),
        ("Train and evaluate candidate model", [interpreter, "-m", "shoplifting.train", "--features", str(features), "--output", str(candidate), "--epochs", str(epochs), "--seed", str(seed), "--device", device, "--threads", str(threads)]),
    ])
    if head_type != "mil":
        commands[-1][1].extend(["--head-type", head_type])
    if hidden_dim != 128:
        commands[-1][1].extend(["--hidden-dim", str(hidden_dim)])
    if feature_config:
        for key, flag in (("window_seconds", "--window-seconds"), ("stride_seconds", "--stride-seconds"), ("max_windows", "--max-windows")):
            if key in feature_config:
                commands[-2][1].extend([flag, str(feature_config[key])])
    if frozen_transfer is not None:
        commands.append(("Compare frozen and context-trained heads using validation only", [interpreter, str(project_root / "scripts/select_context_model.py"), "--candidate", str(candidate), "--baseline", str(frozen_transfer), "--features", str(features)]))
    return commands


def run_step(title: str, command: list[str], project_root: Path, log_stream) -> None:
    print(title, flush=True)
    log_stream.write(f"\n{title}\n{json.dumps(command)}\n")
    log_stream.flush()
    process = subprocess.Popen(command, cwd=str(project_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_stream.write(line)
        log_stream.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def promote_candidate(candidate: Path, current: Path) -> Path | None:
    """Rotate a backup, then rename the successful candidate, rolling back errors."""
    if not (candidate / "model.pt").is_file():
        raise ValueError("Candidate has no model.pt; promotion stopped.")
    backup = None
    if current.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup = current.with_name(f"model-backup-{stamp}-{uuid.uuid4().hex[:6]}")
        current.rename(backup)
    try:
        candidate.rename(current)
    except OSError:
        if backup is not None:
            backup.rename(current)
        raise
    return backup


def run_pipeline(*, project_root: Path = PROJECT_ROOT, data_dir: Path | None = None,
                 output: Path | None = None, features: Path | None = None,
                 download_data: bool = False, promote: bool = False,
                 epochs: int = 100, seed: int = 42, threads: int = 4,
                 device: str = "cpu", log_path: Path | None = None,
                 manifest_path: Path | None = None) -> dict:
    root = Path(project_root).resolve()
    if epochs < 1 or threads < 1:
        raise ValueError("epochs and threads must be positive")
    if data_dir is not None and download_data:
        raise ValueError("Use either existing videos with --data, or --download-data for the public dataset.")
    interpreter = canonical_interpreter(root)
    verification = subprocess.run([interpreter, str(root / "scripts/verify_project.py"), "--dependencies-only", "--json"],
                                  cwd=str(root), text=True, capture_output=True)
    if verification.returncode:
        raise ValueError("Project dependencies are not ready. Run SETUP.command first.\n" + verification.stdout + verification.stderr)
    encoder = check_encoder(root / "artifacts/torch/checkpoints/r3d_18-b3b3357e.pth")
    if encoder["status"] != "ready":
        raise ValueError(f"{encoder['detail']} {encoder['action']}")
    config = active_dataset(root)
    real_mode = bool(config and config["kind"] == "ucf_crime" and data_dir is None)
    if real_mode:
        manifest_path = Path(manifest_path).resolve() if manifest_path else root / config["manifest"]
        if download_data:
            subprocess.run([interpreter, str(root / "scripts/download_ucf_crime.py")], cwd=root, check=True)
        if not manifest_path.is_file():
            if not any((root / config["source_dir"]).glob("*.mp4")):
                raise ValueError("The real-retail source videos are missing. Restore the complete project or explicitly use --download-data.")
            if config.get("preparation") == "whole_event_context":
                subprocess.run([interpreter, str(root / "scripts/prepare_ucf_context.py"), "--videos", str(root / config["source_dir"]), "--annotations", str(root / config["annotations"]), "--output", str(manifest_path.parent), "--splits", str(root / config["split_reference"])], cwd=root, check=True)
            else:
                subprocess.run([interpreter, str(root / "scripts/prepare_ucf_crime.py"), "--videos", str(root / config["source_dir"]), "--annotations", str(root / config["annotations"]), "--output", str(root / config["clips_dir"]), "--manifest", str(manifest_path)], cwd=root, check=True)
        rebase_manifest(manifest_path, root)
    elif manifest_path is not None:
        manifest_path = Path(manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise ValueError(f"The supplied manifest does not exist: {manifest_path}")
    data = Path(data_dir).expanduser().resolve() if data_dir is not None else root / config["clips_dir"] if real_mode else root / "data/raw/mnnit"
    available = inspect_data(data)["labelled_videos_available"]
    needs_archive = download_data and not real_mode
    if not available and not download_data and manifest_path is None:
        if data_dir is not None:
            raise ValueError(f"No usable pair of Normal/ and Shoplifting/ video folders in {data}. No download was attempted.")
        verify_local_archive(root / "data/raw/Dataset.zip")
        needs_archive = True
    target = Path(output).expanduser().resolve() if output is not None else root / "artifacts/model-new"
    current = root / "artifacts/model"
    if target == current.resolve() or target in current.resolve().parents or current.resolve() in target.parents:
        raise ValueError("Candidate output must be separate from artifacts/model. Use --promote only after candidate training succeeds.")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"Candidate output already exists and is not empty: {target}. Choose a new --output directory.")
    if promote and target.parent != current.parent:
        raise ValueError("For atomic promotion, the candidate --output must be directly inside artifacts/.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".training-candidate-{uuid.uuid4().hex}"
    manifest = manifest_path if manifest_path is not None else root / "data/manifest.csv"
    feature_dir = Path(features).expanduser().resolve() if features is not None else root / (config["feature_dir"] if real_mode else "artifacts/features")
    commands = build_commands(project_root=root, interpreter=interpreter, data_dir=data,
                              manifest=manifest, features=feature_dir, candidate=staging,
                              extract_archive=needs_archive, epochs=epochs, seed=seed,
                              threads=threads, device=device, allow_download=download_data,
                              manifest_ready=manifest_path is not None,
                              head_type=config.get("neural_head_type", "mil") if real_mode else "mil",
                              feature_config=config.get("feature_config") if real_mode else None,
                              hidden_dim=config.get("neural_hidden_dim", 128) if real_mode else 128,
                              frozen_transfer=root / config["frozen_transfer_checkpoint"] if real_mode and config.get("frozen_transfer_checkpoint") else None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log = Path(log_path).expanduser().resolve() if log_path else root / f"artifacts/pipeline-{stamp}-{uuid.uuid4().hex[:6]}.log"
    if log == current.resolve() or current.resolve() in log.parents or log == target or target in log.parents:
        raise ValueError("Log path must be outside the current and candidate model directories.")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pipeline log: {log}", flush=True)
    print(f"Candidate output: {target}; current model is retained until successful explicit promotion.", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"Project: {root}\nInterpreter: {interpreter}\n")
        try:
            for number, (title, command) in enumerate(commands, 1):
                run_step(f"[{number}/{len(commands)}] {title}", command, root, stream)
            required = ("model.pt", "metrics.json", "split_manifest.json", "test_predictions.csv", "mean_logistic.pkl")
            if any(not (staging / name).is_file() or (staging / name).stat().st_size == 0 for name in required):
                raise ValueError("Training finished without all required candidate artifacts. The current model was retained.")
            evidence = {}
            for path in [manifest, feature_dir / "index.json", *(staging / name for name in required)]:
                if path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                    evidence[path.name] = digest.hexdigest()
            (staging / "provenance.json").write_text(json.dumps({
                "schema_version": 1,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "seed": seed,
                "artifact_sha256": evidence,
                "active_dataset": config,
                "training_log": str(log),
                "selection": "Checkpoint, threshold and deployment selected using validation only; see metrics.json.",
                "assistance": "Implementation developed with Codex assistance."
            }, indent=2) + "\n")
            if target.exists():
                target.rmdir()  # Only an empty, preflight-checked directory is removed.
            staging.rename(target)
            backup = promote_candidate(target, current) if promote else None
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            stream.write(f"\nFAILED: {error}\nStaging directory: {staging}\n")
            raise
    final = current if promote else target
    print(f"Completed. Model and evaluation artifacts: {final}", flush=True)
    if backup:
        print(f"Previous model backup: {backup}", flush=True)
    return {"model_directory": str(final), "backup": str(backup) if backup else None, "log": str(log)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Existing labelled-video root containing Normal/ and Shoplifting/ folders")
    parser.add_argument("--manifest", type=Path, dest="manifest_path", help="Use an audited manifest while preserving original source-video groups")
    parser.add_argument("--download-data", action="store_true", help="Explicitly allow downloading the public dataset if its verified local archive is unavailable")
    parser.add_argument("--output", type=Path, help="New, empty candidate directory; default: artifacts/model-new")
    parser.add_argument("--features", type=Path, help="Feature cache directory; default: artifacts/features")
    parser.add_argument("--promote", action="store_true", help="After all steps succeed, promote the candidate and retain a backup of artifacts/model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log", type=Path, dest="log_path")
    args = parser.parse_args(argv)
    try:
        run_pipeline(data_dir=args.data, output=args.output, features=args.features,
                     download_data=args.download_data, promote=args.promote, epochs=args.epochs,
                     seed=args.seed, threads=args.threads, device=args.device, log_path=args.log_path, manifest_path=args.manifest_path)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
