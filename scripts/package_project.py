"""Package the finished Final_AI source, models and active real-retail data.

Only allowlisted project files are included. Environments, IDE workspace data,
old exports and inactive staged datasets are never traversed or published.
This command makes a local ZIP; it does not upload or publish anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import zipfile

ARCHIVE_ROOT = "Final_AI"
DEFAULT_OUTPUT = Path("exports/Final_AI_Complete.zip")
REQUIRED_FILES = (
    "README.md", "requirements.txt", "requirements-lock.txt", ".gitignore",
    "SETUP.command", "BUILD_MODEL.command", "detect.py",
    "START_UI.command", "run_ui.py", ".run/Retail_Review_UI.run.xml",
    "ui/index.html", "ui/style.css", "ui/app.js", "shoplifting/web_app.py", "shoplifting/review_suggestions.py",
    "shoplifting/__init__.py", "shoplifting/predict.py",
    "scripts/package_project.py", "scripts/download_ucf_crime.py", "scripts/prepare_ucf_crime.py",
    "artifacts/model/model.pt", "artifacts/model/mean_logistic.pkl", "artifacts/model/metrics.json",
    "artifacts/model/provenance.json", "artifacts/model/split_manifest.json", "artifacts/model/test_predictions.csv",
    "artifacts/model/mil_model.pt", "artifacts/model/mean_logistic_model.pt",
    "artifacts/torch/checkpoints/r3d_18-b3b3357e.pth", "data/active_dataset.json",
    "report_support/README.md", "report_support/Writing_Prompts.md", "report_support/Report_Outline.docx",
    "report_support/latex/main.tex", "report_support/latex/README.md", "report_support/latex/COMPXXXX.cls",
    "report_support/latex/COMPXXXX.bst", "report_support/latex/references_AI.bib",
)
IGNORED_DIRECTORIES = {".git", ".idea", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "downloads", "exports", "tmp", "scratch", "qa", "renders", "rendered", "previews", "preview", "intermediates"}
STORED_SUFFIXES = {".zip", ".npz", ".mp4", ".pt", ".pth", ".pkl", ".docx"}
LEGACY_ENTRY_FILES = frozenset({
    "START_HERE.command", "RUN_DEMO.command", "run_demo.py",
    "shoplifting/launcher.py", "tests/test_launcher.py",
    ".run/Shoplifting_Demo.run.xml", ".run/Choose_Video.run.xml",
})


def _safe_path(root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Package input escapes the project: {path}") from error
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"Refusing to package a symlink: {current}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Package input escapes the project: {path}")
    return path


def _relative_input(root: Path, value, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Active dataset needs a nonempty {field} path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe active-dataset {field}: {value}")
    return _safe_path(root, root / relative)


def _walk_selected(root: Path, relative, suffixes: set[str]):
    directory = _safe_path(root, root / relative)
    if not directory.is_dir():
        return
    for current, directories, filenames in os.walk(directory, followlinks=False):
        kept = []
        for name in sorted(directories):
            if name.startswith(".") or name in IGNORED_DIRECTORIES:
                continue
            _safe_path(root, Path(current) / name)
            kept.append(name)
        directories[:] = kept
        for name in sorted(filenames):
            if name.startswith(".") or Path(name).suffix.lower() not in suffixes:
                continue
            path = _safe_path(root, Path(current) / name)
            if path.is_file():
                yield path


def _active_inputs(root: Path) -> dict[str, Path]:
    ledger_path = root / "data/active_dataset.json"
    if not ledger_path.is_file():
        raise ValueError("Active dataset is missing. Finish preparing the real-retail data first.")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError("Cannot read data/active_dataset.json.") from error
    if ledger.get("kind") != "ucf_crime":
        raise ValueError("The finished package must use the active UCF-Crime real-retail subset.")
    fields = ("manifest", "provenance", "source_dir", "clips_dir", "annotations", "feature_dir")
    paths = {field: _relative_input(root, ledger.get(field), field) for field in fields}
    data_base = root / "data/real_retail"
    for field in fields[:-1]:
        if not paths[field].is_relative_to(data_base):
            raise ValueError(f"Active dataset {field} must be inside data/real_retail.")
    if not paths["feature_dir"].is_relative_to(root / "artifacts"):
        raise ValueError("Active feature cache must be inside artifacts/.")
    for field in ("manifest", "provenance", "annotations"):
        if not paths[field].is_file() or paths[field].stat().st_size == 0:
            raise ValueError(f"Active dataset file is missing or empty: {paths[field]}")
    for field in ("source_dir", "clips_dir", "feature_dir"):
        if not paths[field].is_dir():
            raise ValueError(f"Active dataset directory is missing: {paths[field]}")
    return paths


def _validate_feature_index(root: Path, directory: Path, files: set[Path]) -> None:
    path = directory / "index.json"
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError("Feature index is missing or unreadable; finish extraction first.") from error
    records = index.get("records")
    if index.get("complete") is False or not isinstance(records, list) or not records:
        raise ValueError("Feature cache is incomplete; finish extraction first.")
    expected = index.get("expected_video_count")
    if expected is not None and expected != len(records):
        raise ValueError("Feature index record count is incomplete.")
    for record in records:
        filename = record.get("feature_file") if isinstance(record, dict) else None
        if not isinstance(filename, str) or not filename:
            raise ValueError("Each feature record needs a feature_file path.")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npz":
            raise ValueError(f"Unsafe feature_file path: {filename}")
        selected = _safe_path(root, directory / relative)
        if selected not in files or selected.stat().st_size == 0:
            raise ValueError(f"Feature cache file is missing or empty: {filename}")


def _validate_manifest(root: Path, manifest: Path, clips: Path, files: set[Path]) -> None:
    with manifest.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records or {row.get("label") for row in records} != {"0", "1"}:
        raise ValueError("The active manifest must contain both normal and event-labelled clips.")
    ids = [row.get("id") for row in records]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError("The active manifest contains missing or duplicate IDs.")
    for row in records:
        if not row.get("group", "").startswith("ucf:"):
            raise ValueError("Active clips must preserve their UCF source-video groups.")
        value = Path(row.get("path", ""))
        if value.is_absolute():
            if value.is_relative_to(root):
                selected = _safe_path(root, value)
            else:
                # Archives retain the original auditable manifest. After a
                # move, use its class and filename to locate the local copy;
                # never follow a path back to another machine/project.
                category = "Normal" if row["label"] == "0" else "Shoplifting"
                selected = _safe_path(root, clips / category / value.name)
        else:
            candidates = [root / value, manifest.parent / value]
            selected = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            selected = _safe_path(root, selected)
        if not selected.is_relative_to(clips) or selected not in files or selected.stat().st_size == 0:
            raise ValueError(f"A manifest clip is missing or outside the active clips directory: {selected}")


def collect_files(root: str | Path) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Project directory not found: {root}")
    files: set[Path] = set()
    for directory in ("shoplifting", "scripts", "tests"):
        suffixes = {".py", ".js"} if directory == "tests" else {".py"}
        files.update(_walk_selected(root, directory, suffixes))
    files.update(_walk_selected(root, "ui", {".html", ".css", ".js", ".svg"}))
    # The UI is the sole shared run entry. Old configurations left in copied
    # folders must not reintroduce separate demo/chooser flows into an export.
    run_configuration = _safe_path(root, root / ".run/Retail_Review_UI.run.xml")
    if run_configuration.is_file():
        files.add(run_configuration)
    # Portable language-compatibility inspections only; never include IDE
    # workspace state, SDK registrations, histories or other personal settings.
    for relative in (".idea/inspectionProfiles/Project_Default.xml", ".idea/inspectionProfiles/profiles_settings.xml"):
        path = _safe_path(root, root / relative)
        if path.is_file():
            files.add(path)
    for directory, suffixes in (
        ("artifacts/model", {".pt", ".pth", ".pkl", ".json", ".csv"}),
        ("artifacts/baseline-short-clips", {".pt", ".pkl", ".json", ".csv"}),
        ("artifacts/features", {".npz", ".json"}),
        ("artifacts/validation", {".json", ".txt", ".md"}),
        ("artifacts/torch/checkpoints", {".pt", ".pth"}),
        ("artifacts/predictions", {".mp4", ".json", ".csv"}),
        ("outputs", {".mp4", ".json", ".csv"}),
        ("examples", {".mp4", ".json", ".csv", ".md"}),
    ):
        selected = _walk_selected(root, directory, suffixes)
        files.update(path for path in selected if not path.is_relative_to(root / "outputs/ui"))
    for path in (root / "artifacts").glob("pipeline-*.log"):
        if path.is_file():
            files.add(_safe_path(root, path))
    # Keep editable authored sources and their figure assets, not Word/LaTeX
    # render checks, temporary pages or compiler intermediates.
    for path in _walk_selected(root, "report_support", {".docx", ".md", ".tex", ".bib", ".cls", ".bst", ".sty", ".png", ".jpg", ".jpeg", ".svg"}):
        if not path.name.lower().startswith(("page-", "page_", "render-", "preview", "thumbnail", "qa-", "overlay-check")):
            files.add(path)
    # LaTeX uses the vector PDF plots directly, so an export must retain them.
    # Include the current review copy without collecting old template/QA PDFs.
    current_reports = {
        root / "report_support/Retail_Review_AI_Draft.pdf",
        root / "report_support/Report_Preparation/output/pdf/Retail_Review_AI_Draft.pdf",
    }
    for path in _walk_selected(root, "report_support", {".pdf"}):
        relative = path.relative_to(root / "report_support")
        if "figures" in relative.parts or path in current_reports:
            files.add(path)
    for path in root.iterdir():
        if path.name == ".gitignore" or path.name.startswith(("README", "LICENSE", "NOTICE", "COPYING")) or (path.name.startswith("requirements") and path.suffix == ".txt") or path.suffix in {".command", ".py"}:
            _safe_path(root, path)
            if path.is_file():
                files.add(path)
    ledger = _safe_path(root, root / "data/active_dataset.json")
    if ledger.is_file():
        files.add(ledger)
    active = _active_inputs(root)
    files.update(active[field] for field in ("manifest", "provenance", "annotations"))
    for field in ("source_dir", "clips_dir"):
        files.update(_walk_selected(root, active[field].relative_to(root), {".mp4", ".json", ".txt"}))
    # All preparation/download provenance stays beside the active manifest.
    files.update(_walk_selected(root, active["manifest"].parent.relative_to(root), {".json", ".txt", ".csv", ".mp4"}))
    files.update(_walk_selected(root, active["provenance"].parent.relative_to(root), {".json", ".txt", ".csv", ".mp4"}))
    files.update(_walk_selected(root, active["feature_dir"].relative_to(root), {".npz", ".json"}))
    files.difference_update(root / name for name in LEGACY_ENTRY_FILES)
    missing = [name for name in REQUIRED_FILES if root / name not in files]
    if missing:
        raise ValueError("Complete package is missing required files:\n  " + "\n  ".join(missing))
    for name in ("artifacts/model/model.pt", "artifacts/torch/checkpoints/r3d_18-b3b3357e.pth"):
        if (root / name).stat().st_size == 0:
            raise ValueError(f"Required model is empty: {name}")
    for field in ("source_dir", "clips_dir"):
        if not any(path.is_relative_to(active[field]) and path.suffix == ".mp4" for path in files):
            raise ValueError(f"No real-retail videos found in active {field}.")
    if not any(path.is_relative_to(root / "examples") and path.suffix == ".mp4" for path in files):
        raise ValueError("Stable example videos are missing. Save the demonstration videos under examples/ before packaging.")
    _validate_feature_index(root, active["feature_dir"], files)
    _validate_manifest(root, active["manifest"], active["clips_dir"], files)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_package(root: str | Path, output: str | Path | None = None) -> Path:
    root = Path(root).resolve()
    files = collect_files(root)
    output = Path(output) if output is not None else DEFAULT_OUTPUT
    if not output.is_absolute():
        output = root / output
    if output.suffix.lower() != ".zip":
        raise ValueError("Package output must end in .zip.")
    if output.is_relative_to(root):
        _safe_path(root, output)
    output = output.resolve()
    if output in files:
        raise ValueError("Package output must not overwrite a project input.")
    if output.is_relative_to(root):
        relative = output.relative_to(root)
        if relative.parts[0] in {"data", "artifacts", ".venv", ".git", ".idea", "downloads"}:
            raise ValueError("Write packages in exports/, not inside data, models, environments or old exports.")
        if relative.parts[0] == "exports" and len(relative.parts) != 2:
            raise ValueError("Write directly into exports/, not an extracted project copy.")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".final-ai-package-", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True, strict_timestamps=False) as archive:
            for path in files:
                archive.write(path, f"{ARCHIVE_ROOT}/{path.relative_to(root).as_posix()}", compress_type=zipfile.ZIP_STORED if path.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(temporary) as archive:
            names = set(archive.namelist())
            if any(f"{ARCHIVE_ROOT}/{name}" not in names for name in REQUIRED_FILES):
                raise ValueError("Package verification found missing required entries.")
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"Package CRC verification failed for {corrupt}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help=f"Default: {DEFAULT_OUTPUT}")
    args = parser.parse_args(argv)
    try:
        output = build_package(Path(__file__).resolve().parents[1], args.output)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Packaging failed: {error}\n")
    print(f"Verified complete local project: {output}")
    print(f"Size: {output.stat().st_size:,} bytes")
    print("Includes source, launchers, real-retail data, trained models, features and provenance. Nothing was uploaded.")


if __name__ == "__main__":
    main()
