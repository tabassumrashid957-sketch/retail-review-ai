# Retail Review AI

A local research prototype that scores recorded retail video, suggests approximate
time ranges to inspect, and records a human review separately from the model.
It does not establish theft, payment, identity, or a complete shelf-to-pocket
action sequence.

## Install and run

Python 3.10 or newer is required. The recorded environment used Python 3.10;
`requirements-lock.txt` records its package versions. Dependency availability
and floating-point results can vary by operating system and package version.

From the root of this repository, on macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_encoder.py
.venv/bin/python scripts/verify_project.py --model-check-only
.venv/bin/python run_ui.py
```

The encoder command downloads the official Torchvision R3D-18 weights, about
128 MiB, to the ignored `artifacts/torch/` cache and verifies its published
SHA-256 prefix. The small trained review head is included in the repository.
No API key is needed. After packages and weights are installed, inference runs
locally without an external inference service.

On macOS, `SETUP.command` performs dependency and encoder setup. Then
double-click `START_UI.command`, or open the repository in PyCharm, select the
project's `.venv/bin/python` interpreter, select **Retail Review UI**, and click
Run. If execution permissions were lost in a ZIP download, use
`/bin/zsh SETUP.command` and `/bin/zsh START_UI.command` from Terminal.

The browser interface opens at `http://127.0.0.1:8765`. Keep its Python process
running while using it. To use another port or suppress automatic browser opening:

```sh
.venv/bin/python run_ui.py --port 8766 --no-open
```

Upload a video that you are authorised to process, click **Analyse clip**, watch
the proposed ranges and full footage, and save **Possible shoplifting visible**,
**No visible shoplifting** or **Unclear** with optional notes. The downloadable
JSON keeps the human assessment separate from the model output. Uploads and
reviews are stored in ignored `outputs/ui/`; do not add them to Git.

Example videos are not distributed here. Example buttons appear only after the
relevant research clips have been reconstructed locally. Uploading your own
authorised clip does not require the training dataset.

## Reconstruct the exact research data

This repository provides manifests, frame bounds, original source assignments,
download checksums and reconstruction code. It excludes raw and derived video.

The experiment uses 21 temporally annotated original UCF-Crime Shoplifting
recordings from the original test partition. The current context dataset
contains 46 derived clips, 23 positive and 23 non-event intervals, from 19 source
recordings. The clip split is 20 train, 14 validation and 12 test, over 8, 5 and 6
source groups. This is a custom exploratory resplit, not an official UCF benchmark
or a pristine new holdout: the follow-up reused previously evaluated sources.

After installing dependencies and the encoder, run:

```sh
.venv/bin/python scripts/download_ucf_crime.py
.venv/bin/python scripts/prepare_ucf_context.py --splits data/real_retail/ucf_crime/context/source_splits.json
```

The downloader requests only the specified members of the official archive,
about 612 MB after decompression. The full remote archive is about 103 GB; the
script refuses a server response that ignores the requested byte range. Source
files are checked by size, ZIP CRC32 and SHA-256. Availability of the authors'
server is outside this project's control.

The explicit preparation step is required because the repository already
contains the audited manifest, while its media files are omitted. Preparation
retains complete recorded incident intervals plus up to two seconds of context.
Negatives are duration-matched intervals outside annotations in the same source
recording, with a one-second guard. Two sources have no suitable equal-duration
negative interval and are excluded. Negative labels are not independent proof
that no theft occurred.

Original group assignments are preserved in `context/source_splits.json`.
Frame bounds, source hashes and clip identifiers are in `context/metadata.json`
and `context/manifest.csv`. Paths are project-relative in this distribution;
preparation and inference resolve them to the local checkout. Re-encoding on
another platform may change MP4 byte hashes even when the same source frames
are selected; retain source hashes, frame intervals and split assignments when
checking scientific equivalence.

The older short-clip preparation is retained for historical reconstruction:

```sh
.venv/bin/python scripts/prepare_ucf_crime.py
```

This optional command reconstructs the earlier 104-clip protocol. Its metrics
are separate from the current 46-clip context experiment.

## Reproduce training and evaluate a candidate

After reconstructing the context clips:

```sh
.venv/bin/python scripts/run_pipeline.py --output artifacts/model-new
```

Use a new empty output directory for each run. This extracts all two-second
R3D-18 windows at a one-second stride, trains the context GRU/logistic candidates,
and compares them with the preserved earlier MIL head using validation data.
The repository includes the frozen baseline head and original split metadata
needed for that comparison. A candidate does not replace `artifacts/model/`
unless `--promote` is explicitly passed. `BUILD_MODEL.command` requests promotion
and retains a backup; use it only when you intend to replace the local model.

The frozen R3D-18 encoder produces 512-dimensional features from 16 RGB frames
per window. The deployed head is a transferred MIL MLP with a 128-unit hidden
layer and top-25% logit pooling. The 64-unit bidirectional GRU is an experimental
comparison, recorded under `models.mil` in the metrics. The separate
`selected_model=frozen_transfer` identifies deployment. The learned heads and
application were developed with Codex assistance; these are adaptations, not a
faithful reproduction of the cited paper's loss or official evaluation protocol.

Saved experiment results are in `artifacts/model/metrics.json`, with validation
and test predictions. The context experiment selected deployment by validation
average precision, then validation F1. Its ordinary binary test result at
threshold 0.5 is 50% accuracy on 12 clips, AP 0.7042, ROC AUC 0.6944, F1 0.4000,
and confusion matrix `[[4,2],[4,2]]` (rows true normal/positive, columns predicted
normal/positive).

The separate deployed selective policy uses positive bound 0.6415769 and no
supported negative bound. Its saved fresh-inference audit flags one positive
test clip and returns inconclusive on the other 11: coverage 1/12 and positive
review recall 1/6. One correct flag does not establish reliable detection.

The eight-second context rule and technical visibility checks are engineering
guards, not action-completeness or pocket-placement detectors. Review timestamps
are approximate navigation aids. Their boundaries and reviewer effectiveness
have not been quantitatively evaluated. No face identification, payment
verification, live-camera evaluation or automatic retraining from human reviews
is implemented.

## Software checks

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_project.py --model-check-only
```

The Python suite contains 110 software tests. Recognition accuracy is measured
separately. Run the tests in your environment. Training data is reported as
absent until it is reconstructed. Historical provenance retains earlier test
counts and hashes; see `DISTRIBUTION_NOTES.md` for their scope.

The browser review-save tests use Node.js, which is optional for running the
application:

```sh
node tests/test_review_save_race.js
```

## Attribution, rights and assistance

- UCF-Crime: Waqas Sultani, Chen Chen and Mubarak Shah, *Real-World Anomaly
  Detection in Surveillance Videos*, CVPR 2018.
  [Official project](https://www.crcv.ucf.edu/projects/real-world/) and
  [original annotations](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018/blob/master/Temporal_Anomaly_Annotation.txt).
- The exact archive-member catalogue and source hashes used by this project are
  recorded in `data/real_retail/ucf_crime/sources.json`.
- Frozen encoder: Torchvision R3D-18 with KINETICS400_V1 pretrained weights.
  [Official model/transform documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html).
  The associated reference is Du Tran and colleagues, *A Closer Look at
  Spatiotemporal Convolutions for Action Recognition*, CVPR 2018
  ([paper](https://arxiv.org/abs/1711.11248)).
- Code development, debugging, data preparation, experiments and documentation
  received extensive OpenAI Codex assistance. This repository is not evidence of
  unaided authorship. Follow applicable coursework rules and disclose assistance
  accurately if using it for assessment.

Public download access does not establish permission to redistribute source
footage. No explicit dataset redistribution licence was established by this
project, so raw/derived videos are omitted. The authors describe the dataset as
real-world surveillance; incidents, payment/intent, store locations and worldwide
representation were not independently verified. Third-party data, pretrained
weights and dependencies retain their respective rights and terms. No blanket
MIT or Creative Commons licence is asserted for them or for this repository.
