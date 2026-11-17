# Scope of this repository snapshot

This staging copy was prepared on 17 November 2026 from the working local
application. Preparing it did not alter that application or its saved reviews.

Included: application source, tests, dependency specifications, one UI run
configuration and launch/setup scripts; current small model heads and prior
baseline heads; decision policy, metrics and numeric predictions; exact source
and split manifests, annotation intervals, checksums and reconstruction scripts.

Excluded: virtual environments, large pretrained encoder, source/derived/example
video, private uploads and human-review records, generated annotated videos,
exports, screenshots, IDE state, maintenance backups and report drafts/support.

Copied JSON/CSV/text metadata has user-specific absolute project paths replaced
with project-relative paths. Labels, source groups, frame bounds, split
assignments, scientific metrics and checkpoint tensor values were not changed.
The active data ledger is marked `metadata_only_media_not_distributed` because
the footage must be restored separately. Training-head checkpoint files remain
byte-for-byte identical to the canonical snapshots.

`distribution_manifest.json` records original and distributed hashes for copied
files. Historical `provenance.json` hashes and test counts describe their original
experiment snapshots, before path normalisation and later UI cleanup. They are
not a claim that all current distributed text files have those historical byte
hashes. The model-head hashes remain valid. The current source inventory and
distribution checks are recorded separately.

The staging adaptation adds `scripts/download_encoder.py` and invokes it from
`SETUP.command`, because the original privately packaged project supplied its
large encoder cache. It downloads only the official Torchvision weights, checks
the published hash prefix, and leaves the trained head unchanged. The README
documents the download and the explicit context-clip preparation step.

Two existing software tests were adapted to this clean distribution: interpreter
recovery uses a temporary interpreter fixture instead of requiring the author's
real virtual environment, and the UI test expects an empty example list while
third-party videos are omitted. No model or review decision logic was changed.

No third-party footage is uploaded by these preparation steps. Publishing or
pushing this staging repository is a separate action; no remote operation was
performed by its preparation agent.
