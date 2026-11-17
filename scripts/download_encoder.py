"""Fetch the official pretrained R3D-18 weights into the ignored local cache.

No dataset or private video is downloaded. The trained review head included in
this repository is not changed. The official checkpoint is about 128 MiB.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import certifi
    import torch
    from torchvision.models.video import R3D_18_Weights
    from scripts.verify_project import check_encoder

    checkpoint = ROOT / "artifacts/torch/checkpoints/r3d_18-b3b3357e.pth"
    status = check_encoder(checkpoint)
    if status["status"] == "ready":
        print("Official R3D-18 cache is already verified.")
        return 0
    if checkpoint.exists():
        raise ValueError("Existing encoder cache failed verification. Remove or repair that file before retrying.")
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    (ROOT / "artifacts/torch").mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(ROOT / "artifacts/torch"))
    print("Downloading official Torchvision KINETICS400_V1 R3D-18 weights (about 128 MiB).")
    R3D_18_Weights.KINETICS400_V1.get_state_dict(progress=True, check_hash=True)
    status = check_encoder(checkpoint)
    if status["status"] != "ready":
        raise ValueError("The downloaded official encoder failed the published SHA-256-prefix check.")
    print("Official encoder verified. The repository's trained head was retained.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Encoder setup stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
