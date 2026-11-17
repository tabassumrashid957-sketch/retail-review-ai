"""Frozen pretrained R3D-18 action features, not an identity-recognition model."""
import os
from pathlib import Path
import certifi
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18


def configure_device(device="cpu", threads=4):
    torch.set_num_threads(max(1, threads))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu on this computer.")
    return torch.device(device)


class VideoEncoder:
    def __init__(self, device="cpu", cache_dir="artifacts/torch", threads=4):
        self.device = configure_device(device, threads)
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(Path(cache_dir).resolve()))
        self.weights = R3D_18_Weights.KINETICS400_V1
        try:
            self.model = r3d_18(weights=self.weights)
        except Exception as exc:
            raise RuntimeError("Pretrained R3D-18 weights could not be loaded. Internet is needed once to download the official PyTorch weights; random weights are never substituted.") from exc
        self.model.fc = torch.nn.Identity()
        self.model.eval().requires_grad_(False).to(self.device)
        self.transform = self.weights.transforms()

    @torch.inference_mode()
    def encode(self, clips):
        # clips: list of uint8 arrays [T,H,W,RGB]. Official transforms expect T,C,H,W.
        tensor = torch.stack([self.transform(torch.from_numpy(clip).permute(0, 3, 1, 2)) for clip in clips]).to(self.device)
        result = self.model(tensor).float().cpu().numpy()
        if result.ndim != 2 or result.shape[1] != 512 or not np.isfinite(result).all():
            raise RuntimeError("Encoder returned invalid video features.")
        return result
