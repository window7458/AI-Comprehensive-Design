"""Frozen vision-foundation-model teachers.

Every teacher takes the Ultralytics training tensor as-is (RGB, float, 0..1, BxCxHxW) and
returns a dense patch-feature map (B x C_t x h x w, float32). Normalisation, resizing to a
multiple of the patch size and removal of CLS/register tokens are handled here, so the
student side never needs to know which teacher it is talking to.

Teachers are kept in a process-wide registry instead of being attributes of the YOLO model:
Ultralytics deep-copies the model for EMA and pickles it into every checkpoint, and we do not
want a 100M-400M parameter ViT copied or saved with the student.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> (kind, source id, patch size, normalisation)
TEACHER_SPECS = {
    "siglip2_b": ("siglip", "google/siglip2-base-patch16-512", 16, "half"),
    "dinov3_b": ("dinov3", "facebook/dinov3-vitb16-pretrain-lvd1689m", 16, "imagenet"),
    "dinov2_b": ("dinov2", "facebook/dinov2-base", 14, "imagenet"),
    "radio_v2.5_b": ("radio", "radio_v2.5-b", 16, None),
    "cradio_v3_b": ("radio", "c-radio_v3-b|nvidia/C-RADIOv3-B", 16, None),
    "cradio_v4_so400m": ("radio", "c-radio_v4-so400m|nvidia/C-RADIOv4-SO400M", 16, None),
    "dummy": ("dummy", "", 16, None),  # tiny random CNN, only for CPU smoke tests
}

_REGISTRY: dict[str, "Teacher"] = {}


class Teacher(nn.Module):
    """Frozen backbone wrapper that maps 0..1 RGB images to a dense feature map."""

    def __init__(self, name: str, res: int = 512):
        super().__init__()
        if name not in TEACHER_SPECS:
            raise KeyError(f"unknown teacher '{name}', choose from {sorted(TEACHER_SPECS)}")
        self.name = name
        self.kind, source, self.patch, norm = TEACHER_SPECS[name]
        self.res = res
        self.num_prefix = 0
        self.backbone = self._load(source)
        mean, std = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD), "half": ((0.5,) * 3, (0.5,) * 3)}.get(
            norm, ((0.0,) * 3, (1.0,) * 3)
        )
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.eval().requires_grad_(False)
        with torch.no_grad():
            self.dim = self.forward(torch.rand(1, 3, 64, 64)).shape[1]

    def _load(self, source: str) -> nn.Module:
        token = os.environ.get("HF_TOKEN")
        if self.kind == "siglip":
            from transformers import SiglipVisionModel

            return SiglipVisionModel.from_pretrained(source, token=token)
        if self.kind == "dinov3":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(source, token=token)
            self.num_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))
            return model
        if self.kind == "dinov2":
            from transformers import AutoModel

            self.num_prefix = 1
            return AutoModel.from_pretrained(source, token=token)
        if self.kind == "radio":
            return _load_radio(source)
        # dummy: stride-16 random CNN
        return nn.Sequential(nn.Conv2d(3, 32, 16, 16), nn.GELU(), nn.Conv2d(32, 64, 1))

    def train(self, mode: bool = True):
        return super().train(False)  # always frozen / eval

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        s = self.res / max(h, w)
        th = max(self.patch, round(h * s / self.patch) * self.patch)
        tw = max(self.patch, round(w * s / self.patch) * self.patch)
        return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False, antialias=True)

    @torch.no_grad()
    def forward(self, img01: torch.Tensor) -> torch.Tensor:
        x = self._resize(img01.float())
        gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        x = (x - self.mean) / self.std
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if x.is_cuda else None
        with torch.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=amp_dtype is not None):
            if self.kind == "siglip":
                tok = self.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            elif self.kind in {"dinov3", "dinov2"}:
                tok = self.backbone(pixel_values=x).last_hidden_state[:, self.num_prefix:]
            elif self.kind == "radio":
                out = self.backbone(x)
                tok = out[1] if isinstance(out, (tuple, list)) else out.features
            else:
                tok = self.backbone(x)
        tok = tok.float()
        if tok.dim() == 4:  # already NCHW
            return tok
        b, n, c = tok.shape
        if n != gh * gw:  # defensive: keep the last gh*gw tokens (patch tokens come after prefix tokens)
            tok = tok[:, -gh * gw:]
        return tok.transpose(1, 2).reshape(b, c, gh, gw)


def _load_radio(source: str) -> nn.Module:
    """Load RADIO / C-RADIO. 'hub_version|hf_id': torch.hub first, Hugging Face as fallback."""
    hub_version, _, hf_id = source.partition("|")
    errors = []
    try:
        model = torch.hub.load("NVlabs/RADIO", "radio_model", version=hub_version, progress=True,
                               skip_validation=True, trust_repo=True)
        return model
    except Exception as e:  # noqa: BLE001
        errors.append(f"torch.hub({hub_version}): {e}")
    if hf_id:
        try:
            from transformers import AutoModel

            return AutoModel.from_pretrained(hf_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN"))
        except Exception as e:  # noqa: BLE001
            errors.append(f"hf({hf_id}): {e}")
    raise RuntimeError("could not load RADIO teacher:\n  " + "\n  ".join(errors))


def get_teacher(name: str, res: int = 512, device: torch.device | str | None = None) -> Teacher:
    """Return the (cached) frozen teacher, optionally moved to `device`."""
    key = f"{name}@{res}"
    if key not in _REGISTRY:
        _REGISTRY[key] = Teacher(name, res)
    t = _REGISTRY[key]
    if device is not None and t.mean.device != torch.device(device):
        t.to(device)
    return t

