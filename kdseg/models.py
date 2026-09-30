"""YOLO11-seg students with (a) feature distillation from a frozen teacher and (b) teacher feature fusion.

* DistillSegModel  (track B): teacher is used only in the training loss. The deployable model is a
  plain YOLO11s-seg; `export_student()` strips the distillation adapters.
* FusionSegModel   (track A): the frozen teacher runs at inference too and its features are added
  to chosen YOLO layers through zero-initialised projections (upper-bound reference).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import RANK

from .teachers import get_teacher


def _run_layers(model: SegmentationModel, x: torch.Tensor, capture=(), inject=None):
    """Ultralytics `_predict_once` with optional feature capture and a per-layer injection hook."""
    y, feats = [], {}
    for m in model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        if inject is not None:
            x = inject(m.i, x)
        if m.i in capture:
            feats[m.i] = x
        y.append(x if m.i in model.save else None)
    return x, feats


class DistillAdapter(nn.Module):
    """Student feature (C_s) -> teacher space (C_t)."""

    def __init__(self, c_s: int, c_t: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_s, c_t, 1, bias=False), nn.BatchNorm2d(c_t), nn.GELU(), nn.Conv2d(c_t, c_t, 1)
        )

    def forward(self, x):
        return self.net(x)


def _layer_channels(model: SegmentationModel, layers, imgsz: int = 256) -> dict[int, int]:
    p = next(model.parameters())
    with torch.no_grad():
        _, feats = _run_layers(model, torch.zeros(1, 3, imgsz, imgsz, device=p.device, dtype=p.dtype), capture=layers)
    return {i: f.shape[1] for i, f in feats.items()}


class DistillSegModel(SegmentationModel):
    """SegmentationModel + feature-distillation loss against one or more frozen teachers."""

    def __init__(self, cfg="yolo11s-seg.yaml", ch=3, nc=None, verbose=True, kd=None):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        kd = dict(kd or {})
        self.kd = {
            "teachers": list(kd.get("teachers", [])),
            "layers": [int(i) for i in kd.get("layers", [4, 6, 10])],
            "weight": float(kd.get("weight", 1.0)),
            "loss": kd.get("loss", "cosine"),
            "teacher_res": int(kd.get("teacher_res", 512)),
        }
        chans = _layer_channels(self, self.kd["layers"])
        self.kd_adapters = nn.ModuleDict()
        for t in self.kd["teachers"]:
            c_t = get_teacher(t, self.kd["teacher_res"]).dim
            for i in self.kd["layers"]:
                self.kd_adapters[f"{t.replace('.', '_')}__{i}"] = DistillAdapter(chans[i], c_t)

    def _kd_loss(self, feats: dict[int, torch.Tensor], img: torch.Tensor) -> torch.Tensor:
        total, n = img.new_zeros(()), 0
        for t in self.kd["teachers"]:
            tf = get_teacher(t, self.kd["teacher_res"], img.device)(img)
            for i in self.kd["layers"]:
                s = self.kd_adapters[f"{t.replace('.', '_')}__{i}"](feats[i]).float()
                tt = F.interpolate(tf, size=s.shape[-2:], mode="bilinear", align_corners=False)
                if self.kd["loss"] == "mse":  # MSE on per-location layer-normalised features
                    total = total + F.mse_loss(F.layer_norm(s.transpose(1, 3), s.shape[1:2]),
                                               F.layer_norm(tt.transpose(1, 3), tt.shape[1:2]))
                else:  # 1 - cosine similarity per location
                    total = total + (1 - F.cosine_similarity(s, tt, dim=1)).mean()
                n += 1
        return total / max(n, 1)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is not None or not self.training or not self.kd["teachers"]:  # validation / no teacher
            loss, items = self.criterion(self.forward(batch["img"]) if preds is None else preds, batch)
            return loss, {**items, "kd_loss": torch.zeros((), device=loss.device)}
        preds, feats = _run_layers(self, batch["img"], capture=self.kd["layers"])
        loss, items = self.criterion(preds, batch)
        kd = self._kd_loss(feats, batch["img"])
        bs = batch["img"].shape[0]
        # criterion returns a per-component vector already scaled by batch size; the trainer sums it
        loss = torch.cat([loss.flatten(), (self.kd["weight"] * kd * bs).reshape(1)])
        return loss, {**items, "kd_loss": kd.detach()}


class FusionSegModel(SegmentationModel):
    """SegmentationModel whose chosen layers receive projected frozen-teacher features."""

    def __init__(self, cfg="yolo11s-seg.yaml", ch=3, nc=None, verbose=True, fusion=None):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        fusion = dict(fusion or {})
        self.fusion = {
            "teacher": fusion["teacher"],
            "layers": [int(i) for i in fusion.get("layers", [4, 6, 10])],
            "teacher_res": int(fusion.get("teacher_res", 512)),
        }
        chans = _layer_channels(self, self.fusion["layers"])
        c_t = get_teacher(self.fusion["teacher"], self.fusion["teacher_res"]).dim
        self.fusion_proj = nn.ModuleDict()
        for i in self.fusion["layers"]:
            proj = nn.Sequential(nn.Conv2d(c_t, chans[i], 1, bias=False), nn.BatchNorm2d(chans[i]))
            nn.init.zeros_(proj[1].weight)  # starts exactly as the baseline
            self.fusion_proj[str(i)] = proj

    def _predict_once(self, x, profile=False, embed=None):
        if not hasattr(self, "fusion_proj") or profile or embed:
            return super()._predict_once(x, profile, embed)
        tf = get_teacher(self.fusion["teacher"], self.fusion["teacher_res"], x.device)(x)
        layers = set(self.fusion["layers"])

        def inject(i, feat):
            if i not in layers:
                return feat
            t = F.interpolate(tf, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            return feat + self.fusion_proj[str(i)](t.to(feat.dtype))

        return _run_layers(self, x, inject=inject)[0]


def make_trainer(model_cls, extra_key: str, extra: dict):
    """Build a SegmentationTrainer subclass that instantiates `model_cls(..., **{extra_key: extra})`."""

    class _Trainer(SegmentationTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = model_cls(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1,
                              **{extra_key: extra})
            model = self.set_model_names_for_load(model)
            if weights:
                model.load(weights)
            return model

    _Trainer.__name__ = f"{model_cls.__name__}Trainer"
    return _Trainer


def export_student(ckpt_path: str | Path, out_path: str | Path) -> Path:
    """Convert a DistillSegModel checkpoint into a plain YOLO11-seg checkpoint (adapters removed)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt.get("ema") or ckpt["model"]
    student = SegmentationModel(deepcopy(src.yaml), ch=src.yaml.get("channels", 3), nc=len(src.names), verbose=False)
    sd = {k: v for k, v in src.float().state_dict().items() if not k.startswith("kd_adapters.")}
    student.load_state_dict(sd, strict=True)
    for attr in ("names", "args", "stride", "task"):
        if hasattr(src, attr):
            setattr(student, attr, getattr(src, attr))
    out = dict(ckpt)
    out["model"], out["ema"] = student.half(), None
    out.pop("optimizer", None)
    out_path = Path(out_path)
    torch.save(out, out_path)
    return out_path
