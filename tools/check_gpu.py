#!/usr/bin/env python3
"""Check that PyTorch sees the NVIDIA GPU and that the training code path works on it (~1 min, no downloads).

1. torch / CUDA / cuDNN versions, GPU name, memory, bf16 support
2. a fp16 matmul on the GPU
3. one AMP forward + backward of DistillSegModel and FusionSegModel (YOLO11s-seg, 2 x 640 images)
   with the random "dummy" teacher — the exact CUDA code path the real teachers use

  python tools/check_gpu.py            # exits 1 if anything fails
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    print(f"torch {torch.__version__} | built for CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        print("FAIL torch.cuda.is_available() is False.\n"
              "     - CPU-only torch wheel? -> reinstall torch for the server's CUDA (see `nvidia-smi` top right)\n"
              "     - or the container has no GPU attached")
        sys.exit(1)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} | {p.total_memory / 2**30:.1f} GiB | compute capability {p.major}.{p.minor}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()} (teachers run in {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'})")

    dev = torch.device("cuda:0")
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    print(f"OK   fp16 matmul ({20 * 2 * 2048**3 / (time.time() - t0) / 1e12:.1f} TFLOPS)")

    from kdseg import DistillSegModel, FusionSegModel
    from ultralytics.cfg import get_cfg

    names = {i: f"c{i}" for i in range(10)}
    for cls, kw in [(DistillSegModel, {"kd": {"teachers": ["dummy"], "teacher_res": 512}}),
                    (FusionSegModel, {"fusion": {"teacher": "dummy", "teacher_res": 512}})]:
        try:
            m = cls("yolo11s-seg.yaml", nc=10, verbose=False, **kw).to(dev).train()
            m.args, m.names = get_cfg(), names
            b = 2
            batch = {
                "img": torch.rand(b, 3, 640, 640, device=dev),
                "batch_idx": torch.tensor([0.0, 1.0], device=dev),
                "cls": torch.tensor([[0.0], [8.0]], device=dev),
                "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.3, 0.4, 0.1, 0.2]], device=dev),
                "masks": torch.zeros(b, 160, 160, device=dev),
            }
            batch["masks"][0, 60:100, 70:90] = 1
            batch["masks"][1, 50:70, 40:60] = 1  # overlap_mask: instance index per image
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.float16):
                loss, items = m(batch)
            loss.sum().backward()
            torch.cuda.synchronize()
            ok = torch.isfinite(loss).all().item()
            print(f"{'OK  ' if ok else 'FAIL'} {cls.__name__:15s} AMP fwd+bwd on GPU | losses "
                  f"{ {k: round(float(v), 3) for k, v in items.items()} } | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            if not ok:
                sys.exit(1)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {cls.__name__}: {type(e).__name__}: {e}")
            sys.exit(1)
    print("GPU check passed")


if __name__ == "__main__":
    main()
