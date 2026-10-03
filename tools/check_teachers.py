#!/usr/bin/env python3
"""Download every teacher once and run a forward pass — do this before starting the long pilot.

Catches missing HF_TOKEN / un-accepted licences (DINOv3 is gated) and blocked hosts early.
  python tools/check_teachers.py                       # teachers used by the default pilot
  python tools/check_teachers.py dinov2_b cradio_v4_so400m
  python tools/check_teachers.py --cpu                 # no GPU (slow)
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kdseg.teachers import get_teacher  # noqa: E402

DEFAULT = ["siglip2_b", "dinov3_b", "radio_v2.5_b", "cradio_v3_b"]


def main():
    args = [a for a in sys.argv[1:] if a != "--cpu"]
    names = args or DEFAULT
    if torch.cuda.is_available():
        dev = "cuda"
        print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__} CUDA {torch.version.cuda} | "
              f"teacher dtype {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")
    elif "--cpu" in sys.argv:
        dev = "cpu"
    else:
        print("FAIL no CUDA GPU visible to torch — run `python tools/check_gpu.py` (or pass --cpu to test anyway)")
        sys.exit(1)
    x = torch.rand(8, 3, 640, 640, device=dev)
    failed = []
    for n in names:
        try:
            t0 = time.time()
            t = get_teacher(n, 512, dev)
            load_s = time.time() - t0
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            f = t(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            params = sum(p.numel() for p in t.parameters()) / 1e6
            mem = f" peak={torch.cuda.max_memory_allocated() / 2**30:4.1f}GiB" if dev == "cuda" else ""
            ok = torch.isfinite(f).all().item() and f.std().item() > 0
            print(f"{'OK  ' if ok else 'BAD '} {n:18s} dim={t.dim:5d} grid={tuple(f.shape[-2:])} params={params:6.1f}M "
                  f"load={load_s:5.1f}s fwd(8x640)={time.time() - t0:5.2f}s{mem}"
                  + ("" if ok else "  <- NaN/constant features"))
            if not ok:
                failed.append(n)
        except Exception as e:  # noqa: BLE001
            failed.append(n)
            print(f"FAIL {n:18s} {type(e).__name__}: {str(e).splitlines()[0][:300]}")
    if failed:
        print(f"\nfailed: {failed}  (DINOv3: accept the licence on its HF page and `export HF_TOKEN=...`)")
        sys.exit(1)


if __name__ == "__main__":
    main()
