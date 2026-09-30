#!/usr/bin/env python3
"""Download every teacher once and run a forward pass — do this before starting the long pilot.

Catches missing HF_TOKEN / un-accepted licences (DINOv3 is gated) and blocked hosts early.
  python tools/check_teachers.py                       # teachers used by the default pilot
  python tools/check_teachers.py dinov2_b cradio_v4_so400m
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kdseg.teachers import get_teacher  # noqa: E402

DEFAULT = ["siglip2_b", "dinov3_b", "radio_v2.5_b", "cradio_v3_b"]


def main():
    names = sys.argv[1:] or DEFAULT
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
            print(f"OK   {n:18s} dim={t.dim:5d} grid={tuple(f.shape[-2:])} params={params:6.1f}M "
                  f"load={load_s:5.1f}s fwd(8x640)={time.time() - t0:5.2f}s")
        except Exception as e:  # noqa: BLE001
            failed.append(n)
            print(f"FAIL {n:18s} {type(e).__name__}: {str(e).splitlines()[0][:300]}")
    if failed:
        print(f"\nfailed: {failed}  (DINOv3: accept the licence on its HF page and `export HF_TOKEN=...`)")
        sys.exit(1)


if __name__ == "__main__":
    main()
