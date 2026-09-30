#!/usr/bin/env python3
"""Create a tiny synthetic AI-Hub-style (CVAT XML) dataset for smoke tests.

Class frequencies follow the real instance table (pole/car very common, scooter/wheelchair rare),
so the balanced sampler can be checked without the real data.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import load_classes  # noqa: E402

REAL_INSTANCES = {
    "car": 147132, "pole": 98463, "tree_trunk": 97018, "person": 47192, "traffic_sign": 38918, "bollard": 37266,
    "truck": 33211, "traffic_light": 26799, "movable_signage": 20203, "bus": 11321, "bicycle": 10003,
    "motorcycle": 9039, "potted_plant": 9031, "bench": 6215, "power_controller": 5262, "barricade": 4323,
    "stop": 4311, "traffic_light_controller": 3986, "chair": 3643, "fire_hydrant": 2593, "carrier": 1479,
    "table": 1317, "kiosk": 1198, "stroller": 485, "scooter": 351, "wheelchair": 204,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=24)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    names = load_classes("configs/classes.yaml")["source_names"]
    rng = np.random.default_rng(args.seed)
    p = np.array([REAL_INSTANCES[n] for n in names], dtype=float) ** 0.7
    p /= p.sum()
    colors = rng.integers(0, 255, (len(names), 3))
    W, H = 320, 240
    for g in range(args.groups):
        d = args.out / f"Polygon_{g:04d}"
        d.mkdir(parents=True, exist_ok=True)
        lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>", "  <version>1.1</version>"]
        for k in range(args.frames):
            img = np.full((H, W, 3), 90, np.uint8)
            name = f"MP_SEL_{g:04d}_{k:05d}.jpg"
            lines.append(f'  <image id="{k}" name="{name}" width="{W}" height="{H}">')
            for _ in range(rng.integers(1, 6)):
                c = int(rng.choice(len(names), p=p))
                cx, cy, r = rng.integers(30, W - 30), rng.integers(30, H - 30), rng.integers(8, 28)
                ang = np.sort(rng.random(6)) * 2 * np.pi
                pts = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1).astype(np.int32)
                cv2.fillPoly(img, [pts], tuple(int(v) for v in colors[c]))
                pstr = ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)
                lines.append(f'    <polygon label="{names[c]}" occluded="0" points="{pstr}" z_order="0"/>')
            if k == 0:  # a label outside classes.yaml, must be dropped
                lines.append('    <polygon label="dog" occluded="0" points="1,1;5,1;5,5" z_order="0"/>')
            lines.append("  </image>")
            cv2.imwrite(str(d / name), img)
        lines.append("</annotations>")
        (d / f"Polygon_{g:04d}.xml").write_text("\n".join(lines) + "\n")
    print(f"synthetic dataset: {args.groups} groups x {args.frames} frames -> {args.out}")


if __name__ == "__main__":
    main()
