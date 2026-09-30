#!/usr/bin/env python3
"""Train + evaluate one experiment (E0 / B1..B5 / A-best) and write <project>/<exp>/results.json.

Example
  python tools/train.py --exp B2 --data data/processed/splits/pilot/data.yaml --project runs/pilot
  python tools/train.py --exp A-best --teacher dinov3_b ...    # fusion with an explicit teacher
  python tools/train.py --exp B1 --set epochs=3 batch=4 fraction=0.1   # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kdseg import DistillSegModel, FusionSegModel, export_student, make_trainer  # noqa: E402
from tools.common import load_classes, load_yaml  # noqa: E402


def parse_sets(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = yaml.safe_load(v)
    return out


def best_single_teacher(project: Path, cfg: dict) -> str:
    """Pick the teacher of the single-teacher distillation run with the highest val mask mAP50-95."""
    best, best_map = None, -1.0
    for name, e in cfg["experiments"].items():
        if e.get("track") != "distill" or len(e.get("teachers", [])) != 1:
            continue
        f = project / name / "results.json"
        if f.exists():
            m = json.loads(f.read_text())["val"]["mask_map50_95"]
            if m > best_map:
                best, best_map = e["teachers"][0], m
    if best is None:
        raise SystemExit("A-best needs finished single-teacher B* runs (or pass --teacher)")
    print(f"[train] A-best teacher = {best} (val mask mAP50-95 {best_map:.4f})")
    return best


def evaluate(weights: Path, data: str, split: str, args_common: dict, save_dir: Path, classes: dict) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=data, split=split, imgsz=args_common["imgsz"], batch=args_common["batch"], plots=False,
                  project=str(save_dir), name=f"val_{split}", exist_ok=True, verbose=False)
    names = classes["names"]
    per_class = {n: None for n in names}
    for k, c in enumerate(m.seg.ap_class_index):
        per_class[names[int(c)]] = float(m.seg.ap[k])

    def mean_of(group):
        v = [per_class[n] for n in group if per_class.get(n) is not None]
        return float(np.mean(v)) if v else None

    return {
        "mask_map50_95": float(m.seg.map), "mask_map50": float(m.seg.map50),
        "box_map50_95": float(m.box.map), "box_map50": float(m.box.map50),
        "key_mask_map50_95": mean_of(classes.get("key_classes", [])),
        "rare_mask_map50_95": mean_of(classes.get("rare_classes", [])),
        "per_class_mask_map50_95": per_class,
        "speed_ms": {k: float(v) for k, v in m.speed.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", required=True, help="data.yaml written by make_subset.py")
    ap.add_argument("--project", default="runs/pilot")
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--classes", default="configs/classes.yaml")
    ap.add_argument("--teacher", default=None, help="fusion teacher override (A-best)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--eval-test", action="store_true", help="also evaluate on the held-out test split")
    ap.add_argument("--set", nargs="*", default=[], help="extra Ultralytics args, key=value")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    classes = load_classes(args.classes)
    exp = cfg["experiments"][args.exp]
    project = Path(args.project).resolve()
    common = {**cfg["common"], **parse_sets(args.set)}
    track = exp["track"]

    model_path = common.pop("model")
    if track == "baseline":
        from ultralytics.models.yolo.segment import SegmentationTrainer as trainer
        info = {}
    elif track == "distill":
        info = {**cfg["distill"], "teachers": exp["teachers"]}
        trainer = make_trainer(DistillSegModel, "kd", info)
    elif track == "fusion":
        teacher = args.teacher or exp.get("teacher", "auto")
        if teacher == "auto":
            teacher = best_single_teacher(project, cfg)
        info = {**cfg["fusion"], "teacher": teacher}
        trainer = make_trainer(FusionSegModel, "fusion", info)
    else:
        raise SystemExit(f"unknown track {track}")

    from ultralytics import YOLO

    print(f"[train] {args.exp} ({exp.get('desc', '')}) track={track} {info}")
    t0 = time.time()
    model = YOLO(model_path)
    model.train(trainer=trainer, data=args.data, project=str(project), name=args.exp, exist_ok=True,
                device=args.device, **common)
    run_dir = project / args.exp
    best = run_dir / "weights" / "best.pt"
    deploy = best
    if track == "distill":  # evaluate the plain student that would be deployed
        deploy = export_student(best, run_dir / "weights" / "student.pt")
    train_h = (time.time() - t0) / 3600

    results = {"exp": args.exp, "desc": exp.get("desc", ""), "track": track, "config": info,
               "train_args": {**common, "model": model_path}, "train_hours": round(train_h, 3),
               "weights": str(deploy), "val": evaluate(deploy, args.data, "val", common, run_dir, classes)}
    if args.eval_test:
        results["test"] = evaluate(deploy, args.data, "test", common, run_dir, classes)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    v = results["val"]
    print(f"[train] {args.exp} done: mask mAP50-95 {v['mask_map50_95']:.4f}  key {v['key_mask_map50_95']}  "
          f"rare {v['rare_mask_map50_95']}  scooter {v['per_class_mask_map50_95'].get('scooter')}")


if __name__ == "__main__":
    main()
