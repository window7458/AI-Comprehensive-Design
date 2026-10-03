#!/usr/bin/env python3
"""Collect <project>/*/results.json into summary.csv / summary.md and name the best encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import load_classes  # noqa: E402
from tools.make_subset import to_markdown  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="runs/pilot")
    ap.add_argument("--classes", default="configs/classes.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    names = load_classes(args.classes)["names"]
    project = Path(args.project)
    rows, per_class = [], {}
    for f in sorted(project.glob("*/results.json")):
        r = json.loads(f.read_text())
        v = r.get(args.split)
        if v is None:
            continue
        cfg = r.get("config", {})
        teacher = "+".join(cfg.get("teachers", [])) or cfg.get("teacher", "-")
        rows.append({
            "exp": r["exp"], "track": r["track"], "teacher": teacher,
            "mask_mAP50-95": v["mask_map50_95"], "mask_mAP50": v["mask_map50"],
            "box_mAP50-95": v["box_map50_95"], "key_mAP": v["key_mask_map50_95"],
            "rare_mAP": v["rare_mask_map50_95"], "focus_mAP": v.get("focus_mask_map50_95"),
            "scooter": v["per_class_mask_map50_95"].get("scooter"),
            "traffic_light": v["per_class_mask_map50_95"].get("traffic_light"),
            "infer_ms": v["speed_ms"].get("inference"), "train_h": r.get("train_hours"),
        })
        per_class[r["exp"]] = v["per_class_mask_map50_95"]
    if not rows:
        print(f"no results in {project}")
        return
    df = pd.DataFrame(rows).sort_values("mask_mAP50-95", ascending=False)
    base = df.loc[df["exp"] == "E0", "mask_mAP50-95"]
    if len(base):
        df.insert(4, "Δ_vs_E0", df["mask_mAP50-95"] - float(base.iloc[0]))
    pc = pd.DataFrame(per_class).reindex(names)
    df.round(4).to_csv(project / f"summary_{args.split}.csv", index=False)
    pc.round(4).to_csv(project / f"per_class_{args.split}.csv")

    distill = df[df["track"] == "distill"]
    best = distill.iloc[0] if len(distill) else None
    md = [f"# Pilot summary ({args.split})", "", to_markdown(df.round(4)), ""]
    if best is not None:
        md += [f"**Best distillation run:** {best['exp']} ({best['teacher']}) — mask mAP50-95 "
               f"{best['mask_mAP50-95']:.4f}", ""]
    md += ["## Per-class mask mAP50-95", "", to_markdown(pc.round(3).reset_index(names="class"))]
    (project / f"summary_{args.split}.md").write_text("\n".join(md) + "\n")
    print(df.round(4).to_string(index=False))
    print(f"wrote {project / f'summary_{args.split}.md'}")


if __name__ == "__main__":
    main()
