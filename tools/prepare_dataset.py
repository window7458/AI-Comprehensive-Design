#!/usr/bin/env python3
"""Convert the raw Google-Drive dataset into one YOLO-seg tree + a per-image index.

Supported inputs (auto-detected):
  * cvat : AI-Hub 인도보행 Polygon style — per-sequence folders with a CVAT-for-images XML
           (<image name= width= height=><polygon label= points="x,y;x,y;..."/></image>).
  * yolo : already converted YOLO-seg ( .../images/**.jpg + .../labels/**.txt ).

Output (OUT = --out):
  OUT/images/<group>/<file>      symlink (or copy) to the original image
  OUT/labels/<group>/<stem>.txt  YOLO-seg polygon labels
  OUT/index.csv                  image, group, order, width, height, n_inst, <source-class counts...>

Labels are written with the training ids (configs/classes.yaml `map`: 26 source classes -> 10),
while index.csv keeps the 26 source-class counts so sampling can balance at the fine level.
  OUT/prepare_report.txt         dropped labels / missing images summary

`group` is the video sequence / folder. Consecutive frames of one sequence are near-duplicates,
so all later splits are done per group to avoid train/val leakage.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import IMG_EXTS, load_classes  # noqa: E402


def norm_label(s: str) -> str:
    return s.strip().lower()


def find_images(root: Path) -> dict[str, Path]:
    out = {}
    for dp, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            if Path(f).suffix.lower() in IMG_EXTS:
                out.setdefault(f, Path(dp) / f)
    return out


def parse_cvat(src: Path, name_to_id, dropped: Counter, missing: list):
    """Yield (image_path, group, width, height, [(cls, [(x, y), ...]), ...])."""
    xmls = sorted(src.rglob("*.xml"))
    if not xmls:
        raise SystemExit(f"no *.xml found under {src}")
    img_cache: dict[Path, dict[str, Path]] = {}
    xml_per_dir = Counter(xp.parent for xp in xmls)
    global_imgs = None
    for xp in xmls:
        folder = xp.parent
        if folder not in img_cache:
            img_cache[folder] = find_images(folder)
        imgs = img_cache[folder]
        if not imgs:  # annotations stored apart from the images -> search the whole source tree
            global_imgs = global_imgs if global_imgs is not None else find_images(src)
            imgs = global_imgs
        group = folder.name if xml_per_dir[folder] == 1 and folder != src else xp.stem
        for _, el in ET.iterparse(xp, events=("end",)):
            if el.tag != "image":
                continue
            name = el.get("name", "")
            path = imgs.get(Path(name).name)
            if path is None:
                missing.append(f"{xp.name}:{name}")
                el.clear()
                continue
            w, h = float(el.get("width")), float(el.get("height"))
            objs = []
            for p in el.iter("polygon"):
                lab = norm_label(p.get("label", ""))
                cid = name_to_id(lab)
                if cid is None:
                    dropped[lab] += 1
                    continue
                pts = [tuple(map(float, xy.split(","))) for xy in p.get("points", "").split(";") if xy]
                if len(pts) >= 3:
                    objs.append((cid, [(x / w, y / h) for x, y in pts]))
            yield path, group, int(w), int(h), objs
            el.clear()


def parse_yolo(src: Path, group_regex: str | None, remap: dict[int, int] | None):
    img_paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS and "images" in p.parts)
    if not img_paths:
        raise SystemExit(f"no images under an 'images' folder in {src}")
    rx = re.compile(group_regex) if group_regex else None
    for ip in img_paths:
        parts = list(ip.parts)
        k = len(parts) - 1 - parts[::-1].index("images")
        lp = Path(*parts[:k], "labels", *parts[k + 1:]).with_suffix(".txt")
        if rx:
            m = rx.search(ip.stem)
            group = m.group(1) if m else ip.parent.name
        else:
            group = ip.parent.name if ip.parent.name != "images" else ip.stem.rsplit("_", 1)[0]
        objs = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                v = line.split()
                if len(v) < 7:
                    continue
                c = int(float(v[0]))
                c = remap.get(c) if remap is not None else c
                if c is None:
                    continue
                xy = list(map(float, v[1:]))
                objs.append((c, list(zip(xy[0::2], xy[1::2]))))
        yield ip, group, 0, 0, objs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="raw dataset root (downloaded from Google Drive)")
    ap.add_argument("--out", required=True, type=Path, help="processed dataset root")
    ap.add_argument("--classes", default="configs/classes.yaml")
    ap.add_argument("--format", choices=["auto", "cvat", "yolo"], default="auto")
    ap.add_argument("--link", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--group-regex", default=None, help="yolo format: regex on file stem, group(1) = sequence id")
    ap.add_argument("--src-names", default=None, help="yolo format: data.yaml of the source, remap classes by name")
    args = ap.parse_args()

    cls = load_classes(args.classes)
    names, train_names, s2t = cls["source_names"], cls["names"], cls["source_to_train"]
    lookup = {n: i for i, n in enumerate(names)}
    lookup.update({norm_label(a): lookup[c] for a, c in cls.get("aliases", {}).items()})

    def name_to_id(lab):
        return lookup.get(lab, lookup.get(lab.replace(" ", "_")))

    fmt = args.format
    if fmt == "auto":
        fmt = "cvat" if next(args.src.rglob("*.xml"), None) else "yolo"
    print(f"[prepare] format={fmt} src={args.src}")

    dropped, missing = Counter(), []
    if fmt == "cvat":
        items = parse_cvat(args.src, name_to_id, dropped, missing)
    else:
        remap = None
        if args.src_names:
            src_names = load_classes(args.src_names)["source_names"]
            remap = {i: name_to_id(norm_label(n)) for i, n in enumerate(src_names)}
            remap = {k: v for k, v in remap.items() if v is not None}
        items = parse_yolo(args.src, args.group_regex, remap)

    out_img, out_lbl = args.out / "images", args.out / "labels"
    rows, seen = [], set()
    for n, (ip, group, w, h, objs) in enumerate(items, 1):
        group = re.sub(r"[^\w.-]", "_", group)
        dst = out_img / group / ip.name
        if dst in seen:  # same file name referenced twice
            continue
        seen.add(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and not dst.is_symlink():
            if args.link == "symlink":
                dst.symlink_to(ip.resolve())
            else:
                shutil.copy2(ip, dst)
        lp = out_lbl / group / (ip.stem + ".txt")
        lp.parent.mkdir(parents=True, exist_ok=True)
        counts = [0] * len(names)
        lines = []
        for c, pts in objs:
            pts = [(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in pts]
            counts[c] += 1
            if s2t[c] is not None:  # ignored source classes (e.g. traffic_sign) become background
                lines.append(f"{s2t[c]} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
        lp.write_text("\n".join(lines) + ("\n" if lines else ""))
        rows.append([str(dst), group, 0, w, h, sum(counts), *counts])
        if n % 5000 == 0:
            print(f"[prepare] {n} images")

    # frame order inside each group (sorted by file name) — used to spread samples over time
    rows.sort(key=lambda r: (r[1], Path(r[0]).name))
    last, k = None, 0
    for r in rows:
        k = k + 1 if r[1] == last else 0
        last, r[2] = r[1], k

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "index.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "group", "order", "width", "height", "n_inst", *names])
        wr.writerows(rows)

    totals = [sum(r[6 + i] for r in rows) for i in range(len(names))]
    train_totals = [sum(t for t, k in zip(totals, s2t) if k == j) for j in range(len(train_names))]
    report = [f"images: {len(rows)}  groups: {len({r[1] for r in rows})}", "instances per source class:"]
    report += [f"  {i:2d} {n:26s} {totals[i]:>8d}  -> {train_names[s2t[i]] if s2t[i] is not None else '(ignored)'}"
               for i, n in enumerate(names)]
    report += ["instances per training class:"]
    report += [f"  {j:2d} {n:26s} {train_totals[j]:>8d}" for j, n in enumerate(train_names)]
    report += [f"dropped labels (not in classes.yaml): {dict(dropped.most_common())}",
               f"annotated images without a file: {len(missing)}", *[f"  {m}" for m in missing[:50]]]
    (args.out / "prepare_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report[:3 + len(names) + len(train_names) + 2]))
    print(f"[prepare] wrote {args.out / 'index.csv'}")


if __name__ == "__main__":
    main()
