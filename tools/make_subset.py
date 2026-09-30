#!/usr/bin/env python3
"""Class-balanced, sequence-aware train/val/test selection (instead of random sampling).

1. Group split (once, cached in <data>/splits/groups.json)
   Whole sequences (groups) go to train / val / test with iterative stratification: the rarest
   class is placed first, so scooter / wheelchair / stroller ... end up in every split in roughly
   the requested ratio and consecutive frames never leak between splits.

2. Image selection per split (budget, e.g. 6000 train images for the pilot; 0 = all images)
   a. quota phase — classes from rarest to most common: pick images containing the class until it
      has `min-per-class` images (or all of them if fewer exist; scooter keeps all 224).
      Candidates are thinned along time and penalised per sequence so picks are spread out.
   b. fill phase — the rest of the budget is drawn with repeat-factor weights
      (w = max over classes in the image of sqrt(t / f_c)), again thinned along time.

3. Optional repeat-factor sampling (LVIS RFS) for the train list: images with rare classes are
   listed more than once in train.txt (--rfs-t, 0 disables).

Outputs in --out: train.txt, val.txt, test.txt, data.yaml, subset_report.md, subset_report.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import load_classes  # noqa: E402

SPLITS = ("train", "val", "test")


def split_groups(df: pd.DataFrame, names: list[str], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Iterative stratification at sequence level. Returns {group: split}."""
    rng = np.random.default_rng(seed)
    g = df.groupby("group")
    inst = g[names].sum()
    n_img = g.size().reindex(inst.index)
    groups = list(inst.index)
    M = inst.to_numpy(dtype=np.float64)
    order = rng.permutation(len(groups))  # random tie-breaking
    want = {s: ratios[s] * M.sum(0) for s in SPLITS}
    want_img = {s: ratios[s] * n_img.sum() for s in SPLITS}
    have = {s: np.zeros(M.shape[1]) for s in SPLITS}
    have_img = dict.fromkeys(SPLITS, 0.0)
    remaining = set(range(len(groups)))
    assign = {}

    def put(i, c):
        def deficit(s):
            dc = (want[s][c] - have[s][c]) / max(want[s][c], 1e-9) if c is not None else 0.0
            di = (want_img[s] - have_img[s]) / max(want_img[s], 1e-9)
            return (dc, di) if ratios[s] > 0 else (-math.inf, -math.inf)

        s = max(SPLITS, key=deficit)
        assign[groups[i]] = s
        have[s] += M[i]
        have_img[s] += n_img.iloc[i]
        remaining.discard(i)

    while remaining:
        rem = np.array(sorted(remaining))
        left = M[rem].sum(0)
        if left.sum() == 0:  # only unlabelled groups left
            for i in rng.permutation(rem):
                put(int(i), None)
            break
        c = int(np.argmin(np.where(left > 0, left, np.inf)))
        cand = rem[M[rem, c] > 0]
        cand = sorted(cand, key=lambda i: (-M[i, c], order[i]))
        for i in cand:
            put(int(i), c)
    return assign


def thin_along_time(idx: np.ndarray, df: pd.DataFrame, keep: int) -> np.ndarray:
    """Keep ~`keep` indices evenly spread over (group, frame order)."""
    if len(idx) <= keep:
        return idx
    sub = df.iloc[idx][["group", "order"]]
    idx = idx[np.lexsort((sub["order"].to_numpy(), sub["group"].to_numpy()))]
    step = len(idx) / keep
    return idx[(np.arange(keep) * step).astype(int)]


def select_images(df: pd.DataFrame, names: list[str], budget: int, min_per_class: int, fill_t: float,
                  rng: np.random.Generator) -> np.ndarray:
    n = len(df)
    if budget <= 0 or budget >= n:
        return np.ones(n, dtype=bool)
    present = df[names].to_numpy() > 0
    avail = present.sum(0)
    quota = np.minimum(avail, min_per_class)
    groups = pd.factorize(df["group"])[0]
    sel = np.zeros(n, dtype=bool)
    have = np.zeros(len(names))
    gsel = np.zeros(groups.max() + 1)

    # a. quota phase: rarest class first
    for c in np.argsort(avail):
        need = int(quota[c] - have[c])
        if need <= 0 or sel.sum() >= budget:
            continue
        need = min(need, budget - int(sel.sum()))
        cand = np.flatnonzero(present[:, c] & ~sel)
        cand = thin_along_time(cand, df, 3 * need)
        picked = []
        for _ in range(min(need, len(cand))):
            short = np.clip(quota - have, 0, None) / np.maximum(quota, 1)  # classes still under quota
            score = present[cand] @ short + 1e-3 * rng.random(len(cand))
            score = score / (1.0 + 0.25 * gsel[groups[cand]])  # spread over sequences
            j = int(np.argmax(score))
            i = cand[j]
            picked.append(i)
            have += present[i]
            gsel[groups[i]] += 1
            cand = np.delete(cand, j)
        sel[picked] = True

    # b. fill phase: repeat-factor weighted sampling of the remaining budget
    rest = budget - int(sel.sum())
    if rest > 0:
        f = np.maximum(avail / n, 1e-9)
        rf = np.maximum(1.0, np.sqrt(fill_t / f))
        cand = thin_along_time(np.flatnonzero(~sel), df, 3 * rest)
        w = np.where(present[cand].any(1), (present[cand] * rf).max(1), 0.2)
        pick = rng.choice(cand, size=min(rest, len(cand)), replace=False, p=w / w.sum())
        sel[pick] = True
    return sel


def repeat_factors(df: pd.DataFrame, names: list[str], t: float, rng: np.random.Generator) -> np.ndarray:
    """LVIS repeat-factor sampling: r_i = max_c max(1, sqrt(t / f_c)), stochastic rounding."""
    reps = np.ones(len(df), dtype=int)
    if t <= 0 or len(df) == 0:
        return reps
    present = df[names].to_numpy() > 0
    f = np.maximum(present.mean(0), 1e-9)
    r = np.where(present, np.maximum(1.0, np.sqrt(t / f)), 1.0).max(1)
    return np.floor(r).astype(int) + (rng.random(len(r)) < r - np.floor(r))


def to_markdown(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="processed root containing index.csv")
    ap.add_argument("--out", required=True, type=Path, help="output dir, e.g. <data>/splits/pilot")
    ap.add_argument("--classes", default="configs/classes.yaml")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--resplit", action="store_true", help="recompute the cached sequence split")
    ap.add_argument("--train-images", type=int, default=6000, help="0 = all train images (full run)")
    ap.add_argument("--val-images", type=int, default=1500, help="0 = all val images")
    ap.add_argument("--test-images", type=int, default=0, help="0 = all test images")
    ap.add_argument("--min-train-per-class", type=int, default=300)
    ap.add_argument("--min-eval-per-class", type=int, default=60)
    ap.add_argument("--rfs-t", type=float, default=0.1, help="repeat-factor threshold for train.txt, 0 = off")
    ap.add_argument("--fill-t", type=float, default=0.3, help="repeat-factor threshold used to weight the fill phase")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cls = load_classes(args.classes)
    df = pd.read_csv(args.data / "index.csv")
    missing = [n for n in cls["source_names"] if n not in df.columns]
    if missing:
        raise SystemExit(f"index.csv lacks class columns {missing}; re-run prepare_dataset.py with the same classes")
    # balance on the fine source classes that are actually trained (ignored ones such as traffic_sign are skipped)
    names = [n for n, t in zip(cls["source_names"], cls["source_to_train"]) if t is not None]
    train_names = cls["names"]
    members = {j: [n for n, t in zip(cls["source_names"], cls["source_to_train"]) if t == j]
               for j in range(len(train_names))}

    gfile = args.data / "splits" / "groups.json"
    ratios = {"train": 1 - args.val_ratio - args.test_ratio, "val": args.val_ratio, "test": args.test_ratio}
    if gfile.exists() and not args.resplit:
        assign = json.loads(gfile.read_text())["assign"]
        new = set(df["group"]) - set(assign)
        if new:
            raise SystemExit(f"{len(new)} groups not in {gfile}; use --resplit")
        print(f"[subset] reusing sequence split {gfile}")
    else:
        assign = split_groups(df, names, ratios, args.seed)
        gfile.parent.mkdir(parents=True, exist_ok=True)
        gfile.write_text(json.dumps({"ratios": ratios, "seed": args.seed, "assign": assign}, indent=1))
        print(f"[subset] wrote sequence split {gfile}")
    df["split"] = df["group"].map(assign)

    rng = np.random.default_rng(args.seed)
    budgets = {"train": args.train_images, "val": args.val_images, "test": args.test_images}
    mins = {"train": args.min_train_per_class, "val": args.min_eval_per_class, "test": args.min_eval_per_class}
    args.out.mkdir(parents=True, exist_ok=True)
    chosen = {}
    for s in SPLITS:
        part = df[df["split"] == s].reset_index(drop=True)
        part = part[select_images(part, names, budgets[s], mins[s], args.fill_t, rng)].reset_index(drop=True)
        reps = repeat_factors(part, names, args.rfs_t, rng) if s == "train" else np.ones(len(part), dtype=int)
        part["repeat"] = reps
        chosen[s] = part
        lines = [p for p, r in zip(part["image"], reps) for _ in range(r)]
        (args.out / f"{s}.txt").write_text("\n".join(lines) + "\n")
        print(f"[subset] {s}: {len(part)} images ({len(lines)} list entries), {part['group'].nunique()} sequences")

    data_yaml = {
        "path": str(args.data.resolve()),
        "train": str((args.out / "train.txt").resolve()),
        "val": str((args.out / "val.txt").resolve()),
        "test": str((args.out / "test.txt").resolve()),
        "names": dict(enumerate(train_names)),
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))

    # report: source classes (what the sampler balances) and training classes (what the model sees)
    def class_row(label, cols, **extra):
        has = lambda d: d[cols].sum(axis=1) > 0 if cols else pd.Series(False, index=d.index)  # noqa: E731
        inst = lambda d: int(d[cols].to_numpy().sum()) if cols else 0  # noqa: E731
        r = {**extra, "class": label, "all_inst": inst(df), "all_img": int(has(df).sum()),
             "all_seq": int(df.loc[has(df), "group"].nunique())}
        for s, part in chosen.items():
            r[f"{s}_img"], r[f"{s}_inst"] = int(has(part).sum()), inst(part)
        tr = chosen["train"]
        r["train_inst_rfs"] = int((tr[cols].sum(axis=1) * tr["repeat"]).sum()) if cols else 0
        return r

    t_rows = [class_row(n, members[j], id=j, merged=", ".join(members[j]) or "-") for j, n in enumerate(train_names)]
    s_rows = [class_row(n, [n], train_class=train_names[t])
              for n, t in zip(cls["source_names"], cls["source_to_train"]) if t is not None]
    t_rep = pd.DataFrame(t_rows)
    t_rep = t_rep[["id", "class", *[c for c in t_rep.columns if c not in ("id", "class", "merged")], "merged"]]
    s_rep = pd.DataFrame(s_rows).sort_values("all_inst", ascending=False)
    t_rep.to_csv(args.out / "subset_report.csv", index=False)
    s_rep.to_csv(args.out / "subset_report_source.csv", index=False)
    md = ["# Subset report", "",
          f"budgets: {budgets}, min per source class: {mins}, rfs_t: {args.rfs_t}, seed: {args.seed}", "",
          "## Training classes", "", to_markdown(t_rep), "",
          "## Source classes (balancing level)", "", to_markdown(s_rep)]
    empty = [f"{r['class']}({s})" for r in t_rows for s in SPLITS if ratios[s] > 0 and r[f"{s}_inst"] == 0]
    if empty:
        md += ["", f"**WARNING** training classes without instances: {', '.join(empty)} "
                   "(stairs is expected to be empty until Surface caution_zone data is added)"]
    (args.out / "subset_report.md").write_text("\n".join(md) + "\n")
    print(t_rep.to_string(index=False))
    if empty:
        print(f"[subset] WARNING training classes without instances: {', '.join(empty)}")
    print(f"[subset] wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
