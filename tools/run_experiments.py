#!/usr/bin/env python3
"""Run the pilot matrix sequentially (one fresh process per experiment), then summarise.

Finished runs (results.json present) are skipped, so the script can simply be restarted after an
interruption. A failing run is logged and the next one starts.

  python tools/run_experiments.py --data data/processed/splits/pilot/data.yaml --project runs/pilot
  python tools/run_experiments.py ... --only E0 B2 --set epochs=30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import ROOT, load_yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", default="runs/pilot")
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--classes", default="configs/classes.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="subset of experiment ids")
    ap.add_argument("--with-optional", action="store_true", help="also run B2b (DINOv2) and B5b (C-RADIOv4)")
    ap.add_argument("--force", action="store_true", help="re-run finished experiments")
    ap.add_argument("--device", default="0")
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ids = args.only or [e for e in cfg["order"] if args.with_optional or not cfg["experiments"][e].get("optional")]
    project = Path(args.project)
    log_dir = project / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for exp in ids:
        if (project / exp / "results.json").exists() and not args.force:
            print(f"[run] {exp}: already done, skipping")
            continue
        cmd = [sys.executable, str(ROOT / "tools" / "train.py"), "--exp", exp, "--data", args.data,
               "--project", str(project), "--config", args.config, "--classes", args.classes,
               "--device", args.device]
        if args.eval_test:
            cmd.append("--eval-test")
        if args.set:
            cmd += ["--set", *args.set]
        print(f"[run] {exp}: {' '.join(cmd)}  (log: {log_dir / (exp + '.log')})", flush=True)
        with open(log_dir / f"{exp}.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = proc.wait()
        if rc != 0:
            print(f"[run] {exp}: FAILED (exit {rc}), see {log_dir / (exp + '.log')}")
            failed.append(exp)

    subprocess.run([sys.executable, str(ROOT / "tools" / "summarize.py"), "--project", str(project),
                    "--classes", args.classes], check=False)
    if failed:
        print(f"[run] failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
