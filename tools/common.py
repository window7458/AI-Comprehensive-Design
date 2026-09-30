"""Small helpers shared by the tools."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes(path: str | Path) -> dict:
    """Return the class config with `names` as an id-ordered list."""
    cfg = load_yaml(path)
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    cfg["names"] = list(names)
    return cfg
