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


def _as_list(names) -> list[str]:
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def load_classes(path: str | Path) -> dict:
    """Return the class config with id-ordered lists.

    names           : training classes
    source_names    : raw annotation classes (= names when no mapping is configured)
    source_to_train : source id -> training id, or None when the source class is ignored
    """
    cfg = load_yaml(path)
    cfg["names"] = _as_list(cfg["names"])
    if "source_names" not in cfg:
        cfg["source_names"] = list(cfg["names"])
        cfg["source_to_train"] = list(range(len(cfg["names"])))
        return cfg
    src = cfg["source_names"] = _as_list(cfg["source_names"])
    train_id = {n: i for i, n in enumerate(cfg["names"])}
    s2t: dict[str, int | None] = dict.fromkeys(cfg.get("ignore") or [])
    for t, members in (cfg.get("map") or {}).items():
        if t not in train_id:
            raise ValueError(f"map target '{t}' is not in names")
        for s in members or []:
            if s in s2t:
                raise ValueError(f"source class '{s}' is mapped twice")
            s2t[s] = train_id[t]
    unknown = set(s2t) - set(src)
    unmapped = [s for s in src if s not in s2t]
    if unknown or unmapped:
        raise ValueError(f"class map mismatch: unknown {sorted(unknown)}, unmapped {unmapped}")
    cfg["source_to_train"] = [s2t[s] for s in src]
    return cfg
