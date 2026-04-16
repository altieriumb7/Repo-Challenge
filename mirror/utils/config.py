from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str | Path, scenario_name: str | None = None) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if scenario_name and scenario_name in (cfg.get("scenarios") or {}):
        cfg = deep_merge(cfg, cfg["scenarios"][scenario_name])
    return cfg
