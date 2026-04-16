from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def git_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def env_snapshot() -> dict:
    try:
        import importlib.metadata as md

        packages = {d.metadata["Name"]: d.version for d in md.distributions() if d.metadata.get("Name")}
    except Exception:
        packages = {}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "packages": dict(sorted(packages.items())),
        "openrouter_model_fast": os.getenv("MODEL_FAST", "openai/gpt-4o-mini"),
        "openrouter_model_strong": os.getenv("MODEL_STRONG", "openai/gpt-4.1"),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
