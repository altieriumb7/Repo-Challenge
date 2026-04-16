from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class MemoryStore:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "memory.json"
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "suspicious_entities": [],
            "risky_merchants": [],
            "temporal_patterns": [],
            "geo_anomalies": [],
            "message_flags": [],
        }

    def update(self, key: str, values: list[Any]) -> None:
        existing = set(map(str, self.state.get(key, [])))
        for value in values:
            existing.add(str(value))
        self.state[key] = sorted(existing)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=True), encoding="utf-8")

    def save_frame(self, name: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.base_dir / f"{name}.parquet", index=False)
