from __future__ import annotations

import csv
from pathlib import Path


class ExperimentRegistry:
    HEADER = [
        "run_id",
        "scenario",
        "config_hash",
        "runtime_sec",
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "suspect_rate",
        "top_pattern_families",
        "artifacts_dir",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADER)
                writer.writeheader()

    def append(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writerow({k: row.get(k, "") for k in self.HEADER})
