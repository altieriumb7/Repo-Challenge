from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import pandas as pd


@dataclass
class AgentEvidence:
    agent: str
    transaction_id: str
    score: float
    confidence: float
    reasons: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    name: str
    evidence: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    scenario_name: str
    input_dir: str
    output_dir: str
    config: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)
    features: dict[str, pd.DataFrame] = field(default_factory=dict)
    agent_outputs: dict[str, AgentResult] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMBudget:
    max_calls: int
    max_strong_calls: int
    calls: int = 0
    strong_calls: int = 0
    wait_time_seconds: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def try_acquire(self, model: str, wait_time_seconds: float = 0.0) -> bool:
        with self._lock:
            is_strong = "strong" in str(model).lower()
            if self.calls >= self.max_calls:
                return False
            if is_strong and self.strong_calls >= self.max_strong_calls:
                return False
            self.calls += 1
            if is_strong:
                self.strong_calls += 1
            self.wait_time_seconds += max(0.0, float(wait_time_seconds))
            return True

    def usage(self) -> dict[str, float]:
        with self._lock:
            return {
                "calls": float(self.calls),
                "strong_calls": float(self.strong_calls),
                "wait_time_seconds": float(self.wait_time_seconds),
            }
