from __future__ import annotations

from dataclasses import dataclass, field
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
