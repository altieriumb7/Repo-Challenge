from __future__ import annotations

from collections import Counter

import pandas as pd


def summarize(decisions: pd.DataFrame, agent_outputs: dict) -> dict:
    reason_counter: Counter[str] = Counter()
    for reasons in decisions.get("reasons", []):
        for r in reasons:
            if isinstance(r, list):
                reason_counter.update(r)
    out = {
        "suspect_rate": float(decisions["is_suspect"].mean()),
        "avg_score": float(decisions["score"].mean()),
        "top_reasons": reason_counter.most_common(8),
        "per_agent_rows": {k: len(v.evidence) for k, v in agent_outputs.items()},
    }
    return out
