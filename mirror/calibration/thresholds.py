from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_stress(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stressed = df.copy()
    if len(stressed) == 0:
        return stressed
    idx = rng.choice(len(stressed), size=max(1, len(stressed) // 20), replace=False)
    stressed.loc[idx, "score"] = (stressed.loc[idx, "score"] + rng.uniform(0.2, 0.5, size=len(idx))).clip(0, 1)
    return stressed


def choose_threshold(scores: pd.Series, min_frac: float, max_frac: float, target_frac: float) -> float:
    if scores.empty:
        return 1.0
    std = float(scores.std(ddof=0)) if len(scores) > 1 else 0.0
    size_adj = min(0.02, 2.0 / max(len(scores), 1))
    uncertainty_adj = min(0.03, (0.18 - std) * 0.12) if std < 0.18 else -min(0.02, (std - 0.18) * 0.05)
    adaptive_target = float(np.clip(target_frac + size_adj + uncertainty_adj, min_frac, max_frac))
    quantile = float(np.clip(1 - adaptive_target, 1 - max_frac, 1 - min_frac))
    return float(scores.quantile(quantile))


def apply_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["is_suspect"] = out["score"] >= threshold
    return out
