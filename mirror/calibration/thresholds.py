from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_stress(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stressed = df.copy()
    idx = rng.choice(len(stressed), size=max(1, len(stressed) // 20), replace=False)
    stressed.loc[idx, "score"] = (stressed.loc[idx, "score"] + rng.uniform(0.2, 0.5, size=len(idx))).clip(0, 1)
    return stressed


def choose_threshold(scores: pd.Series, min_frac: float, max_frac: float, target_frac: float) -> float:
    quantile = max(min(1 - target_frac, 1 - min_frac), 1 - max_frac)
    return float(scores.quantile(quantile))


def apply_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["is_suspect"] = out["score"] >= threshold
    return out
