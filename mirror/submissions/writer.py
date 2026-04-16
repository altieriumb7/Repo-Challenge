from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_submission(df: pd.DataFrame, path: str | Path, min_frac: float, max_frac: float) -> Path:
    ordered = df.sort_values("score", ascending=False)
    suspects = ordered[ordered["is_suspect"]].copy()
    total = len(df)
    frac = len(suspects) / max(total, 1)
    if len(suspects) == 0 or frac < min_frac:
        suspects = ordered.head(max(1, int(total * min_frac)))
    if frac > max_frac:
        suspects = ordered.head(max(1, int(total * max_frac)))
    ids = suspects["transaction_id"].astype(str).drop_duplicates()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids.tolist()), encoding="ascii", errors="ignore")
    return out
