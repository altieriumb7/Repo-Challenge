from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _is_ascii_safe(value: str) -> bool:
    try:
        value.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def write_submission(
    df: pd.DataFrame,
    eval_transactions: pd.DataFrame,
    path: str | Path,
    min_frac: float,
    max_frac: float,
    suspect_rate_bounds: tuple[float, float] | None = None,
) -> Path:
    ordered = df.sort_values("score", ascending=False)
    suspects = ordered[ordered["is_suspect"]].copy()
    total = len(df)
    frac = len(suspects) / max(total, 1)
    if len(suspects) == 0 or frac < min_frac:
        suspects = ordered.head(max(1, int(total * min_frac)))
    if frac > max_frac:
        suspects = ordered.head(max(1, int(total * max_frac)))

    ids = suspects["transaction_id"].astype(str)
    if ids.empty:
        raise ValueError("submission safety check failed: no IDs selected.")
    if not ids.map(_is_ascii_safe).all():
        bad = ids[~ids.map(_is_ascii_safe)].head(3).tolist()
        raise ValueError(f"submission safety check failed: non-ASCII IDs detected, examples={bad}")
    if ids.duplicated().any():
        raise ValueError("submission safety check failed: duplicate IDs detected.")

    eval_ids = set(eval_transactions["transaction_id"].astype(str).tolist())
    invalid = [i for i in ids.tolist() if i not in eval_ids]
    if invalid:
        raise ValueError(f"submission safety check failed: {len(invalid)} IDs are not in eval transactions.")
    if len(ids) == len(eval_ids):
        raise ValueError("submission safety check failed: all eval rows flagged.")

    rate = len(ids) / max(1, len(eval_ids))
    bounds = suspect_rate_bounds or (min_frac, max_frac)
    if not (bounds[0] <= rate <= bounds[1]):
        raise ValueError(
            f"submission safety check failed: suspect rate {rate:.4f} outside configured bounds [{bounds[0]:.4f}, {bounds[1]:.4f}]."
        )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids.tolist()), encoding="ascii", errors="strict")

    report = {
        "rows_total": len(eval_ids),
        "rows_flagged": len(ids),
        "suspect_rate": rate,
        "bounds": {"min": bounds[0], "max": bounds[1]},
    }
    (out.parent / f"{out.stem}.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
