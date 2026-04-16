from __future__ import annotations

from typing import Any

import pandas as pd


TRANSACTION_RENAME = {
    "id": "transaction_id",
    "timestamp": "event_time",
    "sender": "sender_id",
    "recipient": "recipient_id",
    "user_id": "sender_id",
}


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in TRANSACTION_RENAME.items() if k in df.columns}).copy()
    required = ["transaction_id", "event_time", "amount"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required transaction column: {col}")
    if "sender_id" not in df.columns:
        df["sender_id"] = "unknown_sender"
    if "recipient_id" not in df.columns:
        df["recipient_id"] = "unknown_recipient"
    df["transaction_id"] = df["transaction_id"].astype(str)
    df["sender_id"] = df["sender_id"].astype(str)
    df["recipient_id"] = df["recipient_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df.sort_values("event_time").reset_index(drop=True)


def normalize_records(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)
