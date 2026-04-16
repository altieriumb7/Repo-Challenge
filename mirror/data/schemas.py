from __future__ import annotations

from typing import Any

import pandas as pd

TRANSACTION_RENAME = {
    "id": "transaction_id",
    "tx_id": "transaction_id",
    "transactionId": "transaction_id",
    "timestamp": "event_time",
    "time": "event_time",
    "datetime": "event_time",
    "sender": "sender_id",
    "senderId": "sender_id",
    "recipient": "recipient_id",
    "recipientId": "recipient_id",
    "user_id": "sender_id",
    "value": "amount",
}

OPTIONAL_MODALITY_RENAMES = {
    "users": {"id": "user_id", "uid": "user_id"},
    "locations": {"uid": "user_id", "timestamp": "event_time", "lng": "lon", "longitude": "lon", "latitude": "lat"},
    "sms": {"message": "text", "content": "text", "uid": "user_id"},
    "mails": {"message": "body", "content": "body", "uid": "user_id", "email_body": "body"},
}


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in TRANSACTION_RENAME.items() if k in df.columns}).copy()
    required = ["transaction_id", "event_time", "amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"transactions.csv missing required columns {missing}. Expected aliases include: {sorted(TRANSACTION_RENAME)}"
        )
    if "sender_id" not in df.columns:
        df["sender_id"] = "unknown_sender"
    if "recipient_id" not in df.columns:
        df["recipient_id"] = "unknown_recipient"

    df["transaction_id"] = df["transaction_id"].astype(str)
    df["sender_id"] = df["sender_id"].astype(str)
    df["recipient_id"] = df["recipient_id"].astype(str)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    if df["event_time"].isna().all():
        raise ValueError("transactions.csv has no parseable event_time values.")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if df["amount"].isna().all():
        raise ValueError("transactions.csv has no parseable amount values.")
    df["amount"] = df["amount"].fillna(0.0)
    return df.sort_values("event_time").reset_index(drop=True)


def normalize_records(rows: list[dict[str, Any]], modality: str | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = pd.json_normalize(rows)
    if modality and modality in OPTIONAL_MODALITY_RENAMES:
        out = out.rename(columns={k: v for k, v in OPTIONAL_MODALITY_RENAMES[modality].items() if k in out.columns})
    if "event_time" in out.columns:
        out["event_time"] = pd.to_datetime(out["event_time"], utc=True, errors="coerce")
    return out
