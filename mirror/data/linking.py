from __future__ import annotations

import pandas as pd


def link_entities(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tx = data["transactions"].copy()
    users = data.get("users", pd.DataFrame()).copy()

    if not users.empty:
        if "user_id" in users.columns:
            users["user_id"] = users["user_id"].astype(str)
            keep_cols = [c for c in ["user_id", "job", "salary", "residence"] if c in users.columns]
            tx = tx.merge(users[keep_cols], left_on="sender_id", right_on="user_id", how="left")

    tx["first_seen_sender"] = tx.groupby("sender_id")["event_time"].transform("min")
    tx["first_seen_recipient"] = tx.groupby("recipient_id")["event_time"].transform("min")
    return tx
