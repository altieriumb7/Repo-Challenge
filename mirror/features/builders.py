from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_transaction_features(tx: pd.DataFrame) -> pd.DataFrame:
    f = tx[["transaction_id", "sender_id", "recipient_id", "event_time", "amount"]].copy()
    f["log_amount"] = np.log1p(f["amount"].clip(lower=0))
    f["hour"] = f["event_time"].dt.hour.fillna(0).astype(int)
    f["dow"] = f["event_time"].dt.dayofweek.fillna(0).astype(int)
    f["off_hours"] = ((f["hour"] < 6) | (f["hour"] > 22)).astype(int)
    f["sender_tx_count"] = f.groupby("sender_id").cumcount() + 1
    f["recipient_tx_count"] = f.groupby("recipient_id").cumcount() + 1
    f["is_new_recipient_for_sender"] = (
        f.groupby(["sender_id", "recipient_id"]).cumcount() == 0
    ).astype(int)
    sender_mean = f.groupby("sender_id")["amount"].transform("mean").replace(0, 1)
    f["sender_amount_zproxy"] = (f["amount"] - sender_mean) / sender_mean
    last_tx_time = f.groupby("sender_id")["event_time"].shift(1)
    recency_minutes = (f["event_time"] - last_tx_time).dt.total_seconds() / 60.0
    f["recency_min"] = recency_minutes.fillna(1e6).clip(lower=0)
    f["burst_30m"] = (f["recency_min"] <= 30).astype(int)
    return f


def build_geo_features(tx: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    out = tx[["transaction_id"]].copy()
    out["geo_risk"] = 0.0
    if locations.empty or "user_id" not in locations.columns:
        return out
    loc = locations.copy()
    for c in ["lat", "lon"]:
        if c not in loc.columns:
            loc[c] = 0.0
    tx_loc = tx.merge(loc[["user_id", "lat", "lon"]], left_on="sender_id", right_on="user_id", how="left")
    out["geo_risk"] = tx_loc[["lat", "lon"]].isna().any(axis=1).astype(float)
    return out


def build_comms_features(tx: pd.DataFrame, sms: pd.DataFrame, mails: pd.DataFrame) -> pd.DataFrame:
    suspicious_tokens = ["urgent", "verify", "password", "wire", "gift card", "click", "otp"]

    def _score(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["user_id", "comms_score"])
        text_col = "text" if "text" in df.columns else ("body" if "body" in df.columns else None)
        user_col = "user_id" if "user_id" in df.columns else ("sender_id" if "sender_id" in df.columns else None)
        if text_col is None or user_col is None:
            return pd.DataFrame(columns=["user_id", "comms_score"])
        s = df[text_col].fillna("").str.lower()
        score = sum(s.str.contains(tok).astype(int) for tok in suspicious_tokens)
        return pd.DataFrame({"user_id": df[user_col].astype(str), "comms_score": score}).groupby("user_id", as_index=False).mean()

    sms_s = _score(sms)
    mail_s = _score(mails)
    comb = pd.concat([sms_s, mail_s], ignore_index=True)
    if not comb.empty:
        comb = comb.groupby("user_id", as_index=False)["comms_score"].mean()
    out = tx[["transaction_id", "sender_id"]].merge(comb, left_on="sender_id", right_on="user_id", how="left")
    out["comms_score"] = out["comms_score"].fillna(0.0)
    return out[["transaction_id", "comms_score"]]


def build_graph_features(tx: pd.DataFrame) -> pd.DataFrame:
    g = nx.DiGraph()
    for row in tx.itertuples(index=False):
        g.add_edge(str(row.sender_id), str(row.recipient_id))
    out = tx[["transaction_id", "sender_id", "recipient_id"]].copy()
    out["sender_out_degree"] = out["sender_id"].map(lambda n: g.out_degree(str(n)))
    out["recipient_in_degree"] = out["recipient_id"].map(lambda n: g.in_degree(str(n)))
    out["edge_novelty"] = (tx.groupby(["sender_id", "recipient_id"]).cumcount() == 0).astype(int)
    return out[["transaction_id", "sender_out_degree", "recipient_in_degree", "edge_novelty"]]


def build_feature_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    linked = data["linked_transactions"]
    feat = build_transaction_features(linked)
    feat = feat.merge(build_geo_features(linked, data.get("locations", pd.DataFrame())), on="transaction_id", how="left")
    feat = feat.merge(build_comms_features(linked, data.get("sms", pd.DataFrame()), data.get("mails", pd.DataFrame())), on="transaction_id", how="left")
    feat = feat.merge(build_graph_features(linked), on="transaction_id", how="left")
    return feat.fillna(0)
