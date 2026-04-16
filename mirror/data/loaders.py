from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schemas import normalize_records, normalize_transactions


def _load_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("items", [])
    return normalize_records(rows)


def load_modalities(input_dir: str | Path, audio_enabled: bool = False) -> dict[str, pd.DataFrame]:
    root = Path(input_dir)
    tx_path = root / "transactions.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"transactions.csv not found in {root}")

    transactions = normalize_transactions(pd.read_csv(tx_path))
    users = _load_json(root / "users.json")
    locations = _load_json(root / "locations.json")
    sms = _load_json(root / "sms.json")
    mails = _load_json(root / "mails.json")

    audio_df = pd.DataFrame()
    if audio_enabled:
        audio_dir = root / "audio"
        files = sorted(audio_dir.glob("*.mp3")) if audio_dir.exists() else []
        audio_df = pd.DataFrame({"audio_path": [str(p) for p in files]})

    return {
        "transactions": transactions,
        "users": users,
        "locations": locations,
        "sms": sms,
        "mails": mails,
        "audio": audio_df,
    }
