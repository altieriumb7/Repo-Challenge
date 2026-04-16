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


def _safe_transcribe(audio_paths: list[str], max_files: int) -> pd.DataFrame:
    rows = []
    try:
        import whisper  # type: ignore

        model = whisper.load_model("tiny")
        for path in audio_paths[:max_files]:
            try:
                out = model.transcribe(path)
                rows.append({"audio_path": path, "transcript": out.get("text", "")})
            except Exception:
                rows.append({"audio_path": path, "transcript": "", "transcription_error": True})
    except Exception:
        return pd.DataFrame({"audio_path": audio_paths[:max_files], "transcript": ["" for _ in audio_paths[:max_files]], "transcription_error": [True for _ in audio_paths[:max_files]]})
    return pd.DataFrame(rows)


def load_modalities(input_dir: str | Path, config: dict | None = None) -> dict[str, pd.DataFrame]:
    config = config or {}
    root = Path(input_dir)
    tx_path = root / "transactions.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"transactions.csv not found in {root}")

    transactions = normalize_transactions(pd.read_csv(tx_path))
    users = _load_json(root / "users.json")
    locations = _load_json(root / "locations.json")
    sms = _load_json(root / "sms.json")
    mails = _load_json(root / "mails.json")

    run_cfg = config.get("run", {})
    audio_df = pd.DataFrame()
    if run_cfg.get("audio_enabled", False) and not run_cfg.get("disable_audio", False):
        audio_dir = root / "audio"
        files = sorted(audio_dir.glob("*.mp3")) if audio_dir.exists() else []
        paths = [str(p) for p in files]
        max_files = int(run_cfg.get("max_audio_files_to_transcribe", 10))
        if run_cfg.get("transcribe_audio", False):
            audio_df = _safe_transcribe(paths, max_files=max_files)
        else:
            audio_df = pd.DataFrame({"audio_path": paths[:max_files]})

    return {
        "transactions": transactions,
        "users": users,
        "locations": locations,
        "sms": sms,
        "mails": mails,
        "audio": audio_df,
    }
