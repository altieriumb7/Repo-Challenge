from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .schemas import normalize_records, normalize_transactions

SUPPORTED_OPTIONAL = ("users", "locations", "sms", "mails")


def _load_json(path: Path, modality: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("items", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path.name} must contain a JSON list or an object with an 'items' list.")
    return normalize_records(rows, modality=modality)


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
        return pd.DataFrame(
            {
                "audio_path": audio_paths[:max_files],
                "transcript": ["" for _ in audio_paths[:max_files]],
                "transcription_error": [True for _ in audio_paths[:max_files]],
            }
        )
    return pd.DataFrame(rows)


def _validate_contract(root: Path, transactions: pd.DataFrame, optional_modalities: dict[str, pd.DataFrame]) -> None:
    if transactions.empty:
        raise ValueError(f"transactions.csv in {root} is empty after normalization.")
    for modality, frame in optional_modalities.items():
        if frame.empty:
            continue
        if modality in {"sms", "mails"}:
            expected_text_cols = {"text", "body", "transcript"}
            if not any(c in frame.columns for c in expected_text_cols):
                raise ValueError(
                    f"{modality}.json is present but does not contain a supported text column ({sorted(expected_text_cols)})."
                )


def load_modalities(input_dir: str | Path, config: dict | None = None) -> dict[str, pd.DataFrame]:
    config = config or {}
    root = Path(input_dir)
    tx_path = root / "transactions.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"transactions.csv not found in {root}")

    transactions = normalize_transactions(pd.read_csv(tx_path))
    optional = {name: _load_json(root / f"{name}.json", name) for name in SUPPORTED_OPTIONAL}

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

    _validate_contract(root, transactions, optional)

    return {
        "transactions": transactions,
        "users": optional["users"],
        "locations": optional["locations"],
        "sms": optional["sms"],
        "mails": optional["mails"],
        "audio": audio_df,
    }
