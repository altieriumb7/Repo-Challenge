from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .schemas import TEXTUAL_COLUMN_ALIASES, normalize_records, normalize_transactions

SUPPORTED_OPTIONAL = ("users", "locations", "sms", "mails")
LOGGER = logging.getLogger(__name__)


def _load_json(path: Path, modality: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("items", [])
    if not isinstance(rows, list):
        raise ValueError(f"{path.name} must contain a JSON list or an object with an 'items' list.")
    return normalize_records(rows, modality=modality)


def _infer_user_id_from_audio_path(path: str) -> str:
    stem = Path(path).stem
    match = re.match(r"([A-Za-z0-9]+)[_-].*", stem)
    return match.group(1) if match else "unknown_user"


def _safe_transcribe(audio_paths: list[str], max_files: int, max_workers: int = 2) -> tuple[pd.DataFrame, dict]:
    rows = []
    subset = audio_paths[:max_files]
    try:
        import whisper  # type: ignore

        model = whisper.load_model("tiny")

        def _transcribe_one(path: str) -> dict:
            try:
                out = model.transcribe(path)
                return {"audio_path": path, "text": out.get("text", ""), "user_id": _infer_user_id_from_audio_path(path)}
            except Exception:
                return {"audio_path": path, "text": "", "user_id": _infer_user_id_from_audio_path(path), "transcription_error": True}

        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(subset) or 1))) as pool:
            futures = {pool.submit(_transcribe_one, path): idx for idx, path in enumerate(subset)}
            ordered = [None for _ in subset]
            for fut in as_completed(futures):
                ordered[futures[fut]] = fut.result()
        rows = [r for r in ordered if r is not None]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Audio transcription unavailable: %s", exc)
        return (
            pd.DataFrame(
                {
                    "audio_path": subset,
                    "text": ["" for _ in subset],
                    "user_id": [_infer_user_id_from_audio_path(path) for path in subset],
                    "transcription_error": [True for _ in subset],
                }
            ),
            {"backend_available": False, "warning": f"transcription unavailable: {exc}"},
        )
    return pd.DataFrame(rows), {"backend_available": True, "warning": ""}


def _validate_contract(root: Path, transactions: pd.DataFrame, optional_modalities: dict[str, pd.DataFrame]) -> None:
    if transactions.empty:
        raise ValueError(f"transactions.csv in {root} is empty after normalization.")
    for modality, frame in optional_modalities.items():
        if frame.empty:
            continue
        if modality in {"sms", "mails"}:
            expected_text_cols = set(TEXTUAL_COLUMN_ALIASES)
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
    audio_dir = root / "audio"
    files = sorted(audio_dir.glob("*.mp3")) if audio_dir.exists() else []
    paths = [str(p) for p in files]
    modality_diag: dict[str, object] = {
        "sms_loaded": not optional["sms"].empty,
        "mails_loaded": not optional["mails"].empty,
        "audio_detected": bool(files),
        "audio_enabled": bool(run_cfg.get("audio_enabled", False)),
        "transcribe_audio": bool(run_cfg.get("transcribe_audio", False)),
        "audio_files_detected": len(files),
        "audio_files_considered": 0,
        "audio_files_transcribed": 0,
        "transcripts_produced": False,
        "audio_warning": "",
    }
    if run_cfg.get("audio_enabled", False) and not run_cfg.get("disable_audio", False):
        max_files = int(run_cfg.get("max_audio_files_to_transcribe", 10))
        modality_diag["audio_files_considered"] = min(max_files, len(files))
        if run_cfg.get("transcribe_audio", False):
            max_audio_workers = int(run_cfg.get("max_audio_workers", 2))
            audio_df, audio_diag = _safe_transcribe(paths, max_files=max_files, max_workers=max_audio_workers)
            modality_diag["audio_warning"] = audio_diag.get("warning", "")
        else:
            audio_df = pd.DataFrame({"audio_path": paths[:max_files], "user_id": [_infer_user_id_from_audio_path(path) for path in paths[:max_files]]})
            modality_diag["audio_warning"] = "transcription disabled by config"
        modality_diag["audio_files_transcribed"] = int((audio_df.get("text", pd.Series(dtype=str)).astype(str).str.strip() != "").sum()) if "text" in audio_df.columns else 0
        modality_diag["transcripts_produced"] = bool(modality_diag["audio_files_transcribed"])
    elif run_cfg.get("disable_audio", False):
        modality_diag["audio_warning"] = "audio disabled by config"

    _validate_contract(root, transactions, optional)

    return {
        "transactions": transactions,
        "users": optional["users"],
        "locations": optional["locations"],
        "sms": optional["sms"],
        "mails": optional["mails"],
        "audio": audio_df,
        "modality_diagnostics": modality_diag,
    }
