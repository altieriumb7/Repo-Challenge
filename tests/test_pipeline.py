from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mirror.data.loaders import load_modalities
from mirror.features.builders import build_feature_matrix
from mirror.pipeline import run_pipeline


def _make_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3"],
            "event_time": ["2026-01-01T10:00:00Z", "2026-01-01T10:05:00Z", "2026-01-02T23:10:00Z"],
            "amount": [10, 5000, 35],
            "sender_id": ["u1", "u1", "u2"],
            "recipient_id": ["r1", "r2", "r1"],
        }
    ).to_csv(root / "transactions.csv", index=False)
    (root / "users.json").write_text(json.dumps([{"user_id": "u1", "job": "engineer", "salary": 100000, "residence": "X"}]))
    (root / "locations.json").write_text(json.dumps([{"user_id": "u1", "lat": 1.0, "lon": 2.0}]))
    (root / "sms.json").write_text(json.dumps([{"user_id": "u1", "text": "urgent verify otp now"}]))
    (root / "mails.json").write_text(json.dumps([]))


def test_loaders_and_features(tmp_path: Path):
    d = tmp_path / "Deus Ex - train"
    _make_dataset(d)
    data = load_modalities(d)
    data["linked_transactions"] = data["transactions"]
    feats = build_feature_matrix(data)
    assert "log_amount" in feats.columns
    assert len(feats) == 3


def test_run_pipeline_and_submission(tmp_path: Path):
    d = tmp_path / "Brave New World - train"
    _make_dataset(d)
    out = tmp_path / "out"
    cfg = {
        "run": {"random_seed": 42, "llm_enabled": False, "audio_enabled": False, "output_submission_name": "submission.txt"},
        "thresholding": {"min_suspect_fraction": 0.01, "max_suspect_fraction": 0.5, "target_suspect_fraction": 0.2},
        "llm": {"timeout_seconds": 1, "max_retries": 0},
    }
    result = run_pipeline(str(d), str(out), cfg)
    sub = Path(result["submission"])
    assert sub.exists()
    assert sub.read_text(encoding="ascii").strip() != ""


def test_missing_modalities_no_crash(tmp_path: Path):
    d = tmp_path / "The Truman Show - train"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "transaction_id": ["x1"],
            "event_time": ["2026-01-01T00:00:00Z"],
            "amount": [1.0],
            "sender_id": ["a"],
            "recipient_id": ["b"],
        }
    ).to_csv(d / "transactions.csv", index=False)
    data = load_modalities(d)
    assert data["sms"].empty
    assert data["mails"].empty
