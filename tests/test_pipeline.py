from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mirror.agents.agents import CommsRiskAgent
from mirror.data.loaders import load_modalities
from mirror.features.builders import build_feature_matrix
from mirror.pipeline import run_pipeline
from mirror.types import LLMBudget, PipelineContext


def _make_dataset(root: Path, prefix: str = "t") -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "transaction_id": [f"{prefix}1", f"{prefix}2", f"{prefix}3"],
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
    d = tmp_path / "sample"
    _make_dataset(d)
    data = load_modalities(d, config={"run": {"audio_enabled": False}})
    data["linked_transactions"] = data["transactions"]
    feats = build_feature_matrix(data)
    assert "log_amount" in feats.columns
    assert len(feats) == 3


def test_run_pipeline_and_submission(tmp_path: Path):
    train = tmp_path / "scenario - train"
    eval_dir = tmp_path / "scenario - eval"
    _make_dataset(train, "tr")
    _make_dataset(eval_dir, "ev")
    out = tmp_path / "out"
    cfg = {
        "run": {
            "random_seed": 42,
            "llm_enabled": False,
            "audio_enabled": False,
            "parallel_agents": True,
            "output_submission_name": "submission.txt",
        },
        "thresholding": {"min_suspect_fraction": 0.01, "max_suspect_fraction": 0.5, "target_suspect_fraction": 0.2},
        "llm": {"timeout_seconds": 1, "max_retries": 0},
    }
    result = run_pipeline(str(train), str(eval_dir), str(out), cfg)
    sub = Path(result["submission"])
    assert sub.exists()
    assert sub.read_text(encoding="ascii").strip() != ""
    assert (out / "submission.report.json").exists()
    assert (out / "cases.parquet").exists()
    assert (out / "patterns.json").exists()
    assert (out / "diagnostics.json").exists()
    assert (out / "config.snapshot.json").exists()


def test_missing_modalities_no_crash(tmp_path: Path):
    d = tmp_path / "minimal"
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
    data = load_modalities(d, config={"run": {"audio_enabled": False}})
    assert data["sms"].empty
    assert data["mails"].empty


def _pipeline_cfg(parallel_agents: bool) -> dict:
    return {
        "run": {
            "random_seed": 42,
            "llm_enabled": False,
            "disable_llm": True,
            "audio_enabled": False,
            "parallel_agents": parallel_agents,
            "max_agent_workers": 4,
            "output_submission_name": "submission.txt",
        },
        "thresholding": {
            "min_suspect_fraction": 0.01,
            "max_suspect_fraction": 0.8,
            "target_suspect_fraction": 0.4,
            "suspect_rate_lower_bound": 0.01,
            "suspect_rate_upper_bound": 0.9,
        },
        "llm": {"timeout_seconds": 1, "max_retries": 0},
    }


def test_serial_parallel_structural_equivalence(tmp_path: Path):
    train = tmp_path / "scenario - train"
    eval_dir = tmp_path / "scenario - eval"
    _make_dataset(train, "tr")
    _make_dataset(eval_dir, "ev")

    out_serial = tmp_path / "out_serial"
    out_parallel = tmp_path / "out_parallel"
    serial = run_pipeline(str(train), str(eval_dir), str(out_serial), _pipeline_cfg(parallel_agents=False))
    parallel = run_pipeline(str(train), str(eval_dir), str(out_parallel), _pipeline_cfg(parallel_agents=True))

    serial_diag = serial["diagnostics"]
    parallel_diag = parallel["diagnostics"]
    assert set(serial_diag["per_agent_rows"].keys()) == set(parallel_diag["per_agent_rows"].keys())
    assert serial_diag["per_agent_rows"] == parallel_diag["per_agent_rows"]
    assert serial_diag["orchestrator"]["parallel_agents"] is False
    assert parallel_diag["orchestrator"]["parallel_agents"] is True


def test_stage_dependencies_and_output_order(tmp_path: Path):
    train = tmp_path / "scenario - train"
    eval_dir = tmp_path / "scenario - eval"
    _make_dataset(train, "tr")
    _make_dataset(eval_dir, "ev")
    out = tmp_path / "out"
    result = run_pipeline(str(train), str(eval_dir), str(out), _pipeline_cfg(parallel_agents=True))
    trace = result["diagnostics"]["profile"]["agent_stages_seconds"]
    assert list(trace.keys()) == ["stage_0_profile", "stage_1_risk", "stage_2_synthesis", "stage_3_decision"]


def test_llm_budget_not_exceeded_parallel():
    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt: str, model: str) -> str:
            self.calls += 1
            return json.dumps({"scam_probability": 0.9, "urgency_score": 0.9})

    tx = pd.DataFrame({"transaction_id": [f"t{i}" for i in range(6)], "sender_id": [f"u{i}" for i in range(6)], "burst_30m": [1] * 6})
    sms = pd.DataFrame({"user_id": [f"u{i}" for i in range(6)], "text": ["urgent verify account wire"] * 6})
    fake = _FakeLLM()
    llm_budget = LLMBudget(max_calls=2, max_strong_calls=1)
    ctx = PipelineContext(
        scenario_name="x",
        input_dir=".",
        output_dir=".",
        config={
            "run": {"llm_enabled": True, "disable_llm": False, "max_messages_for_llm_review": 6, "max_llm_workers": 3},
            "llm": {"max_messages_for_llm_review": 6},
        },
        data={"sms": sms, "mails": pd.DataFrame(), "audio": pd.DataFrame(), "llm_client": fake, "llm_budget": llm_budget},
        features={"matrix": tx},
    )
    result = CommsRiskAgent().run(ctx)
    assert fake.calls <= 2
    assert llm_budget.usage()["calls"] <= 2


def test_pipeline_parallel_disabled_mode(tmp_path: Path):
    train = tmp_path / "scenario - train"
    eval_dir = tmp_path / "scenario - eval"
    _make_dataset(train, "tr")
    _make_dataset(eval_dir, "ev")
    out = tmp_path / "out_serial_only"
    result = run_pipeline(str(train), str(eval_dir), str(out), _pipeline_cfg(parallel_agents=False))
    assert result["diagnostics"]["orchestrator"]["parallel_agents"] is False
