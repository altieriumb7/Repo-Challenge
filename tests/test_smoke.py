from __future__ import annotations

from pathlib import Path

from mirror.pipeline import run_pipeline


def test_smoke_tiny_fixture(tmp_path: Path):
    train = Path("tests/fixtures/tiny_scenario/train")
    eval_dir = Path("tests/fixtures/tiny_scenario/eval")
    out = tmp_path / "smoke"
    cfg = {
        "run": {"llm_enabled": False, "disable_llm": True, "audio_enabled": False, "scenario_name": "tiny"},
        "thresholding": {
            "min_suspect_fraction": 0.01,
            "max_suspect_fraction": 0.8,
            "target_suspect_fraction": 0.4,
            "suspect_rate_lower_bound": 0.01,
            "suspect_rate_upper_bound": 0.9,
        },
        "llm": {"timeout_seconds": 1, "max_retries": 0},
    }
    result = run_pipeline(str(train), str(eval_dir), str(out), cfg)
    assert Path(result["submission"]).exists()
    assert (out / "submission.report.json").exists()
    assert (out / "diagnostics.json").exists()
    assert (out / "environment.snapshot.json").exists()
