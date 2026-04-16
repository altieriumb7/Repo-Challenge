from __future__ import annotations

import json
from pathlib import Path

import typer

from mirror.pipeline import run_pipeline
from mirror.utils.config import load_config

app = typer.Typer(help="Reply Mirror agent-based fraud pipeline")


def _resolved_cfg(config: str, scenario_name: str) -> dict:
    cfg = load_config(config, scenario_name=scenario_name)
    cfg.setdefault("run", {})["scenario_name"] = scenario_name
    return cfg


def _scenario_slug(name: str) -> str:
    return name.lower().replace(" - train", "").replace(" - eval", "").replace(" ", "_")


@app.command("run-scenario")
def run_scenario(
    train_dir: str,
    eval_dir: str,
    name: str,
    output_dir: str = "outputs",
    config: str = "configs/default.yaml",
) -> None:
    out = Path(output_dir) / _scenario_slug(name)
    result = run_pipeline(train_dir=train_dir, eval_dir=eval_dir, output_dir=str(out), config=_resolved_cfg(config, name))
    typer.echo(json.dumps(result, indent=2))


@app.command("run-all")
def run_all(root_dir: str, output_root: str = "outputs", config: str = "configs/default.yaml") -> None:
    root = Path(root_dir)
    train_dirs = sorted([p for p in root.glob("*train") if p.is_dir() and (p / "transactions.csv").exists()])
    results = {}
    for train in train_dirs:
        scenario = train.name.replace(" - train", "")
        eval_dir = root / f"{scenario} - eval"
        if not eval_dir.exists():
            continue
        out_dir = Path(output_root) / _scenario_slug(scenario)
        results[scenario] = run_pipeline(
            train_dir=str(train),
            eval_dir=str(eval_dir),
            output_dir=str(out_dir),
            config=_resolved_cfg(config, scenario),
        )
    typer.echo(json.dumps(results, indent=2))


@app.command("backtest")
def backtest(train_dir: str, config: str = "configs/default.yaml", output_dir: str = "outputs/backtest") -> None:
    out = run_pipeline(train_dir=train_dir, eval_dir=train_dir, output_dir=output_dir, config=_resolved_cfg(config, Path(train_dir).name))
    typer.echo(f"Backtest diagnostics: {json.dumps(out['diagnostics'], indent=2)}")


@app.command("make-submission")
def make_submission(train_dir: str, eval_dir: str, output: str, config: str = "configs/default.yaml") -> None:
    scenario = Path(eval_dir).name.replace(" - eval", "")
    output_dir = str(Path(output).parent)
    cfg = _resolved_cfg(config, scenario)
    cfg.setdefault("run", {})["output_submission_name"] = Path(output).name
    out = run_pipeline(train_dir=train_dir, eval_dir=eval_dir, output_dir=output_dir, config=cfg)
    typer.echo(f"Submission generated at {out['submission']}")


@app.command("compare-runs")
def compare_runs(registry_csv: str) -> None:
    lines = Path(registry_csv).read_text(encoding="utf-8")
    typer.echo(lines)
