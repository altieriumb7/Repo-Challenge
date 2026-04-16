from __future__ import annotations

import json
from pathlib import Path

import typer

from mirror.pipeline import run_pipeline
from mirror.utils.config import load_config

app = typer.Typer(help="Reply Mirror agent-based fraud pipeline")


def _resolved_cfg(config: str, input_dir: str) -> dict:
    return load_config(config, scenario_name=Path(input_dir).name)


@app.command()
def run(input_dir: str, output_dir: str, config: str = "configs/default.yaml") -> None:
    result = run_pipeline(input_dir, output_dir, _resolved_cfg(config, input_dir))
    typer.echo(json.dumps(result, indent=2))


@app.command()
def inspect(input_dir: str, config: str = "configs/default.yaml") -> None:
    out = run_pipeline(input_dir, output_dir="outputs/inspect", config=_resolved_cfg(config, input_dir))
    typer.echo(json.dumps(out["diagnostics"], indent=2))


@app.command()
def backtest(input_dir: str, output_dir: str, config: str = "configs/default.yaml") -> None:
    out = run_pipeline(input_dir, output_dir=output_dir, config=_resolved_cfg(config, input_dir))
    typer.echo(f"Backtest diagnostics: {json.dumps(out['diagnostics'], indent=2)}")


@app.command("make-submission")
def make_submission(input_dir: str, output: str, config: str = "configs/default.yaml") -> None:
    output_dir = str(Path(output).parent)
    cfg = _resolved_cfg(config, input_dir)
    cfg.setdefault("run", {})["output_submission_name"] = Path(output).name
    out = run_pipeline(input_dir, output_dir=output_dir, config=cfg)
    typer.echo(f"Submission generated at {out['submission']}")
