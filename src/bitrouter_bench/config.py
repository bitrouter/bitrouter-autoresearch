"""Bench harness configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BenchConfig(BaseModel):
    """Top-level configuration for a benchmark run."""

    bitrouter_url: str = "http://localhost:8787"
    openclaw_bin: str = "openclaw"

    tasks_dir: Path = Path("tasks")
    results_dir: Path = Path("results/runs")
    openclaw_config_dir: Path = Path("configs")

    conditions: list[str] = Field(
        default=["bitrouter_auto", "direct_opus"],
    )
    repeats: int = 3
    default_max_turns: int = 20

    # Budget defaults per difficulty (USD)
    budget_easy: float = 0.05
    budget_medium: float = 0.10
    budget_hard: float = 0.20

    # Models used by the harness itself (NOT through BitRouter)
    judge_model: str = "claude-sonnet-4-6-20250514"
    user_agent_model: str = "claude-sonnet-4-6-20250514"

    # Scoring weights
    weight_completion: float = 0.5
    weight_interaction: float = 0.3
    weight_efficiency: float = 0.2


def load_config(path: Path | None = None) -> BenchConfig:
    """Load config from YAML file, falling back to defaults."""
    if path is None:
        path = Path("configs/bench.yaml")
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return BenchConfig(**data)
    return BenchConfig()


def budget_for_difficulty(config: BenchConfig, difficulty: str) -> float:
    """Return the dollar budget for a given difficulty tier."""
    return {
        "easy": config.budget_easy,
        "medium": config.budget_medium,
        "hard": config.budget_hard,
    }.get(difficulty, config.budget_medium)
