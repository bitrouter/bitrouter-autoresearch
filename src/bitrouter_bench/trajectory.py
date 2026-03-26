"""Trajectory models and JSONL storage for trial recordings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """A single turn in the conversation."""

    turn_number: int
    role: str  # "user" or "agent"
    content: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    openclaw_raw: dict | None = None  # raw JSON from OpenClaw (agent turns)
    cost_snapshot_usd: float | None = None  # cumulative cost at this point


class Trajectory(BaseModel):
    """Complete record of a single trial."""

    trial_id: str  # "{timestamp}_{task_id_slug}_{condition}"
    task_id: str
    condition: str  # "bitrouter_auto" or "direct_opus"
    turns: list[Turn] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    stop_reason: str = ""  # "user_stop", "budget_exceeded", "max_turns", "timeout", "error"
    started_at: str = ""
    ended_at: str = ""
    metrics_before: dict = Field(default_factory=dict)
    metrics_after: dict = Field(default_factory=dict)


def make_trial_id(task_id: str, condition: str) -> str:
    """Generate a unique trial ID from task and condition."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = task_id.replace("/", "_").replace(" ", "-")
    return f"{ts}_{slug}_{condition}"


def trial_dir(results_dir: Path, trial_id: str) -> Path:
    """Return the directory path for a trial's output."""
    return results_dir / trial_id


def save_turn(output_dir: Path, turn: Turn) -> None:
    """Append a single turn to trajectory.jsonl (streaming write)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "trajectory.jsonl"
    with open(path, "a") as f:
        f.write(turn.model_dump_json() + "\n")


def save_metadata(output_dir: Path, trajectory: Trajectory) -> None:
    """Write trial metadata (everything except individual turns)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "trial_id": trajectory.trial_id,
        "task_id": trajectory.task_id,
        "condition": trajectory.condition,
        "total_cost_usd": trajectory.total_cost_usd,
        "stop_reason": trajectory.stop_reason,
        "started_at": trajectory.started_at,
        "ended_at": trajectory.ended_at,
        "turn_count": len(trajectory.turns),
        "metrics_before": trajectory.metrics_before,
        "metrics_after": trajectory.metrics_after,
    }
    path = output_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def load_trajectory(output_dir: Path) -> Trajectory:
    """Load a trajectory from its output directory."""
    meta_path = output_dir / "metadata.json"
    traj_path = output_dir / "trajectory.jsonl"

    with open(meta_path) as f:
        meta = json.load(f)

    turns: list[Turn] = []
    if traj_path.exists():
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(Turn.model_validate_json(line))

    return Trajectory(
        trial_id=meta["trial_id"],
        task_id=meta["task_id"],
        condition=meta["condition"],
        turns=turns,
        total_cost_usd=meta.get("total_cost_usd", 0.0),
        stop_reason=meta.get("stop_reason", ""),
        started_at=meta.get("started_at", ""),
        ended_at=meta.get("ended_at", ""),
        metrics_before=meta.get("metrics_before", {}),
        metrics_after=meta.get("metrics_after", {}),
    )
