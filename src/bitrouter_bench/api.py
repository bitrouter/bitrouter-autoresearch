"""Read-only FastAPI backend for live trial observation.

All endpoints are GET-only. The website cannot start or modify trials.
SSE stream at /api/trials/live/stream pushes events as they happen.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from bitrouter_bench.config import BenchConfig, budget_for_difficulty, load_config
from bitrouter_bench.events import event_bus
from bitrouter_bench.task_loader import discover_tasks

logger = logging.getLogger(__name__)


def create_app(config: BenchConfig | None = None) -> FastAPI:
    """Create the FastAPI application with all routes."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title="BitRouter Bench",
        description="Read-only API for observing benchmark trials",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "https://*.vercel.app",
        ],
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    app.state.config = config

    # ── Tasks ──────────────────────────────────────────────────────

    @app.get("/api/tasks")
    async def list_tasks(difficulty: str | None = None) -> list[dict]:
        """List all benchmark tasks."""
        tasks = discover_tasks(config.tasks_dir, difficulty)
        return [
            {
                "task_id": t.task_id,
                "name": t.meta.name,
                "description": t.meta.description,
                "difficulty": t.meta.difficulty,
                "category": t.meta.category,
                "budget_usd": t.meta.budget_usd
                or budget_for_difficulty(config, t.meta.difficulty),
                "max_turns": t.meta.max_turns,
            }
            for t in tasks
        ]

    @app.get("/api/tasks/{difficulty}/{name}")
    async def get_task(difficulty: str, name: str) -> dict:
        """Get a single task with full scenario (excluding eval criteria)."""
        task_id = f"{difficulty}/{name}"
        tasks = discover_tasks(config.tasks_dir)
        task = next((t for t in tasks if t.task_id == task_id), None)
        if task is None:
            raise HTTPException(404, f"Task not found: {task_id}")
        return {
            "task_id": task.task_id,
            "name": task.meta.name,
            "description": task.meta.description,
            "difficulty": task.meta.difficulty,
            "category": task.meta.category,
            "budget_usd": task.meta.budget_usd
            or budget_for_difficulty(config, task.meta.difficulty),
            "max_turns": task.meta.max_turns,
            "scenario": {
                "persona": task.scenario.persona,
                "known_info": task.scenario.known_info,
                "instructions": task.scenario.instructions,
                # Note: unknown_info excluded — it's for the UserAgent only
            },
        }

    # ── Trials (historical) ───────────────────────────────────────

    @app.get("/api/trials")
    async def list_trials(
        condition: str | None = None,
        task_id: str | None = None,
    ) -> list[dict]:
        """List all completed trials with metadata."""
        runs_dir = config.results_dir
        if not runs_dir.exists():
            return []

        trials: list[dict] = []
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            meta_path = run_dir / "metadata.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            if condition and meta.get("condition") != condition:
                continue
            if task_id and meta.get("task_id") != task_id:
                continue

            verdict_path = run_dir / "verdict.json"
            verdict = None
            if verdict_path.exists():
                with open(verdict_path) as f:
                    verdict = json.load(f)

            trials.append({**meta, "verdict": verdict})

        return trials

    @app.get("/api/trials/{trial_id}")
    async def get_trial(trial_id: str) -> dict:
        """Get a single trial's metadata and verdict."""
        trial_path = config.results_dir / trial_id
        if not trial_path.is_dir():
            raise HTTPException(404, f"Trial not found: {trial_id}")

        meta_path = trial_path / "metadata.json"
        if not meta_path.exists():
            raise HTTPException(404, f"Trial metadata not found: {trial_id}")

        with open(meta_path) as f:
            meta = json.load(f)

        verdict = None
        verdict_path = trial_path / "verdict.json"
        if verdict_path.exists():
            with open(verdict_path) as f:
                verdict = json.load(f)

        return {**meta, "verdict": verdict}

    @app.get("/api/trials/{trial_id}/trajectory")
    async def get_trajectory(trial_id: str) -> list[dict]:
        """Get the full conversation transcript for a trial."""
        traj_path = config.results_dir / trial_id / "trajectory.jsonl"
        if not traj_path.exists():
            raise HTTPException(404, f"Trajectory not found: {trial_id}")

        turns: list[dict] = []
        with open(traj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    turns.append(json.loads(line))
        return turns

    # ── Live stream (SSE) ─────────────────────────────────────────

    @app.get("/api/trials/live/stream")
    async def live_stream() -> EventSourceResponse:
        """SSE stream of the currently running trial.

        Events: turn, cost, status, verdict, done
        Late joiners receive replay of all events so far.
        """

        async def generate():
            if event_bus.active_trial_id is None:
                yield {
                    "event": "status",
                    "data": json.dumps({"status": "idle", "message": "No trial running"}),
                }
                return

            yield {
                "event": "status",
                "data": json.dumps({
                    "status": "running",
                    "trial_id": event_bus.active_trial_id,
                }),
            }

            async for event in event_bus.subscribe():
                yield {
                    "event": event.type,
                    "data": json.dumps({
                        "trial_id": event.trial_id,
                        "timestamp": event.timestamp,
                        **event.data,
                    }),
                }

        return EventSourceResponse(generate())

    # ── Live status ───────────────────────────────────────────────

    @app.get("/api/live/status")
    async def live_status() -> dict:
        """Check if a trial is currently running."""
        return {
            "active": event_bus.active_trial_id is not None,
            "trial_id": event_bus.active_trial_id,
        }

    # ── Results ───────────────────────────────────────────────────

    @app.get("/api/results")
    async def get_results() -> list[dict]:
        """Aggregate results across all judged trials."""
        runs_dir = config.results_dir
        if not runs_dir.exists():
            return []

        rows: list[dict] = []
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            meta_path = run_dir / "metadata.json"
            verdict_path = run_dir / "verdict.json"
            if not meta_path.exists() or not verdict_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)
            with open(verdict_path) as f:
                verdict = json.load(f)

            rows.append({
                "task_id": meta["task_id"],
                "condition": meta["condition"],
                "composite_score": verdict["composite_score"],
                "completion_score": verdict["completion_score"],
                "interaction_quality": verdict["interaction_quality"],
                "resource_efficiency": verdict["resource_efficiency"],
                "cost_usd": meta.get("total_cost_usd", 0),
                "turn_count": meta.get("turn_count", 0),
                "stop_reason": meta.get("stop_reason", ""),
            })

        return rows

    @app.get("/api/results/compare")
    async def compare_results() -> dict:
        """Statistical comparison between conditions."""
        from analysis.aggregate import aggregate, compare_conditions

        rows = aggregate(config.results_dir)
        if not rows:
            return {"conditions": {}, "comparison": None}
        return compare_conditions(rows)

    return app
