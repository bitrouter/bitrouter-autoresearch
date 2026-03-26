"""Trial runner and benchmark orchestrator."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bitrouter_bench.config import BenchConfig, budget_for_difficulty
from bitrouter_bench.cost_meter import CostMeter
from bitrouter_bench.openclaw import OpenClawRunner
from bitrouter_bench.task_loader import Task, discover_tasks
from bitrouter_bench.trajectory import (
    Trajectory,
    Turn,
    make_trial_id,
    save_metadata,
    save_turn,
    trial_dir,
)
from bitrouter_bench.events import Event, event_bus
from bitrouter_bench.user_agent import UserAgent

logger = logging.getLogger(__name__)


class TrialRunner:
    """Run a single trial: one task × one condition."""

    def __init__(
        self,
        task: Task,
        condition: str,
        config: BenchConfig,
        live: bool = False,
    ) -> None:
        self._task = task
        self._condition = condition
        self._config = config
        self._live = live
        self._trial_id = ""

    async def _emit(self, event_type: str, data: dict) -> None:
        """Publish an event to the live stream (no-op if --live not set)."""
        if not self._live:
            return
        await event_bus.publish(Event(
            type=event_type,
            data=data,
            trial_id=self._trial_id,
        ))

    async def run(self) -> Trajectory:
        """Execute the trial and return the trajectory."""
        trial_id = make_trial_id(self._task.task_id, self._condition)
        self._trial_id = trial_id
        output = trial_dir(self._config.results_dir, trial_id)
        output.mkdir(parents=True, exist_ok=True)

        budget = (
            self._task.meta.budget_usd
            if self._task.meta.budget_usd is not None
            else budget_for_difficulty(self._config, self._task.meta.difficulty)
        )
        max_turns = self._task.meta.max_turns or self._config.default_max_turns
        session_id = str(uuid.uuid4())

        # Set up workspace isolation
        workspace = Path(tempfile.mkdtemp(prefix=f"bench_{trial_id}_"))

        # Resolve OpenClaw state dir for this condition
        state_dir = self._config.openclaw_config_dir / f"openclaw_{self._condition}"

        openclaw = OpenClawRunner(
            openclaw_bin=self._config.openclaw_bin,
            state_dir=state_dir,
            timeout_seconds=300,
            workspace=workspace,
        )
        user_agent = UserAgent(
            scenario=self._task.scenario,
            model=self._config.user_agent_model,
        )
        cost_meter = CostMeter(self._config.bitrouter_url)

        trajectory = Trajectory(
            trial_id=trial_id,
            task_id=self._task.task_id,
            condition=self._condition,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        if self._live:
            await event_bus.start_trial(trial_id)
        await self._emit("status", {
            "status": "starting",
            "task_id": self._task.task_id,
            "condition": self._condition,
            "budget_usd": budget,
            "max_turns": max_turns,
        })

        # Capture baseline metrics
        try:
            baseline = await cost_meter.start()
            trajectory.metrics_before = baseline.raw
        except Exception as exc:
            logger.warning("Could not capture baseline metrics: %s", exc)

        # Generate initial user message
        try:
            user_msg = await user_agent.get_initial_message()
        except Exception as exc:
            logger.error("UserAgent failed to generate initial message: %s", exc)
            trajectory.stop_reason = "error"
            trajectory.ended_at = datetime.now(timezone.utc).isoformat()
            save_metadata(output, trajectory)
            return trajectory

        turn_num = 0
        stop_reason = "max_turns"

        while turn_num < max_turns:
            # Record user turn
            user_turn = Turn(
                turn_number=turn_num,
                role="user",
                content=user_msg,
            )
            trajectory.turns.append(user_turn)
            save_turn(output, user_turn)
            await self._emit("turn", user_turn.model_dump())
            turn_num += 1

            # Send to OpenClaw
            agent_resp = await openclaw.send_message(user_msg, session_id)

            if agent_resp.status == "error" or agent_resp.status == "timeout":
                logger.error("OpenClaw error: %s", agent_resp.error)
                stop_reason = "error" if agent_resp.status == "error" else "timeout"
                break

            # Record agent turn
            agent_turn = Turn(
                turn_number=turn_num,
                role="agent",
                content=agent_resp.text,
                openclaw_raw=agent_resp.raw,
            )
            trajectory.turns.append(agent_turn)
            save_turn(output, agent_turn)
            await self._emit("turn", agent_turn.model_dump())
            turn_num += 1

            # Check budget
            try:
                current_cost = await cost_meter.current_cost()
                await self._emit("cost", {
                    "cumulative_usd": current_cost,
                    "budget_usd": budget,
                })
                if current_cost > budget:
                    logger.info("Budget exceeded ($%.4f limit)", budget)
                    stop_reason = "budget_exceeded"
                    break
            except Exception as exc:
                logger.debug("Budget check failed: %s", exc)

            # Get next user message
            try:
                user_msg = await user_agent.respond_to_agent(agent_resp.text)
            except Exception as exc:
                logger.error("UserAgent failed: %s", exc)
                stop_reason = "error"
                break

            if user_agent.stopped:
                stop_reason = "user_stop"
                # Record the final STOP message
                stop_turn = Turn(
                    turn_number=turn_num,
                    role="user",
                    content=user_msg,
                )
                trajectory.turns.append(stop_turn)
                save_turn(output, stop_turn)
                break

        trajectory.stop_reason = stop_reason
        trajectory.ended_at = datetime.now(timezone.utc).isoformat()

        # Capture final metrics
        try:
            final_snapshot, total_cost = await cost_meter.finish()
            trajectory.metrics_after = final_snapshot.raw
            trajectory.total_cost_usd = total_cost
        except Exception as exc:
            logger.warning("Could not capture final metrics: %s", exc)

        save_metadata(output, trajectory)

        await self._emit("done", {
            "stop_reason": stop_reason,
            "total_cost_usd": trajectory.total_cost_usd,
            "turn_count": len(trajectory.turns),
        })
        if self._live:
            await event_bus.end_trial()

        logger.info(
            "Trial %s complete: %s, %d turns, $%.4f, %s",
            trial_id,
            stop_reason,
            len(trajectory.turns),
            trajectory.total_cost_usd,
            self._task.task_id,
        )
        return trajectory


class BenchRunner:
    """Orchestrate a full benchmark run across tasks × conditions × repeats."""

    def __init__(self, config: BenchConfig, live: bool = False) -> None:
        self._config = config
        self._live = live

    async def run(
        self,
        task_filter: str | None = None,
        difficulty_filter: str | None = None,
        condition_filter: str | None = None,
        repeats: int | None = None,
    ) -> list[Trajectory]:
        """Run the benchmark and return all trajectories."""
        tasks = discover_tasks(self._config.tasks_dir, difficulty_filter)

        if task_filter:
            tasks = [t for t in tasks if task_filter in t.task_id]

        if not tasks:
            logger.warning("No tasks found matching filters")
            return []

        conditions = (
            [condition_filter] if condition_filter else self._config.conditions
        )
        num_repeats = repeats if repeats is not None else self._config.repeats

        trajectories: list[Trajectory] = []
        total = len(tasks) * len(conditions) * num_repeats

        logger.info(
            "Running %d trials (%d tasks × %d conditions × %d repeats)",
            total,
            len(tasks),
            len(conditions),
            num_repeats,
        )

        # Group by condition to minimize config switching
        for condition in conditions:
            logger.info("=== Condition: %s ===", condition)
            for task in tasks:
                for repeat in range(num_repeats):
                    logger.info(
                        "  [%d/%d] %s (repeat %d)",
                        len(trajectories) + 1,
                        total,
                        task.task_id,
                        repeat + 1,
                    )
                    trial = TrialRunner(task, condition, self._config, live=self._live)
                    trajectory = await trial.run()
                    trajectories.append(trajectory)

        return trajectories
