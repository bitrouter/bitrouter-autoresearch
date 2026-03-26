"""Hybrid evaluation: programmatic assertions + LLM-as-judge via OpenClaw."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid

from pydantic import BaseModel

from bitrouter_bench.openclaw import OpenClawRunner
from bitrouter_bench.task_loader import Task
from bitrouter_bench.trajectory import Trajectory

logger = logging.getLogger(__name__)


class JudgmentResult(BaseModel):
    """Result of evaluating a single trial."""

    completion_score: float = 0.0
    interaction_quality: float = 0.0
    resource_efficiency: float = 0.0
    composite_score: float = 0.0
    assertion_results: dict[str, bool] = {}
    judge_reasoning: str = ""
    budget_usd: float = 0.0
    actual_cost_usd: float = 0.0


class Judge:
    """Hybrid evaluator combining programmatic checks and LLM judgment.

    The LLM evaluation runs through the bench-judge OpenClaw agent via
    the gateway, keeping all three agents on the same infrastructure.
    """

    def __init__(
        self,
        openclaw: OpenClawRunner,
        agent_id: str = "bench-judge",
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> None:
        self._openclaw = openclaw
        self._agent_id = agent_id
        self._w_completion, self._w_interaction, self._w_efficiency = weights

    async def evaluate(
        self,
        task: Task,
        trajectory: Trajectory,
        workspace: str | None = None,
        budget_usd: float = 0.10,
    ) -> JudgmentResult:
        """Run full hybrid evaluation on a completed trial."""
        # 1. Programmatic assertions
        completion_score, assertion_results = self._run_assertions(
            task, workspace or "/tmp",
        )

        # 2. LLM-as-judge (blind — no condition labels)
        interaction_quality, reasoning = await self._llm_judge(task, trajectory)

        # 3. Resource efficiency
        if budget_usd > 0:
            resource_efficiency = max(
                0.0,
                1.0 - (trajectory.total_cost_usd / budget_usd),
            )
        else:
            resource_efficiency = 1.0

        # 4. Composite score
        composite = (
            self._w_completion * completion_score
            + self._w_interaction * interaction_quality
            + self._w_efficiency * resource_efficiency
        )

        return JudgmentResult(
            completion_score=round(completion_score, 4),
            interaction_quality=round(interaction_quality, 4),
            resource_efficiency=round(resource_efficiency, 4),
            composite_score=round(composite, 4),
            assertion_results=assertion_results,
            judge_reasoning=reasoning,
            budget_usd=budget_usd,
            actual_cost_usd=trajectory.total_cost_usd,
        )

    def _run_assertions(
        self,
        task: Task,
        workspace: str,
    ) -> tuple[float, dict[str, bool]]:
        """Run programmatic assertion commands and return (score, details)."""
        assertions = task.eval_criteria.programmatic_assertions
        if not assertions:
            return 1.0, {}

        results: dict[str, bool] = {}
        for cmd in assertions:
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    timeout=30,
                    env={
                        **os.environ,
                        "BENCH_WORKSPACE": workspace,
                    },
                )
                results[cmd] = proc.returncode == 0
            except subprocess.TimeoutExpired:
                logger.warning("Assertion timed out: %s", cmd)
                results[cmd] = False
            except Exception as exc:
                logger.warning("Assertion failed: %s — %s", cmd, exc)
                results[cmd] = False

        passed = sum(1 for v in results.values() if v)
        score = passed / len(results) if results else 0.0
        return score, results

    async def _llm_judge(
        self,
        task: Task,
        trajectory: Trajectory,
    ) -> tuple[float, str]:
        """Call LLM judge via bench-judge agent. Returns (score, reasoning)."""
        if not task.eval_criteria.llm_judge_prompt:
            return 0.5, "No LLM judge prompt defined; defaulting to 0.5"

        transcript = self._format_transcript(trajectory)
        session_id = f"bench-judge-{uuid.uuid4().hex[:12]}"

        prompt = f"""\
Evaluate this AI assistant interaction. Respond with ONLY a JSON object.

## Task Description
{task.meta.description}

## Conversation Transcript
{transcript}

## Evaluation Focus
{task.eval_criteria.llm_judge_prompt}

## Expected Output Format
Respond with a JSON object containing:
- "interaction_quality": a float between 0.0 and 1.0
- "reasoning": a brief explanation citing specific transcript evidence

Rate the interaction and respond with JSON only."""

        try:
            resp = await self._openclaw.send_message(
                prompt,
                agent_id=self._agent_id,
                session_id=session_id,
                thinking="medium",
            )
            if resp.status != "ok":
                logger.error("Judge agent error: %s", resp.error)
                return 0.5, f"Judge agent error: {resp.error}"

            # Parse JSON from response — may be wrapped in markdown code block
            text = resp.text.strip()
            if text.startswith("```"):
                # Strip markdown code fences
                lines = text.splitlines()
                text = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )

            result = json.loads(text)
            quality = float(result.get("interaction_quality", 0.5))
            reasoning = result.get("reasoning", "")
            return max(0.0, min(1.0, quality)), reasoning

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse judge response: %s", exc)
            return 0.5, f"Judge parse error: {exc}"
        except Exception as exc:
            logger.error("Judge evaluation failed: %s", exc)
            return 0.5, f"Judge error: {exc}"

    @staticmethod
    def _format_transcript(trajectory: Trajectory) -> str:
        """Format trajectory as a blind transcript (no condition labels)."""
        lines: list[str] = []
        for turn in trajectory.turns:
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"**{role}**: {turn.content}")
        return "\n\n".join(lines)


def save_verdict(output_dir: str | os.PathLike, result: JudgmentResult) -> None:
    """Write verdict.json to the trial directory."""
    from pathlib import Path

    path = Path(output_dir) / "verdict.json"
    with open(path, "w") as f:
        f.write(result.model_dump_json(indent=2))
