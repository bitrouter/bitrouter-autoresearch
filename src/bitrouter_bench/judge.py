"""Hybrid evaluation: programmatic assertions + LLM-as-judge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess

import httpx
from pydantic import BaseModel

from bitrouter_bench.task_loader import Task
from bitrouter_bench.trajectory import Trajectory

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

_JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge evaluating an AI assistant's performance on a user task.
You will see a conversation transcript between a user and an assistant.
You do NOT know which system or model produced the assistant's responses.
Evaluate objectively based solely on the conversation quality.

Respond with a JSON object containing:
- "interaction_quality": a float between 0.0 and 1.0
- "reasoning": a brief explanation of your rating

Scoring guide for interaction_quality:
- 1.0: Exceptional — natural conversation, efficient, asked great clarifying questions
- 0.8: Good — mostly smooth, minor inefficiencies
- 0.6: Adequate — completed the task but interaction was clunky or verbose
- 0.4: Below average — significant issues in communication or approach
- 0.2: Poor — major problems, unhelpful responses, wrong assumptions
- 0.0: Failed — completely unhelpful or harmful interaction
"""


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
    """Hybrid evaluator combining programmatic checks and LLM judgment."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6-20250514",
        api_key: str | None = None,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
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
        """Call LLM judge with a blind transcript. Returns (score, reasoning)."""
        if not task.eval_criteria.llm_judge_prompt:
            return 0.5, "No LLM judge prompt defined; defaulting to 0.5"

        transcript = self._format_transcript(trajectory)

        user_prompt = f"""\
## Task Description
{task.meta.description}

## Conversation Transcript
{transcript}

## Evaluation Focus
{task.eval_criteria.llm_judge_prompt}

Rate the interaction and respond with JSON only."""

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self._model,
            "max_tokens": 1024,
            "system": _JUDGE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    ANTHROPIC_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block["text"]

            # Parse JSON from response
            result = json.loads(text)
            quality = float(result.get("interaction_quality", 0.5))
            reasoning = result.get("reasoning", "")
            return max(0.0, min(1.0, quality)), reasoning

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse judge response: %s", exc)
            return 0.5, f"Judge parse error: {exc}"
        except httpx.HTTPError as exc:
            logger.error("Judge API call failed: %s", exc)
            return 0.5, f"Judge API error: {exc}"

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
