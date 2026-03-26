"""Parameterized task generation for benchmark trials.

Generates TASK.md files on-the-fly from a structured configuration,
producing varied but reproducible benchmark tasks with measurable criteria.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from bitrouter_bench.openclaw import OpenClawRunner
from bitrouter_bench.task_loader import EvalCriteria, Task, TaskMeta, UserScenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

class CategoryConfig(BaseModel):
    """A task category with generation weight and tool requirements."""

    name: str
    weight: float = 0.25
    examples: list[str] = Field(default_factory=list)
    requires_tools: list[str] = Field(default_factory=list)


class PersonaArchetype(BaseModel):
    """A user persona with behavioral parameters."""

    id: str
    verbosity: int = 3          # 1-5
    patience: int = 5           # 1-10
    disclosure_style: str = "only_when_asked"  # volunteer | only_when_asked | contradicts
    completion_bar: str = "medium"             # low | medium | high


class EvalConstraints(BaseModel):
    """Constraints on generated evaluation criteria."""

    min_assertions: int = 1
    max_assertions: int = 5
    assertion_types: list[str] = Field(default_factory=lambda: [
        "file_exists",
        "content_match",
        "command_exit_0",
    ])
    require_expected_outcome: bool = True
    require_key_steps: bool = True
    max_budget_usd: float = 0.25
    max_turns: int = 25


class GenerationConfig(BaseModel):
    """Top-level task generation configuration."""

    max_tasks_per_run: int = 20
    seed: int | None = None

    categories: list[CategoryConfig] = Field(default_factory=lambda: [
        CategoryConfig(
            name="file_operations",
            weight=0.3,
            examples=[
                "create, edit, or organize files",
                "parse and transform data between formats",
                "search and filter file contents",
            ],
            requires_tools=["filesystem"],
        ),
        CategoryConfig(
            name="code_tasks",
            weight=0.3,
            examples=[
                "debug a broken script",
                "write tests for existing code",
                "refactor with specific constraints",
                "implement a small utility function",
            ],
            requires_tools=["filesystem"],
        ),
        CategoryConfig(
            name="system_admin",
            weight=0.2,
            examples=[
                "diagnose a process or config issue",
                "set up a local development environment",
                "write a shell script to automate a workflow",
            ],
            requires_tools=["filesystem"],
        ),
        CategoryConfig(
            name="analysis",
            weight=0.2,
            examples=[
                "analyze a dataset and produce a summary",
                "compare two files and report differences",
                "generate a report from structured data",
            ],
            requires_tools=["filesystem"],
        ),
    ])

    difficulty_distribution: dict[str, float] = Field(default_factory=lambda: {
        "easy": 0.3,
        "medium": 0.4,
        "hard": 0.3,
    })

    personas: list[PersonaArchetype] = Field(default_factory=lambda: [
        PersonaArchetype(
            id="impatient_exec",
            verbosity=1,
            patience=2,
            disclosure_style="volunteer",
            completion_bar="low",
        ),
        PersonaArchetype(
            id="curious_beginner",
            verbosity=4,
            patience=9,
            disclosure_style="volunteer",
            completion_bar="high",
        ),
        PersonaArchetype(
            id="precise_engineer",
            verbosity=3,
            patience=5,
            disclosure_style="only_when_asked",
            completion_bar="high",
        ),
        PersonaArchetype(
            id="vague_requester",
            verbosity=2,
            patience=6,
            disclosure_style="contradicts",
            completion_bar="medium",
        ),
    ])

    eval_constraints: EvalConstraints = Field(default_factory=EvalConstraints)


# ---------------------------------------------------------------------------
# Generated task model (extends base Task with extra fields)
# ---------------------------------------------------------------------------

class GeneratedTaskSpec(BaseModel):
    """A task specification produced by the generator, before validation."""

    name: str
    description: str
    category: str
    difficulty: str
    persona_id: str
    persona_params: dict = Field(default_factory=dict)

    # Scenario
    persona_description: str = ""
    known_info: str = ""
    unknown_info: str = ""
    instructions: str = ""

    # Evaluation
    assertions: list[str] = Field(default_factory=list)
    llm_judge_prompt: str = ""
    expected_outcome: str = ""
    key_steps: list[str] = Field(default_factory=list)

    # Metadata
    budget_usd: float = 0.10
    max_turns: int = 20


# ---------------------------------------------------------------------------
# Generator prompt
# ---------------------------------------------------------------------------

_GENERATOR_PROMPT = """\
You are a benchmark task designer. Generate ONE self-contained task for
evaluating an AI coding assistant.

## Constraints

- **Category**: {category}
- **Difficulty**: {difficulty}
- **Available tools**: filesystem (read, write, edit files), shell commands
- **Budget**: ${max_budget_usd} max
- **Max turns**: {max_turns}

## Category Examples
{category_examples}

## Requirements

1. The task MUST be completable by an AI agent with filesystem access only.
2. All outcomes MUST be measurable via shell assertion commands.
3. The task MUST be self-contained — no external APIs, no web access, no
   dependencies that might not be installed.
4. For medium/hard tasks, include "unknown info" that changes the outcome
   if the agent discovers it through clarifying questions.
5. Include an expected outcome description for the judge.
6. Include ordered key steps a good agent would follow.

## Difficulty Guidelines

- **easy**: Single-step task, 1-2 assertions, no hidden info needed.
  Example: "Create a file with specific content."
- **medium**: Multi-step task, 2-3 assertions, some progressive disclosure.
  Example: "Debug a script — user knows the error, but the fix also requires
  handling an edge case the user will mention only if asked."
- **hard**: Complex multi-step task, 3-5 assertions, ambiguous initial request,
  hidden requirements. Example: "Build a data pipeline — user starts vague,
  has specific format requirements they'll share if asked, plus an edge case
  in the data they'll mention only if the agent asks about data quality."

## Assertions Format

Use shell commands that exit 0 on success. Available variables:
- $BENCH_WORKSPACE — the working directory

Examples:
- `test -f $BENCH_WORKSPACE/output.txt` (file exists)
- `grep -q "expected text" $BENCH_WORKSPACE/output.txt` (content match)
- `python3 -c "import json; json.load(open('$BENCH_WORKSPACE/data.json'))"` (valid JSON)

## Output Format

Respond with ONLY a JSON object (no markdown fences):

{{
  "name": "kebab-case-task-name",
  "description": "One-sentence task description",
  "persona_description": "Who the user is (matches difficulty — easy=simple user, hard=complex persona)",
  "known_info": "What the user knows upfront",
  "unknown_info": "Info revealed only through clarifying questions (empty string for easy tasks)",
  "instructions": "The user's goal in their own words",
  "assertions": ["shell command 1", "shell command 2"],
  "llm_judge_prompt": "What the judge should focus on when evaluating",
  "expected_outcome": "What a successful completion looks like",
  "key_steps": ["step 1", "step 2", "step 3"],
  "budget_usd": 0.10,
  "max_turns": 15
}}"""


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------

class TaskGenerator:
    """Generate benchmark tasks using an LLM via OpenClaw."""

    def __init__(
        self,
        openclaw: OpenClawRunner,
        config: GenerationConfig | None = None,
        agent_id: str = "bench-judge",
    ) -> None:
        self._openclaw = openclaw
        self._config = config or GenerationConfig()
        self._agent_id = agent_id
        self._rng = random.Random(self._config.seed)

    async def generate(self, count: int | None = None) -> list[Task]:
        """Generate a batch of validated tasks.

        Returns only tasks that pass validation.
        """
        n = min(count or self._config.max_tasks_per_run, self._config.max_tasks_per_run)
        tasks: list[Task] = []
        attempts = 0
        max_attempts = n * 3  # allow retries for validation failures

        while len(tasks) < n and attempts < max_attempts:
            attempts += 1
            category = self._pick_category()
            difficulty = self._pick_difficulty()
            persona = self._pick_persona()

            try:
                spec = await self._generate_one(category, difficulty, persona)
                task = self._spec_to_task(spec, persona)
                errors = self._validate(spec)
                if errors:
                    logger.warning(
                        "Task '%s' failed validation: %s", spec.name, "; ".join(errors),
                    )
                    continue
                tasks.append(task)
                logger.info("Generated task: %s (%s/%s)", spec.name, difficulty, category.name)
            except Exception as exc:
                logger.warning("Generation attempt %d failed: %s", attempts, exc)

        return tasks

    async def generate_and_save(
        self,
        output_dir: Path,
        count: int | None = None,
    ) -> list[Task]:
        """Generate tasks and write them as TASK.md files."""
        tasks = await self.generate(count)
        for task in tasks:
            task_dir = output_dir / task.meta.difficulty / task.meta.name
            task_dir.mkdir(parents=True, exist_ok=True)
            self._write_task_md(task_dir / "TASK.md", task)
            logger.info("Saved: %s", task_dir / "TASK.md")
        return tasks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_category(self) -> CategoryConfig:
        weights = [c.weight for c in self._config.categories]
        return self._rng.choices(self._config.categories, weights=weights, k=1)[0]

    def _pick_difficulty(self) -> str:
        dist = self._config.difficulty_distribution
        difficulties = list(dist.keys())
        weights = [dist[d] for d in difficulties]
        return self._rng.choices(difficulties, weights=weights, k=1)[0]

    def _pick_persona(self) -> PersonaArchetype:
        return self._rng.choice(self._config.personas)

    async def _generate_one(
        self,
        category: CategoryConfig,
        difficulty: str,
        persona: PersonaArchetype,
    ) -> GeneratedTaskSpec:
        """Generate a single task spec via LLM."""
        prompt = _GENERATOR_PROMPT.format(
            category=category.name,
            difficulty=difficulty,
            category_examples="\n".join(f"- {ex}" for ex in category.examples),
            max_budget_usd=self._config.eval_constraints.max_budget_usd,
            max_turns=self._config.eval_constraints.max_turns,
        )

        session_id = f"taskgen-{uuid.uuid4().hex[:12]}"
        resp = await self._openclaw.send_message(
            prompt,
            agent_id=self._agent_id,
            session_id=session_id,
            thinking="medium",
        )

        if resp.status != "ok":
            raise RuntimeError(f"Task generation failed: {resp.error}")

        # Parse JSON from response
        text = resp.text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        data = json.loads(text)
        return GeneratedTaskSpec(
            category=category.name,
            difficulty=difficulty,
            persona_id=persona.id,
            persona_params={
                "verbosity": persona.verbosity,
                "patience": persona.patience,
                "disclosure_style": persona.disclosure_style,
                "completion_bar": persona.completion_bar,
            },
            **data,
        )

    def _spec_to_task(self, spec: GeneratedTaskSpec, persona: PersonaArchetype) -> Task:
        """Convert a GeneratedTaskSpec into a Task object."""
        meta = TaskMeta(
            name=spec.name,
            description=spec.description,
            difficulty=spec.difficulty,
            category=spec.category,
            budget_usd=spec.budget_usd,
            max_turns=spec.max_turns,
        )
        scenario = UserScenario(
            persona=spec.persona_description,
            known_info=spec.known_info,
            unknown_info=spec.unknown_info,
            instructions=spec.instructions,
        )
        eval_criteria = EvalCriteria(
            programmatic_assertions=spec.assertions,
            llm_judge_prompt=spec.llm_judge_prompt,
        )
        return Task(
            task_id=f"{spec.difficulty}/{spec.name}",
            path=Path(f"generated/{spec.difficulty}/{spec.name}/TASK.md"),
            meta=meta,
            scenario=scenario,
            eval_criteria=eval_criteria,
        )

    def _validate(self, spec: GeneratedTaskSpec) -> list[str]:
        """Validate a generated task spec. Returns list of error strings."""
        errors: list[str] = []
        constraints = self._config.eval_constraints

        # Name check
        if not spec.name or " " in spec.name:
            errors.append("name must be non-empty kebab-case (no spaces)")

        # Assertion count
        if len(spec.assertions) < constraints.min_assertions:
            errors.append(
                f"need >= {constraints.min_assertions} assertions, got {len(spec.assertions)}"
            )
        if len(spec.assertions) > constraints.max_assertions:
            errors.append(
                f"need <= {constraints.max_assertions} assertions, got {len(spec.assertions)}"
            )

        # Assertions must reference $BENCH_WORKSPACE
        for cmd in spec.assertions:
            if "$BENCH_WORKSPACE" not in cmd and "BENCH_WORKSPACE" not in cmd:
                errors.append(f"assertion missing $BENCH_WORKSPACE: {cmd}")

        # Medium/hard tasks need hidden info
        if spec.difficulty in ("medium", "hard") and not spec.unknown_info.strip():
            errors.append(f"{spec.difficulty} task must have unknown_info for progressive disclosure")

        # Expected outcome required
        if constraints.require_expected_outcome and not spec.expected_outcome.strip():
            errors.append("expected_outcome is required")

        # Key steps required
        if constraints.require_key_steps and len(spec.key_steps) < 2:
            errors.append("need at least 2 key_steps")

        # Budget cap
        if spec.budget_usd > constraints.max_budget_usd:
            errors.append(
                f"budget ${spec.budget_usd} exceeds max ${constraints.max_budget_usd}"
            )

        # No eval leakage into scenario
        scenario_text = f"{spec.known_info} {spec.instructions}".lower()
        for word in ("assertion", "score", "evaluate", "benchmark", "judge"):
            if word in scenario_text:
                errors.append(f"eval term '{word}' found in scenario text (leakage)")

        return errors

    @staticmethod
    def _write_task_md(path: Path, task: Task) -> None:
        """Write a Task as a TASK.md file with frontmatter."""
        assertions_md = "\n".join(f"- `{a}`" for a in task.eval_criteria.programmatic_assertions)

        content = f"""\
---
name: {task.meta.name}
description: "{task.meta.description}"
metadata:
  difficulty: {task.meta.difficulty}
  category: {task.meta.category}
  budget_usd: {task.meta.budget_usd}
  max_turns: {task.meta.max_turns}
  version: "1.0"
  generated: true
---

# User Scenario

## Persona
{task.scenario.persona}

## Known Info
{task.scenario.known_info}

## Unknown Info
{task.scenario.unknown_info}

## Instructions
{task.scenario.instructions}

---

# Evaluation Criteria

## Programmatic Assertions
{assertions_md}

## LLM Judge
{task.eval_criteria.llm_judge_prompt}
"""
        with open(path, "w") as f:
            f.write(content)
