"""Parse TASK.md files into structured Task objects."""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter
from pydantic import BaseModel, Field


class TaskMeta(BaseModel):
    """Metadata from TASK.md frontmatter."""

    name: str
    description: str
    difficulty: str  # "easy" | "medium" | "hard"
    category: str = "general"
    budget_usd: float | None = None
    max_turns: int = 20
    version: str = "1.0"


class UserScenario(BaseModel):
    """What the UserAgent sees — never includes evaluation criteria."""

    persona: str = ""
    known_info: str = ""
    unknown_info: str = ""
    instructions: str = ""


class EvalCriteria(BaseModel):
    """What the Judge sees — never shown to UserAgent."""

    programmatic_assertions: list[str] = Field(default_factory=list)
    llm_judge_prompt: str = ""


class Task(BaseModel):
    """A fully parsed benchmark task."""

    task_id: str  # e.g. "easy/create-hello-file"
    path: Path
    meta: TaskMeta
    scenario: UserScenario
    eval_criteria: EvalCriteria


def discover_tasks(
    tasks_dir: Path,
    difficulty: str | None = None,
) -> list[Task]:
    """Walk tasks/{easy,medium,hard}/*/TASK.md and return parsed tasks."""
    tasks: list[Task] = []
    difficulties = [difficulty] if difficulty else ["easy", "medium", "hard"]

    for diff in difficulties:
        diff_dir = tasks_dir / diff
        if not diff_dir.is_dir():
            continue
        for task_dir in sorted(diff_dir.iterdir()):
            task_file = task_dir / "TASK.md"
            if task_file.is_file():
                tasks.append(parse_task(task_file))

    return tasks


def parse_task(path: Path) -> Task:
    """Parse a single TASK.md file into a Task object."""
    post = frontmatter.load(str(path))

    metadata = post.metadata.get("metadata", {})
    meta = TaskMeta(
        name=post.metadata["name"],
        description=post.metadata["description"],
        difficulty=metadata.get("difficulty", _infer_difficulty(path)),
        category=metadata.get("category", "general"),
        budget_usd=_parse_float(metadata.get("budget_usd")),
        max_turns=int(metadata.get("max_turns", 20)),
        version=str(metadata.get("version", "1.0")),
    )

    body = post.content
    scenario = _parse_user_scenario(body)
    eval_criteria = _parse_eval_criteria(body)

    task_id = f"{meta.difficulty}/{meta.name}"

    return Task(
        task_id=task_id,
        path=path,
        meta=meta,
        scenario=scenario,
        eval_criteria=eval_criteria,
    )


def _infer_difficulty(path: Path) -> str:
    """Infer difficulty from directory structure: tasks/easy/... → 'easy'."""
    parts = path.parts
    for diff in ("easy", "medium", "hard"):
        if diff in parts:
            return diff
    return "medium"


def _parse_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_user_scenario(body: str) -> UserScenario:
    """Extract User Scenario sections from markdown body."""
    return UserScenario(
        persona=_extract_section(body, "Persona"),
        known_info=_extract_section(body, "Known Info"),
        unknown_info=_extract_section(body, "Unknown Info"),
        instructions=_extract_section(body, "Instructions"),
    )


def _parse_eval_criteria(body: str) -> EvalCriteria:
    """Extract Evaluation Criteria sections from markdown body."""
    assertions_text = _extract_section(body, "Programmatic Assertions")
    assertions = _parse_assertion_list(assertions_text)

    llm_judge_prompt = _extract_section(body, "LLM Judge")

    return EvalCriteria(
        programmatic_assertions=assertions,
        llm_judge_prompt=llm_judge_prompt,
    )


def _extract_section(body: str, heading: str) -> str:
    """Extract content under a ## heading, stopping at the next ## or --- or # heading."""
    pattern = rf"##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\n---|\n#\s|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _parse_assertion_list(text: str) -> list[str]:
    """Parse a markdown list of backtick-wrapped shell commands."""
    commands: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: - `command here` or - command here
        backtick_match = re.match(r"^-\s+`(.+?)`", line)
        if backtick_match:
            commands.append(backtick_match.group(1))
            continue
        # Match bare list items starting with -
        bare_match = re.match(r"^-\s+(.+)", line)
        if bare_match:
            commands.append(bare_match.group(1))
    return commands
