"""LLM-based user simulator running through OpenClaw gateway.

The UserAgent plays the role of a human user interacting with the execution
agent (OpenClaw bench-test).  It receives the task's UserScenario (persona,
known info, unknown info, instructions) but NEVER the evaluation criteria.

Design principles (from tau-bench):
- Progressive disclosure: reveal Unknown Info only when asked.
- Emit "STOP" when the task goal is satisfied.
- Role-flip trick: the LLM generates "assistant" messages that become
  user utterances, while the real agent's replies are fed as "user" messages.
"""

from __future__ import annotations

import logging
import re
import uuid

from bitrouter_bench.openclaw import OpenClawRunner
from bitrouter_bench.task_loader import UserScenario

logger = logging.getLogger(__name__)


def _build_session_prompt(
    scenario: UserScenario,
    persona_params: dict | None = None,
) -> str:
    """Build the per-session system instruction for the user simulator.

    This is sent as the first message to the bench-user agent, instructing it
    on how to behave for this specific trial.  The SOUL.md in the agent's
    workspace provides the base personality; this adds the scenario specifics.
    """
    params = persona_params or {}
    verbosity = params.get("verbosity", 3)
    patience = params.get("patience", 5)
    disclosure = params.get("disclosure_style", "only_when_asked")
    completion_bar = params.get("completion_bar", "medium")

    return f"""\
=== SCENARIO START ===
You are now role-playing as a human user for this benchmark trial.
Stay in character for the entire conversation.

## Your Persona
{scenario.persona or "A general user."}

## Behavioral Parameters
- verbosity: {verbosity} (1=terse, 5=chatty)
- patience: {patience} (1=easily frustrated, 10=endlessly patient)
- disclosure_style: {disclosure}
- completion_bar: {completion_bar}

## What You Know
{scenario.known_info or "No specific information provided."}

## What You Also Know (reveal ONLY per disclosure_style rules)
{scenario.unknown_info or "Nothing additional."}

## Your Goal
{scenario.instructions or "Complete the task described above."}

## Rules
- Start by stating your initial request based on What You Know.
- If disclosure_style is "only_when_asked": reveal hidden info only when the
  assistant asks a directly relevant question.
- If disclosure_style is "volunteer": you may naturally mention hidden info
  when it feels relevant, but don't dump everything at once.
- If disclosure_style is "contradicts": occasionally give slightly inconsistent
  info, then correct yourself when pressed.
- When your goal has been accomplished to your satisfaction (per completion_bar),
  respond with exactly the word STOP on its own line.
- Be realistic. Match the verbosity and patience levels.
- NEVER mention evaluation criteria, scores, or that you are a simulator.
- If the assistant asks for information outside your scenario, say you're not
  sure or defer to their judgment.

Now generate your opening message to the AI assistant.
=== SCENARIO END ==="""


class UserAgent:
    """Simulated user that drives multi-turn interaction via OpenClaw gateway."""

    def __init__(
        self,
        scenario: UserScenario,
        openclaw: OpenClawRunner,
        agent_id: str = "bench-user",
        persona_params: dict | None = None,
    ) -> None:
        self._scenario = scenario
        self._openclaw = openclaw
        self._agent_id = agent_id
        self._session_id = f"bench-user-{uuid.uuid4().hex[:12]}"
        self._persona_params = persona_params
        self._turn_count = 0
        self.stopped = False

    async def get_initial_message(self) -> str:
        """Generate the opening user message based on Known Info.

        Sends the scenario prompt to the bench-user agent, which generates
        the simulated user's opening message.
        """
        prompt = _build_session_prompt(self._scenario, self._persona_params)
        resp = await self._openclaw.send_message(
            prompt,
            agent_id=self._agent_id,
            session_id=self._session_id,
            thinking="low",
        )
        if resp.status != "ok":
            raise RuntimeError(f"UserAgent failed to generate initial message: {resp.error}")

        self._turn_count += 1
        return resp.text

    async def respond_to_agent(self, agent_message: str) -> str:
        """Generate the next user message in response to the agent's reply.

        We send the test agent's response to the bench-user agent, which
        generates the simulated user's next message.
        """
        # Prefix with context so the user-simulator knows this is the
        # assistant's reply it should respond to
        prompt = (
            f"The AI assistant replied:\n\n{agent_message}\n\n"
            "Now respond as the user. Remember your persona and rules."
        )

        resp = await self._openclaw.send_message(
            prompt,
            agent_id=self._agent_id,
            session_id=self._session_id,
            thinking="low",
        )
        if resp.status != "ok":
            raise RuntimeError(f"UserAgent failed: {resp.error}")

        self._turn_count += 1

        if self._is_stop(resp.text):
            self.stopped = True

        return resp.text

    @staticmethod
    def _is_stop(message: str) -> bool:
        """Check if the user's message indicates task completion."""
        return bool(re.search(r"(?m)^\s*STOP\s*$", message))
