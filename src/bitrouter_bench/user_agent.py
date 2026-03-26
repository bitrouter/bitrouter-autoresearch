"""LLM-based user simulator (tau-bench style).

The UserAgent plays the role of a human user interacting with the execution
agent (OpenClaw).  It receives the task's UserScenario (persona, known info,
unknown info, instructions) but NEVER the evaluation criteria.

Design principles (from tau-bench):
- Progressive disclosure: reveal Unknown Info only when asked.
- Emit "STOP" when the task goal is satisfied.
- Role-flip trick: the LLM generates "assistant" messages that become
  user utterances, while the real agent's replies are fed as "user" messages.
"""

from __future__ import annotations

import logging
import os
import re

import httpx

from bitrouter_bench.task_loader import UserScenario

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

_SYSTEM_PROMPT_TEMPLATE = """\
You are role-playing as a human user interacting with an AI assistant.
Stay in character for the entire conversation.

## Your Persona
{persona}

## What You Know
{known_info}

## What You Also Know (reveal ONLY when the assistant asks a relevant question)
{unknown_info}

## Your Goal
{instructions}

## Rules
- Start by stating your initial request based on What You Know.
- Only reveal information from "What You Also Know" when the assistant
  asks a question that directly relates to that information.
- When your goal has been accomplished satisfactorily, respond with
  exactly the word STOP on its own line (you may add a brief thank-you
  before it).
- Be a realistic user: ask follow-ups if something is unclear, express
  preferences, push back if the assistant makes wrong assumptions.
- Keep your messages concise and natural — do not dump all information
  at once.
- NEVER mention evaluation criteria, scores, or that you are a simulator.
- If the assistant asks for clarification you cannot answer based on your
  scenario, say you are not sure or defer to the assistant's judgment.
"""


class UserAgent:
    """Simulated user that drives multi-turn interaction with the agent."""

    def __init__(
        self,
        scenario: UserScenario,
        model: str = "claude-sonnet-4-6-20250514",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._max_tokens = max_tokens
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            persona=scenario.persona or "A general user.",
            known_info=scenario.known_info or "No specific information provided.",
            unknown_info=scenario.unknown_info or "Nothing additional.",
            instructions=scenario.instructions or "Complete the task described above.",
        )
        # Role-flipped history: UserAgent's own messages = "assistant",
        # real agent's messages = "user"
        self._messages: list[dict[str, str]] = []
        self.stopped = False

    async def get_initial_message(self) -> str:
        """Generate the opening user message based on Known Info."""
        response = await self._call_llm(
            extra_instruction=(
                "Generate your opening message to the AI assistant. "
                "State your request based on What You Know. "
                "Keep it natural and concise."
            ),
        )
        self._messages.append({"role": "assistant", "content": response})
        return response

    async def respond_to_agent(self, agent_message: str) -> str:
        """Generate the next user message in response to the agent's reply.

        The agent's message is added as role="user" (role-flip) and the
        LLM generates the next "assistant" completion which becomes the
        user's reply.
        """
        self._messages.append({"role": "user", "content": agent_message})

        response = await self._call_llm()
        self._messages.append({"role": "assistant", "content": response})

        if self._is_stop(response):
            self.stopped = True

        return response

    async def _call_llm(self, extra_instruction: str | None = None) -> str:
        """Call the Anthropic Messages API directly (not through BitRouter)."""
        system = self._system_prompt
        if extra_instruction:
            system += f"\n\n## Current Instruction\n{extra_instruction}"

        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": self._messages if self._messages else [
                {"role": "user", "content": "Begin."},
            ],
        }

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                ANTHROPIC_API_URL,
                json=payload,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

        # Extract text from content blocks
        content_blocks = data.get("content", [])
        text_parts = [
            block["text"]
            for block in content_blocks
            if block.get("type") == "text"
        ]
        return "\n".join(text_parts)

    @staticmethod
    def _is_stop(message: str) -> bool:
        """Check if the user's message indicates task completion."""
        # Match STOP on its own line (possibly with surrounding whitespace/thanks)
        return bool(re.search(r"(?m)^\s*STOP\s*$", message))
