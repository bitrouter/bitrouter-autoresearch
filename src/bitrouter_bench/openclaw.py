"""Programmatic interface to OpenClaw CLI — gateway mode."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class OpenClawResponse:
    """Parsed response from an OpenClaw agent invocation."""

    text: str = ""
    status: str = "ok"
    raw: dict = field(default_factory=dict)
    error: str | None = None


class OpenClawRunner:
    """Invoke OpenClaw agents via the Gateway.

    Each agent is identified by its agent ID (e.g. bench-test, bench-user,
    bench-judge).  Messages are routed through the running OpenClaw gateway,
    giving agents access to the full tool suite (browser, filesystem, etc.).
    """

    def __init__(
        self,
        openclaw_bin: str = "openclaw",
        timeout_seconds: int = 600,
    ) -> None:
        self._bin = openclaw_bin
        self._timeout = timeout_seconds

    async def send_message(
        self,
        message: str,
        *,
        agent_id: str,
        session_id: str,
        thinking: str = "medium",
    ) -> OpenClawResponse:
        """Send a message to a named OpenClaw agent via the gateway.

        Runs: openclaw agent --agent <id> --message "<msg>"
              --session-id <sid> --json --timeout <T> --thinking <level>
        """
        cmd = [
            self._bin,
            "agent",
            "--agent", agent_id,
            "--message", message,
            "--session-id", session_id,
            "--json",
            "--timeout", str(self._timeout),
            "--thinking", thinking,
        ]

        env = dict(os.environ)

        logger.debug("OpenClaw cmd: %s", " ".join(cmd))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout + 30,
            )
        except asyncio.TimeoutError:
            logger.error("OpenClaw timed out after %ds", self._timeout)
            return OpenClawResponse(error="timeout", status="timeout")
        except FileNotFoundError:
            return OpenClawResponse(
                error=f"openclaw binary not found: {self._bin}",
                status="error",
            )

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            logger.error(
                "OpenClaw agent=%s exited %d: %s",
                agent_id, proc.returncode, err_msg,
            )
            return OpenClawResponse(error=err_msg, status="error")

        return self._parse_output(stdout.decode(errors="replace"))

    def _parse_output(self, stdout: str) -> OpenClawResponse:
        """Parse OpenClaw --json stdout into a response."""
        stdout = stdout.strip()
        if not stdout:
            return OpenClawResponse(error="empty output", status="error")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # OpenClaw may emit non-JSON preamble; try last line
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                return OpenClawResponse(
                    text=stdout,
                    status="parse_error",
                    error="Could not parse JSON from OpenClaw output",
                )

        text = self._extract_text(data)
        return OpenClawResponse(text=text, status="ok", raw=data)

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Extract the agent's reply text from OpenClaw JSON output.

        The response structure is:
          { "result": { "payloads": [{ "text": "..." }] }, "summary": "completed" }

        The actual agent text is in result.payloads[*].text.
        The top-level "summary" is just a status indicator ("completed").
        """
        # Primary: result.payloads[].text
        result = data.get("result", {})
        if isinstance(result, dict):
            payloads = result.get("payloads", [])
            if payloads:
                parts = [
                    p.get("text", "")
                    for p in payloads
                    if isinstance(p, dict) and p.get("text")
                ]
                if parts:
                    return "\n".join(parts)

        # Fallback: top-level text field
        if data.get("text"):
            return data["text"]

        # Last resort
        return data.get("summary", "")
