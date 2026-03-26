"""Programmatic interface to OpenClaw CLI."""

from __future__ import annotations

import asyncio
import json
import logging
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
    """Invoke OpenClaw as a subprocess for each agent turn."""

    def __init__(
        self,
        openclaw_bin: str,
        state_dir: Path,
        timeout_seconds: int = 300,
        workspace: Path | None = None,
    ) -> None:
        self._bin = openclaw_bin
        self._state_dir = state_dir
        self._timeout = timeout_seconds
        self._workspace = workspace

    async def send_message(
        self,
        message: str,
        session_id: str,
        thinking: str = "medium",
    ) -> OpenClawResponse:
        """Send a message to OpenClaw and return the agent's response.

        Runs: openclaw agent --local --message "<msg>" --session-id <id>
              --json --timeout <T> --thinking <level>
        """
        cmd = [
            self._bin,
            "agent",
            "--local",
            "--message",
            message,
            "--session-id",
            session_id,
            "--json",
            "--timeout",
            str(self._timeout),
            "--thinking",
            thinking,
        ]

        env = {
            "OPENCLAW_STATE_DIR": str(self._state_dir.resolve()),
            "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin",
            "HOME": str(Path.home()),
        }

        # Inherit common env vars
        import os

        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "BITROUTER_API_KEY"):
            val = os.environ.get(key)
            if val:
                env[key] = val

        cwd = str(self._workspace) if self._workspace else None

        logger.debug("OpenClaw cmd: %s", " ".join(cmd))
        logger.debug("OpenClaw state_dir: %s", self._state_dir)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout + 30,  # grace period
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
            logger.error("OpenClaw exited %d: %s", proc.returncode, err_msg)
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

        text = data.get("text", "") or data.get("summary", "") or data.get("result", "")
        return OpenClawResponse(text=text, status="ok", raw=data)
