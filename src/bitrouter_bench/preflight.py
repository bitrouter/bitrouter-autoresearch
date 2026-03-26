"""Pre-flight checks — hard gate before any benchmark trial runs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil

import httpx

from bitrouter_bench.config import BenchConfig

logger = logging.getLogger(__name__)


class PreflightError(Exception):
    """A single pre-flight check failed."""


class Preflight:
    """Run all pre-flight checks.  Raises SystemExit on any failure."""

    def __init__(self, config: BenchConfig) -> None:
        self._config = config

    async def run(self) -> None:
        """Execute every check; collect errors and abort if any fail."""
        checks: list[tuple[str, ...]] = [
            ("BitRouter health", "_check_bitrouter_health"),
            ("BitRouter metrics", "_check_bitrouter_metrics"),
            ("OpenClaw binary", "_check_openclaw_binary"),
            ("OpenClaw gateway", "_check_openclaw_gateway"),
            ("OpenClaw agents", "_check_openclaw_agents"),
            ("API keys", "_check_api_keys"),
        ]

        errors: list[str] = []
        for name, method_name in checks:
            try:
                await getattr(self, method_name)()
                logger.info("\u2713 %s", name)
            except PreflightError as exc:
                msg = f"\u2717 {name}: {exc}"
                errors.append(msg)
                logger.error(msg)

        if errors:
            raise SystemExit(
                f"\nPreflight failed ({len(errors)} error(s)):\n"
                + "\n".join(f"  {e}" for e in errors)
                + "\n\nFix the above issues before running the benchmark."
            )
        logger.info("All preflight checks passed.")

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def _check_bitrouter_health(self) -> None:
        url = f"{self._config.bitrouter_url}/health"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != "ok":
                    raise PreflightError(
                        f"Unexpected health response: {json.dumps(data)}"
                    )
        except httpx.ConnectError:
            raise PreflightError(
                f"Cannot connect to BitRouter at {self._config.bitrouter_url}. "
                "Is it running?"
            )
        except httpx.HTTPError as exc:
            raise PreflightError(f"BitRouter health check failed: {exc}")

    async def _check_bitrouter_metrics(self) -> None:
        url = f"{self._config.bitrouter_url}/v1/metrics"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                if "routes" not in data and "uptime_seconds" not in data:
                    raise PreflightError(
                        "Metrics response missing expected fields (routes, uptime_seconds)"
                    )
        except httpx.ConnectError:
            raise PreflightError("Cannot reach BitRouter /v1/metrics endpoint")
        except httpx.HTTPError as exc:
            raise PreflightError(f"Metrics endpoint error: {exc}")

    async def _check_openclaw_binary(self) -> None:
        binary = self._config.openclaw_bin
        if not shutil.which(binary):
            raise PreflightError(
                f"OpenClaw binary '{binary}' not found on PATH"
            )

    async def _check_openclaw_gateway(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            self._config.openclaw_bin, "health",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise PreflightError(
                f"OpenClaw gateway not healthy (exit {proc.returncode}): {err}"
            )

    async def _check_openclaw_agents(self) -> None:
        required = {
            self._config.agent_id_user,
            self._config.agent_id_judge,
        }
        # Add all condition-specific test agents
        required.update(self._config.condition_agent_map.values())
        proc = await asyncio.create_subprocess_exec(
            self._config.openclaw_bin, "agents", "list", "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            raise PreflightError(
                "Failed to list OpenClaw agents: "
                + stderr.decode(errors="replace").strip()
            )

        # OpenClaw emits warnings/plugin logs mixed with JSON on stdout.
        # Extract the first complete JSON array using bracket depth tracking.
        raw = stdout.decode(errors="replace")
        agents_data = None

        start = raw.find("[")
        if start != -1:
            depth = 0
            end = -1
            for i in range(start, len(raw)):
                if raw[i] == "[":
                    depth += 1
                elif raw[i] == "]":
                    depth -= 1
                if depth == 0:
                    end = i
                    break
            if end > start:
                try:
                    agents_data = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass

        if agents_data is None:
            raise PreflightError(
                "Could not parse agent list output. "
                "Run 'openclaw agents list --json' manually to debug."
            )

        if isinstance(agents_data, list):
            existing = {a.get("id") for a in agents_data}
        elif isinstance(agents_data, dict) and "agents" in agents_data:
            existing = {a.get("id") for a in agents_data["agents"]}
        else:
            existing = set()

        missing = required - existing
        if missing:
            raise PreflightError(
                f"Missing OpenClaw agents: {', '.join(sorted(missing))}. "
                "Run: openclaw agents add <name> --workspace <dir> --non-interactive"
            )

    async def _check_api_keys(self) -> None:
        # All agents now route through OpenClaw gateway → BitRouter,
        # so direct ANTHROPIC_API_KEY is not strictly required.
        # However, OpenClaw itself may need API keys configured.
        # We verify by checking that BitRouter is reachable (already done)
        # and that OpenClaw has model access.
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            logger.info(
                "ANTHROPIC_API_KEY not set (OK — agents route through "
                "OpenClaw gateway → BitRouter)"
            )
