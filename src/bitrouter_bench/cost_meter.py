"""Cost metering via BitRouter's /v1/metrics endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


# Per-million-token pricing (USD).  Extend as needed.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_million, output_per_million)
    "claude-opus-4-6-20250514": (5.0, 25.0),
    "claude-sonnet-4-6-20250514": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4.1": (2.0, 8.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
}

# Fallback pricing if model not found
DEFAULT_PRICING = (3.0, 15.0)


@dataclass
class MetricsSnapshot:
    """A snapshot of BitRouter's metrics at a point in time."""

    raw: dict = field(default_factory=dict)
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @classmethod
    def from_response(cls, data: dict) -> MetricsSnapshot:
        """Parse the /v1/metrics JSON response."""
        total_req = 0
        total_in = 0
        total_out = 0

        # Walk route-level metrics
        routes = data.get("routes", {})
        for _route_name, route_data in routes.items():
            total_req += route_data.get("total_requests", 0)
            total_in += route_data.get("total_input_tokens", 0)
            total_out += route_data.get("total_output_tokens", 0)

        return cls(
            raw=data,
            total_requests=total_req,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )


class CostMeter:
    """Track dollar cost of a trial via BitRouter metrics deltas."""

    def __init__(self, bitrouter_url: str) -> None:
        self._url = bitrouter_url.rstrip("/")
        self._baseline: MetricsSnapshot | None = None

    async def snapshot(self) -> MetricsSnapshot:
        """Fetch current metrics from BitRouter."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self._url}/v1/metrics", timeout=10)
            resp.raise_for_status()
            return MetricsSnapshot.from_response(resp.json())

    async def start(self) -> MetricsSnapshot:
        """Capture baseline metrics before a trial."""
        self._baseline = await self.snapshot()
        return self._baseline

    async def current_cost(self) -> float:
        """Compute the dollar cost since start() was called."""
        if self._baseline is None:
            return 0.0
        current = await self.snapshot()
        return self._calculate_cost(self._baseline, current)

    async def finish(self) -> tuple[MetricsSnapshot, float]:
        """Capture final metrics and return (snapshot, total_cost)."""
        final = await self.snapshot()
        cost = 0.0
        if self._baseline is not None:
            cost = self._calculate_cost(self._baseline, final)
        return final, cost

    def _calculate_cost(
        self,
        before: MetricsSnapshot,
        after: MetricsSnapshot,
    ) -> float:
        """Calculate dollar cost from token deltas using model pricing."""
        delta_in = max(0, after.total_input_tokens - before.total_input_tokens)
        delta_out = max(0, after.total_output_tokens - before.total_output_tokens)

        # Use default pricing for now — could be refined per-route
        input_rate, output_rate = DEFAULT_PRICING
        cost = (delta_in * input_rate + delta_out * output_rate) / 1_000_000
        return round(cost, 6)

    async def exceeds_budget(self, budget_usd: float) -> bool:
        """Check if the current trial has exceeded its dollar budget."""
        return await self.current_cost() > budget_usd
