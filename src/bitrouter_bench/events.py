"""In-process event bus for live trial streaming.

Single-producer (TrialRunner), single-consumer (SSE endpoint) design.
Uses asyncio.Queue — runner and API must share the same event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A typed event emitted during a trial."""

    type: str  # "turn", "cost", "status", "verdict", "done", "error"
    data: dict
    trial_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_sse(self) -> str:
        """Format as SSE text: event + data lines."""
        payload = json.dumps({
            "trial_id": self.trial_id,
            "timestamp": self.timestamp,
            **self.data,
        })
        return f"event: {self.type}\ndata: {payload}\n\n"


class EventBus:
    """Broadcast events from the runner to SSE subscribers.

    Designed for one active trial at a time. Subscribers that connect
    after the trial started receive a replay of past events first.
    """

    def __init__(self, max_replay: int = 500) -> None:
        self._subscribers: list[asyncio.Queue[Event | None]] = []
        self._replay_buffer: list[Event] = []
        self._max_replay = max_replay
        self._active_trial_id: str | None = None
        self._lock = asyncio.Lock()

    @property
    def active_trial_id(self) -> str | None:
        return self._active_trial_id

    async def start_trial(self, trial_id: str) -> None:
        """Signal that a new trial is starting. Clears replay buffer."""
        async with self._lock:
            self._active_trial_id = trial_id
            self._replay_buffer.clear()

    async def end_trial(self) -> None:
        """Signal that the current trial has ended."""
        async with self._lock:
            self._active_trial_id = None
            # Send sentinel to all subscribers so they know the stream ended
            for q in self._subscribers:
                await q.put(None)

    async def publish(self, event: Event) -> None:
        """Publish an event to all current subscribers + replay buffer."""
        async with self._lock:
            if len(self._replay_buffer) < self._max_replay:
                self._replay_buffer.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("Subscriber queue full, dropping event")

    async def subscribe(self) -> AsyncIterator[Event]:
        """Subscribe to the live event stream.

        Yields all replay events first, then live events as they arrive.
        Yields are terminated when the trial ends (None sentinel).
        """
        queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=200)

        async with self._lock:
            # Replay past events for late joiners
            for event in self._replay_buffer:
                await queue.put(event)
            self._subscribers.append(queue)

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            async with self._lock:
                if queue in self._subscribers:
                    self._subscribers.remove(queue)


# Module-level singleton — shared between runner and API
event_bus = EventBus()
