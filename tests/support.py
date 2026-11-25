"""Small helpers shared by the tests (SSE parsing, mostly)."""

from __future__ import annotations

import json


def sse_events(text: str) -> list[dict]:
    """Parse an SSE body into its JSON payloads."""
    events = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


def event_types(text: str) -> list[str]:
    return [event.get("type") for event in sse_events(text)]
