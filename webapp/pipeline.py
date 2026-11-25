"""Deterministic generate → validate → repair loop behind /api/chat.

The deep agent still does the actual modelling (its own tools and prompts keep
working underneath); this module owns stage ordering, attempt numbers and the
retry bound, so event order never depends on the model's mood.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Iterator, List, Optional

from webapp.validation import validate_sysml_code


def _default_max_attempts() -> int:
    """1 initial generation + 3 repairs, unless SYSML_MAX_ATTEMPTS says else."""
    try:
        value = int(os.environ.get("SYSML_MAX_ATTEMPTS", "4"))
    except ValueError:
        return 4
    return max(1, value)


def _diagnostic_bullets(diagnostics: List[dict]) -> str:
    if not diagnostics:
        return "- (the parser reported errors without line details)"
    return "\n".join(
        f"- line {d.get('line', '?')}:{d.get('column', '?')} — {d.get('message', '')}".rstrip(" —")
        for d in diagnostics
    )


def _repair_prompt(original: str, code: str, diagnostics: List[dict], empty: bool) -> str:
    if empty:
        problem = "Your previous candidate was empty — no model text was produced."
    else:
        problem = (
            "Your previous candidate failed ANTLR validation.\n\n"
            "Previous candidate:\n\n"
            f"{code}\n\n"
            "Reported problems:\n"
            f"{_diagnostic_bullets(diagnostics)}"
        )
    return (
        f"{original}\n\n"
        f"{problem}\n\n"
        "Return the complete corrected model — the whole package, not a diff."
    )


def run_sysml_pipeline(
    agent,
    prompt: str,
    thread_id: Optional[str] = None,
    max_attempts: Optional[int] = None,
    validator: Callable[[str], dict] = validate_sysml_code,
) -> Iterator[dict]:
    """Yield SSE event dicts for one sysml_generation request (see plan table)."""
    attempts_allowed = (
        max_attempts if max_attempts is not None else _default_max_attempts()
    )
    current_prompt = prompt
    last_code = ""

    for attempt in range(1, attempts_allowed + 1):
        yield {
            "type": "tool",
            "name": "generate_sysml",
            "status": "start",
            "attempt": attempt,
        }

        started = time.monotonic()
        code = ""
        for event in agent.stream_generation(current_prompt, thread_id):
            etype = event.get("type")
            if etype == "generated":
                code = event.get("code", "")
            elif etype == "error":
                yield event
                return
            elif etype == "token":
                # The model's own text carries the SysML source and its JSON
                # envelope. Chat forwards no token text here; the validated model
                # reaches the client as a single "code" event for the workspace.
                continue
            else:
                yield event
        yield {
            "type": "tool",
            "name": "generate_sysml",
            "status": "done",
            "attempt": attempt,
            "ms": int((time.monotonic() - started) * 1000),
        }
        last_code = code

        yield {
            "type": "tool",
            "name": "validate_sysml",
            "status": "start",
            "attempt": attempt,
        }
        validating = time.monotonic()
        try:
            report = validator(code)
        except Exception as exc:
            # A crashed validator is a runtime failure, never a syntax failure:
            # surface it and stop instead of burning repair attempts on it.
            summary = f"validator crashed: {exc}"
            yield {
                "type": "validation",
                "status": "error",
                "ok": False,
                "attempt": attempt,
                "final": True,
                "diagnostics": [],
                "summary": summary,
                "detail": summary,
            }
            yield {
                "type": "tool",
                "name": "validate_sysml",
                "status": "failed",
                "attempt": attempt,
                "detail": summary,
                "ms": int((time.monotonic() - validating) * 1000),
            }
            yield {"type": "code", "code": code, "attempt": attempt, "final": False}
            yield {"type": "pipeline", "status": "error", "attempts": attempt, "valid": False}
            return

        empty = not code.strip()
        valid = report.get("status") == "valid"
        final = valid and not empty
        yield {
            "type": "validation",
            "status": report.get("status", "error"),
            "ok": final,
            "attempt": attempt,
            "final": final,
            "diagnostics": report.get("diagnostics", []),
            "summary": report.get("summary", ""),
            "detail": report.get("summary", ""),
        }
        yield {
            "type": "tool",
            "name": "validate_sysml",
            "status": "done" if valid else "failed",
            "attempt": attempt,
            "ms": int((time.monotonic() - validating) * 1000),
        }

        if final:
            yield {"type": "code", "code": code, "attempt": attempt, "final": True}
            yield {
                "type": "token",
                "text": (
                    f"**SysML model accepted** by the kernel (attempt {attempt}).\n\n"
                    "The validated model is below; copy it or approve it there."
                ),
            }
            yield {"type": "pipeline", "status": "done", "attempts": attempt, "valid": True}
            return

        if attempt < attempts_allowed:
            current_prompt = _repair_prompt(
                prompt, code, report.get("diagnostics", []), empty
            )

    yield {
        "type": "code",
        "code": last_code,
        "attempt": attempts_allowed,
        "final": False,
    }
    yield {
        "type": "pipeline",
        "status": "exhausted",
        "attempts": attempts_allowed,
        "valid": False,
    }
    yield {
        "type": "token",
        "text": (
            f"Still failing after {attempts_allowed} attempt(s). The last candidate "
            "and its full diagnostics are below — fix the listed "
            "lines and validate again, or ask me to retry."
        ),
    }
