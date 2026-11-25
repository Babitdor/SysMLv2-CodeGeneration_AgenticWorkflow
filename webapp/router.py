"""LLM router: does this message want a SysML v2 model, or just conversation?

Kept deliberately small — one short classification call plus a keyword
fallback, so a slow or unavailable model never blocks the chat.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Canonical routes.
GENERAL_CHAT = "general_chat"
SYSML_GENERATION = "sysml_generation"

# Legacy aliases — existing callers and `mode` frames keep working.
GENERATE = SYSML_GENERATION
CHAT = GENERAL_CHAT

ROUTER_PROMPT = """You route user messages for a SysML v2 modelling tool.

Answer with exactly one word:

sysml_generation - the user wants a SysML v2 model, diagram, package,
                   requirement structure or code produced, changed, fixed or
                   validated, or an older SysML v1 / UML model converted.
general_chat     - anything else: questions, explanations, comparisons, small
                   talk, or questions about attached engineering documents.

Message:
{message}

Answer:"""

# Fallback when the model is unavailable or answers off-contract.
GENERATE_HINTS = re.compile(
    r"\b(sysml|model|diagram|part def|part definition|requirement|package|"
    r"state machine|activity|allocation|constraint|interface|generate|create|"
    r"build|draft|fix|validate|validation|sysml ?v?1|v1 syntax)\b",
    re.IGNORECASE,
)
CHAT_HINTS = re.compile(
    r"^\s*(what|why|how|who|when|where|which|is|are|can|could|should|explain|"
    r"describe|summari[sz]e|tell me)\b",
    re.IGNORECASE,
)
# Older modelling language asking to be handled/converted as v1 input.
V1_HINTS = re.compile(r"\b(sysml\s*v1|sysml\s*1\.|uml|ibd)\b", re.IGNORECASE)

_GENERATION_PREFIXES = ("sysml", "generat")
_CHAT_PREFIXES = ("general", "chat", "answer")
_CHAT_WORDS = {"conversation", "query"}  # old contract words


def _canonical(word: str) -> str | None:
    if word.startswith(_GENERATION_PREFIXES):
        return SYSML_GENERATION
    if word.startswith(_CHAT_PREFIXES) or word in _CHAT_WORDS:
        return GENERAL_CHAT
    return None


def _looks_like_v1(message: str) -> bool:
    """True when the message references SysML v1 / UML-style input."""
    return bool(V1_HINTS.search(message or ""))


def _format_for(route: str, message: str) -> str | None:
    if route != SYSML_GENERATION:
        return None
    return "v1" if _looks_like_v1(message) else "v2"


@dataclass
class RouteDecision:
    """Result of intent routing: which pipeline, why, and the input dialect."""

    route: str
    reason: str
    source: str  # "llm" | "fallback"
    format: Optional[str] = None  # "v1" | "v2" | None (chat)


def guess(message: str) -> str:
    """Keyword heuristic, used alone or as the router's fallback."""
    text = message.strip()
    if CHAT_HINTS.search(text) and not re.search(
        r"\b(generate|create|build|write|produce|fix)\b", text, re.IGNORECASE
    ):
        return CHAT
    return GENERATE if GENERATE_HINTS.search(text) else CHAT


def decide(llm: Any, message: str) -> RouteDecision:
    """Classify the message, preferring the model and falling back to keywords."""
    text = (message or "").strip()
    try:
        reply = llm.invoke(ROUTER_PROMPT.format(message=text))
        answer = getattr(reply, "content", reply) or ""
        if isinstance(answer, list):  # content blocks
            answer = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in answer
            )
        word = re.findall(r"[a-zA-Z]+", str(answer).lower())
        if word:
            route = _canonical(word[0])
            if route:
                return RouteDecision(
                    route=route,
                    reason=f"router model said {word[0]!r}",
                    source="llm",
                    format=_format_for(route, text),
                )
    except Exception as exc:  # model down, quota, timeout …
        logger.warning("⚠️ Router call failed (%s), using keyword fallback", exc)

    fallback = guess(text)
    return RouteDecision(
        route=fallback,
        reason="keyword fallback",
        source="fallback",
        format=_format_for(fallback, text),
    )


def classify(llm: Any, message: str, default: str | None = None) -> str:
    """Back-compat wrapper: canonical route string only."""
    decision = decide(llm, message)
    if decision.source == "llm":
        return decision.route
    if default is not None:
        return _canonical(str(default).strip().lower()) or GENERAL_CHAT
    # Off-contract model answer with no explicit default: keep the old
    # `default or guess` contract — the heuristic decides, GENERAL_CHAT if it
    # finds nothing (an uncluttered chat is the safe failure).
    return decision.route
