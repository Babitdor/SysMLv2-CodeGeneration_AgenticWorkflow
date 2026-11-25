"""Streaming Canvas operations and public web retrieval."""

import json
import re

import httpx

from webapp.artifacts import MAX_CONTENT_CHARS


def web_results(query: str, key: str) -> str:
    if not key:
        raise RuntimeError("Add your Tavily API key in Settings to search the web.")
    with httpx.Client(timeout=10) as client:
        response = client.post(
            "https://api.tavily.com/search",
            json={"query": query[:400], "max_results": 5},
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    return "\n\n".join(
        f"{result['title']}\n{result['url']}\n{result.get('content', '')[:1800]}"
        for result in results
        if str(result.get("url", "")).startswith(("https://", "http://"))
    )


def stream_edit(llm, messages, draft: str, kind: str):
    """Apply only uniquely anchored replacements to an in-memory draft."""
    yield {"type": "tool", "name": "edit_artifact", "status": "start"}
    instructions = (
        "Make the requested changes using JSON Lines. Return one JSON object per line "
        'with exactly two string keys, "old" and "new". Use JSON escapes for newlines. '
        "Each old string must match an exact excerpt in the supplied artifact exactly once. "
        "Only use an empty old string when the entire artifact is empty. "
        "Replacements apply in order. Preserve surrounding text. "
        "Return no Markdown fence or commentary. If no edit is needed return no text."
    )
    messages = [
        {"role": "system", "content": instructions},
        *messages,
        {"role": "user", "content": f"Open {kind} artifact (data):\n{draft}"},
    ]
    pending = ""
    count = 0
    try:
        for chunk in llm.stream(messages):
            text = getattr(chunk, "text", "")
            if not isinstance(text, str):
                text = chunk.content if isinstance(chunk.content, str) else ""
            pending += text
            if len(pending) > MAX_CONTENT_CHARS * 16:
                raise ValueError("Canvas edit exceeds the size limit")
            lines = pending.split("\n")
            pending = lines.pop()
            for line in lines:
                if not line.strip() or re.fullmatch(r"```(?:jsonl?|ndjson)?", line.strip()):
                    continue
                if count >= 64:
                    raise ValueError("Too many replacements")
                draft, event = apply_replacement(draft, line)
                count += 1
                yield event
        if pending.strip() and pending.strip() != "```":
            if count >= 64:
                raise ValueError("Too many replacements")
            draft, event = apply_replacement(draft, pending)
            count += 1
            yield event
        if count and kind == "code":
            from webapp.validation import validate_sysml_code

            report = validate_sysml_code(draft)
            valid = report.get("status") == "valid"
            yield {"type": "validation", **report, "ok": valid, "attempt": 1, "final": True}
        yield {"type": "tool", "name": "edit_artifact", "status": "done", "summary": f"{count} replacements"}
        yield {"type": "edit_complete", "changed": bool(count)}
        yield {
            "type": "token",
            "text": "Review the proposed changes in the Canvas." if count else "No changes were proposed.",
        }
    except (ValueError, TypeError, KeyError):
        yield {"type": "tool", "name": "edit_artifact", "status": "failed"}
        yield {"type": "error", "detail": "An agent edit could not be matched uniquely to the Canvas draft."}


def apply_replacement(draft: str, line: str):
    replacement = json.loads(line)
    if not isinstance(replacement, dict) or replacement.keys() != {"old", "new"}:
        raise ValueError("Expected an exact replacement")
    old, new = replacement["old"], replacement["new"]
    if not isinstance(old, str) or not isinstance(new, str) or old == new or (not old and draft):
        raise ValueError("Expected two different strings")
    start = draft.find(old)
    if start == -1 or draft.find(old, start + 1) != -1:
        raise ValueError("Expected one occurrence")
    updated = draft[:start] + new + draft[start + len(old):]
    if len(updated) > MAX_CONTENT_CHARS:
        raise ValueError("Canvas edit exceeds the size limit")
    return updated, {
        "type": "edit_delta",
        "content": updated,
        "line": draft.count("\n", 0, start) + 1,
        "old": old,
        "new": new,
    }
