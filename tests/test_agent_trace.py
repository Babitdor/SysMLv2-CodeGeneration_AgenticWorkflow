"""The tool trace: which call, with what arguments, and what came back.

No network and no model: a fake deep agent yields the same chunk shapes the
provider does (arguments split across AIMessageChunks, then a ToolMessage), so
the events asserted here are exactly the ones the browser renders.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, ToolMessage

from agents.SysMLAgent import SUMMARY_CHARS, TRACE_CHARS, SysMLAgent

BODY = "line one\nline two"


def _agent(chunks):
    deep_agent = SimpleNamespace(stream=lambda *args, **kwargs: iter(chunks))
    with patch("agents.SysMLAgent.create_deep_agent", return_value=deep_agent):
        return SysMLAgent(llm=object())


def _call(call_id, name=None, args="", index=0):
    fragment = {"index": index, "id": call_id, "args": args}
    if name is not None:
        # The name rides on the first fragment only, as providers do it.
        fragment["name"] = name
    return AIMessageChunk(content="", tool_call_chunks=[fragment])


def _trace(chunks):
    return list(_agent(chunks).stream_generation("model a robot"))


def _by_status(events, status, name=None):
    return [
        e
        for e in events
        if e.get("status") == status and (name is None or e.get("name") == name)
    ]


def test_start_announces_the_call_and_done_reports_what_came_back():
    events = _trace(
        [
            _call("call_a", "read_file", '{"file_'),
            _call("call_a", args='path": "/skills/x/SKILL.md"}'),
            ToolMessage(content=BODY, tool_call_id="call_a", name="read_file"),
        ]
    )

    starts = _by_status(events, "start", "read_file")
    done = _by_status(events, "done", "read_file")
    assert len(starts) == 1
    assert len(done) == 1

    # A fragment usually does not parse yet: the summary must stay a string.
    assert starts[0]["call_id"] == "call_a"
    assert isinstance(starts[0]["summary"], str)

    assert done[0]["call_id"] == "call_a"
    assert done[0]["summary"] == "/skills/x/SKILL.md"
    assert json.loads(done[0]["args"]) == {"file_path": "/skills/x/SKILL.md"}
    assert done[0]["output"] == BODY
    assert done[0]["chars"] == len(BODY)
    assert done[0]["truncated"] is False
    assert isinstance(done[0]["ms"], int) and done[0]["ms"] >= 0


def test_the_same_tool_twice_stays_two_rows():
    events = _trace(
        [
            _call("call_a", "read_file", '{"file_path": "a"}', index=0),
            ToolMessage(content="A", tool_call_id="call_a", name="read_file"),
            _call("call_b", "read_file", '{"file_path": "b"}', index=1),
            ToolMessage(content="B", tool_call_id="call_b", name="read_file"),
        ]
    )

    done = _by_status(events, "done", "read_file")
    assert [e["call_id"] for e in done] == ["call_a", "call_b"]
    assert [e["summary"] for e in done] == ["a", "b"]
    assert [e["output"] for e in done] == ["A", "B"]


def test_a_tool_error_is_a_failed_row():
    events = _trace(
        [
            _call("call_a", "shell", '{"command": "cat missing"}'),
            ToolMessage(
                content="no such file",
                tool_call_id="call_a",
                name="shell",
                status="error",
            ),
        ]
    )

    failed = _by_status(events, "failed", "shell")
    assert len(failed) == 1
    assert failed[0]["call_id"] == "call_a"
    assert failed[0]["summary"] == "cat missing"
    assert failed[0]["output"] == "no such file"
    assert not _by_status(events, "done", "shell")


def test_a_big_result_is_capped_but_its_length_is_reported():
    big = "x" * 50_000
    events = _trace(
        [
            _call("call_a", "read_file", '{"file_path": "/workspace/big"}'),
            ToolMessage(content=big, tool_call_id="call_a", name="read_file"),
        ]
    )

    done = _by_status(events, "done", "read_file")[0]
    assert done["chars"] == 50_000
    assert done["truncated"] is True
    assert len(done["output"]) == TRACE_CHARS


def test_the_cap_is_configurable(monkeypatch):
    monkeypatch.setenv("SYSML_TRACE_CHARS", "10")
    events = _trace(
        [
            _call("call_a", "read_file", '{"file_path": "/workspace/big"}'),
            ToolMessage(content="y" * 40, tool_call_id="call_a", name="read_file"),
        ]
    )

    done = _by_status(events, "done", "read_file")[0]
    assert done["output"] == "y" * 10
    assert done["truncated"] is True
    assert done["chars"] == 40


def test_a_block_list_result_is_flattened():
    events = _trace(
        [
            _call("call_a", "read_file", '{"file_path": "/workspace/x"}'),
            ToolMessage(
                content=[{"type": "text", "text": "first "}, {"type": "text", "text": "second"}],
                tool_call_id="call_a",
                name="read_file",
            ),
        ]
    )

    assert _by_status(events, "done", "read_file")[0]["output"] == "first second"

def test_an_unmapped_tool_summarises_by_its_first_text_argument():
    # write_todos and the reference tools are not in the summary map, and their
    # argument names read as noise in a row: the first text value names the call.
    events = _trace(
        [
            _call("call_a", "sysml_syntax_reference_tool", '{"construct_type": "part def"}'),
            ToolMessage(
                content="Part Definition Syntax: part def <n> { ... }",
                tool_call_id="call_a",
                name="sysml_syntax_reference_tool",
            ),
        ]
    )

    assert _by_status(events, "done", "sysml_syntax_reference_tool")[0]["summary"] == "part def"


def test_a_long_summary_stays_one_line():
    events = _trace(
        [
            _call("call_a", "write_todos", '{"todos": "%s"}' % ("x" * 400)),
            ToolMessage(content="ok", tool_call_id="call_a", name="write_todos"),
        ]
    )

    assert _by_status(events, "done", "write_todos")[0]["summary"] == "x" * SUMMARY_CHARS

def test_parallel_calls_pair_by_id_not_by_arrival_order():
    # A deep agent issues several calls in one message and the tool messages come
    # back in another order: only the call id can pair them.
    events = _trace(
        [
            _call("call_t", "write_todos", '{"todos": "plan"}', index=0),
            _call("call_r1", "sysml_syntax_reference_tool", '{"construct_type": "part def"}', index=1),
            _call("call_r2", "sysml_syntax_reference_tool", '{"construct_type": "connection"}', index=2),
            ToolMessage(
                content="connection usage",
                tool_call_id="call_r2",
                name="sysml_syntax_reference_tool",
            ),
            ToolMessage(content="todos ok", tool_call_id="call_t", name="write_todos"),
            ToolMessage(
                content="part def usage",
                tool_call_id="call_r1",
                name="sysml_syntax_reference_tool",
            ),
        ]
    )

    done = {e["call_id"]: e for e in events if e.get("status") == "done"}
    assert done["call_t"]["name"] == "write_todos"
    assert done["call_t"]["output"] == "todos ok"
    assert done["call_r1"]["output"] == "part def usage"
    assert done["call_r1"]["summary"] == "part def"
    assert done["call_r2"]["summary"] == "connection"

