"""Offline tests for the streaming assembler - no network, no openai import needed."""
from __future__ import annotations

from types import SimpleNamespace as NS

from bilal_agent.llm import _assemble_stream


def chunk(delta=None, usage=None, empty_choices=False, finish=None):
    return NS(
        choices=[] if empty_choices else [NS(delta=NS(**(delta or {})), finish_reason=finish)],
        usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None,
    )


def tool_delta(index=0, id=None, name=None, arguments=None):
    return {"tool_calls": [NS(index=index, id=id, function=NS(name=name, arguments=arguments))]}


def test_content_stream_assembles_and_calls_on_delta():
    pieces = []
    msg, usage, finish = _assemble_stream(
        [chunk({"content": "Hel"}), chunk({"content": "lo"}), chunk({"content": " world"}, finish="stop")],
        on_delta=pieces.append,
    )
    assert msg == {"role": "assistant", "content": "Hello world"}
    assert pieces == ["Hel", "lo", " world"]
    assert usage is None
    assert finish == "stop"


def test_tool_call_arguments_split_across_chunks():
    msg, _, _ = _assemble_stream([
        chunk(tool_delta(index=0, id="a1", name="grep", arguments="")),
        chunk(tool_delta(index=0, arguments='{"pat')),
        chunk(tool_delta(index=0, name="grep", arguments='tern": "x"}')),  # repeated name ignored
        chunk(finish="tool_calls"),
    ])
    assert msg["role"] == "assistant"
    assert msg["content"] == ""
    assert msg["tool_calls"] == [
        {"id": "a1", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}}
    ]


def test_parallel_tool_calls_keep_index_order():
    msg, _, _ = _assemble_stream([
        chunk(tool_delta(index=1, id="b", name="write_file", arguments="{}")),
        chunk(tool_delta(index=0, id="a", name="read_file", arguments="{}")),
    ])
    names = [tc["function"]["name"] for tc in msg["tool_calls"]]
    assert names == ["read_file", "write_file"]  # index order, not arrival order


def test_missing_tool_call_id_gets_fallback():
    msg, _, _ = _assemble_stream([chunk(tool_delta(index=2, name="ls", arguments="{}"))])
    assert msg["tool_calls"][0]["id"] == "call_2"


def test_usage_only_final_chunk():
    msg, usage, _ = _assemble_stream([
        chunk({"content": "hi"}),
        chunk(empty_choices=True, usage=(120, 30)),
    ])
    assert msg["content"] == "hi"
    assert usage == {"prompt": 120, "completion": 30}


def test_finish_reason_taken_from_last_chunk_that_has_one():
    _, _, finish = _assemble_stream([
        chunk({"content": "a"}, finish=None),
        chunk(empty_choices=True),
        chunk({"content": "b"}, finish="length"),
    ])
    assert finish == "length"
