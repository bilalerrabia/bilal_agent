"""The Task tool: spawn a read-only sub-agent with a FRESH context window.

Why it matters: heavy exploration (grep 40 files, read 10) would bloat the main
conversation and force compaction. A sub-agent burns its own tokens, keeps only
its report, and the main ledger grows by exactly one tool result.

The Tool definition lives here; the Agent binds the real function (it needs the
LLM client and the loop), see Agent._run_subagent.
"""
from __future__ import annotations

from .base import P, Tool

task_tool = Tool(
    name="task",
    description=(
        "Spawn a read-only sub-agent with a fresh context window to do heavy "
        "exploration and return only its report (search many files, read a lot, "
        "summarize). Use it for broad searches or deep dives that would bloat the "
        "main conversation. The sub-agent CANNOT edit files or run commands - it "
        "reads, greps and reports."
    ),
    parameters=P(
        {
            "prompt": {
                "type": "string",
                "description": "Complete, self-contained research brief: what to find, "
                "where to look, and what the report should contain.",
            },
            "label": {
                "type": "string",
                "description": "Short mission label shown to the user (e.g. 'find all auth code').",
            },
        },
        ["prompt"],
    ),
    risk="safe",
    fn=lambda **kw: "error: task tool not bound",  # replaced by Agent._run_subagent
)
