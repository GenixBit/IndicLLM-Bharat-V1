"""Tool-Calling Protocol and Parser for IndicLLM-Bharat.

Defines special tokens, prompt formatters, and regex/JSON parsers for autonomous tool invocation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from bharat.agent.tools import ToolDefinition, ToolResult

TOOL_CALL_START = "<|tool_call|>"
TOOL_CALL_END = "</|tool_call|>"
TOOL_RESPONSE_START = "<|tool_response|>"
TOOL_RESPONSE_END = "</|tool_response|>"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw_text: str


def format_agent_system_prompt(tools: list[ToolDefinition]) -> str:
    """Format system prompt equipping the model with tool descriptions."""
    tool_docs = []
    for t in tools:
        t_dict = t.to_dict()
        tool_docs.append(json.dumps(t_dict, indent=2, ensure_ascii=False))

    tools_block = "\n\n".join(tool_docs)

    return (
        "You are **IndicLLM-Bharat Agent**, an autonomous intelligent AI assistant for India and the world.\n"
        "You have access to the following sovereign tools:\n\n"
        f"{tools_block}\n\n"
        "To invoke a tool, respond with a JSON object wrapped in `<|tool_call|>` and `</|tool_call|>` tags:\n"
        "<|tool_call|>\n"
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "</|tool_call|>\n\n"
        "When you receive a `<|tool_response|> ... </|tool_response|>`, analyze the result and continue your reasoning "
        "or formulate your final response in the user's requested language."
    )


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Format a tool invocation string."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_START}\n{payload}\n{TOOL_CALL_END}"


def format_tool_response(result: ToolResult) -> str:
    """Format tool execution result to feed back into the model."""
    res_dict = result.to_dict()
    payload = json.dumps(res_dict, ensure_ascii=False)
    return f"\n{TOOL_RESPONSE_START}\n{payload}\n{TOOL_RESPONSE_END}\n"


def extract_tool_calls(text: str) -> list[ToolCall]:
    """Extract and parse any tool calls contained in the model response."""
    pattern = re.compile(
        re.escape(TOOL_CALL_START) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CALL_END),
        re.DOTALL,
    )

    calls: list[ToolCall] = []
    for match in pattern.finditer(text):
        raw_json = match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "name" in parsed:
                args = parsed.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                calls.append(
                    ToolCall(
                        name=str(parsed["name"]),
                        arguments=args,
                        raw_text=match.group(0),
                    )
                )
        except json.JSONDecodeError:
            continue

    return calls
