"""Deterministic Tool Execution and ReAct Agent Framework for IndicLLM-Bharat.

Provides safe local tool execution (Python calculation, units, search) and autonomous
multi-step reasoning loops across English and 22 Scheduled Indian Languages.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from bharat.serving.openai_server import BharatInferenceEngine, ChatMessage


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]


@dataclass
class ToolExecutionResult:
    tool_name: str
    arguments: dict[str, Any]
    output: str
    success: bool
    error: str | None = None


class SafePythonExecutor:
    """Safely executes mathematical and algorithmic Python code in an isolated scope."""

    SAFE_BUILTINS: ClassVar[dict[str, Any]] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "math": math,
    }

    @classmethod
    def execute(cls, code: str) -> str:
        """Execute snippet and return output string."""
        # Sanitize unsafe operations
        if any(
            bad in code
            for bad in ["import os", "import sys", "import subprocess", "__", "open(", "eval("]
        ):
            return "Error: Security violation - unauthorized operation"

        local_vars: dict[str, Any] = {}
        try:
            # If code is a single expression, eval it; otherwise exec
            stripped = code.strip()
            if (
                "\n" not in stripped
                and not stripped.startswith("def ")
                and not stripped.startswith("for ")
            ):
                res = eval(stripped, {"__builtins__": cls.SAFE_BUILTINS}, local_vars)
                return str(res)

            exec(code, {"__builtins__": cls.SAFE_BUILTINS}, local_vars)
            # Return last assigned variable or success status
            if "result" in local_vars:
                return str(local_vars["result"])
            return f"Executed successfully. Variables: {list(local_vars.keys())}"
        except Exception as e:
            return f"Execution Error: {e}"


class UnitConverter:
    """Converts currency, distances, data sizes, and temperatures."""

    @staticmethod
    def convert(value: float, from_unit: str, to_unit: str) -> str:
        f_u = from_unit.lower().strip()
        t_u = to_unit.lower().strip()

        # Distance
        to_meters = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mile": 1609.34, "ft": 0.3048}
        if f_u in to_meters and t_u in to_meters:
            meters = value * to_meters[f_u]
            res = meters / to_meters[t_u]
            return f"{value} {from_unit} = {res:.4f} {to_unit}"

        # Data
        to_bytes = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
        if f_u in to_bytes and t_u in to_bytes:
            bytes_val = value * to_bytes[f_u]
            res = bytes_val / to_bytes[t_u]
            return f"{value} {from_unit} = {res:.4f} {to_unit}"

        # Temperature
        if f_u in ("c", "celsius") and t_u in ("f", "fahrenheit"):
            res = (value * 9 / 5) + 32
            return f"{value}°C = {res:.2f}°F"
        if f_u in ("f", "fahrenheit") and t_u in ("c", "celsius"):
            res = (value - 32) * 5 / 9
            return f"{value}°F = {res:.2f}°C"

        return f"Unknown unit conversion: {from_unit} -> {to_unit}"


class ToolRegistry:
    """Registry managing available tools and their execution dispatch."""

    def __init__(self) -> None:
        self.tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.register(
            ToolDefinition(
                name="python_calculator",
                description="Execute Python math expression or algorithm safely.",
                parameters={
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
                func=lambda code: SafePythonExecutor.execute(code),
            )
        )
        self.register(
            ToolDefinition(
                name="unit_converter",
                description="Convert values across units (km, mile, c, f, gb, mb).",
                parameters={
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "from_unit": {"type": "string"},
                        "to_unit": {"type": "string"},
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
                func=lambda value, from_unit, to_unit: UnitConverter.convert(
                    value, from_unit, to_unit
                ),
            )
        )

    def register(self, tool_def: ToolDefinition) -> None:
        self.tools[tool_def.name] = tool_def

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Dispatch and execute tool call."""
        if name not in self.tools:
            return ToolExecutionResult(
                tool_name=name,
                arguments=arguments,
                output="",
                success=False,
                error=f"Tool '{name}' is not registered in ToolRegistry",
            )

        tool = self.tools[name]
        try:
            out = tool.func(**arguments)
            return ToolExecutionResult(
                tool_name=name,
                arguments=arguments,
                output=str(out),
                success=True,
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name=name,
                arguments=arguments,
                output="",
                success=False,
                error=str(e),
            )

    def get_tool_descriptions(self) -> str:
        """Return formatted JSON descriptions of available tools."""
        tool_specs = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self.tools.values()
        ]
        return json.dumps(tool_specs, indent=2)


@dataclass
class AgentStep:
    thought: str
    tool_call: dict[str, Any] | None = None
    observation: str | None = None
    final_answer: str | None = None


class SovereignToolAgent:
    """Multi-step ReAct agent executing tools in a loop."""

    def __init__(
        self,
        tier: str = "1b",
        checkpoint_path: str | Path | None = None,
        registry: ToolRegistry | None = None,
        device: str = "auto",
    ) -> None:
        self.engine = BharatInferenceEngine(
            tier=tier,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.registry = registry or ToolRegistry()

    def run(self, user_task: str, max_steps: int = 3) -> dict[str, Any]:
        """Execute ReAct tool-calling agent loop."""
        tool_specs = self.registry.get_tool_descriptions()

        system_instruction = (
            f"You are IndicLLM-Bharat ReAct Tool Agent. You have access to the following tools:\n"
            f"{tool_specs}\n\n"
            f"To use a tool, output in this exact format:\n"
            f"<thought>reasoning</thought>\n"
            f'<tool_call>{{"name": "tool_name", "arguments": {{"arg": "val"}}}}</tool_call>\n\n'
            f"When you have the final answer, output:\n"
            f"<final_answer>your response</final_answer>"
        )

        history: list[ChatMessage] = [
            ChatMessage(role="system", content=system_instruction),
            ChatMessage(role="user", content=user_task),
        ]

        steps: list[AgentStep] = []

        for _ in range(max_steps):
            prompt = self.engine.format_chat_prompt(history)
            response = self.engine.generate(prompt, max_new_tokens=128, temperature=0.2)

            # Check for tool call
            match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
            if match:
                raw_json = match.group(1).strip()
                try:
                    tool_data = json.loads(raw_json)
                    tool_name = tool_data.get("name", "")
                    tool_args = tool_data.get("arguments", {})

                    exec_res = self.registry.execute_tool(tool_name, tool_args)
                    obs = exec_res.output if exec_res.success else f"Error: {exec_res.error}"

                    steps.append(
                        AgentStep(
                            thought=response,
                            tool_call=tool_data,
                            observation=obs,
                        )
                    )

                    history.append(ChatMessage(role="assistant", content=response))
                    history.append(
                        ChatMessage(role="user", content=f"<observation>{obs}</observation>")
                    )
                    continue
                except Exception:
                    pass

            # If no tool call or final answer returned
            final_match = re.search(r"<final_answer>(.*?)</final_answer>", response, re.DOTALL)
            final_ans = final_match.group(1).strip() if final_match else response.strip()

            steps.append(AgentStep(thought=response, final_answer=final_ans))
            return {
                "task": user_task,
                "final_answer": final_ans,
                "steps_taken": len(steps),
                "steps": [
                    {
                        "thought": s.thought,
                        "tool_call": s.tool_call,
                        "observation": s.observation,
                    }
                    for s in steps
                ],
            }

        # Return best effort after max_steps
        return {
            "task": user_task,
            "final_answer": steps[-1].thought if steps else "Task timed out",
            "steps_taken": len(steps),
            "steps": [
                {
                    "thought": s.thought,
                    "tool_call": s.tool_call,
                    "observation": s.observation,
                }
                for s in steps
            ],
        }
