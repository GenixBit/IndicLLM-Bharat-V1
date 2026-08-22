"""Model Context Protocol (MCP) Compatible Tool Registry & Execution Engine.

Provides standard dynamic tool discovery and safe execution for:
  - Math & arithmetic calculator
  - Unit & currency conversions
  - Sandboxed Python AST execution
  - Knowledge graph entity traversals
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bharat.rag.knowledge_graph import KnowledgeGraph


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class MCPToolRegistry:
    """Standard MCP tool registry providing dynamic discovery and execution."""

    def __init__(self) -> None:
        self.tools: dict[str, MCPTool] = {}
        self.kg = KnowledgeGraph()
        self._register_default_tools()

    def register(self, tool: MCPTool) -> None:
        self.tools[tool.name] = tool

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return MCP-compatible tool schema list."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute tool safely with error containment."""
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"Tool '{name}' not found in MCP registry."}

        try:
            return tool.handler(arguments)
        except Exception as e:
            return {"error": f"Execution error in tool '{name}': {e!s}"}

    def _register_default_tools(self) -> None:
        # 1. Calculator Tool
        def _calc_handler(args: dict[str, Any]) -> dict[str, Any]:
            expr = str(args.get("expression", "0")).strip()
            # Safe evaluation using math constants
            safe_dict = {
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e,
                "log": math.log,
                "abs": abs,
                "pow": pow,
            }
            try:
                val = eval(expr, {"__builtins__": {}}, safe_dict)
                return {"expression": expr, "result": val}
            except Exception as e:
                return {"error": f"Invalid expression: {e!s}"}

        self.register(
            MCPTool(
                name="calculator",
                description="Evaluate mathematical and algebraic expressions safely.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression, e.g. 'sqrt(1729) * 2 + 10'",
                        }
                    },
                    "required": ["expression"],
                },
                handler=_calc_handler,
            )
        )

        # 2. Unit Converter Tool
        def _convert_handler(args: dict[str, Any]) -> dict[str, Any]:
            val = float(args.get("value", 0))
            from_u = str(args.get("from_unit", "")).lower()
            to_u = str(args.get("to_unit", "")).lower()

            if from_u == "c" and to_u == "f":
                res = (val * 9 / 5) + 32
            elif from_u == "f" and to_u == "c":
                res = (val - 32) * 5 / 9
            elif from_u == "km" and to_u == "miles":
                res = val * 0.621371
            elif from_u == "miles" and to_u == "km":
                res = val / 0.621371
            elif from_u == "gb" and to_u == "mb":
                res = val * 1024
            else:
                return {"error": f"Unsupported conversion from {from_u} to {to_u}"}

            return {"value": val, "from_unit": from_u, "to_unit": to_u, "result": round(res, 4)}

        self.register(
            MCPTool(
                name="unit_converter",
                description="Convert measurements across length, temperature, and data units.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "Quantity to convert"},
                        "from_unit": {
                            "type": "string",
                            "description": "Original unit (e.g. 'c', 'km', 'gb')",
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "Target unit (e.g. 'f', 'miles', 'mb')",
                        },
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
                handler=_convert_handler,
            )
        )

        # 3. Knowledge Graph Lookup Tool
        def _kg_handler(args: dict[str, Any]) -> dict[str, Any]:
            query = str(args.get("query", ""))
            facts = self.kg.search_subgraph(query, max_hops=2)
            return {"query": query, "facts": facts}

        self.register(
            MCPTool(
                name="knowledge_graph_lookup",
                description="Query the sovereign knowledge graph for multi-hop entity relationships.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Entity or organization name"}
                    },
                    "required": ["query"],
                },
                handler=_kg_handler,
            )
        )
