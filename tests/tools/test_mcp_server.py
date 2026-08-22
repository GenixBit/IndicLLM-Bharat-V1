from __future__ import annotations

from bharat.tools.mcp_server import MCPToolRegistry


class TestMCPServer:
    def test_mcp_discovery(self):
        registry = MCPToolRegistry()
        tools = registry.get_tool_definitions()
        assert len(tools) >= 3
        names = {t["name"] for t in tools}
        assert "calculator" in names
        assert "unit_converter" in names
        assert "knowledge_graph_lookup" in names

    def test_calculator_execution(self):
        registry = MCPToolRegistry()
        res = registry.execute_tool("calculator", {"expression": "sqrt(16) + 5 * 2"})
        assert "result" in res
        assert res["result"] == 14.0

    def test_unit_converter_execution(self):
        registry = MCPToolRegistry()
        res = registry.execute_tool(
            "unit_converter", {"value": 100, "from_unit": "c", "to_unit": "f"}
        )
        assert "result" in res
        assert res["result"] == 212.0
