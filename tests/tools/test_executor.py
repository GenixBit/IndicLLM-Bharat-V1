from __future__ import annotations

from bharat.tools.executor import (
    SafePythonExecutor,
    SovereignToolAgent,
    ToolRegistry,
    UnitConverter,
)
from scripts.run_tool_agent import main as tool_agent_main
from scripts.run_tool_agent import parse_args


class TestToolExecutor:
    def test_safe_python_executor(self):
        res = SafePythonExecutor.execute("2 ** 10 + 24")
        assert res == "1048"

        res_math = SafePythonExecutor.execute("math.sqrt(144)")
        assert float(res_math) == 12.0

        unsafe = SafePythonExecutor.execute("import os; os.listdir('.')")
        assert "Security violation" in unsafe

    def test_unit_converter(self):
        km_to_mile = UnitConverter.convert(10.0, "km", "mile")
        assert "6.2137" in km_to_mile

        c_to_f = UnitConverter.convert(100.0, "c", "f")
        assert "212.00°F" in c_to_f

    def test_tool_registry(self):
        registry = ToolRegistry()
        res = registry.execute_tool("python_calculator", {"code": "100 * 5"})
        assert res.success
        assert res.output == "500"

        missing = registry.execute_tool("non_existent_tool", {})
        assert not missing.success

    def test_tool_agent_run(self):
        agent = SovereignToolAgent(tier="tiny", device="cpu")
        res = agent.run("Calculate 15 * 12", max_steps=2)

        assert "task" in res
        assert "final_answer" in res
        assert "steps" in res
        assert res["steps_taken"] > 0

    def test_cli_parse_args(self):
        args = parse_args(
            ["--task", "Convert 50 km to miles", "--tier", "small", "--max-steps", "2"]
        )
        assert args.task == "Convert 50 km to miles"
        assert args.tier == "small"
        assert args.max_steps == 2

    def test_cli_main(self):
        code = tool_agent_main(
            ["--task", "Calculate 5 + 5", "--tier", "tiny", "--device", "cpu", "--json"]
        )
        assert code == 0
