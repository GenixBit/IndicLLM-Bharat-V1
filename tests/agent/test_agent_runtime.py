from __future__ import annotations

from bharat.agent.protocol import (
    extract_tool_calls,
    format_agent_system_prompt,
    format_tool_call,
    format_tool_response,
)
from bharat.agent.runtime import AgentStep, BharatAgent
from bharat.agent.tools import ToolRegistry, ToolResult
from scripts.run_bharat_agent import parse_args


class TestAgentRuntime:
    def test_protocol_formatting_and_parsing(self):
        reg = ToolRegistry()
        sys_prompt = format_agent_system_prompt(reg.get_definitions())
        assert "IndicLLM-Bharat Agent" in sys_prompt
        assert "python_interpreter" in sys_prompt

        tool_call_str = format_tool_call("math_calculator", {"expression": "2 + 2"})
        assert "<|tool_call|>" in tool_call_str
        assert "</|tool_call|>" in tool_call_str

        extracted = extract_tool_calls(tool_call_str)
        assert len(extracted) == 1
        assert extracted[0].name == "math_calculator"
        assert extracted[0].arguments == {"expression": "2 + 2"}

        res_str = format_tool_response(ToolResult(success=True, output="4"))
        assert "<|tool_response|>" in res_str
        assert "4" in res_str

    def test_agent_run_tiny(self):
        agent = BharatAgent(tier="tiny", device="cpu", max_iterations=2)
        response = agent.run("What is 10 + 20?")
        assert response.query == "What is 10 + 20?"
        assert response.final_answer != ""
        assert len(response.steps) >= 1
        d = response.to_dict()
        assert "steps" in d
        assert "final_answer" in d

    def test_agent_with_callback(self):
        agent = BharatAgent(tier="tiny", device="cpu", max_iterations=1)
        steps_recorded: list[AgentStep] = []

        def callback(s: AgentStep):
            steps_recorded.append(s)

        response = agent.run("Hello Bharat", step_callback=callback)
        assert response.final_answer != ""
        assert len(steps_recorded) == 1
        assert steps_recorded[0].iteration == 1

    def test_cli_parse_args(self):
        args = parse_args(["--query", "Solve quadratic equation", "--tier", "1b"])
        assert args.query == "Solve quadratic equation"
        assert args.tier == "1b"
