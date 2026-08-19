from __future__ import annotations

from bharat.agent.tools import (
    IndicLanguageTool,
    KnowledgeRetrievalTool,
    MathCalculatorTool,
    PythonCodeInterpreterTool,
    ToolRegistry,
)


class TestAgentTools:
    def test_python_interpreter_success(self):
        tool = PythonCodeInterpreterTool()
        code = "a = 15\nb = 25\nprint(f'Sum: {a + b}')"
        res = tool.execute(code=code)
        assert res.success is True
        assert "Sum: 40" in res.output
        assert res.error is None

    def test_python_interpreter_math_module(self):
        tool = PythonCodeInterpreterTool()
        code = "print(math.factorial(5))"
        res = tool.execute(code=code)
        assert res.success is True
        assert res.output == "120"

    def test_python_interpreter_syntax_error(self):
        tool = PythonCodeInterpreterTool()
        code = "def bad_fn(: print('hello')"
        res = tool.execute(code=code)
        assert res.success is False
        assert "SyntaxError" in (res.error or "")

    def test_math_calculator_eval(self):
        tool = MathCalculatorTool()
        res = tool.execute(expression="sqrt(144) + 2**8")
        assert res.success is True
        assert res.output == "268.0"

    def test_math_calculator_trig(self):
        tool = MathCalculatorTool()
        res = tool.execute(expression="sin(pi/2)")
        assert res.success is True
        assert float(res.output) == 1.0

    def test_math_calculator_invalid(self):
        tool = MathCalculatorTool()
        res = tool.execute(expression="unknown_fn(123)")
        assert res.success is False
        assert "Math Evaluation Error" in (res.error or "")

    def test_knowledge_retrieval_query(self):
        tool = KnowledgeRetrievalTool()
        res = tool.execute(query="Standard Model")
        assert res.success is True
        assert "Standard Model" in res.output

    def test_indic_language_tool(self):
        tool = IndicLanguageTool()
        res = tool.execute(text="नमस्ते भारत", target_language="Tamil")
        assert res.success is True
        assert "Tamil" in res.output

    def test_tool_registry(self):
        reg = ToolRegistry()
        defs = reg.get_definitions()
        assert len(defs) >= 4

        res = reg.execute_tool("math_calculator", expression="5 * 5")
        assert res.success is True
        assert res.output == "25"

        bad_res = reg.execute_tool("non_existent_tool")
        assert bad_res.success is False
        assert "not registered" in (bad_res.error or "")
