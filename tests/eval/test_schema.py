from __future__ import annotations

import pytest

from bharat.eval.schema import EvalExample


class TestEvalExample:
    def test_minimal_valid(self) -> None:
        ex = EvalExample(example_id="qa_001", task_type="qa", prompt="What?", expected="Answer")
        assert ex.example_id == "qa_001"
        assert ex.digest()

    def test_to_dict_roundtrip(self) -> None:
        ex1 = EvalExample(
            example_id="qa_001",
            task_type="qa",
            prompt="What?",
            expected="Answer",
            choices=("a", "b"),
            metadata={"source": "test"},
        )
        d = ex1.to_dict()
        ex2 = EvalExample.from_dict(d)
        assert ex1 == ex2

    def test_digest_deterministic(self) -> None:
        ex1 = EvalExample(example_id="qa_001", task_type="qa", prompt="What?", expected="A")
        ex2 = EvalExample(example_id="qa_001", task_type="qa", prompt="What?", expected="A")
        assert ex1.digest() == ex2.digest()

    def test_digest_changes_with_field(self) -> None:
        ex1 = EvalExample(example_id="qa_001", task_type="qa", prompt="What?", expected="A")
        ex2 = EvalExample(example_id="qa_002", task_type="qa", prompt="What?", expected="A")
        assert ex1.digest() != ex2.digest()

    def test_empty_example_id_raises(self) -> None:
        with pytest.raises(ValueError, match="example_id"):
            EvalExample(example_id="", task_type="qa", prompt="What?", expected="A")

    def test_empty_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="prompt"):
            EvalExample(example_id="x", task_type="qa", prompt="", expected="A")

    def test_unsupported_task_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported task_type"):
            EvalExample(example_id="x", task_type="code", prompt="What?", expected="A")

    def test_classification_requires_choices(self) -> None:
        with pytest.raises(ValueError, match="classification tasks must have at least one choice"):
            EvalExample(example_id="x", task_type="classification", prompt="What?", expected="A")

    def test_classification_with_choices_ok(self) -> None:
        ex = EvalExample(
            example_id="x",
            task_type="classification",
            prompt="What?",
            expected="A",
            choices=("A", "B"),
        )
        assert ex.choices == ("A", "B")
