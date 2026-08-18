from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_production_tokenizer_evidence_readiness import (
    main as readiness_main,
)
from scripts.generate_bharatbench_predictions import main as predictions_main


class TestEvidenceAndEvalScripts:
    def test_generate_bharatbench_predictions_cli(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        examples_file = tmp_path / "examples.jsonl"
        example_records = [
            json.dumps(
                {
                    "example_id": "qa_001",
                    "task_category": "language",
                    "task_name": "reading_comprehension",
                    "task_type": "qa",
                    "language": "hi",
                    "prompt": "भारत की राजधानी क्या है?",
                    "expected": "नई दिल्ली",
                }
            ),
            json.dumps(
                {
                    "example_id": "qa_002",
                    "task_category": "reasoning",
                    "task_name": "math_reasoning",
                    "task_type": "qa",
                    "language": "en",
                    "prompt": "What is 2 + 2?",
                    "expected": "4",
                }
            ),
        ]
        examples_file.write_text("\n".join(example_records), encoding="utf-8")

        out_preds = tmp_path / "predictions.jsonl"
        ret = predictions_main(
            [
                "--examples",
                str(examples_file),
                "--output",
                str(out_preds),
                "--adapter",
                "expected",
                "--json",
            ]
        )
        assert ret == 0
        assert out_preds.is_file()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["examples"] == 2
        assert data["predictions"] == 2
        assert data["adapter"] == "expected"

    def test_check_production_tokenizer_evidence_readiness_cli_missing(
        self, tmp_path: Path
    ) -> None:
        missing_file = tmp_path / "non_existent_manifest.json"
        ret = readiness_main([str(missing_file)])
        assert ret == 2
