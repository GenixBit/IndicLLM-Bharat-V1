"""Step-by-Step Reasoning Verifier and Benchmark Evaluator for IndicLLM-Bharat.

Evaluates and validates structured Chain-of-Thought (<think>...</think><answer>...</answer>)
traces across Mathematics, STEM, and Multilingual deduction.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bharat.reasoning.cot_engine import SOVEREIGN_REASONING_PROBLEMS
from bharat.serving.openai_server import BharatInferenceEngine


@dataclass
class StepVerificationResult:
    problem_id: str
    domain: str
    language: str
    has_think_tag: bool
    has_answer_tag: bool
    is_valid_structure: bool
    thought_text: str
    answer_text: str
    full_output: str


@dataclass
class ReasoningEvaluationReport:
    model_tier: str
    total_problems: int
    structure_valid_count: int
    structure_valid_pct: float
    per_domain_validity: dict[str, float]
    detailed_results: list[StepVerificationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_tier": self.model_tier,
            "total_problems": self.total_problems,
            "structure_valid_count": self.structure_valid_count,
            "structure_valid_pct": self.structure_valid_pct,
            "per_domain_validity": self.per_domain_validity,
            "detailed_results": [asdict(r) for r in self.detailed_results],
        }


class ReasoningVerifier:
    """Evaluates and validates step-by-step reasoning traces."""

    def __init__(
        self,
        tier: str = "1b",
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.engine = BharatInferenceEngine(
            tier=tier,
            checkpoint_path=checkpoint_path,
            device=device,
        )

    def parse_reasoning_trace(self, text: str) -> tuple[bool, bool, str, str]:
        """Extract <think> and <answer> blocks from generated output."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

        has_think = think_match is not None
        has_answer = answer_match is not None

        thought = think_match.group(1).strip() if think_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""

        return has_think, has_answer, thought, answer

    def evaluate_problems(
        self,
        problems: list[dict[str, Any]] | None = None,
        max_new_tokens: int = 128,
    ) -> ReasoningEvaluationReport:
        """Run verification across test problems."""
        target_problems = problems or SOVEREIGN_REASONING_PROBLEMS
        results: list[StepVerificationResult] = []
        domain_counts: dict[str, list[bool]] = {}

        for p in target_problems:
            dom = p["domain"]
            if dom not in domain_counts:
                domain_counts[dom] = []

            prompt = (
                f"You are IndicLLM-Bharat, a sovereign reasoning AI model. "
                f"Think step-by-step using <think>...</think> tags, then provide the final verified solution in <answer>...</answer> tags.\n\n"
                f"Problem: {p['problem']}\n\n"
                f"Assistant: "
            )

            output = self.engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
            )

            has_think, has_answer, thought, answer = self.parse_reasoning_trace(output)
            is_valid = has_think and has_answer

            domain_counts[dom].append(is_valid)

            results.append(
                StepVerificationResult(
                    problem_id=p["id"],
                    domain=dom,
                    language=p["language"],
                    has_think_tag=has_think,
                    has_answer_tag=has_answer,
                    is_valid_structure=is_valid,
                    thought_text=thought,
                    answer_text=answer,
                    full_output=output,
                )
            )

        valid_count = sum(1 for r in results if r.is_valid_structure)
        valid_pct = (valid_count / max(1, len(results))) * 100.0

        per_dom = {
            dom: (sum(1 for v in vals if v) / max(1, len(vals))) * 100.0
            for dom, vals in domain_counts.items()
        }

        return ReasoningEvaluationReport(
            model_tier=self.engine.tier,
            total_problems=len(results),
            structure_valid_count=valid_count,
            structure_valid_pct=round(valid_pct, 2),
            per_domain_validity=per_dom,
            detailed_results=results,
        )

    def export_reports(
        self,
        report: ReasoningEvaluationReport,
        output_dir: str | Path = "docs/benchmarks",
    ) -> tuple[Path, Path]:
        """Export Markdown benchmark report and JSON audit file."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        md_p = out_p / "REASONING_REPORT.md"
        json_p = out_p / "reasoning_audit.json"

        md_lines = [
            "# 🧠 IndicLLM-Bharat Chain-of-Thought (CoT) Reasoning Benchmark Report",
            "",
            f"- **Model Tier**: `{report.model_tier.upper()}`",
            f"- **Generated**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            f"- **Reasoning Structure Compliance**: **{report.structure_valid_pct:.1f}%** ({report.structure_valid_count}/{report.total_problems})",
            "",
            "## Domain Breakdown",
            "",
            "| Domain | Structure Compliance (%) |",
            "|:---|:---:|",
        ]
        for dom, score in report.per_domain_validity.items():
            md_lines.append(f"| `{dom}` | {score:.1f}% |")

        md_lines.extend(
            [
                "",
                "## Test Results Summary",
                "",
                "| Problem ID | Domain | Language | <think> Tag | <answer> Tag | Valid Structure |",
                "|:---|:---|:---:|:---:|:---:|:---:|",
            ]
        )
        for r in report.detailed_results:
            status = "✅ PASS" if r.is_valid_structure else "❌ FAIL"
            md_lines.append(
                f"| `{r.problem_id}` | `{r.domain}` | `{r.language}` | {r.has_think_tag} | {r.has_answer_tag} | {status} |"
            )

        with open(md_p, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        return md_p, json_p
