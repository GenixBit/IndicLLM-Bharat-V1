"""Sovereign Multilingual Chain-of-Thought (CoT) Reasoning Engine for IndicLLM-Bharat.

Generates deep reflective reasoning traces with structured <think>...</think><answer>...</answer>
formats across Mathematics, STEM proofs, Algorithms, and Indic linguistic deduction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOVEREIGN_REASONING_PROBLEMS: list[dict[str, Any]] = [
    {
        "id": "reasoning_math_ramanujan_01",
        "domain": "mathematics_number_theory",
        "language": "en",
        "problem": "Find the smallest positive integer that can be expressed as the sum of two cubes in two different ways (the Hardy-Ramanujan number) and prove its uniqueness.",
        "thought_trace": (
            "1. Let the target integer be N.\n"
            "2. We require N = a^3 + b^3 = c^3 + d^3 where (a, b) != (c, d) and a, b, c, d are positive integers.\n"
            "3. Testing small cubes:\n"
            "   1^3 = 1, 2^3 = 8, 3^3 = 27, 4^3 = 64, 5^3 = 125, 6^3 = 216, 7^3 = 343, 8^3 = 512, 9^3 = 729, 10^3 = 1000, 12^3 = 1728.\n"
            "4. Notice: 1^3 + 12^3 = 1 + 1728 = 1729.\n"
            "5. Notice: 9^3 + 10^3 = 729 + 1000 = 1729.\n"
            "6. Checking for smaller candidate numbers:\n"
            "   All pairs of cubes below 1729 yield unique sums upon exhaustive enumeration.\n"
            "7. Thus, 1729 is the minimal such integer."
        ),
        "final_answer": (
            "The smallest positive integer expressible as the sum of two cubes in two different ways is 1729 (the Hardy-Ramanujan number), "
            "where 1729 = 1^3 + 12^3 = 9^3 + 10^3."
        ),
    },
    {
        "id": "reasoning_physics_orbital_02",
        "domain": "astrodynamics_physics",
        "language": "en",
        "problem": "Derive the vis-viva equation for spacecraft orbital velocity in an elliptical orbit around Earth.",
        "thought_trace": (
            "1. By conservation of specific mechanical energy in a two-body gravitational system:\n"
            "   E = (1/2)*v^2 - mu/r, where mu = G*M.\n"
            "2. For an elliptical orbit with semi-major axis 'a', the total specific mechanical energy is E = -mu / (2*a).\n"
            "3. Equating the two expressions for energy:\n"
            "   (1/2)*v^2 - mu/r = -mu / (2*a)\n"
            "4. Multiplying both sides by 2:\n"
            "   v^2 - 2*mu/r = -mu/a\n"
            "5. Adding 2*mu/r to both sides:\n"
            "   v^2 = mu * (2/r - 1/a)\n"
            "6. Taking the square root gives the orbital velocity v = sqrt(mu * (2/r - 1/a))."
        ),
        "final_answer": (
            "The vis-viva equation is: v = sqrt(mu * (2/r - 1/a)), where v is the orbital velocity, "
            "mu = GM is the standard gravitational parameter, r is the current orbital distance from the central body, "
            "and a is the semi-major axis of the elliptical orbit."
        ),
    },
    {
        "id": "reasoning_algo_dp_03",
        "domain": "algorithms_computer_science",
        "language": "en",
        "problem": "Write an optimal O(n) dynamic programming solution in Python to find the maximum sum of a contiguous subarray (Kadane's Algorithm).",
        "thought_trace": (
            "1. Problem: Given an integer array nums, find the contiguous subarray with the largest sum.\n"
            "2. Let dp[i] represent the maximum subarray sum ending at index i.\n"
            "3. Recurrence relation: dp[i] = max(nums[i], dp[i-1] + nums[i]).\n"
            "4. Space optimization: We only need the previous value, so maintain max_current and max_global in O(1) auxiliary memory.\n"
            "5. Initialization: max_current = max_global = nums[0].\n"
            "6. Iterate from index 1 to len(nums)-1:\n"
            "   max_current = max(nums[i], max_current + nums[i])\n"
            "   max_global = max(max_global, max_current)\n"
            "7. Return max_global in O(n) time and O(1) space."
        ),
        "final_answer": (
            "```python\ndef max_sub_array(nums: list[int]) -> int:\n"
            "    if not nums:\n"
            "        return 0\n"
            "    max_current = max_global = nums[0]\n"
            "    for x in nums[1:]:\n"
            "        max_current = max(x, max_current + x)\n"
            "        max_global = max(max_global, max_current)\n"
            "    return max_global\n```"
        ),
    },
    {
        "id": "reasoning_indic_math_hindi_04",
        "domain": "mathematics_calculus",
        "language": "hi",
        "problem": "फलन f(x) = x^3 - 6x^2 + 9x + 15 के स्थानीय उच्चिष्ठ (local maxima) और निम्निष्ठ (local minima) मान ज्ञात कीजिए।",
        "thought_trace": (
            "1. दिया गया फलन: f(x) = x^3 - 6x^2 + 9x + 15\n"
            "2. प्रथम अवकलज (first derivative) f'(x) ज्ञात करें:\n"
            "   f'(x) = 3x^2 - 12x + 9 = 3(x^2 - 4x + 3) = 3(x - 1)(x - 3)\n"
            "3. क्रांतिक बिंदु (critical points) के लिए f'(x) = 0 रखें:\n"
            "   3(x - 1)(x - 3) = 0 => x = 1 और x = 3.\n"
            "4. द्वितीय अवकलज (second derivative) f''(x) ज्ञात करें:\n"
            "   f''(x) = 6x - 12\n"
            "5. x = 1 पर परीक्षण:\n"
            "   f''(1) = 6(1) - 12 = -6 < 0 => x = 1 पर स्थानीय उच्चिष्ठ (local maximum) है।\n"
            "   f(1) = 1^3 - 6(1)^2 + 9(1) + 15 = 1 - 6 + 9 + 15 = 19.\n"
            "6. x = 3 पर परीक्षण:\n"
            "   f''(3) = 6(3) - 12 = +6 > 0 => x = 3 पर स्थानीय निम्निष्ठ (local minimum) है।\n"
            "   f(3) = 3^3 - 6(3)^2 + 9(3) + 15 = 27 - 54 + 27 + 15 = 15.\n"
            "7. निष्कर्ष: स्थानीय उच्चिष्ठ मान 19 (x = 1 पर) और स्थानीय निम्निष्ठ मान 15 (x = 3 पर) है।"
        ),
        "final_answer": (
            "फलन f(x) का x = 1 पर स्थानीय उच्चिष्ठ मान (Local Maximum) 19 है, "
            "तथा x = 3 पर स्थानीय निम्निष्ठ मान (Local Minimum) 15 है।"
        ),
    },
]


@dataclass
class ReasoningTraceSample:
    problem_id: str
    domain: str
    language: str
    problem: str
    thought_trace: str
    final_answer: str

    def to_formatted_cot_text(self) -> str:
        """Format into standardized reflective <think>...</think><answer>...</answer> structure."""
        return f"<think>\n{self.thought_trace.strip()}\n</think>\n<answer>\n{self.final_answer.strip()}\n</answer>"

    def to_sft_record(self) -> dict[str, Any]:
        return {
            "prompt": self.problem,
            "response": self.to_formatted_cot_text(),
            "metadata": {
                "problem_id": self.problem_id,
                "domain": self.domain,
                "language": self.language,
            },
        }


class CoTReasoningEngine:
    """Manages reasoning curriculum generation and synthesis."""

    def __init__(self, problems: list[dict[str, Any]] | None = None) -> None:
        self.problems = problems or SOVEREIGN_REASONING_PROBLEMS

    def get_samples(self) -> list[ReasoningTraceSample]:
        """Generate structured reasoning trace samples."""
        samples: list[ReasoningTraceSample] = []
        for p in self.problems:
            samples.append(
                ReasoningTraceSample(
                    problem_id=p["id"],
                    domain=p["domain"],
                    language=p["language"],
                    problem=p["problem"],
                    thought_trace=p["thought_trace"],
                    final_answer=p["final_answer"],
                )
            )
        return samples

    def export_curriculum(
        self,
        output_dir: str | Path = "data/reasoning",
        multiplier: int = 1,
    ) -> Path:
        """Export reasoning curriculum dataset in JSONL format."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        base_samples = self.get_samples()
        expanded: list[ReasoningTraceSample] = []
        for _ in range(max(1, multiplier)):
            expanded.extend(base_samples)

        jsonl_file = out_p / "reasoning_curriculum.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for s in expanded:
                f.write(json.dumps(s.to_sft_record(), ensure_ascii=False) + "\n")

        return jsonl_file
