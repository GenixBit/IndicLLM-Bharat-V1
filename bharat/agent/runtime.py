"""Sovereign Multi-Turn Autonomous Agent Runtime for IndicLLM-Bharat.

Executes ReAct reasoning loops, parses tool invocations, runs sandboxed tools,
and generates final multilingual answers across all 22 Indian Languages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from bharat.agent.protocol import (
    extract_tool_calls,
    format_agent_system_prompt,
    format_tool_response,
)
from bharat.agent.tools import ToolRegistry
from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer, load_tokenizer
from bharat.training.scale_trainer import get_scale_tier_config


@dataclass
class AgentStep:
    iteration: int
    thought: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    query: str
    final_answer: str
    steps: list[AgentStep]
    total_iterations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BharatAgent:
    """Autonomous Multi-Turn Agent for IndicLLM-Bharat."""

    def __init__(
        self,
        tier: str = "1b",
        checkpoint_path: str | Path | None = None,
        registry: ToolRegistry | None = None,
        device: str = "cpu",
        max_iterations: int = 5,
    ) -> None:
        self.tier = tier
        self.device = torch.device(device)
        self.max_iterations = max_iterations
        self.registry = registry or ToolRegistry()
        self.tokenizer: BharatTokenizer = load_tokenizer("gpt2")

        if tier == "tiny":
            self.config = BharatModelConfig(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=4096,
            )
        else:
            self.config = get_scale_tier_config(tier, vocab_size=self.tokenizer.vocab_size)

        self.model = BharatForCausalLM(self.config).to(self.device)
        self.model.eval()

        if checkpoint_path and Path(checkpoint_path).is_file():
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"], strict=False)
            elif "state_dict" in state:
                self.model.load_state_dict(state["state_dict"], strict=False)

    def _generate_reply(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Run standard text generation."""
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.long, device=self.device
        ).unsqueeze(0)
        # Cap input length
        if input_ids.shape[1] > 2048:
            input_ids = input_ids[:, -2048:]

        generated = input_ids
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.model(generated)
                next_token_logits = out.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                # Check for EOS
                if next_token.item() == getattr(self.tokenizer, "eos_token_id", 50256):
                    break

        new_tokens = generated[0, input_ids.shape[1] :].tolist()
        return self.tokenizer.decode(new_tokens)

    def run(
        self,
        query: str,
        step_callback: Callable[[AgentStep], None] | None = None,
    ) -> AgentResponse:
        """Execute autonomous agent reasoning and tool-calling loop."""
        system_prompt = format_agent_system_prompt(self.registry.get_definitions())
        conversation = f"System: {system_prompt}\n\nUser: {query}\n\nAssistant: "

        steps: list[AgentStep] = []
        final_answer = ""

        for i in range(self.max_iterations):
            # Generate assistant step
            reply = self._generate_reply(conversation, max_new_tokens=256)
            conversation += reply

            # Detect tool calls
            tool_calls = extract_tool_calls(reply)

            if not tool_calls:
                # No more tools called; final answer reached
                final_answer = reply.strip()
                step = AgentStep(
                    iteration=i + 1,
                    thought=reply.strip(),
                    tool_calls=[],
                    tool_results=[],
                )
                steps.append(step)
                if step_callback:
                    step_callback(step)
                break

            # Execute tool calls
            executed_calls: list[dict[str, Any]] = []
            executed_results: list[dict[str, Any]] = []

            for call in tool_calls:
                executed_calls.append({"name": call.name, "arguments": call.arguments})
                res = self.registry.execute_tool(call.name, **call.arguments)
                executed_results.append(res.to_dict())

                # Append response to conversation
                res_formatted = format_tool_response(res)
                conversation += res_formatted

            step = AgentStep(
                iteration=i + 1,
                thought=reply.strip(),
                tool_calls=executed_calls,
                tool_results=executed_results,
            )
            steps.append(step)
            if step_callback:
                step_callback(step)

        if not final_answer:
            final_answer = reply.strip() or "Task completed."

        return AgentResponse(
            query=query,
            final_answer=final_answer,
            steps=steps,
            total_iterations=len(steps),
        )
