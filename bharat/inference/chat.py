"""Interactive Sovereign Terminal Chat Client for IndicLLM-Bharat.

Provides multi-turn conversational interface across 22 Scheduled Indian Languages
and English with streaming typewriter output and throughput metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from bharat.serving.openai_server import BharatInferenceEngine, ChatMessage

SOVEREIGN_SYSTEM_PROMPT = (
    "You are IndicLLM-Bharat, a sovereign artificial intelligence foundation model developed for India and the world. "
    "You are deeply knowledgeable in all 22 Scheduled Indian Languages, STEM, algorithms, history, and modern technology. "
    "Respond accurately, politely, and helpfully."
)


@dataclass
class ChatTurn:
    user_input: str
    assistant_response: str
    generation_time_sec: float
    token_count: int
    tokens_per_sec: float


class InteractiveChatSession:
    """Multi-turn conversational chat session."""

    def __init__(
        self,
        tier: str = "1b",
        checkpoint_path: str | Path | None = None,
        system_prompt: str = SOVEREIGN_SYSTEM_PROMPT,
        device: str = "auto",
    ) -> None:
        self.engine = BharatInferenceEngine(
            tier=tier,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.system_prompt = system_prompt
        self.history: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
        self.turns: list[ChatTurn] = []

    def send_message(
        self,
        user_text: str,
        stream: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Send user message, receive assistant response, and update history."""
        self.history.append(ChatMessage(role="user", content=user_text))
        prompt = self.engine.format_chat_prompt(self.history)

        start = time.perf_counter()
        accumulated: list[str] = []

        if stream:
            print("\n🤖 Assistant: ", end="", flush=True)
            for chunk in self.engine.generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                print(chunk, end="", flush=True)
                accumulated.append(chunk)
            print("\n")
        else:
            resp = self.engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            accumulated.append(resp)

        full_response = "".join(accumulated).strip()
        elapsed = time.perf_counter() - start
        token_count = len(self.engine.tokenizer.encode(full_response))
        tps = token_count / max(1e-5, elapsed)

        self.history.append(ChatMessage(role="assistant", content=full_response))
        self.turns.append(
            ChatTurn(
                user_input=user_text,
                assistant_response=full_response,
                generation_time_sec=elapsed,
                token_count=token_count,
                tokens_per_sec=tps,
            )
        )

        return full_response

    def reset(self) -> None:
        """Clear conversation history except system prompt."""
        self.history = [ChatMessage(role="system", content=self.system_prompt)]
        self.turns = []


def run_interactive_terminal(session: InteractiveChatSession) -> None:
    """Launch interactive terminal REPL."""
    print("=" * 65)
    print("🇮🇳 Welcome to IndicLLM-Bharat Interactive Sovereign Terminal")
    print(f"  • Model Tier:   {session.engine.tier.upper()}")
    print(f"  • Device:       {session.engine.device}")
    print("  • Type 'exit', 'quit', or Ctrl+C to stop.")
    print("  • Type 'clear' or 'reset' to reset conversation history.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("👤 User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nDhanyavaad! Goodbye.\n")
                break
            if user_input.lower() in ("clear", "reset"):
                session.reset()
                print("\n🧹 Conversation history cleared.\n")
                continue

            session.send_message(user_input, stream=True)
        except (KeyboardInterrupt, EOFError):
            print("\n\nDhanyavaad! Goodbye.\n")
            break
