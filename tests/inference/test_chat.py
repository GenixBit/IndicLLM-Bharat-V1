from __future__ import annotations

from bharat.inference.chat import (
    InteractiveChatSession,
)
from scripts.chat import main as chat_main
from scripts.chat import parse_args


class TestChatSession:
    def test_interactive_chat_session(self):
        session = InteractiveChatSession(tier="tiny", device="cpu")
        resp = session.send_message("Hello Bharat AI!", stream=False, max_new_tokens=5)

        assert isinstance(resp, str)
        assert len(session.history) == 3  # system, user, assistant
        assert len(session.turns) == 1
        assert session.turns[0].tokens_per_sec >= 0.0

    def test_session_reset(self):
        session = InteractiveChatSession(tier="tiny", device="cpu")
        session.send_message("First message", stream=False, max_new_tokens=3)
        assert len(session.history) == 3

        session.reset()
        assert len(session.history) == 1  # Only system prompt
        assert len(session.turns) == 0

    def test_cli_parse_args(self):
        args = parse_args(["--tier", "350m", "--prompt", "Namaste!"])
        assert args.tier == "350m"
        assert args.prompt == "Namaste!"

    def test_cli_main_prompt(self):
        code = chat_main(["--tier", "tiny", "--prompt", "Hello!", "--device", "cpu"])
        assert code == 0
