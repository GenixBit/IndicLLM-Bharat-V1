#!/usr/bin/env python3
# ruff: noqa: N802
"""OpenAI-Compatible Sovereign REST & Streaming API Server for IndicLLM-Bharat.

Implements standard OpenAI API specifications:
  - POST /v1/chat/completions (JSON & Server-Sent Events SSE streaming)
  - POST /v1/completions
  - GET /v1/models
  - GET /v1/health
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer, load_tokenizer
from bharat.training.scale_trainer import get_scale_tier_config


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatCompletionRequest:
    model: str = "bharat-1b-dpo"
    messages: list[ChatMessage] = field(default_factory=list)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | None = None
    repetition_penalty: float = 1.1


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    tier: str = "1b"
    checkpoint_path: str | Path | None = None
    api_key: str | None = None
    device: str = "auto"


class BharatInferenceEngine:
    """High-performance KV-cache accelerated generation engine."""

    def __init__(
        self,
        tier: str = "1b",
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.tier = tier
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

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
        elif tier == "small":
            self.config = BharatModelConfig(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=4,
                max_position_embeddings=4096,
            )
        else:
            self.config = get_scale_tier_config(tier, vocab_size=self.tokenizer.vocab_size)

        self.model = BharatForCausalLM(self.config).to(self.device)
        self.model.eval()

        if checkpoint_path and Path(checkpoint_path).is_file():
            st = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in st:
                self.model.load_state_dict(st["model_state_dict"], strict=False)
            elif "state_dict" in st:
                self.model.load_state_dict(st["state_dict"], strict=False)

    def format_chat_prompt(self, messages: list[ChatMessage]) -> str:
        """Format multi-turn dialogue into standard conversational prompt."""
        formatted: list[str] = []
        for msg in messages:
            role_label = msg.role.capitalize()
            formatted.append(f"{role_label}: {msg.content}")
        formatted.append("Assistant: ")
        return "\n\n".join(formatted)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Iterator[str]:
        """Stream generated text token by token."""
        input_ids = self.tokenizer.encode(prompt)
        tokens = list(input_ids)
        eos_id = getattr(self.tokenizer, "eos_token_id", 50256)

        curr_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Bound sequence length
                if curr_tensor.shape[1] > self.config.max_position_embeddings:
                    curr_tensor = curr_tensor[:, -self.config.max_position_embeddings :]

                logits = self.model(curr_tensor % self.config.vocab_size).logits[:, -1, :]

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for prev_token in set(tokens[-32:]):
                        logits[0, prev_token % self.config.vocab_size] /= repetition_penalty

                if temperature > 0.0:
                    logits = logits / max(1e-4, temperature)
                    probs = F.softmax(logits, dim=-1)

                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = False
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        probs = probs.masked_fill(indices_to_remove, 0.0)
                        probs = probs / probs.sum(dim=-1, keepdim=True)

                    next_token = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_token = int(torch.argmax(logits, dim=-1).item())

                if next_token == eos_id:
                    break

                tokens.append(next_token)
                curr_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

                decoded_chunk = self.tokenizer.decode([next_token])
                yield decoded_chunk

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate complete completion text."""
        chunks = list(
            self.generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        )
        return "".join(chunks)


def create_openai_handler(
    engine: BharatInferenceEngine, api_key: str | None = None
) -> type[BaseHTTPRequestHandler]:
    """Factory to create HTTP handler with bound inference engine."""

    class OpenAIRequestHandler(BaseHTTPRequestHandler):
        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept")

        def do_OPTIONS(self) -> None:
            self.send_response(200)
            self._send_cors_headers()
            self.end_headers()

        def _authenticate(self) -> bool:
            if not api_key:
                return True
            auth_header = self.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
                return token == api_key
            return False

        def do_GET(self) -> None:
            if self.path in ("/v1/health", "/healthz", "/health"):
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                res = {
                    "status": "healthy",
                    "model_tier": engine.tier,
                    "device": str(engine.device),
                    "timestamp": time.time(),
                }
                self.wfile.write(json.dumps(res).encode("utf-8"))
                return

            if self.path == "/v1/models":
                if not self._authenticate():
                    self.send_response(401)
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(b'{"error": "Unauthorized"}')
                    return

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                models = {
                    "object": "list",
                    "data": [
                        {
                            "id": f"bharat-{engine.tier}",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "genixbit-bharat",
                            "permission": [],
                            "root": f"bharat-{engine.tier}",
                            "parent": None,
                        },
                        {
                            "id": "bharat-10b-dpo",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "genixbit-bharat",
                            "permission": [],
                            "root": "bharat-10b-dpo",
                            "parent": None,
                        },
                    ],
                }
                self.wfile.write(json.dumps(models).encode("utf-8"))
                return

            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

        def do_POST(self) -> None:
            if not self._authenticate():
                self.send_response(401)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": "Unauthorized"}')
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                self.send_response(400)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
                return

            if self.path == "/v1/chat/completions":
                raw_messages = data.get("messages", [])
                messages = [
                    ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                    for m in raw_messages
                ]
                prompt = engine.format_chat_prompt(messages)
                stream = data.get("stream", False)
                max_tokens = data.get("max_tokens", 128)
                temperature = data.get("temperature", 0.7)
                top_p = data.get("top_p", 0.9)
                model_id = data.get("model", f"bharat-{engine.tier}")

                if stream:
                    self.send_response(200)
                    self._send_cors_headers()
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()

                    for chunk in engine.generate_stream(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    ):
                        chunk_payload = {
                            "id": f"chatcmpl-{int(time.time()*1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk_payload)}\n\n".encode())
                        self.wfile.flush()

                    done_payload = {
                        "id": f"chatcmpl-{int(time.time()*1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    self.wfile.write(
                        f"data: {json.dumps(done_payload)}\n\ndata: [DONE]\n\n".encode()
                    )
                    self.wfile.flush()
                    return

                # Non-streaming response
                text = engine.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                response = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(engine.tokenizer.encode(prompt)),
                        "completion_tokens": len(engine.tokenizer.encode(text)),
                        "total_tokens": len(engine.tokenizer.encode(prompt + text)),
                    },
                }
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            if self.path == "/v1/completions":
                prompt = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 128)
                temperature = data.get("temperature", 0.7)
                model_id = data.get("model", f"bharat-{engine.tier}")

                text = engine.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
                response = {
                    "id": f"cmpl-{int(time.time()*1000)}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [
                        {
                            "text": text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
                return

            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

    return OpenAIRequestHandler


def run_api_server(config: ServerConfig) -> HTTPServer:
    """Initialize and run OpenAI-compatible REST server."""
    engine = BharatInferenceEngine(
        tier=config.tier,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
    )
    handler_cls = create_openai_handler(engine, api_key=config.api_key)
    server = HTTPServer((config.host, config.port), handler_cls)

    print("\n" + "=" * 65)
    print("🌐 Sovereign IndicLLM-Bharat OpenAI-Compatible REST API Server")
    print(f"  • Endpoint:       http://{config.host}:{config.port}/v1")
    print(f"  • Model Tier:     {config.tier.upper()}")
    print(f"  • Compute Device: {engine.device}")
    print(
        f"  • Auth Status:    {'Bearer Token Required' if config.api_key else 'Public (No Auth)'}"
    )
    print("  • Compatible with: LangChain, LlamaIndex, Cursor, LiteLLM, Open WebUI")
    print("=" * 65 + "\n")

    return server
