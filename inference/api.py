#!/usr/bin/env python3
"""
IndicLLM-Bharat-V1 — OpenAI-compatible Inference API

Serves a trained IndicLLM checkpoint via FastAPI with
OpenAI-compatible /v1/chat/completions endpoint.

Usage:
  python inference/api.py --checkpoint checkpoints/gpt2-124m-sft/final.pt
  python inference/api.py --checkpoint checkpoints/gpt2-124m-dpo/final.pt --port 8000

Test:
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"indicllm","messages":[{"role":"user","content":"Hello!"}]}'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.pretrain import GPT, GPTConfig

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise SystemExit("Install inference deps: pip install fastapi uvicorn pydantic")


# ── Global state ─────────────────────────────────────────────
MODEL: GPT | None = None
TOKENIZER = None
DEVICE = "cpu"
MODEL_NAME = "indicllm-bharat-v1"
APP_START = time.time()


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="IndicLLM-Bharat-V1 API",
    description="OpenAI-compatible inference API for IndicLLM-Bharat foundation model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response schemas ─────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    stream: bool = False


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.95


# ── Generation ───────────────────────────────────────────────
@torch.no_grad()
def generate(prompt_ids: list[int], max_new_tokens: int,
             temperature: float, top_p: float) -> list[int]:
    global MODEL, DEVICE
    model = MODEL
    ctx = nullcontext() if DEVICE in ("cpu", "mps") \
          else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    block_size = model.config.block_size
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    generated = []
    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]
        with ctx:
            logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Top-p (nucleus) sampling
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]

        tok_id = next_token.item()
        generated.append(tok_id)
        x = torch.cat([x, next_token.view(1, 1)], dim=1)

        # Stop at EOT
        if tok_id in (50256, 50257):
            break

    return generated


def format_chat_prompt(messages: list[Message]) -> str:
    """Convert chat messages to a single prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "model": MODEL_NAME,
        "status": "running",
        "uptime_seconds": int(time.time() - APP_START),
        "device": DEVICE,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": int(APP_START)}]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(req: ChatRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt_text = format_chat_prompt(req.messages)
    prompt_ids = TOKENIZER.encode(prompt_text, add_special_tokens=False)

    gen_ids = generate(prompt_ids, req.max_tokens, req.temperature, req.top_p)
    completion_text = TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[ChatChoice(
            index=0,
            message=Message(role="assistant", content=completion_text),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(gen_ids),
            total_tokens=len(prompt_ids) + len(gen_ids),
        ),
    )


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prompt_ids = TOKENIZER.encode(req.prompt, add_special_tokens=False)
    gen_ids = generate(prompt_ids, req.max_tokens, req.temperature, req.top_p)
    completion_text = TOKENIZER.decode(gen_ids, skip_special_tokens=True)
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"text": completion_text, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(gen_ids)},
    }


# ── Startup ───────────────────────────────────────────────────
def load_model(checkpoint: Path):
    global MODEL, TOKENIZER, DEVICE, MODEL_NAME

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    print(f"\n  Loading IndicLLM checkpoint: {checkpoint}")
    print(f"  Device: {DEVICE.upper()}")

    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", ckpt.get("model_args", {}))
    if not model_cfg:
        raise ValueError("Checkpoint missing model config.")

    MODEL = GPT(GPTConfig(**model_cfg)).to(DEVICE)
    MODEL.load_state_dict(ckpt["model"])
    MODEL.eval()

    params = sum(p.numel() for p in MODEL.parameters()) / 1e6
    print(f"  Model loaded: {params:.1f}M parameters")

    from transformers import GPT2TokenizerFast
    TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"  Tokenizer: GPT-2 (vocab={TOKENIZER.vocab_size})")

    MODEL_NAME = checkpoint.parent.name
    print(f"  Ready! Serving as '{MODEL_NAME}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicLLM Inference API")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--workers",    type=int, default=1)
    args = parser.parse_args()

    load_model(args.checkpoint)

    print(f"  API running at http://{args.host}:{args.port}")
    print(f"  Docs at      http://{args.host}:{args.port}/docs")
    print(f"  Test: curl http://localhost:{args.port}/health\n")

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
