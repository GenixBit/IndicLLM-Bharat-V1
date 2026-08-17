#!/usr/bin/env python3
"""IndicLLM-Bharat-V1 — OpenAI-compatible Inference API.

Serves a trained IndicLLM checkpoint via FastAPI with
OpenAI-compatible /v1/chat/completions, /v1/completions, /v1/models, and /health endpoints.

Usage:
  python inference/api.py --checkpoint checkpoints/bharat-350m/final.pt
  python inference/api.py --checkpoint checkpoints/bharat-350m/final.pt --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from bharat.models.bharat_model import BharatForCausalLM
from bharat.models.config import BharatModelConfig
from bharat.tokenizer import BharatTokenizer
from bharat.tokenizer import load_tokenizer as load_bharat_tokenizer
from train.pretrain import GPT, GPTConfig

# ── Global state ─────────────────────────────────────────────
MODEL: torch.nn.Module | None = None
TOKENIZER: BharatTokenizer | None = None
DEVICE: str = "cpu"
MODEL_NAME: str = "indicllm-bharat-v1"
APP_START: float = time.time()


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="IndicLLM-Bharat-V1 API",
    description="OpenAI-compatible inference API for IndicLLM-Bharat foundation model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
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
    stream: bool = False


# ── Generation ───────────────────────────────────────────────
@torch.no_grad()
def generate(
    prompt_ids: list[int], max_new_tokens: int, temperature: float, top_p: float
) -> list[int]:
    global MODEL, DEVICE, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        raise RuntimeError("Model or tokenizer not initialized")

    ctx = (
        nullcontext()
        if DEVICE in ("cpu", "mps")
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

    block_size = getattr(
        MODEL.config,
        "max_position_embeddings",
        getattr(MODEL.config, "block_size", 1024),
    )
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    generated: list[int] = []
    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]
        with ctx:
            out = MODEL(x_cond)
            logits = out.logits if hasattr(out, "logits") else out[0]
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        # Top-p (nucleus) sampling
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]

        tok_id = int(next_token.item())
        generated.append(tok_id)
        x = torch.cat([x, next_token.view(1, 1)], dim=1)

        # Stop at EOS
        if tok_id == TOKENIZER.eos_token_id:
            break

    return generated


def format_chat_prompt(messages: list[Message]) -> str:
    """Convert chat messages to a single prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|system|>\n{msg.content}")
        elif msg.role == "user":
            parts.append(f"<|instruction|>\n{msg.content}")
        elif msg.role == "assistant":
            parts.append(f"<|response|>\n{msg.content}")
    parts.append("<|response|>\n")
    return "\n".join(parts)


async def stream_chat_generator(
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    chat_id: str,
) -> AsyncGenerator[str, None]:
    global MODEL, DEVICE, TOKENIZER
    if MODEL is None or TOKENIZER is None:
        return

    ctx = (
        nullcontext()
        if DEVICE in ("cpu", "mps")
        else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    )

    block_size = getattr(
        MODEL.config,
        "max_position_embeddings",
        getattr(MODEL.config, "block_size", 1024),
    )
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_tokens):
        x_cond = x[:, -block_size:]
        with ctx:
            out = MODEL(x_cond)
            logits = out.logits if hasattr(out, "logits") else out[0]
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum()
        next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]

        tok_id = int(next_token.item())
        delta_text = TOKENIZER.decode([tok_id], skip_special_tokens=True)
        x = torch.cat([x, next_token.view(1, 1)], dim=1)

        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"

        if tok_id == TOKENIZER.eos_token_id:
            break

    final_payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "status": "healthy" if MODEL is not None else "unloaded",
        "uptime_seconds": int(time.time() - APP_START),
        "device": DEVICE,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(APP_START),
                "owned_by": "genixbit",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None,
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> Any:
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt_text = format_chat_prompt(req.messages)
    prompt_ids = TOKENIZER.encode(prompt_text, add_special_tokens=False)

    if req.stream:
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        return StreamingResponse(
            stream_chat_generator(prompt_ids, req.max_tokens, req.temperature, req.top_p, chat_id),
            media_type="text/event-stream",
        )

    gen_ids = generate(prompt_ids, req.max_tokens, req.temperature, req.top_p)
    completion_text = TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=completion_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(gen_ids),
            total_tokens=len(prompt_ids) + len(gen_ids),
        ),
    )


@app.post("/v1/completions")
def completions(req: CompletionRequest) -> dict[str, Any]:
    if MODEL is None or TOKENIZER is None:
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


# ── Model Loader ─────────────────────────────────────────────
def load_model(
    checkpoint: str | Path,
    device_override: str | None = None,
    tokenizer_override: str | None = None,
) -> None:
    global MODEL, TOKENIZER, DEVICE, MODEL_NAME

    if device_override:
        DEVICE = device_override
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    ckpt_path = Path(checkpoint).resolve()
    print(f"\n  Loading IndicLLM checkpoint: {ckpt_path}")
    print(f"  Device: {DEVICE.upper()}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Detect if Native Bharat Model
    is_bharat = False
    model_cfg_obj: Any = None

    if "metadata" in ckpt and hasattr(ckpt["metadata"], "model_config"):
        model_cfg_dict = ckpt["metadata"].model_config
        model_cfg_obj = BharatModelConfig.from_dict(model_cfg_dict)
        is_bharat = True
    elif "model_config" in ckpt:
        m_cfg = ckpt["model_config"]
        model_cfg_obj = BharatModelConfig.from_dict(
            m_cfg if isinstance(m_cfg, dict) else m_cfg.__dict__
        )
        is_bharat = True
    elif "config" in ckpt and "hidden_size" in ckpt.get("config", {}):
        model_cfg_obj = BharatModelConfig.from_dict(ckpt["config"])
        is_bharat = True
    elif "model" in ckpt:
        keys = list(ckpt["model"].keys())
        if any("layers." in k for k in keys) or any("model.embed_tokens" in k for k in keys):
            is_bharat = True
            model_cfg_obj = BharatModelConfig()

    state = ckpt.get("model", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    if is_bharat:
        if model_cfg_obj is None:
            model_cfg_obj = BharatModelConfig()
        MODEL = BharatForCausalLM(model_cfg_obj).to(DEVICE)
        MODEL.load_state_dict(state, strict=False)
        MODEL.eval()
        params = sum(p_t.numel() for p_t in MODEL.parameters()) / 1e6
        print("  Architecture: BharatForCausalLM (RoPE, RMSNorm, SwiGLU, GQA)")
        print(f"  Parameters  : {params:.1f}M params")
    else:
        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", ckpt.get("model_args", {}))
        if not model_cfg:
            model_cfg = {
                "vocab_size": 50257,
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "block_size": 1024,
            }
        gpt_cfg = GPTConfig(**model_cfg)
        MODEL = GPT(gpt_cfg).to(DEVICE)
        MODEL.load_state_dict(state, strict=False)
        MODEL.eval()
        params = sum(p_t.numel() for p_t in MODEL.parameters()) / 1e6
        print("  Architecture: GPT-2 Legacy")
        print(f"  Parameters  : {params:.1f}M params")

    # Load tokenizer
    tok_src = tokenizer_override
    if not tok_src and "metadata" in ckpt and hasattr(ckpt["metadata"], "tokenizer_type"):
        tok_src = getattr(ckpt["metadata"], "tokenizer_type", None)
    if not tok_src and "tokenizer" in ckpt.get("config", {}):
        tok_src = ckpt["config"]["tokenizer"].get("source")

    TOKENIZER = load_bharat_tokenizer(tok_src)
    MODEL_NAME = ckpt_path.stem
    print(f"  Tokenizer   : {TOKENIZER.tokenizer_type} (vocab: {TOKENIZER.vocab_size:,})")
    print(f"  Ready! Serving as '{MODEL_NAME}'\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IndicLLM OpenAI-Compatible Inference API")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path or model ID")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        load_model(args.checkpoint, device_override=args.device, tokenizer_override=args.tokenizer)
    except Exception as e:
        print(f"error loading model: {e}", file=sys.stderr)
        return 1

    import uvicorn

    print(f"  API running at http://{args.host}:{args.port}")
    print(f"  Docs at      http://{args.host}:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
