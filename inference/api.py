#!/usr/bin/env python3
"""
OpenAI-compatible chat completions API.

Usage:
  uvicorn inference.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="llm-lab API", version="0.1.0")

_model = None
_tokenizer = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str = "llm-lab"
    messages: list[Message]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]


def verify_api_key(authorization: str | None) -> None:
    expected = os.environ.get("API_KEY", "dev-key-change-me")
    if not authorization or authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = os.environ.get("MODEL_PATH", "checkpoints/gpt2-124m-sft")
    if not os.path.isdir(path):
        path = "gpt2"
    _tokenizer = AutoTokenizer.from_pretrained(path)
    _model = AutoModelForCausalLM.from_pretrained(path)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    _model.to(device)
    _model.eval()
    return _model, _tokenizer


def build_prompt(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        parts.append(f"{m.role}: {m.content}")
    parts.append("assistant:")
    return "\n".join(parts)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, authorization: str | None = Header(default=None)):
    verify_api_key(authorization)
    import torch

    model, tokenizer = load_model()
    prompt = build_prompt(req.messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=text.strip()),
                finish_reason="stop",
            )
        ],
    )
