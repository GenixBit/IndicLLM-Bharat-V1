from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as HFTokenizersTokenizer
from transformers import PreTrainedTokenizerFast

from bharat.tokenizer.base import BharatTokenizer


class _GPT2Wrapper(BharatTokenizer):
    """Wraps a transformers GPT2TokenizerFast as a BharatTokenizer."""

    def __init__(self, tok: PreTrainedTokenizerFast) -> None:
        self._tok = tok

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id or 50256

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id or 50256

    @property
    def tokenizer_type(self) -> str:
        return "gpt2"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [self._tok.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    def get_metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "special_tokens": {"eos": self.eos_token_id, "pad": self.pad_token_id},
            "is_fast": True,
        }

    def add_special_tokens(self, special_tokens: dict[str, list[str]]) -> int:
        return self._tok.add_special_tokens(special_tokens)

    def fingerprint(self) -> str:
        backend = getattr(self._tok, "backend_tokenizer", None)
        if backend is not None:
            raw = backend.to_str()
            extra = {
                "eos_id": self.eos_token_id,
                "pad_id": self.pad_token_id,
                "bos_id": self.bos_token_id,
                "unk_id": self.unk_token_id,
                "type": "gpt2",
                "name": getattr(self._tok, "name_or_path", ""),
            }
            data = {"serialized": raw, "config": extra}
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()
        fallback = json.dumps(
            {
                "type": "gpt2",
                "vocab_size": self.vocab_size,
                "eos_id": self.eos_token_id,
                "pad_id": self.pad_token_id,
                "name": getattr(self._tok, "name_or_path", ""),
            },
            sort_keys=True,
        )
        return hashlib.sha256(fallback.encode()).hexdigest()


class _SentencePieceNativeWrapper(BharatTokenizer):
    """Wraps a native google sentencepiece SentencePieceProcessor."""

    def __init__(self, processor: Any) -> None:
        self._sp = processor
        self._vocab_size = processor.vocab_size()
        self._bos_id: int = processor.bos_id()
        self._eos_id: int = processor.eos_id()
        self._pad_id: int = processor.pad_id()
        self._unk_id: int = processor.unk_id()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_id if self._eos_id >= 0 else 1

    @property
    def pad_token_id(self) -> int:
        return self._pad_id if self._pad_id >= 0 else 0

    @property
    def bos_token_id(self) -> int:
        return self._bos_id if self._bos_id >= 0 else 2

    @property
    def unk_token_id(self) -> int:
        return self._unk_id if self._unk_id >= 0 else 3

    @property
    def tokenizer_type(self) -> str:
        return "sentencepiece"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002
        return self._sp.encode(text)

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:  # noqa: ARG002
        return self._sp.encode(texts)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: ARG002
        return self._sp.decode(ids)

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:  # noqa: ARG002
        return self._sp.decode(batch)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "unk_token_id": self.unk_token_id,
            "special_tokens": {
                "eos": self.eos_token_id,
                "pad": self.pad_token_id,
                "bos": self.bos_token_id,
                "unk": self.unk_token_id,
            },
        }

    def fingerprint(self) -> str:
        raw = self._sp.serialized_model_proto()
        extra = {
            "type": "sentencepiece",
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_token_id,
            "pad_id": self.pad_token_id,
            "bos_id": self.bos_token_id,
            "unk_id": self.unk_token_id,
        }
        data = {"proto_bytes": raw.hex(), "config": extra}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class _SentencePieceHFWrapper(BharatTokenizer):
    """Wraps a HuggingFace tokenizers SentencePiece tokenizer as a BharatTokenizer."""

    def __init__(self, tok: HFTokenizersTokenizer) -> None:
        self._tok = tok
        self._vocab_size = tok.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        eos = self._tok.token_to_id("<|endoftext|>")
        if eos is not None:
            return eos
        eos = self._tok.token_to_id("</s>")
        return eos if eos is not None else 1

    @property
    def pad_token_id(self) -> int:
        pad = self._tok.token_to_id("<|pad|>")
        if pad is not None:
            return pad
        pad = self._tok.token_to_id("<pad>")
        return pad if pad is not None else 0

    @property
    def tokenizer_type(self) -> str:
        return "sentencepiece"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoded = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return encoded.ids

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        encoded = self._tok.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [e.ids for e in encoded]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return self._tok.decode_batch(batch, skip_special_tokens=skip_special_tokens)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "special_tokens": {"eos": self.eos_token_id, "pad": self.pad_token_id},
        }

    def fingerprint(self) -> str:
        raw = self._tok.to_str() if hasattr(self._tok, "to_str") else None
        if raw:
            extra = {
                "type": "sentencepiece_hf",
                "eos_id": self.eos_token_id,
                "pad_id": self.pad_token_id,
                "vocab_size": self.vocab_size,
            }
            data = {"serialized": raw, "config": extra}
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()
        data = {
            "type": "sentencepiece",
            "vocab_size": self.vocab_size,
            "vocab": {
                self._tok.id_to_token(i): i
                for i in range(self._tok.get_vocab_size())
                if self._tok.id_to_token(i) is not None
            },
            "normalizer": str(self._tok.normalizer.__class__.__name__)
            if hasattr(self._tok, "normalizer") and self._tok.normalizer
            else None,
            "pre_tokenizer": str(self._tok.pre_tokenizer.__class__.__name__)
            if hasattr(self._tok, "pre_tokenizer") and self._tok.pre_tokenizer
            else None,
            "model_type": str(self._tok.model.__class__.__name__)
            if hasattr(self._tok, "model")
            else None,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


class _HFWrapper(BharatTokenizer):
    """Wraps a generic HuggingFace PreTrainedTokenizerFast as a BharatTokenizer."""

    def __init__(self, tok: PreTrainedTokenizerFast) -> None:
        self._tok = tok

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id or 0

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id or 0

    @property
    def tokenizer_type(self) -> str:
        return "hf"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [self._tok.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    def get_metadata(self) -> dict[str, Any]:
        return {
            "tokenizer_type": self.tokenizer_type,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "special_tokens": {"eos": self.eos_token_id, "pad": self.pad_token_id},
            "is_fast": True,
        }

    def add_special_tokens(self, special_tokens: dict[str, list[str]]) -> int:
        return self._tok.add_special_tokens(special_tokens)

    def fingerprint(self) -> str:
        backend = getattr(self._tok, "backend_tokenizer", None)
        if backend is not None:
            raw = backend.to_str()
            extra = {
                "type": "hf",
                "eos_id": self.eos_token_id,
                "pad_id": self.pad_token_id,
                "bos_id": self.bos_token_id,
                "unk_id": self.unk_token_id,
                "name": getattr(self._tok, "name_or_path", ""),
            }
            data = {"serialized": raw, "config": extra}
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()
        fallback = json.dumps(
            {
                "type": "hf",
                "vocab_size": self.vocab_size,
                "name": getattr(self._tok, "name_or_path", "unknown"),
            },
            sort_keys=True,
        )
        return hashlib.sha256(fallback.encode()).hexdigest()


def _looks_like_path(s: str) -> bool:
    return "/" in s or "\\" in s or s.endswith((".json", ".model")) or Path(s).exists()


def _detect_tokenizer_type(tok: Any) -> str:
    if hasattr(tok, "name_or_path") and "gpt2" in getattr(tok, "name_or_path", "").lower():
        return "gpt2"
    return "hf"


def _detect_model_type(tok: HFTokenizersTokenizer) -> str:
    """Detect whether a tokenizers.Tokenizer is SentencePiece or plain BPE/Unigram."""
    model = tok.model
    model_name = model.__class__.__name__.lower()
    if ("sentencepiece" in model_name or "bpe" in model_name) and (
        hasattr(model, "piece_token") or model_name.startswith("sentencepiece")
    ):
        return "sentencepiece"
    if "unigram" in model_name:
        return "sentencepiece"
    return "bpe"


def _load_from_path(source: Path) -> BharatTokenizer:
    if source.suffix == ".json":
        tok = HFTokenizersTokenizer.from_file(str(source))
        mt = _detect_model_type(tok)
        if mt == "sentencepiece":
            return _SentencePieceHFWrapper(tok)
        from transformers import PreTrainedTokenizerFast as HFPretrainedTokenizerFast

        hf_tok = HFPretrainedTokenizerFast(tokenizer_object=tok)
        return _HFWrapper(hf_tok)
    if source.suffix == ".model":
        import sentencepiece as sp

        processor = sp.SentencePieceProcessor()
        processor.Load(str(source))
        return _SentencePieceNativeWrapper(processor)
    if source.is_dir():
        tok_config = source / "tokenizer.json"
        if tok_config.exists():
            tok = HFTokenizersTokenizer.from_file(str(tok_config))
            mt = _detect_model_type(tok)
            if mt == "sentencepiece":
                return _SentencePieceHFWrapper(tok)
            from transformers import PreTrainedTokenizerFast as HFPretrainedTokenizerFast

            hf_tok = HFPretrainedTokenizerFast(tokenizer_object=tok)
            return _HFWrapper(hf_tok)
        from transformers import PreTrainedTokenizerFast as HFPretrainedTokenizerFast

        try:
            tok = HFPretrainedTokenizerFast.from_pretrained(str(source))
            detected = _detect_tokenizer_type(tok)
            if detected == "gpt2":
                return _GPT2Wrapper(tok)
            return _HFWrapper(tok)
        except Exception as e:
            raise ValueError(f"Could not load tokenizer from directory: {source}") from e
    raise ValueError(f"Unsupported tokenizer path: {source}")


def load_tokenizer(
    source: str | Path | None = None,
    tokenizer_type: str | None = None,
    **kwargs: Any,
) -> BharatTokenizer:
    if source is None:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        return _GPT2Wrapper(tok)

    if isinstance(source, str):
        if source.lower() == "gpt2":
            from transformers import GPT2TokenizerFast

            tok = GPT2TokenizerFast.from_pretrained("gpt2")
            return _GPT2Wrapper(tok)

        if _looks_like_path(source):
            return _load_from_path(Path(source))

        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(source, **kwargs)
            detected = tokenizer_type or _detect_tokenizer_type(tok)
            if detected == "gpt2":
                return _GPT2Wrapper(tok)
            return _HFWrapper(tok)
        except Exception:
            raise ValueError(f"Could not load tokenizer '{source}'")

    if isinstance(source, Path):
        return _load_from_path(source)

    raise ValueError(f"Unsupported tokenizer source: {source}")
