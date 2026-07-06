from __future__ import annotations

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
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

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
            "special_tokens": {
                "eos": self.eos_token_id,
                "pad": self.pad_token_id,
            },
            "is_fast": True,
        }

    def add_special_tokens(self, special_tokens: dict[str, list[str]]) -> int:
        return self._tok.add_special_tokens(special_tokens)

    def fingerprint(self) -> str:
        import hashlib
        import json

        tok = self._tok.backend_tokenizer if hasattr(self._tok, "backend_tokenizer") else None
        if tok is not None:
            data = {
                "type": "gpt2",
                "vocab": tok.get_vocab(),
                "merges": tok.model.merges if hasattr(tok.model, "merges") else [],
                "normalizer": str(tok.normalizer.__class__.__name__) if tok.normalizer else None,
                "pre_tokenizer": str(tok.pre_tokenizer.__class__.__name__)
                if tok.pre_tokenizer
                else None,
                "special_tokens": {
                    t.content: tid for tid, t in tok.get_added_tokens_decoder().items()
                },
                "model_type": str(tok.model.__class__.__name__),
            }
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()

        fallback = json.dumps(
            {
                "type": "gpt2",
                "vocab_size": self.vocab_size,
                "eos_id": self.eos_token_id,
                "pad_id": self.pad_token_id,
                "name": self._tok.name_or_path,
            },
            sort_keys=True,
        )
        return hashlib.sha256(fallback.encode()).hexdigest()


class _SentencePieceWrapper(BharatTokenizer):
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
        return eos if eos is not None else 0

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
            "special_tokens": {
                "eos": self.eos_token_id,
                "pad": self.pad_token_id,
            },
        }

    def fingerprint(self) -> str:
        import hashlib
        import json

        tok = self._tok
        data = {
            "type": "sentencepiece",
            "vocab_size": tok.get_vocab_size(),
            "vocab": {
                tok.id_to_token(i): i
                for i in range(tok.get_vocab_size())
                if tok.id_to_token(i) is not None
            },
            "normalizer": str(tok.normalizer.__class__.__name__)
            if hasattr(tok, "normalizer") and tok.normalizer
            else None,
            "pre_tokenizer": str(tok.pre_tokenizer.__class__.__name__)
            if hasattr(tok, "pre_tokenizer") and tok.pre_tokenizer
            else None,
            "model_type": str(tok.model.__class__.__name__) if hasattr(tok, "model") else None,
            "decoder": str(tok.decoder.__class__.__name__)
            if hasattr(tok, "decoder") and tok.decoder
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
        return self._tok.encode_batch(texts, add_special_tokens=add_special_tokens)

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
            "special_tokens": {
                "eos": self.eos_token_id,
                "pad": self.pad_token_id,
            },
            "is_fast": True,
        }

    def fingerprint(self) -> str:
        import hashlib
        import json

        tok = self._tok.backend_tokenizer if hasattr(self._tok, "backend_tokenizer") else None
        if tok is not None:
            data = {
                "type": "hf",
                "vocab": tok.get_vocab(),
                "normalizer": str(tok.normalizer.__class__.__name__) if tok.normalizer else None,
                "pre_tokenizer": str(tok.pre_tokenizer.__class__.__name__)
                if tok.pre_tokenizer
                else None,
                "model_type": str(tok.model.__class__.__name__),
                "special_tokens": {
                    t.content: tid for tid, t in tok.get_added_tokens_decoder().items()
                },
            }
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
            return _SentencePieceWrapper(tok)
        # For generic tokenizer.json, wrap as HF
        from transformers import PreTrainedTokenizerFast as HFPretrainedTokenizerFast

        hf_tok = HFPretrainedTokenizerFast(tokenizer_object=tok)
        return _HFWrapper(hf_tok)
    if source.suffix == ".model":
        from tokenizers import SentencePieceBPETokenizer

        tok = SentencePieceBPETokenizer.from_file(str(source))
        return _SentencePieceWrapper(tok)
    if source.is_dir():
        tok_config = source / "tokenizer.json"
        if tok_config.exists():
            tok = HFTokenizersTokenizer.from_file(str(tok_config))
            mt = _detect_model_type(tok)
            if mt == "sentencepiece":
                return _SentencePieceWrapper(tok)
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
    """Load a tokenizer and wrap it as a BharatTokenizer.

    Args:
        source: Path to tokenizer directory/file, or model name (e.g. 'gpt2').
        tokenizer_type: One of 'gpt2', 'sentencepiece', 'hf', or None for auto-detection.
        **kwargs: Additional arguments passed to the underlying tokenizer loader.

    Returns:
        A BharatTokenizer instance.

    Raises:
        ValueError: If the tokenizer type cannot be determined or loaded.
    """
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
