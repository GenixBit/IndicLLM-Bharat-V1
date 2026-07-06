from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenizerMetadata:
    tokenizer_type: str
    vocab_size: int
    eos_token_id: int
    pad_token_id: int
    tokenizer_hash: str
    special_tokens: dict[str, int] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    git_sha: str = ""
    data_version: str = ""
    seed: int = 0


def tokenizer_hash(tokenizer: Any) -> str:
    """Compute a deterministic hash of a tokenizer's vocabulary."""
    from bharat.tokenizer.base import BharatTokenizer

    if not isinstance(tokenizer, BharatTokenizer):
        msg = f"Expected BharatTokenizer, got {type(tokenizer)}"
        raise TypeError(msg)

    vocab_size = tokenizer.vocab_size
    sample_tokens: list[str] = []
    for i in range(min(vocab_size, 1000)):
        token = tokenizer.decode([i])
        if token:
            sample_tokens.append(token)

    digest = hashlib.sha256(json.dumps({
        "type": tokenizer.tokenizer_type,
        "vocab_size": vocab_size,
        "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id,
        "sample_tokens": sample_tokens[:100],
    }, sort_keys=True).encode()).hexdigest()

    return digest


def metadata_from_tokenizer(tokenizer: Any, **extra: Any) -> TokenizerMetadata:
    """Build metadata from an existing tokenizer."""
    from bharat.tokenizer.base import BharatTokenizer

    if not isinstance(tokenizer, BharatTokenizer):
        msg = f"Expected BharatTokenizer, got {type(tokenizer)}"
        raise TypeError(msg)

    return TokenizerMetadata(
        tokenizer_type=tokenizer.tokenizer_type,
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tokenizer_hash=tokenizer_hash(tokenizer),
        **extra,
    )


def validate_tokenizer_compatibility(
    checkpoint_meta: TokenizerMetadata,
    current_tokenizer: Any,
) -> None:
    """Raise ValueError if tokenizer hash doesn't match checkpoint metadata."""
    from bharat.tokenizer.base import BharatTokenizer

    if not isinstance(current_tokenizer, BharatTokenizer):
        msg = f"Expected BharatTokenizer, got {type(current_tokenizer)}"
        raise TypeError(msg)

    current_hash = tokenizer_hash(current_tokenizer)
    if current_hash != checkpoint_meta.tokenizer_hash:
        raise ValueError(
            f"Tokenizer mismatch: checkpoint has "
            f"type={checkpoint_meta.tokenizer_type}, "
            f"vocab_size={checkpoint_meta.vocab_size} "
            f"(hash={checkpoint_meta.tokenizer_hash[:12]}...), "
            f"but current tokenizer is "
            f"type={current_tokenizer.tokenizer_type}, "
            f"vocab_size={current_tokenizer.vocab_size} "
            f"(hash={current_hash[:12]}...). "
            f"Use the same tokenizer that was used during training."
        )
