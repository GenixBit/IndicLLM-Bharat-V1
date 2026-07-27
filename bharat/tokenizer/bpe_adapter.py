from __future__ import annotations

from pathlib import Path
from typing import Any

from bharat.tokenizer.base import BharatTokenizer
from bharat.tokenizer.bpe import BPETokenizer


class BharatBPETokenizer(BharatTokenizer):
    """Expose a validated deterministic BPE artifact through BharatTokenizer."""

    def __init__(self, tokenizer: BPETokenizer) -> None:
        tokenizer.validate()
        self._tokenizer = tokenizer

    @classmethod
    def load(cls, path: str | Path) -> BharatBPETokenizer:
        return cls(BPETokenizer.load(Path(path)))

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        self._tokenizer.save(Path(path), overwrite=overwrite)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._required_token_id("<pad>")

    @property
    def unk_token_id(self) -> int:
        return self._required_token_id("<unk>")

    @property
    def bos_token_id(self) -> int:
        return self._required_token_id("<bos>")

    @property
    def eos_token_id(self) -> int:
        return self._required_token_id("<eos>")

    @property
    def tokenizer_type(self) -> str:
        return "bharat_bpe"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self._tokenizer.encode(text)
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def encode_batch(self, texts: list[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    def get_metadata(self) -> dict[str, Any]:
        return {
            "type": self.tokenizer_type,
            "schema_version": self._tokenizer.schema_version,
            "normalization": self._tokenizer.normalization,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        return self._tokenizer.compute_hash()

    def _required_token_id(self, token: str) -> int:
        try:
            return self._tokenizer.special_tokens[token]
        except KeyError:
            msg = f"BPE artifact is missing required special token {token!r}"
            raise ValueError(msg) from None
