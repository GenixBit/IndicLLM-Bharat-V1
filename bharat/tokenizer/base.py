from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BharatTokenizer(ABC):
    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def eos_token_id(self) -> int: ...

    @property
    @abstractmethod
    def pad_token_id(self) -> int: ...

    @property
    def bos_token_id(self) -> int:
        return self.eos_token_id

    @property
    def unk_token_id(self) -> int:
        return self.pad_token_id

    @property
    @abstractmethod
    def tokenizer_type(self) -> str: ...

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    @abstractmethod
    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = True
    ) -> list[list[int]]: ...

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...

    @abstractmethod
    def decode_batch(
        self, batch: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]: ...

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]: ...

    def add_special_tokens(self, special_tokens: dict[str, list[str]]) -> int:
        """Add special tokens. Returns number of added tokens.

        Default implementation raises NotImplementedError.
        Override in subclasses that support dynamic special token addition.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support add_special_tokens")

    @abstractmethod
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint of the complete tokenizer configuration.

        Must include vocabulary, merge rules, normalizer, pre-tokenizer,
        special token mappings, and configuration.
        """
