from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}

_BYTE_ALPHABET = list(range(256))


@dataclass
class _BPEMerge:
    left: int
    right: int
    token: int


@dataclass
class BPETokenizer:
    vocab: dict[str, int]
    merges: tuple[_BPEMerge, ...]
    special_tokens: dict[str, int]
    tokenizer_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab": dict(self.vocab),
            "merges": [(m.left, m.right, m.token) for m in self.merges],
            "special_tokens": dict(self.special_tokens),
            "tokenizer_hash": self.tokenizer_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BPETokenizer:
        merges = tuple(_BPEMerge(*m) for m in data["merges"])
        return cls(
            vocab=data["vocab"],
            merges=merges,
            special_tokens=data["special_tokens"],
            tokenizer_hash=data.get("tokenizer_hash", ""),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def compute_tokenizer_hash(vocab: dict[str, int], merges: tuple[_BPEMerge, ...]) -> str:
    import hashlib

    payload = json.dumps(
        {"vocab": dict(vocab), "merges": [(m.left, m.right, m.token) for m in merges]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _byte_to_token(b: int) -> str:
    return f"<byte_{b:02x}>"


def _build_byte_vocab(special: dict[str, int]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for token_str, token_id in special.items():
        vocab[token_str] = token_id

    next_id = max(vocab.values()) + 1 if vocab else 0

    for b in _BYTE_ALPHABET:
        token_str = _byte_to_token(b)
        vocab[token_str] = next_id
        next_id += 1

    return vocab


def train_bpe(
    corpus_path: Path,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
) -> BPETokenizer:
    import hashlib

    tokens = special_tokens if special_tokens is not None else dict(_SPECIAL_TOKENS)
    total_special = max(tokens.values()) + 1

    byte_vocab = _build_byte_vocab(tokens)

    text = corpus_path.read_text(encoding="utf-8")
    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    pairs: Counter[tuple[int, int]] = Counter()
    ids = []
    next_id = total_special + 256

    for b in text.encode("utf-8"):
        ids.append(b)

    for i in range(len(ids) - 1):
        pairs[(ids[i], ids[i + 1])] += 1

    merges: list[_BPEMerge] = []

    while len(byte_vocab) < vocab_size and pairs:
        most_common = pairs.most_common(1)[0][0]
        new_token = next_id
        next_id += 1

        new_ids: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == most_common:
                new_ids.append(new_token)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids

        left, right = most_common
        token_str = _byte_to_token(left) + _byte_to_token(right)
        byte_vocab[token_str] = new_token
        merges.append(_BPEMerge(left=left, right=right, token=new_token))

        pairs.clear()
        for i in range(len(ids) - 1):
            pairs[(ids[i], ids[i + 1])] += 1

    tokenizer = BPETokenizer(
        vocab=dict(byte_vocab),
        merges=tuple(merges),
        special_tokens=dict(tokens),
    )

    hash_input = json.dumps(
        {
            "corpus_sha256": corpus_sha,
            "special_tokens": dict(tokens),
            "vocab_size": vocab_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    tokenizer.tokenizer_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    return tokenizer
