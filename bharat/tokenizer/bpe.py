from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "bpe-v1"

_NORMALIZATION = "nfc"

_SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}

_RESERVED_TOKENS: dict[str, int] = {}

_BYTE_ALPHABET = list(range(256))


@dataclass(frozen=True)
class BPEMerge:
    left: int
    right: int
    token: int
    rank: int


@dataclass
class BPETokenizer:
    schema_version: str = _SCHEMA_VERSION
    normalization: str = _NORMALIZATION
    special_tokens: dict[str, int] = field(default_factory=lambda: dict(_SPECIAL_TOKENS))
    reserved_tokens: dict[str, int] = field(default_factory=dict)
    byte_value_to_id: dict[int, int] = field(default_factory=dict)
    id_to_bytes: dict[int, bytes] = field(default_factory=dict)
    vocab: dict[str, int] = field(default_factory=dict)
    merges: tuple[BPEMerge, ...] = ()
    tokenizer_hash: str = ""

    @property
    def base_vocab_size(self) -> int:
        return len(self.special_tokens) + len(self.reserved_tokens) + 256

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalization": self.normalization,
            "special_tokens": dict(self.special_tokens),
            "reserved_tokens": dict(self.reserved_tokens),
            "byte_value_to_id": {str(k): v for k, v in sorted(self.byte_value_to_id.items())},
            "id_to_bytes": {str(k): v.hex() for k, v in sorted(self.id_to_bytes.items())},
            "vocab": {k: v for k, v in sorted(self.vocab.items(), key=lambda x: x[1])},
            "merges": [(m.left, m.right, m.token, m.rank) for m in self.merges],
        }

    def compute_hash(self) -> str:
        payload = json.dumps(self._canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalization": self.normalization,
            "special_tokens": dict(self.special_tokens),
            "reserved_tokens": dict(self.reserved_tokens),
            "byte_value_to_id": {str(k): v for k, v in sorted(self.byte_value_to_id.items())},
            "id_to_bytes": {str(k): v.hex() for k, v in sorted(self.id_to_bytes.items())},
            "vocab": {k: v for k, v in sorted(self.vocab.items(), key=lambda x: x[1])},
            "merges": [(m.left, m.right, m.token, m.rank) for m in self.merges],
            "tokenizer_hash": self.tokenizer_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BPETokenizer:
        if data.get("schema_version", _SCHEMA_VERSION) != _SCHEMA_VERSION:
            msg = f"unsupported schema version: {data.get('schema_version')}"
            raise ValueError(msg)

        byte_value_to_id = {int(k): v for k, v in data["byte_value_to_id"].items()}
        id_to_bytes = {int(k): bytes.fromhex(v) for k, v in data["id_to_bytes"].items()}

        raw_merges: list[BPEMerge] = []
        for m in data["merges"]:
            if len(m) == 4:
                raw_merges.append(BPEMerge(left=m[0], right=m[1], token=m[2], rank=m[3]))
            elif len(m) == 3:
                rank = len(raw_merges)
                raw_merges.append(BPEMerge(left=m[0], right=m[1], token=m[2], rank=rank))
            else:
                msg = f"invalid merge entry: {m}"
                raise ValueError(msg)

        t = cls(
            schema_version=data.get("schema_version", _SCHEMA_VERSION),
            normalization=data.get("normalization", _NORMALIZATION),
            special_tokens=data.get("special_tokens", {}),
            reserved_tokens=data.get("reserved_tokens", {}),
            byte_value_to_id=byte_value_to_id,
            id_to_bytes=id_to_bytes,
            vocab=data.get("vocab", {}),
            merges=tuple(raw_merges),
            tokenizer_hash=data.get("tokenizer_hash", ""),
        )

        stored_hash = data.get("tokenizer_hash", "")
        if stored_hash:
            computed = t.compute_hash()
            if stored_hash != computed:
                msg = f"tokenizer hash mismatch: stored {stored_hash} != computed {computed}"
                raise ValueError(msg)

        return t

    def save(self, path: Path, *, overwrite: bool = False) -> None:
        if path.exists() and not overwrite:
            msg = f"refusing to overwrite existing file: {path}"
            raise FileExistsError(msg)

        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            tmp_path.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
            loaded = BPETokenizer.load(tmp_path)
            if loaded.compute_hash() != self.compute_hash():
                msg = "save verification failed: hash mismatch"
                raise RuntimeError(msg)
            tmp_path.rename(path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    @classmethod
    def load(cls, path: Path) -> BPETokenizer:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def encode(self, text: str, *, allow_special: bool = False) -> list[int]:
        for surrogate in re.findall(r"[\ud800-\udfff]", text):
            msg = f"lone surrogate in input: U+{ord(surrogate):04X}"
            raise ValueError(msg)

        special_map: dict[str, int] | None = self.special_tokens if allow_special else None

        ids: list[int] = []
        i = 0
        while i < len(text):
            if special_map is not None:
                matched = False
                for token_str, token_id in sorted(special_map.items(), key=lambda x: -len(x[0])):
                    if text[i:].startswith(token_str):
                        ids.append(token_id)
                        i += len(token_str)
                        matched = True
                        break
                if matched:
                    continue

            byte_val = text[i].encode("utf-8")
            for b in byte_val:
                ids.append(self.byte_value_to_id[b])
            i += 1

        for merge in self.merges:
            new_ids: list[int] = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and (ids[j], ids[j + 1]) == (merge.left, merge.right):
                    new_ids.append(merge.token)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids

        return ids

    def decode(self, ids: Sequence[int]) -> str:
        result = bytearray()
        for tid in ids:
            if tid in self.id_to_bytes:
                result.extend(self.id_to_bytes[tid])
            elif tid in self.special_tokens.values():
                pass
            else:
                msg = f"unknown token ID: {tid}"
                raise ValueError(msg)
        try:
            return result.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            msg = "invalid UTF-8 in decoded output"
            raise ValueError(msg) from e


def _validate_special_tokens(special: dict[str, int]) -> None:
    seen_ids: set[int] = set()
    seen_strings: set[str] = set()
    for token_str, token_id in special.items():
        if not isinstance(token_id, int) or token_id < 0:
            msg = f"special token ID must be non-negative integer, got {token_id}"
            raise ValueError(msg)
        if not token_str:
            msg = "special token string must not be empty"
            raise ValueError(msg)
        if token_id in seen_ids:
            msg = f"duplicate special token ID: {token_id}"
            raise ValueError(msg)
        if token_str in seen_strings:
            msg = f"duplicate special token string: {token_str}"
            raise ValueError(msg)
        seen_ids.add(token_id)
        seen_strings.add(token_str)


def _validate_vocab_size(vocab_size: int, base_size: int) -> None:
    if vocab_size < base_size:
        msg = (
            f"requested vocab_size {vocab_size} is less than "
            f"base vocabulary size {base_size} "
            f"({base_size - 256} special/reserved + 256 byte tokens)"
        )
        raise ValueError(msg)


def _build_base_vocab(
    special_tokens: dict[str, int],
    reserved_tokens: dict[str, int],
) -> tuple[dict[int, int], dict[int, bytes], dict[str, int]]:
    byte_value_to_id: dict[int, int] = {}
    id_to_bytes: dict[int, bytes] = {}
    vocab: dict[str, int] = {}

    for token_str, token_id in special_tokens.items():
        vocab[token_str] = token_id

    for token_str, token_id in reserved_tokens.items():
        vocab[token_str] = token_id

    existing_ids = set(special_tokens.values()) | set(reserved_tokens.values())
    next_id = 0
    while next_id in existing_ids:
        next_id += 1

    for b in _BYTE_ALPHABET:
        byte_value_to_id[b] = next_id
        id_to_bytes[next_id] = bytes([b])
        vocab[f"<byte_{b:02x}>"] = next_id
        next_id += 1

    return byte_value_to_id, id_to_bytes, vocab


def _read_corpus_records(corpus_path: Path, text_field: str = "text") -> list[bytes]:
    records: list[bytes] = []
    for line_num, raw_line in enumerate(corpus_path.read_bytes().split(b"\n"), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            decoded = stripped.decode("utf-8")
        except UnicodeDecodeError as e:
            msg = f"line {line_num}: malformed UTF-8"
            raise ValueError(msg) from e

        try:
            obj = json.loads(decoded)
        except json.JSONDecodeError as e:
            msg = f"line {line_num}: malformed JSON: {e}"
            raise ValueError(msg) from e

        if not isinstance(obj, dict):
            msg = f"line {line_num}: expected JSON object, got {type(obj).__name__}"
            raise ValueError(msg)

        if text_field not in obj:
            msg = f"line {line_num}: missing '{text_field}' field"
            raise ValueError(msg)

        text_val = obj[text_field]
        if not isinstance(text_val, str):
            msg = f"line {line_num}: '{text_field}' must be a string, got {type(text_val).__name__}"
            raise ValueError(msg)

        for surrogate in re.findall(r"[\ud800-\udfff]", text_val):
            msg = f"line {line_num}: lone surrogate U+{ord(surrogate):04X}"
            raise ValueError(msg)

        encoded = text_val.encode("utf-8")
        records.append(encoded)

    return records


def train_bpe(
    corpus_path: Path,
    vocab_size: int,
    special_tokens: dict[str, int] | None = None,
    reserved_tokens: dict[str, int] | None = None,
    text_field: str = "text",
) -> BPETokenizer:
    if special_tokens is None:
        special_tokens = dict(_SPECIAL_TOKENS)
    _validate_special_tokens(special_tokens)

    if reserved_tokens is None:
        reserved_tokens = dict(_RESERVED_TOKENS)

    byte_value_to_id, id_to_bytes, vocab = _build_base_vocab(special_tokens, reserved_tokens)
    _validate_vocab_size(vocab_size, len(vocab))

    records = _read_corpus_records(corpus_path, text_field=text_field)

    all_record_ids: list[list[int]] = []
    for rec_bytes in records:
        rec_ids = [byte_value_to_id[b] for b in rec_bytes]
        all_record_ids.append(rec_ids)

    merges: list[BPEMerge] = []
    next_id = max(vocab.values()) + 1

    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[int, int]] = Counter()
        for rec_ids in all_record_ids:
            for i in range(len(rec_ids) - 1):
                pair_counts[(rec_ids[i], rec_ids[i + 1])] += 1

        if not pair_counts:
            break

        max_freq = pair_counts.most_common(1)[0][1]
        candidates = [pair for pair, freq in pair_counts.items() if freq == max_freq]
        best_pair = min(candidates, key=lambda p: (p[0], p[1]))

        new_token = next_id
        next_id += 1
        rank = len(merges)

        left, right = best_pair
        merges.append(BPEMerge(left=left, right=right, token=new_token, rank=rank))

        new_bytes = id_to_bytes[left] + id_to_bytes[right]
        id_to_bytes[new_token] = new_bytes

        vocab_str = f"<merge_{rank}>"
        vocab[vocab_str] = new_token

        for j, rec_ids in enumerate(all_record_ids):
            new_rec: list[int] = []
            k = 0
            while k < len(rec_ids):
                if k < len(rec_ids) - 1 and (rec_ids[k], rec_ids[k + 1]) == best_pair:
                    new_rec.append(new_token)
                    k += 2
                else:
                    new_rec.append(rec_ids[k])
                    k += 1
            all_record_ids[j] = new_rec

    tokenizer = BPETokenizer(
        schema_version=_SCHEMA_VERSION,
        normalization=_NORMALIZATION,
        special_tokens=special_tokens,
        reserved_tokens=reserved_tokens,
        byte_value_to_id=byte_value_to_id,
        id_to_bytes=id_to_bytes,
        vocab=vocab,
        merges=tuple(merges),
        tokenizer_hash="",
    )

    tokenizer.tokenizer_hash = tokenizer.compute_hash()

    return tokenizer
