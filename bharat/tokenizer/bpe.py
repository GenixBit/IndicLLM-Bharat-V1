from __future__ import annotations

import hashlib
import json
import re
import secrets
import unicodedata
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

    def _compact_serialize(self) -> str:
        payload = self.to_dict()
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

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
        payload = json.dumps(
            self._canonical_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
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

    def validate(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            msg = f"unsupported schema version: {self.schema_version}"
            raise ValueError(msg)
        if self.normalization not in ("nfc", "none"):
            msg = f"unsupported normalization policy: {self.normalization}"
            raise ValueError(msg)

        _validate_special_and_reserved_tokens(self.special_tokens, self.reserved_tokens)

        if len(self.byte_value_to_id) != 256:
            msg = (
                f"byte_value_to_id must have exactly 256 entries, got {len(self.byte_value_to_id)}"
            )
            raise ValueError(msg)
        for b in range(256):
            if b not in self.byte_value_to_id:
                msg = f"missing byte value {b} in byte_value_to_id"
                raise ValueError(msg)
        byte_ids = set(self.byte_value_to_id.values())
        if len(byte_ids) != 256:
            msg = f"byte_value_to_id values must be unique, got {len(byte_ids)} unique IDs"
            raise ValueError(msg)

        special_reserved_ids = set(self.special_tokens.values()) | set(
            self.reserved_tokens.values()
        )
        if byte_ids & special_reserved_ids:
            msg = "byte IDs collide with special/reserved IDs"
            raise ValueError(msg)

        for b, tid in self.byte_value_to_id.items():
            expected = bytes([b])
            if self.id_to_bytes.get(tid) != expected:
                msg = f"id_to_bytes[{tid}] mismatch: expected {expected!r}, got {self.id_to_bytes.get(tid)!r}"
                raise ValueError(msg)

        all_ids = set(self.vocab.values())
        if len(all_ids) != len(self.vocab):
            msg = "vocabulary contains duplicate IDs"
            raise ValueError(msg)

        for token_str, token_id in self.vocab.items():
            if token_id in special_reserved_ids:
                continue
            if token_id in byte_ids:
                continue
            found = any(m.token == token_id for m in self.merges)
            if not found:
                msg = f"token ID {token_id} ({token_str!r}) not found in any category"
                raise ValueError(msg)

        seen_ranks: set[int] = set()
        seen_merge_tokens: set[int] = set()
        base_ids = special_reserved_ids | byte_ids
        for i, m in enumerate(self.merges):
            if m.left not in all_ids:
                msg = f"merge {i}: left ID {m.left} not in vocabulary"
                raise ValueError(msg)
            if m.right not in all_ids:
                msg = f"merge {i}: right ID {m.right} not in vocabulary"
                raise ValueError(msg)
            if m.token in base_ids:
                msg = f"merge {i}: token ID {m.token} collides with base ID"
                raise ValueError(msg)
            if m.token in seen_merge_tokens:
                msg = f"merge {i}: duplicate merge token ID {m.token}"
                raise ValueError(msg)
            if m.rank in seen_ranks:
                msg = f"merge {i}: duplicate rank {m.rank}"
                raise ValueError(msg)
            if m.rank != i:
                msg = f"merge {i}: rank {m.rank} != expected {i}"
                raise ValueError(msg)
            expected_bytes = self.id_to_bytes.get(m.left, b"") + self.id_to_bytes.get(m.right, b"")
            actual_bytes = self.id_to_bytes.get(m.token, b"")
            if actual_bytes != expected_bytes:
                msg = f"merge {i}: id_to_bytes[{m.token}] = {actual_bytes!r}, expected {expected_bytes!r}"
                raise ValueError(msg)
            seen_ranks.add(m.rank)
            seen_merge_tokens.add(m.token)

        computed = self.compute_hash()
        if self.tokenizer_hash and computed != self.tokenizer_hash:
            msg = f"tokenizer hash mismatch: stored {self.tokenizer_hash} != computed {computed}"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BPETokenizer:
        if data.get("schema_version", _SCHEMA_VERSION) != _SCHEMA_VERSION:
            msg = f"unsupported schema version: {data.get('schema_version')}"
            raise ValueError(msg)
        if data.get("normalization", _NORMALIZATION) not in ("nfc", "none"):
            msg = f"unsupported normalization policy: {data.get('normalization')}"
            raise ValueError(msg)

        raw_byte_value_to_id = data.get("byte_value_to_id", {})
        if not isinstance(raw_byte_value_to_id, dict):
            msg = "byte_value_to_id must be a dict"
            raise ValueError(msg)
        byte_value_to_id: dict[int, int] = {}
        for k, v in raw_byte_value_to_id.items():
            if not isinstance(v, int) or isinstance(v, bool):
                msg = f"byte_value_to_id values must be integers, got {v!r}"
                raise ValueError(msg)
            try:
                byte_value_to_id[int(k)] = v
            except (ValueError, TypeError):
                msg = f"byte_value_to_id key must be an integer, got {k!r}"
                raise ValueError(msg)

        raw_id_to_bytes = data.get("id_to_bytes", {})
        if not isinstance(raw_id_to_bytes, dict):
            msg = "id_to_bytes must be a dict"
            raise ValueError(msg)
        id_to_bytes: dict[int, bytes] = {}
        for k, v in raw_id_to_bytes.items():
            if not isinstance(v, str):
                msg = f"id_to_bytes values must be hex strings, got {v!r}"
                raise ValueError(msg)
            try:
                id_to_bytes[int(k)] = bytes.fromhex(v)
            except (ValueError, TypeError):
                msg = f"id_to_bytes contains invalid hex: key={k!r} value={v!r}"
                raise ValueError(msg)

        raw_special: dict[str, int] = data.get("special_tokens", {})
        raw_reserved: dict[str, int] = data.get("reserved_tokens", {})
        for label, raw in [("special_tokens", raw_special), ("reserved_tokens", raw_reserved)]:
            for k, v in raw.items():
                if not isinstance(k, str):
                    msg = f"{label} keys must be strings, got {k!r}"
                    raise ValueError(msg)
                if not isinstance(v, int) or isinstance(v, bool):
                    msg = f"{label} values must be integers, got {v!r}"
                    raise ValueError(msg)
                if v < 0:
                    msg = f"{label} values must be non-negative, got {v}"
                    raise ValueError(msg)

        raw_merges: list[BPEMerge] = []
        for m in data.get("merges", ()):
            if not isinstance(m, list | tuple):
                msg = f"merge entry must be a list/tuple, got {type(m).__name__}"
                raise ValueError(msg)
            if len(m) == 4:
                raw_merges.append(BPEMerge(left=m[0], right=m[1], token=m[2], rank=m[3]))
            elif len(m) == 3:
                rank = len(raw_merges)
                raw_merges.append(BPEMerge(left=m[0], right=m[1], token=m[2], rank=rank))
            else:
                msg = f"invalid merge entry length {len(m)}: {m}"
                raise ValueError(msg)

        raw_vocab = data.get("vocab", {})
        if not isinstance(raw_vocab, dict):
            msg = "vocab must be a dict"
            raise ValueError(msg)
        for k, v in raw_vocab.items():
            if not isinstance(k, str):
                msg = f"vocab keys must be strings, got {k!r}"
                raise ValueError(msg)
            if not isinstance(v, int) or isinstance(v, bool):
                msg = f"vocab values must be integers, got {v!r}"
                raise ValueError(msg)
            if v < 0:
                msg = f"vocab values must be non-negative, got {v}"
                raise ValueError(msg)

        t = cls(
            schema_version=data.get("schema_version", _SCHEMA_VERSION),
            normalization=data.get("normalization", _NORMALIZATION),
            special_tokens=raw_special,
            reserved_tokens=raw_reserved,
            byte_value_to_id=byte_value_to_id,
            id_to_bytes=id_to_bytes,
            vocab=raw_vocab,
            merges=tuple(raw_merges),
            tokenizer_hash=data.get("tokenizer_hash", ""),
        )

        t.validate()

        return t

    def save(self, path: Path, *, overwrite: bool = False) -> None:
        if path.exists() and not overwrite:
            msg = f"refusing to overwrite existing file: {path}"
            raise FileExistsError(msg)

        tmp_name = f".{path.name}.{secrets.token_hex(8)}.tmp"
        tmp_path = path.with_name(tmp_name)
        try:
            serialized = self._compact_serialize()
            tmp_path.write_text(serialized, encoding="utf-8")
            tmp_path.resolve().stat()

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
        normalized = unicodedata.normalize("NFC", text)

        for surrogate in re.findall(r"[\ud800-\udfff]", normalized):
            msg = f"lone surrogate in input: U+{ord(surrogate):04X}"
            raise ValueError(msg)

        special_map: dict[str, int] | None = self.special_tokens if allow_special else None

        ids: list[int] = []
        i = 0
        while i < len(normalized):
            if special_map is not None:
                matched = False
                for token_str, token_id in sorted(special_map.items(), key=lambda x: -len(x[0])):
                    if normalized[i:].startswith(token_str):
                        ids.append(token_id)
                        i += len(token_str)
                        matched = True
                        break
                if matched:
                    continue

            byte_val = normalized[i].encode("utf-8")
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

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        special_ids = set(self.special_tokens.values())
        reserved_ids = set(self.reserved_tokens.values())
        all_special_reserved = special_ids | reserved_ids

        result = bytearray()
        for tid in ids:
            if tid in self.id_to_bytes:
                result.extend(self.id_to_bytes[tid])
            elif tid in all_special_reserved:
                if not skip_special_tokens:
                    for token_str, token_id in self.special_tokens.items():
                        if token_id == tid:
                            result.extend(token_str.encode("utf-8"))
                            break
                    else:
                        for token_str, token_id in self.reserved_tokens.items():
                            if token_id == tid:
                                result.extend(token_str.encode("utf-8"))
                                break
            else:
                msg = f"unknown token ID: {tid}"
                raise ValueError(msg)
        try:
            return result.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            msg = "invalid UTF-8 in decoded output"
            raise ValueError(msg) from e


def _validate_special_and_reserved_tokens(
    special_tokens: dict[str, int],
    reserved_tokens: dict[str, int],
) -> None:
    seen_ids: set[int] = set()
    seen_strings: set[str] = set()

    for label, tokens in [("special", special_tokens), ("reserved", reserved_tokens)]:
        for token_str, token_id in tokens.items():
            if not isinstance(token_str, str) or not token_str:
                msg = f"{label} token string must be a non-empty string, got {token_str!r}"
                raise ValueError(msg)
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                msg = f"{label} token ID must be a non-negative integer, got {token_id!r}"
                raise ValueError(msg)
            if token_id < 0:
                msg = f"{label} token ID must be non-negative, got {token_id}"
                raise ValueError(msg)
            if token_id in seen_ids:
                msg = f"duplicate token ID {token_id} across special/reserved"
                raise ValueError(msg)
            if token_str in seen_strings:
                msg = f"duplicate token string {token_str!r} across special/reserved"
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

        normalized = unicodedata.normalize("NFC", text_val)
        encoded = normalized.encode("utf-8")
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

    if reserved_tokens is None:
        reserved_tokens = dict(_RESERVED_TOKENS)

    _validate_special_and_reserved_tokens(special_tokens, reserved_tokens)

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
    tokenizer.validate()

    return tokenizer
