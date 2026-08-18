"""High-Throughput Memory-Mapped Binary Token Streaming Engine for IndicLLM-Bharat.

Enables zero-copy, memory-efficient token streaming for pretraining models from tiny
up to 10B parameters on billions of tokens across multi-shard datasets.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from bharat.tokenizer import BharatTokenizer


@dataclass
class BinaryShardHeader:
    version: int
    dtype: str  # "uint16" or "uint32"
    vocab_size: int
    num_tokens: int
    num_documents: int
    magic: str = "BHARAT_BIN"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinaryShardHeader:
        return cls(
            version=data["version"],
            dtype=data["dtype"],
            vocab_size=data["vocab_size"],
            num_tokens=data["num_tokens"],
            num_documents=data.get("num_documents", 0),
            magic=data.get("magic", "BHARAT_BIN"),
        )


class BinaryTokenPacker:
    """Packs tokenized text into memory-mapped binary shards."""

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str = "bharat_shard",
        vocab_size: int = 64000,
        dtype: str = "uint16",
        max_tokens_per_shard: int = 10_000_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.vocab_size = vocab_size
        self.dtype_str = dtype
        self.np_dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.max_tokens_per_shard = max_tokens_per_shard

        self.current_shard_idx = 0
        self.current_tokens: list[int] = []
        self.doc_count = 0
        self.total_tokens_written = 0
        self.shard_paths: list[Path] = []

    def add_document(self, token_ids: list[int]) -> None:
        """Add a tokenized document to the packing stream."""
        if not token_ids:
            return

        self.current_tokens.extend(token_ids)
        self.doc_count += 1

        if len(self.current_tokens) >= self.max_tokens_per_shard:
            self.flush()

    def flush(self) -> Path | None:
        """Write current buffered tokens to a new binary shard on disk."""
        if not self.current_tokens:
            return None

        shard_name = f"{self.prefix}_{self.current_shard_idx:05d}"
        bin_path = self.output_dir / f"{shard_name}.bin"
        meta_path = self.output_dir / f"{shard_name}.meta.json"

        arr = np.array(self.current_tokens, dtype=self.np_dtype)
        with open(bin_path, "wb") as f:
            arr.tofile(f)

        header = BinaryShardHeader(
            version=1,
            dtype=self.dtype_str,
            vocab_size=self.vocab_size,
            num_tokens=len(arr),
            num_documents=self.doc_count,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(header.to_dict(), f, indent=2)

        self.total_tokens_written += len(arr)
        self.shard_paths.append(bin_path)
        self.current_shard_idx += 1
        self.current_tokens.clear()
        self.doc_count = 0
        return bin_path

    def close(self) -> list[Path]:
        self.flush()
        return self.shard_paths


class MMapTokenShard:
    """Zero-copy memory-mapped access to a single token shard."""

    def __init__(self, bin_path: str | Path) -> None:
        self.bin_path = Path(bin_path)
        meta_path = self.bin_path.with_name(self.bin_path.stem + ".meta.json")

        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing metadata file for shard: {meta_path}")

        with open(meta_path, encoding="utf-8") as f:
            self.header = BinaryShardHeader.from_dict(json.load(f))

        self.np_dtype = np.uint16 if self.header.dtype == "uint16" else np.uint32
        self.num_tokens = self.header.num_tokens

        # Open memory-mapped array in read-only copy-on-write mode
        self.mmap_array = np.memmap(
            self.bin_path, dtype=self.np_dtype, mode="r", shape=(self.num_tokens,)
        )

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, idx: int | slice) -> np.ndarray:
        return self.mmap_array[idx]


class MMapTokenDataset(Dataset[torch.Tensor]):
    """PyTorch Dataset for fixed-length sequence training from binary shards."""

    def __init__(
        self,
        shard_paths: list[str | Path] | str | Path,
        block_size: int = 512,
    ) -> None:
        self.block_size = block_size
        if isinstance(shard_paths, str | Path):
            p = Path(shard_paths)
            paths = sorted(p.glob("*.bin")) if p.is_dir() else [p]
        else:
            paths = [Path(p) for p in shard_paths]

        if not paths:
            raise ValueError("No binary shards found!")

        self.shards = [MMapTokenShard(p) for p in paths]
        self.total_tokens = sum(len(s) for s in self.shards)

        # Build cumulative token offsets
        self.shard_offsets: list[int] = [0]
        for s in self.shards:
            self.shard_offsets.append(self.shard_offsets[-1] + len(s))

        # Effective number of complete sequences
        self.total_samples = max(0, (self.total_tokens - 1) // self.block_size)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range (0..{self.total_samples})")

        token_offset = idx * self.block_size
        target_len = self.block_size + 1  # x + y target

        # Find starting shard
        shard_idx = 0
        while (
            shard_idx < len(self.shards) - 1 and self.shard_offsets[shard_idx + 1] <= token_offset
        ):
            shard_idx += 1

        local_offset = token_offset - self.shard_offsets[shard_idx]
        current_shard = self.shards[shard_idx]

        if local_offset + target_len <= len(current_shard):
            chunk = np.array(
                current_shard[local_offset : local_offset + target_len], dtype=np.int64
            )
        else:
            # Span across shard boundary
            chunk_parts: list[np.ndarray] = [np.array(current_shard[local_offset:], dtype=np.int64)]
            remaining = target_len - len(chunk_parts[0])
            next_idx = shard_idx + 1
            while remaining > 0 and next_idx < len(self.shards):
                next_shard = self.shards[next_idx]
                take = min(remaining, len(next_shard))
                chunk_parts.append(np.array(next_shard[:take], dtype=np.int64))
                remaining -= take
                next_idx += 1
            chunk = np.concatenate(chunk_parts)
            if len(chunk) < target_len:
                # Pad if end of dataset reached
                pad = np.zeros(target_len - len(chunk), dtype=np.int64)
                chunk = np.concatenate([chunk, pad])

        t = torch.from_numpy(chunk)
        x = t[:-1]
        y = t[1:]
        return x, y


class MMapTokenStream(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Asynchronous iterable token streamer with worker rank splitting."""

    def __init__(
        self,
        shard_paths: list[str | Path] | str | Path,
        block_size: int = 512,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = MMapTokenDataset(shard_paths, block_size=block_size)
        self.block_size = block_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        total_samples = len(self.dataset)

        if total_samples == 0:
            return

        indices = np.arange(total_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        if worker_info is not None:
            # Multi-worker splitting
            per_worker = int(np.ceil(total_samples / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, total_samples)
            indices = indices[start:end]

        for idx in indices:
            yield self.dataset[int(idx)]


def pack_text_corpus(
    tokenizer: BharatTokenizer,
    input_file: str | Path,
    output_dir: str | Path,
    prefix: str = "bharat_train",
    max_tokens_per_shard: int = 1_000_000,
) -> list[Path]:
    """Helper function to read a text/JSONL file and pack it into binary shards."""
    packer = BinaryTokenPacker(
        output_dir=output_dir,
        prefix=prefix,
        vocab_size=tokenizer.vocab_size,
        max_tokens_per_shard=max_tokens_per_shard,
    )

    p = Path(input_file)
    with open(p, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if text.startswith("{") and text.endswith("}"):
                try:
                    data = json.loads(text)
                    text = data.get("text", data.get("content", data.get("response", "")))
                except json.JSONDecodeError:
                    pass
            if text:
                tokens = tokenizer.encode(text, add_special_tokens=True)
                packer.add_document(tokens)

    return packer.close()
