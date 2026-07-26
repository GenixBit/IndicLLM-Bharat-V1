from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

from bharat.serving.gguf_preflight import GGUFMetadataEntry, GGUFPreflightResult

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12


@dataclass(frozen=True)
class GGUFWriteResult:
    output_path: Path
    bytes_written: int
    metadata_count: int
    tensor_count: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "output_path": str(self.output_path),
            "bytes_written": self.bytes_written,
            "metadata_count": self.metadata_count,
            "tensor_count": self.tensor_count,
        }


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_metadata_value(entry: GGUFMetadataEntry) -> tuple[int, bytes]:
    if entry.value_type == "bool":
        return _GGUF_TYPE_BOOL, struct.pack("<?", entry.value)
    if entry.value_type == "float":
        return _GGUF_TYPE_FLOAT64, struct.pack("<d", entry.value)
    if entry.value_type == "int":
        if entry.value < -(2**63) or entry.value > 2**63 - 1:
            raise ValueError(f"metadata {entry.key!r} integer is outside signed 64-bit range")
        return _GGUF_TYPE_INT64, struct.pack("<q", entry.value)
    if entry.value_type == "string":
        return _GGUF_TYPE_STRING, _pack_string(entry.value)
    raise ValueError(f"metadata {entry.key!r} has unsupported value_type {entry.value_type!r}")


def build_gguf_header(preflight: GGUFPreflightResult) -> bytes:
    """Build a deterministic GGUF v3 header containing scalar metadata only."""
    if preflight.tensor_count != 0:
        raise ValueError("GGUF header writer currently requires tensor_count to be 0")

    metadata = tuple(sorted(preflight.metadata, key=lambda item: item.key))
    parts = [
        _GGUF_MAGIC,
        struct.pack("<I", _GGUF_VERSION),
        struct.pack("<Q", 0),
        struct.pack("<Q", len(metadata)),
    ]
    for entry in metadata:
        value_type, value_bytes = _pack_metadata_value(entry)
        parts.extend((_pack_string(entry.key), struct.pack("<I", value_type), value_bytes))
    return b"".join(parts)


def write_gguf_header(preflight: GGUFPreflightResult, output_path: Path) -> GGUFWriteResult:
    """Write a local metadata-only GGUF v3 file without replacing existing output."""
    if output_path.suffix.lower() != ".gguf":
        raise ValueError("output_path must end with .gguf")
    if not output_path.parent.exists() or not output_path.parent.is_dir():
        raise ValueError(f"output parent must exist and be a directory: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(f"output path already exists: {output_path}")

    payload = build_gguf_header(preflight)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        try:
            os.link(tmp_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(f"output path was created concurrently: {output_path}") from exc
        tmp_path.unlink()
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    return GGUFWriteResult(
        output_path=output_path,
        bytes_written=len(payload),
        metadata_count=len(preflight.metadata),
        tensor_count=0,
    )
