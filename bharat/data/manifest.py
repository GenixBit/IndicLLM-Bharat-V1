from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

_MANIFEST_SCHEMA_VERSION = "1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_ALLOWED_MANIFEST_ROOT_KEYS = frozenset(
    {
        "manifest_version",
        "dataset_id",
        "source_id",
        "source_version",
        "created_at",
        "license",
        "language",
        "split",
        "records",
        "bytes_utf8",
        "sha256",
        "processing_config_digest",
        "registry_digest",
        "policy_digest",
        "domain",
        "shards",
    }
)
_ALLOWED_SHARD_KEYS = frozenset(
    {
        "shard_id",
        "index",
        "record_start",
        "record_end",
        "bytes_utf8",
        "sha256",
        "created_at",
    }
)


def _validate_sha256(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if not _SHA256_RE.match(value):
        raise ValueError(
            f"{field_name} must be a 64-character lowercase hex string, "
            f"got '{value[:20]}...' (len={len(value)})"
        )


def _validate_shards(shards: tuple[ShardManifest, ...]) -> None:
    if not shards:
        return
    sorted_shards = sorted(shards, key=lambda s: s.index)
    expected_indices = list(range(len(shards)))
    actual_indices = [s.index for s in sorted_shards]
    if actual_indices != expected_indices:
        raise ValueError(f"Shard indexes must be sequential from 0; got {actual_indices}")
    for i in range(1, len(sorted_shards)):
        prev = sorted_shards[i - 1]
        curr = sorted_shards[i]
        if curr.record_start != prev.record_end:
            raise ValueError(
                f"Non-contiguous shards: shard index {prev.index} ends at "
                f"{prev.record_end} but shard index {curr.index} starts at "
                f"{curr.record_start}"
            )


@dataclass(frozen=True)
class ShardManifest:
    shard_id: str
    index: int
    record_start: int
    record_end: int
    bytes_utf8: int = 0
    sha256: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "shard_id": self.shard_id,
            "index": self.index,
            "record_start": self.record_start,
            "record_end": self.record_end,
        }
        if self.bytes_utf8:
            d["bytes_utf8"] = self.bytes_utf8
        if self.sha256:
            d["sha256"] = self.sha256
        if self.created_at:
            d["created_at"] = self.created_at
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShardManifest:
        unknown = set(data) - _ALLOWED_SHARD_KEYS
        if unknown:
            raise ValueError(f"Unknown shard key(s): {', '.join(sorted(unknown))}")
        shard_id = data.get("shard_id")
        if not isinstance(shard_id, str) or not shard_id:
            raise ValueError("shard_id must be a non-empty string")
        index = data.get("index")
        if not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer")
        record_start = data.get("record_start")
        if not isinstance(record_start, int) or record_start < 0:
            raise ValueError("record_start must be a non-negative integer")
        record_end = data.get("record_end")
        if not isinstance(record_end, int) or record_end < 0:
            raise ValueError("record_end must be a non-negative integer")
        if record_end < record_start:
            raise ValueError(f"record_end ({record_end}) must be >= record_start ({record_start})")
        sha256 = data.get("sha256", "")
        if sha256:
            _validate_sha256(sha256, f"shard '{shard_id}'.sha256")
        return cls(
            shard_id=shard_id,
            index=index,
            record_start=record_start,
            record_end=record_end,
            bytes_utf8=data.get("bytes_utf8", 0),
            sha256=sha256,
            created_at=data.get("created_at", ""),
        )


@dataclass(frozen=True)
class DatasetManifest:
    manifest_version: str
    dataset_id: str
    source_id: str
    source_version: str
    created_at: str
    license: str
    language: str
    split: str
    records: int
    bytes_utf8: int
    sha256: str
    processing_config_digest: str
    registry_digest: str
    policy_digest: str
    domain: str = ""
    shards: tuple[ShardManifest, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "dataset_id": self.dataset_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "created_at": self.created_at,
            "license": self.license,
            "language": self.language,
            "split": self.split,
            "records": self.records,
            "bytes_utf8": self.bytes_utf8,
            "sha256": self.sha256,
            "processing_config_digest": self.processing_config_digest,
            "registry_digest": self.registry_digest,
            "policy_digest": self.policy_digest,
            "shards": [s.to_dict() for s in self.shards],
        }
        if self.domain:
            d["domain"] = self.domain
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        unknown = set(data) - _ALLOWED_MANIFEST_ROOT_KEYS
        if unknown:
            raise ValueError(f"Unknown manifest key(s): {', '.join(sorted(unknown))}")

        manifest_version = data.get("manifest_version")
        if manifest_version != _MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest_version '{manifest_version}', "
                f"expected '{_MANIFEST_SCHEMA_VERSION}'"
            )

        dataset_id = data.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a non-empty string")

        source_id = data.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")

        source_version = data.get("source_version")
        if not isinstance(source_version, str) or not source_version:
            raise ValueError("source_version must be a non-empty string")

        created_at = data.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("created_at must be a non-empty string")

        lic = data.get("license")
        if not isinstance(lic, str) or not lic:
            raise ValueError("license must be a non-empty string")

        language = data.get("language")
        if not isinstance(language, str) or not language:
            raise ValueError("language must be a non-empty string")

        split = data.get("split")
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a non-empty string")

        domain = data.get("domain", "")

        records = data.get("records")
        if not isinstance(records, int) or records < 0:
            raise ValueError("records must be a non-negative integer")

        bytes_utf8 = data.get("bytes_utf8")
        if not isinstance(bytes_utf8, int) or bytes_utf8 < 0:
            raise ValueError("bytes_utf8 must be a non-negative integer")

        sha256 = data.get("sha256")
        if not isinstance(sha256, str) or not sha256:
            raise ValueError("sha256 must be a non-empty string")
        _validate_sha256(sha256, "sha256")

        processing_config_digest = data.get("processing_config_digest")
        if not isinstance(processing_config_digest, str) or not processing_config_digest:
            raise ValueError("processing_config_digest must be a non-empty string")
        _validate_sha256(processing_config_digest, "processing_config_digest")

        registry_digest = data.get("registry_digest")
        if not isinstance(registry_digest, str) or not registry_digest:
            raise ValueError("registry_digest must be a non-empty string")
        _validate_sha256(registry_digest, "registry_digest")

        policy_digest = data.get("policy_digest")
        if not isinstance(policy_digest, str) or not policy_digest:
            raise ValueError("policy_digest must be a non-empty string")
        _validate_sha256(policy_digest, "policy_digest")

        shards_raw = data.get("shards", [])
        if not isinstance(shards_raw, list):
            raise ValueError("shards must be a list")
        shards = tuple(ShardManifest.from_dict(s) for s in shards_raw)

        _validate_shards(shards)

        total_shard_records = sum(s.record_end - s.record_start for s in shards)
        if shards and total_shard_records != records:
            raise ValueError(
                f"Shard record sum ({total_shard_records}) does not match "
                f"manifest records ({records})"
            )

        return cls(
            manifest_version=manifest_version,
            dataset_id=dataset_id,
            source_id=source_id,
            source_version=source_version,
            created_at=created_at,
            license=lic,
            language=language,
            split=split,
            domain=domain,
            records=records,
            bytes_utf8=bytes_utf8,
            sha256=sha256,
            processing_config_digest=processing_config_digest,
            registry_digest=registry_digest,
            policy_digest=policy_digest,
            shards=shards,
        )

    def digest(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.records < 0:
            errors.append("records must be non-negative")
        if self.bytes_utf8 < 0:
            errors.append("bytes_utf8 must be non-negative")
        if not self.sha256:
            errors.append("sha256 must be non-empty")
        elif not _SHA256_RE.match(self.sha256):
            errors.append(f"sha256 has invalid format (len={len(self.sha256)})")
        if not self.processing_config_digest:
            errors.append("processing_config_digest must be non-empty")
        elif not _SHA256_RE.match(self.processing_config_digest):
            errors.append("processing_config_digest has invalid format")
        if not self.registry_digest:
            errors.append("registry_digest must be non-empty")
        elif not _SHA256_RE.match(self.registry_digest):
            errors.append("registry_digest has invalid format")
        if not self.policy_digest:
            errors.append("policy_digest must be non-empty")
        elif not _SHA256_RE.match(self.policy_digest):
            errors.append("policy_digest has invalid format")
        if self.shards:
            total_shard_records = sum(s.record_end - s.record_start for s in self.shards)
            if total_shard_records != self.records:
                errors.append(
                    f"Shard record sum ({total_shard_records}) != "
                    f"manifest records ({self.records})"
                )
            seen_indices: set[int] = set()
            for s in self.shards:
                if s.index in seen_indices:
                    errors.append(f"Duplicate shard index: {s.index}")
                seen_indices.add(s.index)
                if s.record_end < s.record_start:
                    errors.append(f"Shard '{s.shard_id}': record_end < record_start")
            sorted_shards = sorted(self.shards, key=lambda s: s.index)
            expected = list(range(len(self.shards)))
            actual = [s.index for s in sorted_shards]
            if actual != expected:
                errors.append(f"Shard indexes must be sequential from 0; got {actual}")
            for i in range(1, len(sorted_shards)):
                if sorted_shards[i].record_start != sorted_shards[i - 1].record_end:
                    errors.append(
                        f"Non-contiguous shards: shard {sorted_shards[i-1].index} ends at "
                        f"{sorted_shards[i-1].record_end} but shard {sorted_shards[i].index} "
                        f"starts at {sorted_shards[i].record_start}"
                    )
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


def digest_processing_config(config: object) -> str:
    cfg = config
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()
    elif hasattr(cfg, "__dataclass_fields__"):
        from dataclasses import asdict

        cfg = asdict(cfg)  # type: ignore[call-overload]
    if not isinstance(cfg, dict):
        cfg = {"value": str(cfg)}
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_manifest(
    dataset_id: str,
    source_id: str,
    source_version: str,
    license: str,
    language: str,
    split: str,
    records: int,
    bytes_utf8: int,
    sha256: str,
    processing_config_digest: str,
    registry_digest: str,
    policy_digest: str,
    shards: tuple[ShardManifest, ...] = (),
    created_at: str | None = None,
    domain: str = "",
) -> DatasetManifest:
    if created_at is not None:
        if not _ISO_UTC_RE.match(created_at):
            raise ValueError(
                f"created_at must be in ISO 8601 UTC format "
                f"(YYYY-MM-DDTHH:MM:SSZ), got '{created_at}'"
            )
    else:
        created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if shards:
        _validate_shards(shards)
        total_shard_records = sum(s.record_end - s.record_start for s in shards)
        if total_shard_records != records:
            raise ValueError(
                f"Shard record sum ({total_shard_records}) does not match "
                f"manifest records ({records})"
            )
    return DatasetManifest(
        manifest_version=_MANIFEST_SCHEMA_VERSION,
        dataset_id=dataset_id,
        source_id=source_id,
        source_version=source_version,
        created_at=created_at,
        license=license,
        language=language,
        split=split,
        domain=domain,
        records=records,
        bytes_utf8=bytes_utf8,
        sha256=sha256,
        processing_config_digest=processing_config_digest,
        registry_digest=registry_digest,
        policy_digest=policy_digest,
        shards=shards,
    )
