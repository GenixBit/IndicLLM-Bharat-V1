from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SourceStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"


class SourceKind(StrEnum):
    HUGGINGFACE = "huggingface"
    HTTP = "http"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    LOCAL = "local"
    OTHER = "other"


class UsagePurpose(StrEnum):
    PRETRAINING = "pretraining"
    SFT = "sft"
    DPO = "dpo"
    EVALUATION = "evaluation"


class UsageDomain(StrEnum):
    GENERAL = "general"
    CODE = "code"
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCE = "finance"
    CREATIVE = "creative"
    OTHER = "other"


_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*[a-z0-9]$|^[a-z]$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2}))?$")


def validate_slug(value: str, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path}: source_id must be a string, got {type(value).__name__}")
    if not _SLUG_RE.match(value):
        raise ValueError(
            f"{path}: invalid source_id '{value}' — must be a lowercase slug "
            r"(e.g. 'fineweb_edu', 'wiki_2025')"
        )
    return value


def validate_sha256(value: str, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path}: SHA-256 must be a string, got {type(value).__name__}")
    if not _SHA256_RE.match(value):
        raise ValueError(
            f"{path}: invalid SHA-256 '{value}' — must be a 64-character lowercase hex string"
        )
    return value


def validate_date(value: str, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path}: date must be a string, got {type(value).__name__}")
    if not _ISO_DATE_RE.match(value):
        raise ValueError(f"{path}: invalid date '{value}' — must be ISO-8601 (e.g. 2025-06-01)")
    return value


_SECRET_PATTERNS = [
    re.compile(
        r"(?i)(api[_-]?key|api[_-]?secret|password|secret|token|credential)"
        r"\s*[:=]\s*\S+"
    ),
    re.compile(r"(?i)(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36,})"),
    re.compile(r"://[^:/\s]+:[^@/\s]+@"),  # user:pass@ in URI authority
]


def validate_no_secrets(value: str, path: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(value):
            raise ValueError(f"{path}: URI appears to contain embedded credentials or secrets")


def validate_language_tags(tags: list[str], path: str) -> tuple[str, ...]:
    if not tags:
        raise ValueError(f"{path}: languages must be a non-empty list")
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            raise TypeError(f"{path}: language tag must be a string, got {type(tag).__name__}")
        normalized = tag.strip().lower().replace("-", "_")
        if normalized in seen:
            raise ValueError(f"{path}: duplicate language tag '{tag}'")
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


# ---------------------------------------------------------------------------
# Integrity record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataIntegrityRecord:
    revision: str
    sha256: str | None = None
    manifest_uri: str | None = None
    manifest_sha256: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: str) -> DataIntegrityRecord:
        unknown = set(data) - {"revision", "sha256", "manifest_uri", "manifest_sha256"}
        if unknown:
            raise ValueError(f"{path}: unknown integrity key(s): {', '.join(sorted(unknown))}")

        revision = data.get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"{path}: integrity.revision must be a non-empty string")

        sha256 = data.get("sha256")
        if sha256 is not None:
            sha256 = validate_sha256(sha256, path)

        manifest_uri = data.get("manifest_uri")
        if manifest_uri is not None and not isinstance(manifest_uri, str):
            raise TypeError(
                f"{path}: integrity.manifest_uri must be a string, "
                f"got {type(manifest_uri).__name__}"
            )

        manifest_sha256 = data.get("manifest_sha256")
        if manifest_sha256 is not None:
            manifest_sha256 = validate_sha256(manifest_sha256, path)

        return cls(
            revision=revision,
            sha256=sha256,
            manifest_uri=manifest_uri,
            manifest_sha256=manifest_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"revision": self.revision}
        if self.sha256 is not None:
            d["sha256"] = self.sha256
        if self.manifest_uri is not None:
            d["manifest_uri"] = self.manifest_uri
        if self.manifest_sha256 is not None:
            d["manifest_sha256"] = self.manifest_sha256
        return d


# ---------------------------------------------------------------------------
# Data source spec
# ---------------------------------------------------------------------------


_VALID_SOURCE_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "source_id",
        "version",
        "display_name",
        "provider",
        "kind",
        "uri",
        "revision",
        "languages",
        "domains",
        "splits",
        "purposes",
        "status",
        "license",
        "integrity",
        "gated",
        "credentials_env",
        "dataset_card_url",
        "collection_method",
        "upstream_sources",
        "supersedes",
        "created_at",
        "updated_at",
        "notes",
    }
)


@dataclass(frozen=True)
class DataSourceSpec:
    schema_version: int
    source_id: str
    version: str
    display_name: str
    provider: str
    kind: SourceKind
    uri: str
    revision: str
    languages: tuple[str, ...]
    domains: tuple[str, ...]
    splits: tuple[str, ...]
    purposes: tuple[UsagePurpose, ...]
    status: SourceStatus
    license: str
    integrity: DataIntegrityRecord | None
    gated: bool
    credentials_env: str | None
    dataset_card_url: str | None
    collection_method: str | None
    upstream_sources: tuple[str, ...]
    supersedes: str | None
    created_at: str
    updated_at: str
    notes: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any], file_path: str) -> DataSourceSpec:
        unknown = set(data) - _VALID_SOURCE_ROOT_KEYS
        if unknown:
            raise ValueError(f"{file_path}: unknown source key(s): {', '.join(sorted(unknown))}")

        schema_version = data.get("schema_version")
        if isinstance(schema_version, bool):
            raise TypeError(f"{file_path}: schema_version must be an integer, got bool")
        if not isinstance(schema_version, int):
            raise TypeError(
                f"{file_path}: schema_version must be an integer, "
                f"got {type(schema_version).__name__}"
            )
        if schema_version != 1:
            raise ValueError(
                f"{file_path}: unsupported schema_version {schema_version}, expected 1"
            )

        source_id = validate_slug(data.get("source_id", ""), file_path)

        version = data.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError(f"{file_path}: version must be a non-empty string")

        display_name = data.get("display_name")
        if not isinstance(display_name, str) or not display_name:
            raise ValueError(f"{file_path}: display_name must be a non-empty string")

        provider = data.get("provider")
        if not isinstance(provider, str) or not provider:
            raise ValueError(f"{file_path}: provider must be a non-empty string")

        kind_val = data.get("kind")
        if isinstance(kind_val, bool) or not isinstance(kind_val, str):
            raise TypeError(f"{file_path}: kind must be a string, got {type(kind_val).__name__}")
        try:
            kind = SourceKind(kind_val.lower())
        except ValueError:
            valid = ", ".join(sorted(k.value for k in SourceKind))
            raise ValueError(f"{file_path}: unknown kind '{kind_val}'; valid: {valid}")

        uri = data.get("uri")
        if not isinstance(uri, str) or not uri:
            raise ValueError(f"{file_path}: uri must be a non-empty string")
        validate_no_secrets(uri, file_path)

        revision = data.get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"{file_path}: revision must be a non-empty string")
        if kind == SourceKind.HUGGINGFACE and revision in ("main", "master", "latest"):
            raise ValueError(
                f"{file_path}: Hugging Face revision must be an immutable commit SHA, "
                f"got '{revision}'"
            )

        languages = validate_language_tags(data.get("languages", []), file_path)

        domains_raw = data.get("domains", [])
        if not isinstance(domains_raw, list) or not domains_raw:
            raise ValueError(f"{file_path}: domains must be a non-empty list")
        domains = tuple(str(d).lower() for d in domains_raw)

        splits_raw = data.get("splits", [])
        if not isinstance(splits_raw, list) or not splits_raw:
            raise ValueError(f"{file_path}: splits must be a non-empty list")
        splits = tuple(str(s) for s in splits_raw)

        purposes_raw = data.get("purposes", [])
        if not isinstance(purposes_raw, list) or not purposes_raw:
            raise ValueError(f"{file_path}: purposes must be a non-empty list")
        purposes_list: list[UsagePurpose] = []
        for p in purposes_raw:
            if not isinstance(p, str):
                raise TypeError(f"{file_path}: purpose must be a string, got {type(p).__name__}")
            try:
                purposes_list.append(UsagePurpose(p.lower()))
            except ValueError:
                valid = ", ".join(sorted(p.value for p in UsagePurpose))
                raise ValueError(f"{file_path}: unknown purpose '{p}'; valid: {valid}")
        purposes = tuple(purposes_list)

        status_val = data.get("status", "proposed")
        if isinstance(status_val, bool) or not isinstance(status_val, str):
            raise TypeError(
                f"{file_path}: status must be a string, got {type(status_val).__name__}"
            )
        try:
            status = SourceStatus(status_val.lower())
        except ValueError:
            valid = ", ".join(s.value for s in SourceStatus)
            raise ValueError(f"{file_path}: unknown status '{status_val}'; valid: {valid}")

        lic = data.get("license", "")
        if not isinstance(lic, str) or not lic:
            raise ValueError(f"{file_path}: license must be a non-empty string")

        integrity_data = data.get("integrity")
        if integrity_data is None:
            integrity = None
        elif not isinstance(integrity_data, dict):
            raise TypeError(
                f"{file_path}: integrity must be a mapping, got {type(integrity_data).__name__}"
            )
        else:
            integrity = DataIntegrityRecord.from_dict(integrity_data, f"{file_path}.integrity")

        gated = data.get("gated", False)
        if not isinstance(gated, bool):
            raise TypeError(f"{file_path}: gated must be a boolean, got {type(gated).__name__}")

        creds = data.get("credentials_env")
        if creds is not None:
            if not isinstance(creds, str):
                raise TypeError(
                    f"{file_path}: credentials_env must be a string, got {type(creds).__name__}"
                )
            if re.search(r"[\s:=]", creds):
                raise ValueError(
                    f"{file_path}: credentials_env must be an environment-variable name "
                    f"(not a value), got '{creds}'"
                )
            for pattern in _SECRET_PATTERNS:
                if pattern.search(creds):
                    raise ValueError(
                        f"{file_path}: credentials_env appears to contain a secret or API key, "
                        f"got '{creds}'"
                    )

        dataset_card_url = data.get("dataset_card_url")
        if dataset_card_url is not None and not isinstance(dataset_card_url, str):
            raise TypeError(
                f"{file_path}: dataset_card_url must be a string, "
                f"got {type(dataset_card_url).__name__}"
            )

        collection_method = data.get("collection_method")
        if collection_method is not None and not isinstance(collection_method, str):
            raise TypeError(
                f"{file_path}: collection_method must be a string, "
                f"got {type(collection_method).__name__}"
            )

        upstream = data.get("upstream_sources", [])
        if not isinstance(upstream, list):
            raise TypeError(
                f"{file_path}: upstream_sources must be a list, got {type(upstream).__name__}"
            )
        for u in upstream:
            if not isinstance(u, str):
                raise TypeError(
                    f"{file_path}: upstream source must be a string, got {type(u).__name__}"
                )

        supersedes = data.get("supersedes")

        created_at = validate_date(data.get("created_at", ""), f"{file_path}.created_at")
        updated_at = validate_date(data.get("updated_at", ""), f"{file_path}.updated_at")

        notes = data.get("notes")

        return cls(
            schema_version=schema_version,
            source_id=source_id,
            version=version,
            display_name=display_name,
            provider=provider,
            kind=kind,
            uri=uri,
            revision=revision,
            languages=languages,
            domains=tuple(domains),
            splits=splits,
            purposes=purposes,
            status=status,
            license=lic,
            integrity=integrity,
            gated=gated,
            credentials_env=creds,
            dataset_card_url=dataset_card_url,
            collection_method=collection_method,
            upstream_sources=tuple(upstream),
            supersedes=supersedes,
            created_at=created_at,
            updated_at=updated_at,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "version": self.version,
            "display_name": self.display_name,
            "provider": self.provider,
            "kind": self.kind.value,
            "uri": self.uri,
            "revision": self.revision,
            "languages": list(self.languages),
            "domains": list(self.domains),
            "splits": list(self.splits),
            "purposes": [p.value for p in self.purposes],
            "status": self.status.value,
            "license": self.license,
            "gated": self.gated,
            "upstream_sources": list(self.upstream_sources),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.integrity is not None:
            d["integrity"] = self.integrity.to_dict()
        if self.credentials_env is not None:
            d["credentials_env"] = self.credentials_env
        if self.dataset_card_url is not None:
            d["dataset_card_url"] = self.dataset_card_url
        if self.collection_method is not None:
            d["collection_method"] = self.collection_method
        if self.supersedes is not None:
            d["supersedes"] = self.supersedes
        if self.notes is not None:
            d["notes"] = self.notes
        return d
