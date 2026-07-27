from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bharat.data.approval import (
    DatasetApproval,
    validate_approval_for_manifest,
)
from bharat.data.manifest import DatasetManifest
from bharat.data.release import DatasetAuditReport, DatasetRelease

_SCHEMA_VERSION = "1"
_URL_RE = re.compile(r"^(?:https?|ftp|s3|gs|hf):/+", re.IGNORECASE)
_NET_PATH_RE = re.compile(r"^//")


@dataclass(frozen=True)
class SamplerConfig:
    version: str
    seed: int
    max_total_records: int = 0
    max_total_bytes: int = 0
    max_records_per_source: dict[str, int] = field(default_factory=dict)
    max_bytes_per_source: dict[str, int] = field(default_factory=dict)
    max_records_per_language: dict[str, int] = field(default_factory=dict)
    max_bytes_per_language: dict[str, int] = field(default_factory=dict)
    text_field: str = "text"
    language_field: str = "language"
    domain_field: str = "domain"
    exact_dedup: bool = True
    output_corpus: str = ""
    output_manifest: str = ""

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be non-empty")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        if self.max_total_records < 0:
            msg = f"max_total_records must be non-negative, got {self.max_total_records}"
            raise ValueError(msg)
        if self.max_total_bytes < 0:
            msg = f"max_total_bytes must be non-negative, got {self.max_total_bytes}"
            raise ValueError(msg)
        for key, val in self.max_records_per_source.items():
            if val <= 0:
                raise ValueError(f"max_records_per_source[{key!r}] must be positive, got {val}")
        for key, val in self.max_bytes_per_source.items():
            if val <= 0:
                raise ValueError(f"max_bytes_per_source[{key!r}] must be positive, got {val}")
        for key, val in self.max_records_per_language.items():
            if val <= 0:
                raise ValueError(f"max_records_per_language[{key!r}] must be positive, got {val}")
        for key, val in self.max_bytes_per_language.items():
            if val <= 0:
                raise ValueError(f"max_bytes_per_language[{key!r}] must be positive, got {val}")
        if not self.text_field:
            raise ValueError("text_field must be non-empty")
        if not self.language_field:
            raise ValueError("language_field must be non-empty")

    @property
    def dry_run(self) -> bool:
        return not self.output_corpus and not self.output_manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "max_total_records": self.max_total_records,
            "max_total_bytes": self.max_total_bytes,
            "max_records_per_source": dict(self.max_records_per_source),
            "max_bytes_per_source": dict(self.max_bytes_per_source),
            "max_records_per_language": dict(self.max_records_per_language),
            "max_bytes_per_language": dict(self.max_bytes_per_language),
            "text_field": self.text_field,
            "language_field": self.language_field,
            "domain_field": self.domain_field,
            "exact_dedup": self.exact_dedup,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplerConfig:
        return cls(
            version=data["version"],
            seed=data["seed"],
            max_total_records=data.get("max_total_records", 0),
            max_total_bytes=data.get("max_total_bytes", 0),
            max_records_per_source=data.get("max_records_per_source", {}),
            max_bytes_per_source=data.get("max_bytes_per_source", {}),
            max_records_per_language=data.get("max_records_per_language", {}),
            max_bytes_per_language=data.get("max_bytes_per_language", {}),
            text_field=data.get("text_field", "text"),
            language_field=data.get("language_field", "language"),
            domain_field=data.get("domain_field", "domain"),
            exact_dedup=data.get("exact_dedup", True),
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    release_id: str
    manifest_digest: str
    approval_digest: str
    source_id: str
    shard_path: str
    shard_digest: str
    record_index: int
    record_id: str
    language: str
    domain: str
    utf8_bytes: int
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "manifest_digest": self.manifest_digest,
            "approval_digest": self.approval_digest,
            "source_id": self.source_id,
            "shard_path": self.shard_path,
            "shard_digest": self.shard_digest,
            "record_index": self.record_index,
            "record_id": self.record_id,
            "language": self.language,
            "domain": self.domain,
            "utf8_bytes": self.utf8_bytes,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    created_at: str
    sampler_config: SamplerConfig
    releases: tuple[dict[str, str], ...]
    total_candidates: int
    total_selected: int
    exact_dedup_removed: int
    per_source_cap_removed: int
    per_language_cap_removed: int
    global_cap_removed: int
    total_corpus_bytes: int
    per_source_records: dict[str, int]
    per_source_bytes: dict[str, int]
    per_language_records: dict[str, int]
    per_language_bytes: dict[str, int]
    per_domain_records: dict[str, int]
    per_domain_bytes: dict[str, int]
    corpus_sha256: str
    records: tuple[ProvenanceRecord, ...]
    manifest_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "sampler_config": self.sampler_config.to_dict(),
            "releases": list(self.releases),
            "total_candidates": self.total_candidates,
            "total_selected": self.total_selected,
            "exact_dedup_removed": self.exact_dedup_removed,
            "per_source_cap_removed": self.per_source_cap_removed,
            "per_language_cap_removed": self.per_language_cap_removed,
            "global_cap_removed": self.global_cap_removed,
            "total_corpus_bytes": self.total_corpus_bytes,
            "per_source_records": dict(self.per_source_records),
            "per_source_bytes": dict(self.per_source_bytes),
            "per_language_records": dict(self.per_language_records),
            "per_language_bytes": dict(self.per_language_bytes),
            "per_domain_records": dict(self.per_domain_records),
            "per_domain_bytes": dict(self.per_domain_bytes),
            "corpus_sha256": self.corpus_sha256,
            "records": [r.to_dict() for r in self.records],
        }
        if self.manifest_sha256:
            d["manifest_sha256"] = self.manifest_sha256
        return d

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def canonical_json_no_created(self) -> str:
        d = self.to_dict()
        d.pop("created_at", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def compute_digest(self) -> str:
        return hashlib.sha256(self.canonical_json_no_created().encode("utf-8")).hexdigest()


@dataclass
class _Record:
    text: str
    content_sha256: str
    utf8_bytes: int
    selection_key: str
    release_id: str
    manifest_digest: str
    approval_digest: str
    source_id: str
    shard_path: str
    shard_digest: str
    record_index: int
    record_id: str
    language: str
    domain: str


def _reject_remote(path: str | Path) -> Path:
    value = str(path)
    if _URL_RE.match(value) or _NET_PATH_RE.match(value):
        raise ValueError(f"remote or network path not allowed: {value}")
    return Path(value)


def _contains_lone_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(c) <= 0xDFFF for c in text)


def _compute_selection_key(
    version: str,
    seed: int,
    release_id: str,
    source_id: str,
    shard_digest: str,
    record_index: int,
    content_sha256: str,
) -> str:
    payload = json.dumps(
        [
            version,
            seed,
            release_id,
            source_id,
            shard_digest,
            record_index,
            content_sha256,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def serialize_record(text: str, field_name: str) -> bytes:
    line = json.dumps({field_name: text}, ensure_ascii=False, separators=(",", ":"))
    return (line + "\n").encode("utf-8")


def _check_symlink_escape(shard_path: Path, root: Path) -> None:
    resolved_shard = shard_path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_shard.relative_to(resolved_root)
    except ValueError:
        raise ValueError(f"symlink escape: {shard_path} resolves outside {root}")


def sample_tokenizer_corpus(
    release_roots: Sequence[Path],
    manifest_paths: Sequence[Path],
    approval_paths: Sequence[Path],
    config: SamplerConfig,
) -> CorpusManifest:
    if not (len(release_roots) == len(manifest_paths) == len(approval_paths)):
        raise ValueError(
            f"Mismatched input counts: "
            f"{len(release_roots)} release roots, "
            f"{len(manifest_paths)} manifest paths, "
            f"{len(approval_paths)} approval paths — must be equal"
        )

    if config.dry_run:
        if config.output_corpus:
            msg = "dry_run is True but output_corpus is set; use --execute to produce output"
            raise ValueError(msg)
        if config.output_manifest:
            msg = "dry_run is True but output_manifest is set; use --execute to produce output"
            raise ValueError(msg)
    else:
        if not config.output_corpus:
            raise ValueError("output_corpus must be set when executing")
        if not config.output_manifest:
            raise ValueError("output_manifest must be set when executing")

    releases_meta: list[dict[str, str]] = []
    all_records: list[_Record] = []

    for release_root, manifest_path, approval_path in zip(
        release_roots, manifest_paths, approval_paths, strict=True
    ):
        release_root = _reject_remote(release_root)
        manifest_path = _reject_remote(manifest_path)
        approval_path = _reject_remote(approval_path)

        release_path = release_root / "dataset_release.json"
        audit_path = release_root / "audit_report.json"

        if not release_path.exists():
            raise FileNotFoundError(f"dataset_release.json not found: {release_path}")
        if not audit_path.exists():
            raise FileNotFoundError(f"audit_report.json not found: {audit_path}")

        release_data = _load_json(release_path)
        audit_data = _load_json(audit_path)

        release = DatasetRelease(
            release_id=release_data["release_id"],
            dataset_id=release_data["dataset_id"],
            manifest_digest=release_data["manifest_digest"],
            approval_digest=release_data["approval_digest"],
            shard_count=release_data["shard_count"],
            records=release_data["records"],
            bytes_utf8=release_data["bytes_utf8"],
            created_at=release_data["created_at"],
            package_sha256=release_data["package_sha256"],
        )

        audit = DatasetAuditReport(
            dataset_id=audit_data["dataset_id"],
            manifest_digest=audit_data["manifest_digest"],
            approval_digest=audit_data["approval_digest"],
            shard_checks_passed=audit_data["shard_checks_passed"],
            approval_checks_passed=audit_data["approval_checks_passed"],
            total_records=audit_data["total_records"],
            total_bytes_utf8=audit_data["total_bytes_utf8"],
            issues=tuple(audit_data.get("issues", [])),
        )

        if audit.dataset_id != release.dataset_id:
            raise ValueError(
                f"audit dataset_id mismatch: audit={audit.dataset_id!r}, "
                f"release={release.dataset_id!r}"
            )
        if audit.manifest_digest != release.manifest_digest:
            raise ValueError(
                f"audit manifest_digest mismatch: audit={audit.manifest_digest}, "
                f"release={release.manifest_digest}"
            )

        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        manifest = DatasetManifest.from_dict(_load_json(manifest_path))

        if manifest.dataset_id != release.dataset_id:
            raise ValueError(
                f"manifest dataset_id mismatch: manifest={manifest.dataset_id!r}, "
                f"release={release.dataset_id!r}"
            )

        if manifest.digest() != release.manifest_digest:
            raise ValueError(
                f"manifest digest mismatch for {manifest.dataset_id}: "
                f"release={release.manifest_digest}, "
                f"computed={manifest.digest()}"
            )

        if not approval_path.exists():
            raise FileNotFoundError(f"approval not found: {approval_path}")
        approval_data = _load_json(approval_path)
        approval = DatasetApproval(**approval_data)

        approval_issues = validate_approval_for_manifest(approval, manifest)
        if approval_issues:
            raise ValueError(
                f"approval validation failed for {manifest.dataset_id}: "
                f"{'; '.join(approval_issues)}"
            )

        if approval.digest() != release.approval_digest:
            raise ValueError(
                f"approval digest mismatch for {manifest.dataset_id}: "
                f"release={release.approval_digest}, "
                f"computed={approval.digest()}"
            )

        if not approval.approval_status == "approved":
            raise ValueError(
                f"approval status for {manifest.dataset_id} is "
                f"{approval.approval_status!r}, expected 'approved'"
            )
        if not approval.license_reviewed:
            raise ValueError(f"license_reviewed not true for {manifest.dataset_id}")
        if not approval.pii_reviewed:
            raise ValueError(f"pii_reviewed not true for {manifest.dataset_id}")
        if not approval.contamination_reviewed:
            raise ValueError(f"contamination_reviewed not true for {manifest.dataset_id}")
        if not approval.safety_reviewed:
            raise ValueError(f"safety_reviewed not true for {manifest.dataset_id}")

        releases_meta.append(
            {
                "release_id": release.release_id,
                "dataset_id": release.dataset_id,
                "manifest_digest": release.manifest_digest,
                "approval_digest": release.approval_digest,
                "shard_count": str(release.shard_count),
                "records": str(release.records),
                "bytes_utf8": str(release.bytes_utf8),
                "created_at": release.created_at,
                "package_sha256": release.package_sha256,
            }
        )

        shards_dir = manifest_path.parent / "shards"
        if not shards_dir.is_dir():
            raise FileNotFoundError(f"shards directory not found: {shards_dir}")

        for shard in sorted(manifest.shards, key=lambda s: s.index):
            shard_path = shards_dir / shard.shard_id
            if not shard_path.exists():
                raise FileNotFoundError(f"shard not found: {shard_path}")

            _check_symlink_escape(shard_path, release_root)

            shard_bytes = shard_path.read_bytes()
            if shard.sha256:
                computed = hashlib.sha256(shard_bytes).hexdigest()
                if computed != shard.sha256:
                    raise ValueError(
                        f"shard {shard.shard_id} sha256 mismatch: "
                        f"declared={shard.sha256}, computed={computed}"
                    )
                shard_digest = shard.sha256
            else:
                shard_digest = hashlib.sha256(shard_bytes).hexdigest()

            try:
                shard_text = shard_bytes.decode("utf-8")
            except UnicodeDecodeError as e:
                raise ValueError(f"malformed UTF-8 in shard {shard_path}: {e}")

            for record_index, line in enumerate(shard_text.splitlines()):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    record_data = json.loads(stripped)
                except json.JSONDecodeError as e:
                    raise ValueError(f"malformed JSON in {shard_path}:{record_index}: {e}")

                if not isinstance(record_data, dict):
                    raise ValueError(
                        f"record must be a JSON object at " f"{shard_path}:{record_index}"
                    )

                if record_data.get("accepted", True) is not True:
                    continue

                text = record_data.get(config.text_field)
                if text is None or not isinstance(text, str) or not text.strip():
                    continue

                if _contains_lone_surrogate(text):
                    raise ValueError(f"lone surrogate at {shard_path}:{record_index}")

                content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                utf8_bytes = len(serialize_record(text, config.text_field))

                language = record_data.get(config.language_field)
                if not isinstance(language, str) or not language:
                    language = manifest.language

                domain = record_data.get(config.domain_field)
                if not isinstance(domain, str):
                    domain = manifest.domain

                record_id = record_data.get("record_id", "")
                if not isinstance(record_id, str):
                    record_id = ""

                selection_key = _compute_selection_key(
                    version=config.version,
                    seed=config.seed,
                    release_id=release.release_id,
                    source_id=manifest.source_id,
                    shard_digest=shard_digest,
                    record_index=record_index,
                    content_sha256=content_sha256,
                )

                all_records.append(
                    _Record(
                        text=text,
                        content_sha256=content_sha256,
                        utf8_bytes=utf8_bytes,
                        selection_key=selection_key,
                        release_id=release.release_id,
                        manifest_digest=release.manifest_digest,
                        approval_digest=release.approval_digest,
                        source_id=manifest.source_id,
                        shard_path=f"shards/{shard.shard_id}",
                        shard_digest=shard_digest,
                        record_index=record_index,
                        record_id=record_id,
                        language=language,
                        domain=domain,
                    )
                )

    total_candidates = len(all_records)

    if config.exact_dedup:
        groups: dict[str, list[_Record]] = {}
        for r in all_records:
            groups.setdefault(r.content_sha256, []).append(r)
        deduped: list[_Record] = []
        exact_dedup_removed = 0
        for candidates in groups.values():
            min_record = min(
                candidates,
                key=lambda r: (
                    r.selection_key,
                    r.release_id,
                    r.source_id,
                    r.shard_path,
                    r.record_index,
                    r.record_id,
                ),
            )
            deduped.append(min_record)
            exact_dedup_removed += len(candidates) - 1
    else:
        deduped = list(all_records)
        exact_dedup_removed = 0

    deduped.sort(key=lambda r: (r.selection_key, r.record_id))

    source_records: dict[str, int] = {}
    source_bytes: dict[str, int] = {}
    language_records: dict[str, int] = {}
    language_bytes: dict[str, int] = {}
    domain_records: dict[str, int] = {}
    domain_bytes: dict[str, int] = {}
    selected: list[_Record] = []
    per_source_cap_removed = 0
    per_language_cap_removed = 0
    global_cap_removed = 0

    for r in deduped:
        src = r.source_id
        lang = r.language
        r_bytes = r.utf8_bytes

        if (
            src in config.max_records_per_source
            and source_records.get(src, 0) >= config.max_records_per_source[src]
        ):
            per_source_cap_removed += 1
            continue

        if (
            src in config.max_bytes_per_source
            and source_bytes.get(src, 0) + r_bytes > config.max_bytes_per_source[src]
        ):
            per_source_cap_removed += 1
            continue

        if (
            lang in config.max_records_per_language
            and language_records.get(lang, 0) >= config.max_records_per_language[lang]
        ):
            per_language_cap_removed += 1
            continue

        if (
            lang in config.max_bytes_per_language
            and language_bytes.get(lang, 0) + r_bytes > config.max_bytes_per_language[lang]
        ):
            per_language_cap_removed += 1
            continue

        if config.max_total_records > 0 and len(selected) >= config.max_total_records:
            global_cap_removed += 1
            continue

        current_bytes = sum(s.utf8_bytes for s in selected)
        if config.max_total_bytes > 0 and current_bytes + r_bytes > config.max_total_bytes:
            global_cap_removed += 1
            continue

        selected.append(r)
        source_records[src] = source_records.get(src, 0) + 1
        source_bytes[src] = source_bytes.get(src, 0) + r_bytes
        language_records[lang] = language_records.get(lang, 0) + 1
        language_bytes[lang] = language_bytes.get(lang, 0) + r_bytes
        domain_records[r.domain] = domain_records.get(r.domain, 0) + 1
        domain_bytes[r.domain] = domain_bytes.get(r.domain, 0) + r_bytes

    total_corpus_bytes = sum(r.utf8_bytes for r in selected)

    records = tuple(
        ProvenanceRecord(
            release_id=r.release_id,
            manifest_digest=r.manifest_digest,
            approval_digest=r.approval_digest,
            source_id=r.source_id,
            shard_path=r.shard_path,
            shard_digest=r.shard_digest,
            record_index=r.record_index,
            record_id=r.record_id,
            language=r.language,
            domain=r.domain,
            utf8_bytes=r.utf8_bytes,
            content_sha256=r.content_sha256,
        )
        for r in selected
    )

    if config.dry_run:
        return CorpusManifest(
            schema_version=_SCHEMA_VERSION,
            created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            sampler_config=config,
            releases=tuple(releases_meta),
            total_candidates=total_candidates,
            total_selected=len(selected),
            exact_dedup_removed=exact_dedup_removed,
            per_source_cap_removed=per_source_cap_removed,
            per_language_cap_removed=per_language_cap_removed,
            global_cap_removed=global_cap_removed,
            total_corpus_bytes=total_corpus_bytes,
            per_source_records=dict(source_records),
            per_source_bytes=dict(source_bytes),
            per_language_records=dict(language_records),
            per_language_bytes=dict(language_bytes),
            per_domain_records=dict(domain_records),
            per_domain_bytes=dict(domain_bytes),
            corpus_sha256="0" * 64,
            records=records,
        )

    corpus_path = Path(config.output_corpus)
    manifest_path_out = Path(config.output_manifest)

    if corpus_path.exists():
        raise FileExistsError(f"corpus already exists: {corpus_path}")
    if manifest_path_out.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path_out}")

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_out.parent.mkdir(parents=True, exist_ok=True)

    tmp_corpus = corpus_path.parent / f".tmp.{os.getpid()}.{corpus_path.name}"
    tmp_manifest = manifest_path_out.parent / f".tmp.{os.getpid()}.{manifest_path_out.name}"

    corpus_sha = hashlib.sha256()
    total_bytes = 0

    try:
        with tmp_corpus.open("wb") as f:
            for r in selected:
                line_bytes = serialize_record(r.text, config.text_field)
                f.write(line_bytes)
                corpus_sha.update(line_bytes)
                total_bytes += len(line_bytes)

        tmp_corpus_bytes = tmp_corpus.stat().st_size
        if tmp_corpus_bytes != total_bytes:
            raise RuntimeError(
                f"corpus byte count mismatch: " f"expected {total_bytes}, got {tmp_corpus_bytes}"
            )

        computed_corpus_sha = corpus_sha.hexdigest()
        read_back_sha = hashlib.sha256(tmp_corpus.read_bytes()).hexdigest()
        if computed_corpus_sha != read_back_sha:
            raise RuntimeError(
                f"corpus sha256 verification failed: "
                f"write={computed_corpus_sha}, "
                f"read_back={read_back_sha}"
            )

        if total_bytes != tmp_corpus.stat().st_size:
            raise RuntimeError(
                f"corpus size changed after flush: "
                f"expected {total_bytes}, got {tmp_corpus.stat().st_size}"
            )

        base_manifest = CorpusManifest(
            schema_version=_SCHEMA_VERSION,
            created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            sampler_config=config,
            releases=tuple(releases_meta),
            total_candidates=total_candidates,
            total_selected=len(selected),
            exact_dedup_removed=exact_dedup_removed,
            per_source_cap_removed=per_source_cap_removed,
            per_language_cap_removed=per_language_cap_removed,
            global_cap_removed=global_cap_removed,
            total_corpus_bytes=total_bytes,
            per_source_records=dict(source_records),
            per_source_bytes=dict(source_bytes),
            per_language_records=dict(language_records),
            per_language_bytes=dict(language_bytes),
            per_domain_records=dict(domain_records),
            per_domain_bytes=dict(domain_bytes),
            corpus_sha256=computed_corpus_sha,
            records=records,
        )

        manifest_digest = base_manifest.compute_digest()

        final_manifest = CorpusManifest(
            schema_version=base_manifest.schema_version,
            created_at=base_manifest.created_at,
            sampler_config=base_manifest.sampler_config,
            releases=base_manifest.releases,
            total_candidates=base_manifest.total_candidates,
            total_selected=base_manifest.total_selected,
            exact_dedup_removed=base_manifest.exact_dedup_removed,
            per_source_cap_removed=base_manifest.per_source_cap_removed,
            per_language_cap_removed=base_manifest.per_language_cap_removed,
            global_cap_removed=base_manifest.global_cap_removed,
            total_corpus_bytes=base_manifest.total_corpus_bytes,
            per_source_records=base_manifest.per_source_records,
            per_source_bytes=base_manifest.per_source_bytes,
            per_language_records=base_manifest.per_language_records,
            per_language_bytes=base_manifest.per_language_bytes,
            per_domain_records=base_manifest.per_domain_records,
            per_domain_bytes=base_manifest.per_domain_bytes,
            corpus_sha256=base_manifest.corpus_sha256,
            records=base_manifest.records,
            manifest_sha256=manifest_digest,
        )

        tmp_manifest.write_text(final_manifest.canonical_json(), encoding="utf-8")

        os.replace(tmp_corpus, corpus_path)
        os.replace(tmp_manifest, manifest_path_out)

    except BaseException:
        import contextlib

        for p in (tmp_corpus, tmp_manifest):
            with contextlib.suppress(OSError):
                if p.exists():
                    p.unlink()
        if corpus_path.exists():
            with contextlib.suppress(OSError):
                corpus_path.unlink()
        raise

    return final_manifest
