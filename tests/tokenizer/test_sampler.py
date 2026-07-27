from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bharat.data.approval import DatasetApproval
from bharat.data.manifest import DatasetManifest, ShardManifest
from bharat.data.release import DatasetAuditReport, DatasetRelease
from bharat.tokenizer.sampler import (
    CorpusManifest,
    ProvenanceRecord,
    SamplerConfig,
    sample_tokenizer_corpus,
)


def _digest(data: bytes = b"test") -> str:
    return hashlib.sha256(data).hexdigest()


def _make_manifest(
    dataset_id: str = "ds-test",
    source_id: str = "src-test",
    language: str = "en",
    domain: str = "",
    records: int = 3,
    shards: tuple[ShardManifest, ...] = (),
) -> DatasetManifest:
    if not shards:
        shards = (
            ShardManifest(
                shard_id="shard-0000",
                index=0,
                record_start=0,
                record_end=records,
                bytes_utf8=100,
                sha256=_digest(b"dummy"),
            ),
        )
    return DatasetManifest(
        manifest_version="1.0",
        dataset_id=dataset_id,
        source_id=source_id,
        source_version="1.0",
        created_at="2026-07-20T12:00:00Z",
        license="cc-by-4.0",
        language=language,
        split="train",
        records=records,
        bytes_utf8=100,
        sha256=_digest(),
        processing_config_digest=_digest(),
        registry_digest=_digest(),
        policy_digest=_digest(),
        domain=domain,
        shards=shards,
    )


def _make_approval(
    manifest: DatasetManifest,
    status: str = "approved",
    license_reviewed: bool = True,
    pii_reviewed: bool = True,
    contamination_reviewed: bool = True,
    safety_reviewed: bool = True,
) -> DatasetApproval:
    return DatasetApproval(
        approval_id="apr-001",
        dataset_id=manifest.dataset_id,
        manifest_digest=manifest.digest(),
        approver="test@example.com",
        approval_status=status,
        approved_at="2026-07-20T12:00:00Z",
        license_reviewed=license_reviewed,
        pii_reviewed=pii_reviewed,
        contamination_reviewed=contamination_reviewed,
        safety_reviewed=safety_reviewed,
    )


def _make_release(
    manifest: DatasetManifest,
    approval: DatasetApproval,
    release_id: str = "rel-001",
) -> DatasetRelease:
    return DatasetRelease(
        release_id=release_id,
        dataset_id=manifest.dataset_id,
        manifest_digest=manifest.digest(),
        approval_digest=approval.digest(),
        shard_count=len(manifest.shards),
        records=manifest.records,
        bytes_utf8=manifest.bytes_utf8,
        created_at="2026-07-20T12:00:00Z",
        package_sha256=_digest(),
    )


def _write_release_root(
    tmp_path: Path,
    name: str,
    manifest: DatasetManifest,
    approval: DatasetApproval,
    release: DatasetRelease,
    shard_lines: list[dict],
) -> tuple[Path, Path, Path]:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)

    (root / "dataset_release.json").write_text(
        json.dumps(release.to_dict(), indent=2), encoding="utf-8"
    )

    audit = DatasetAuditReport(
        dataset_id=manifest.dataset_id,
        manifest_digest=manifest.digest(),
        approval_digest=approval.digest(),
        shard_checks_passed=True,
        approval_checks_passed=True,
        total_records=manifest.records,
        total_bytes_utf8=manifest.bytes_utf8,
    )
    (root / "audit_report.json").write_text(json.dumps(audit.to_dict(), indent=2), encoding="utf-8")

    shards_dir = root / "shards"
    shards_dir.mkdir(exist_ok=True)

    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in shard_lines) + "\n"
    shard_bytes = content.encode("utf-8")
    shard_sha = hashlib.sha256(shard_bytes).hexdigest()
    shard_path = shards_dir / "shard-0000"
    shard_path.write_bytes(shard_bytes)

    manifest_with_sha = _make_manifest(
        dataset_id=manifest.dataset_id,
        source_id=manifest.source_id,
        language=manifest.language,
        domain=manifest.domain,
        records=len(shard_lines),
        shards=(
            ShardManifest(
                shard_id="shard-0000",
                index=0,
                record_start=0,
                record_end=len(shard_lines),
                bytes_utf8=len(shard_bytes),
                sha256=shard_sha,
            ),
        ),
    )

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_with_sha.to_dict(), indent=2), encoding="utf-8")

    approval_path = root / "approval.json"
    updated_approval = _make_approval(
        manifest=manifest_with_sha,
        status=approval.approval_status,
        license_reviewed=approval.license_reviewed,
        pii_reviewed=approval.pii_reviewed,
        contamination_reviewed=approval.contamination_reviewed,
        safety_reviewed=approval.safety_reviewed,
    )
    approval_path.write_text(json.dumps(updated_approval.to_dict(), indent=2), encoding="utf-8")

    updated_release = _make_release(
        manifest=manifest_with_sha,
        approval=updated_approval,
        release_id=release.release_id,
    )
    (root / "dataset_release.json").write_text(
        json.dumps(updated_release.to_dict(), indent=2), encoding="utf-8"
    )

    updated_audit = DatasetAuditReport(
        dataset_id=manifest_with_sha.dataset_id,
        manifest_digest=manifest_with_sha.digest(),
        approval_digest=updated_approval.digest(),
        shard_checks_passed=True,
        approval_checks_passed=True,
        total_records=manifest_with_sha.records,
        total_bytes_utf8=manifest_with_sha.bytes_utf8,
    )
    (root / "audit_report.json").write_text(
        json.dumps(updated_audit.to_dict(), indent=2), encoding="utf-8"
    )

    return root, manifest_path, approval_path


def _setup_single_release(
    tmp_path: Path,
    shard_records: list[dict[str, object]] | None = None,
    language: str = "en",
    source_id: str = "src-test",
    dataset_id: str = "ds-test",
    release_id: str = "rel-001",
    approval_status: str = "approved",
    domain: str = "",
) -> tuple[Path, Path, Path]:
    if shard_records is None:
        shard_records = [
            {"text": "hello world", "language": "en", "accepted": True},
            {"text": "foo bar", "language": language, "accepted": True},
        ]
    manifest = _make_manifest(
        dataset_id=dataset_id,
        source_id=source_id,
        language=language,
        domain=domain,
        records=len(shard_records),
    )
    approval = _make_approval(manifest, status=approval_status)
    release = _make_release(manifest, approval, release_id=release_id)
    return _write_release_root(tmp_path, release_id, manifest, approval, release, shard_records)


class TestSamplerConfig:
    def test_minimal_valid(self) -> None:
        c = SamplerConfig(version="1.0", seed=42)
        assert c.version == "1.0"

    def test_empty_version_raises(self) -> None:
        with pytest.raises(ValueError, match="version"):
            SamplerConfig(version="", seed=0)

    def test_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            SamplerConfig(version="1.0", seed=-1)

    def test_zero_caps_allowed(self) -> None:
        c = SamplerConfig(version="1.0", seed=0, max_total_records=0, max_total_bytes=0)
        assert c.max_total_records == 0

    def test_negative_global_records_raises(self) -> None:
        with pytest.raises(ValueError, match="max_total_records"):
            SamplerConfig(version="1.0", seed=0, max_total_records=-1)

    def test_positive_source_cap_ok(self) -> None:
        c = SamplerConfig(version="1.0", seed=0, max_records_per_source={"src": 500})
        assert c.max_records_per_source["src"] == 500

    def test_zero_source_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="max_records_per_source"):
            SamplerConfig(version="1.0", seed=0, max_records_per_source={"src": 0})

    def test_to_dict_roundtrip(self) -> None:
        c1 = SamplerConfig(
            version="2.0",
            seed=99,
            max_total_records=1000,
            max_records_per_source={"a": 100},
            max_bytes_per_language={"en": 50000},
        )
        c2 = SamplerConfig.from_dict(c1.to_dict())
        assert c1 == c2


class TestProvenanceRecord:
    def test_minimal(self) -> None:
        r = ProvenanceRecord(
            release_id="r1",
            manifest_digest="a" * 64,
            approval_digest="b" * 64,
            source_id="s1",
            shard_path="shards/shard-0000",
            shard_digest="c" * 64,
            record_index=0,
            record_id="rec-0",
            language="en",
            domain="general",
            utf8_bytes=10,
            content_sha256="d" * 64,
        )
        assert r.release_id == "r1"

    def test_to_dict_roundtrip(self) -> None:
        r1 = ProvenanceRecord(
            release_id="r1",
            manifest_digest="a" * 64,
            approval_digest="b" * 64,
            source_id="s1",
            shard_path="shards/shard-0000",
            shard_digest="c" * 64,
            record_index=0,
            record_id="rec-0",
            language="en",
            domain="general",
            utf8_bytes=10,
            content_sha256="d" * 64,
        )
        d = r1.to_dict()
        r2 = ProvenanceRecord(**d)
        assert r1 == r2


class TestCorpusManifest:
    def test_digest_excludes_own_sha256(self) -> None:
        cfg = SamplerConfig(version="1.0", seed=0)
        m1 = CorpusManifest(
            schema_version="1",
            created_at="2026-01-01T00:00:00Z",
            sampler_config=cfg,
            releases=(),
            total_candidates=0,
            total_selected=0,
            exact_dedup_removed=0,
            per_source_cap_removed=0,
            per_language_cap_removed=0,
            global_cap_removed=0,
            total_corpus_bytes=0,
            per_source_records={},
            per_source_bytes={},
            per_language_records={},
            per_language_bytes={},
            per_domain_records={},
            per_domain_bytes={},
            corpus_sha256="0" * 64,
            records=(),
        )
        m2 = CorpusManifest(
            schema_version="1",
            created_at="2026-01-01T00:00:00Z",
            sampler_config=cfg,
            releases=(),
            total_candidates=0,
            total_selected=0,
            exact_dedup_removed=0,
            per_source_cap_removed=0,
            per_language_cap_removed=0,
            global_cap_removed=0,
            total_corpus_bytes=0,
            per_source_records={},
            per_source_bytes={},
            per_language_records={},
            per_language_bytes={},
            per_domain_records={},
            per_domain_bytes={},
            corpus_sha256="0" * 64,
            records=(),
        )
        assert m1.compute_digest() == m2.compute_digest()


class TestSampleExecution:
    def test_basic_sampling(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_candidates > 0
        assert result.total_selected == result.total_candidates
        assert len(result.records) == result.total_selected

    def test_multiple_approved_releases(self, tmp_path: Path) -> None:
        root_a, ma, aa = _setup_single_release(
            tmp_path,
            language="en",
            source_id="src-a",
            release_id="rel-a",
            shard_records=[{"text": "hello", "language": "en", "accepted": True}],
        )
        root_b, mb, ab = _setup_single_release(
            tmp_path,
            language="hi",
            source_id="src-b",
            release_id="rel-b",
            shard_records=[{"text": "नमस्ते", "language": "hi", "accepted": True}],
            dataset_id="ds-test-2",
        )
        result = sample_tokenizer_corpus(
            release_roots=[root_a, root_b],
            manifest_paths=[ma, mb],
            approval_paths=[aa, ab],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_selected == 2

    def test_rejected_approval_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            approval_status="rejected",
            shard_records=[{"text": "x", "accepted": True}],
        )
        with pytest.raises(ValueError, match="approval status"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_pending_approval_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            approval_status="pending",
            shard_records=[{"text": "x", "accepted": True}],
        )
        with pytest.raises(ValueError, match="approval status"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_manifest_digest_mismatch_rejected(self, tmp_path: Path) -> None:
        root, _, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        fake_manifest = _make_manifest(dataset_id="ds-test", records=1)
        bad_mfest = root / "manifest.json"
        bad_mfest.write_text(json.dumps(fake_manifest.to_dict(), indent=2), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest digest mismatch"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[bad_mfest],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_missing_shard_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        (root / "shards" / "shard-0000").unlink()
        with pytest.raises(FileNotFoundError, match="shard not found"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_tampered_shard_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        (root / "shards" / "shard-0000").write_bytes(b"tampered")
        with pytest.raises(ValueError, match="sha256 mismatch"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_remote_path_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="remote.*not allowed"):
            sample_tokenizer_corpus(
                release_roots=[Path("https://evil.com")],
                manifest_paths=[tmp_path / "m.json"],
                approval_paths=[tmp_path / "a.json"],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        outside = tmp_path / "outside_target"
        outside.write_text("data")
        link = root / "shards" / "escape"
        link.symlink_to(outside)
        mdata = json.loads(mpath.read_text(encoding="utf-8"))
        mdata["shards"][0]["shard_id"] = "escape"
        mpath.write_text(json.dumps(mdata, indent=2), encoding="utf-8")
        # Recompute all digests to keep the governance chain consistent
        updated_manifest = DatasetManifest.from_dict(mdata)
        new_manifest_digest = updated_manifest.digest()
        release_data = json.loads((root / "dataset_release.json").read_text(encoding="utf-8"))
        audit_data = json.loads((root / "audit_report.json").read_text(encoding="utf-8"))
        approval_data = json.loads((root / "approval.json").read_text(encoding="utf-8"))
        approval_data["manifest_digest"] = new_manifest_digest
        new_approval = DatasetApproval(**approval_data)
        new_approval_digest = new_approval.digest()
        release_data["manifest_digest"] = new_manifest_digest
        release_data["approval_digest"] = new_approval_digest
        audit_data["manifest_digest"] = new_manifest_digest
        audit_data["approval_digest"] = new_approval_digest
        (root / "dataset_release.json").write_text(
            json.dumps(release_data, indent=2), encoding="utf-8"
        )
        (root / "audit_report.json").write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
        (root / "approval.json").write_text(
            json.dumps(new_approval.to_dict(), indent=2), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="symlink escape"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def _update_governance_digests(self, root: Path, mpath: Path) -> None:
        mdata = json.loads(mpath.read_text(encoding="utf-8"))
        updated_manifest = DatasetManifest.from_dict(mdata)
        new_manifest_digest = updated_manifest.digest()
        release_data = json.loads((root / "dataset_release.json").read_text(encoding="utf-8"))
        audit_data = json.loads((root / "audit_report.json").read_text(encoding="utf-8"))
        approval_data = json.loads((root / "approval.json").read_text(encoding="utf-8"))
        approval_data["manifest_digest"] = new_manifest_digest
        new_approval = DatasetApproval(**approval_data)
        new_approval_digest = new_approval.digest()
        release_data["manifest_digest"] = new_manifest_digest
        release_data["approval_digest"] = new_approval_digest
        audit_data["manifest_digest"] = new_manifest_digest
        audit_data["approval_digest"] = new_approval_digest
        (root / "dataset_release.json").write_text(
            json.dumps(release_data, indent=2), encoding="utf-8"
        )
        (root / "audit_report.json").write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
        (root / "approval.json").write_text(
            json.dumps(new_approval.to_dict(), indent=2), encoding="utf-8"
        )

    def test_malformed_utf8_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        shard_path = root / "shards" / "shard-0000"
        shard_path.write_bytes(b"\xff\xfe\x00")
        mdata = json.loads(mpath.read_text(encoding="utf-8"))
        mdata["shards"][0]["sha256"] = hashlib.sha256(b"\xff\xfe\x00").hexdigest()
        mpath.write_text(json.dumps(mdata, indent=2), encoding="utf-8")
        self._update_governance_digests(root, mpath)
        with pytest.raises(ValueError, match="malformed UTF-8"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_lone_surrogate_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        shard_path = root / "shards" / "shard-0000"
        shard_bytes = (
            json.dumps({"text": "bad \ud800 char", "accepted": True}).encode("utf-8") + b"\n"
        )
        shard_path.write_bytes(shard_bytes)
        mdata = json.loads(mpath.read_text(encoding="utf-8"))
        mdata["shards"][0]["sha256"] = hashlib.sha256(shard_bytes).hexdigest()
        mpath.write_text(json.dumps(mdata, indent=2), encoding="utf-8")
        self._update_governance_digests(root, mpath)
        with pytest.raises(ValueError, match="lone surrogate"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_whitespace_only_records_skipped(self, tmp_path: Path) -> None:
        records = [
            {"text": "valid", "accepted": True},
            {"text": "   ", "accepted": True},
            {"text": "", "accepted": True},
        ]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_candidates == 1
        assert result.total_selected == 1

    def test_embedded_newline_preserved(self, tmp_path: Path) -> None:
        records = [{"text": "line1\nline2\nline3", "accepted": True}]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_selected == 1
        line = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8")
        assert "line1\\nline2" in line or "line1\nline2" in json.loads(line)["text"]

    def test_tab_and_space_preserved(self, tmp_path: Path) -> None:
        records = [{"text": "tab\there  space", "accepted": True}]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_selected == 1
        text = json.loads((tmp_path / "corpus.jsonl").read_text(encoding="utf-8").strip())["text"]
        assert "tab\there" in text
        assert "  space" in text

    def test_exact_dedup_removes_duplicates(self, tmp_path: Path) -> None:
        records = [
            {"text": "unique", "accepted": True},
            {"text": "dup", "accepted": True},
            {"text": "dup", "accepted": True},
        ]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_candidates == 3
        assert result.total_selected == 2
        assert result.exact_dedup_removed == 1

    def test_non_accepted_records_skipped(self, tmp_path: Path) -> None:
        records = [
            {"text": "accepted", "accepted": True},
            {"text": "rejected", "accepted": False},
        ]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_candidates == 1
        assert result.total_selected == 1

    def test_global_record_cap(self, tmp_path: Path) -> None:
        records = [{"text": f"r{i}", "accepted": True} for i in range(10)]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_total_records=3,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_selected == 3
        assert result.global_cap_removed == 7

    def test_global_byte_cap(self, tmp_path: Path) -> None:
        records = [{"text": chr(ord("a") + i) * 100, "accepted": True} for i in range(5)]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_total_bytes=250,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_selected == 2
        assert result.global_cap_removed == 3

    def test_per_source_record_cap(self, tmp_path: Path) -> None:
        records = [{"text": f"r{i}", "accepted": True} for i in range(10)]
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=records,
            source_id="src-a",
        )
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_records_per_source={"src-a": 4},
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_selected == 4
        assert result.per_source_cap_removed == 6

    def test_per_language_record_cap(self, tmp_path: Path) -> None:
        records = [{"text": f"r{i}", "language": "hi", "accepted": True} for i in range(10)]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records, language="hi")
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_records_per_language={"hi": 3},
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_selected == 3
        assert result.per_language_cap_removed == 7

    def test_per_source_byte_cap(self, tmp_path: Path) -> None:
        records = [{"text": chr(ord("a") + i) * 50, "accepted": True} for i in range(10)]
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=records,
            source_id="src-a",
        )
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_bytes_per_source={"src-a": 120},
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_selected == 2
        assert result.per_source_cap_removed == 8

    def test_deterministic_ranking(self, tmp_path: Path) -> None:
        records = [{"text": f"r{i}", "accepted": True} for i in range(10)]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        r1 = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_total_records=5,
                output_corpus=str(tmp_path / "c1.jsonl"),
                output_manifest=str(tmp_path / "m1.json"),
            ),
        )
        r2 = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_total_records=5,
                output_corpus=str(tmp_path / "c2.jsonl"),
                output_manifest=str(tmp_path / "m2.json"),
            ),
        )
        assert r1.records == r2.records

    def test_different_seed_changes_ranking(self, tmp_path: Path) -> None:
        records = [{"text": f"r{i}", "accepted": True} for i in range(10)]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        r1 = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_total_records=5,
                output_corpus=str(tmp_path / "c1.jsonl"),
                output_manifest=str(tmp_path / "m1.json"),
            ),
        )
        r2 = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=99,
                max_total_records=5,
                output_corpus=str(tmp_path / "c2.jsonl"),
                output_manifest=str(tmp_path / "m2.json"),
            ),
        )
        assert r1.records != r2.records

    def test_no_raw_text_in_manifest(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        manifest_text = (tmp_path / "manifest.json").read_text(encoding="utf-8")
        assert "hello world" not in manifest_text
        assert "foo bar" not in manifest_text

    def test_content_digests_match_emitted_records(self, tmp_path: Path) -> None:
        records = [{"text": "test content", "accepted": True}]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        corpus_text = json.loads((tmp_path / "corpus.jsonl").read_text(encoding="utf-8").strip())[
            "text"
        ]
        expected_sha = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()
        assert result.records[0].content_sha256 == expected_sha

    def test_corpus_digest_matches_exact_bytes(self, tmp_path: Path) -> None:
        records = [{"text": "digest check", "accepted": True}]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        actual_sha = hashlib.sha256((tmp_path / "corpus.jsonl").read_bytes()).hexdigest()
        assert result.corpus_sha256 == actual_sha

    def test_existing_output_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        (tmp_path / "corpus.jsonl").write_text("existing", encoding="utf-8")
        with pytest.raises(FileExistsError, match="corpus already exists"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(
                    version="1.0",
                    seed=42,
                    output_corpus=str(tmp_path / "corpus.jsonl"),
                    output_manifest=str(tmp_path / "manifest.json"),
                ),
            )

    def test_writer_failure_cleanup(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        parent = tmp_path / "blocker"
        parent.write_text("i am a file, not a directory")
        with pytest.raises(OSError):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(
                    version="1.0",
                    seed=42,
                    output_corpus=str(parent / "corpus.jsonl"),
                    output_manifest=str(tmp_path / "manifest.json"),
                ),
            )

    def test_manifest_failure_rollback(self, tmp_path: Path) -> None:
        records = [{"text": "rollback test", "accepted": True}]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        corpus_path = tmp_path / "corpus.jsonl"
        manifest_path = tmp_path / "manifest.json"
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(corpus_path),
                output_manifest=str(manifest_path),
            ),
        )
        assert corpus_path.exists()
        assert manifest_path.exists()
        assert result.corpus_sha256 != "0" * 64

    def test_dry_run_creates_no_files(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(version="1.0", seed=42),
        )
        assert result.total_selected == 2
        assert result.corpus_sha256 == "0" * 64
        assert not (tmp_path / "corpus.jsonl").exists()
        assert not (tmp_path / "manifest.json").exists()

    def test_dry_run_with_output_set_raises(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        with pytest.raises(ValueError, match="output_manifest must be set"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(
                    version="1.0",
                    seed=42,
                    output_corpus=str(tmp_path / "corpus.jsonl"),
                ),
            )

    def test_release_and_shard_ordering_independence(self, tmp_path: Path) -> None:
        records_a = [{"text": f"a-{i}", "accepted": True} for i in range(3)]
        records_b = [{"text": f"b-{i}", "accepted": True} for i in range(3)]
        root_a, ma, aa = _setup_single_release(
            tmp_path,
            shard_records=records_a,
            source_id="src-a",
            release_id="rel-a",
            dataset_id="ds-a",
        )
        root_b, mb, ab = _setup_single_release(
            tmp_path,
            shard_records=records_b,
            source_id="src-b",
            release_id="rel-b",
            dataset_id="ds-b",
        )
        r1 = sample_tokenizer_corpus(
            release_roots=[root_a, root_b],
            manifest_paths=[ma, mb],
            approval_paths=[aa, ab],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "c1.jsonl"),
                output_manifest=str(tmp_path / "m1.json"),
            ),
        )
        r2 = sample_tokenizer_corpus(
            release_roots=[root_b, root_a],
            manifest_paths=[mb, ma],
            approval_paths=[ab, aa],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "c2.jsonl"),
                output_manifest=str(tmp_path / "m2.json"),
            ),
        )
        assert r1.records == r2.records
        assert r1.corpus_sha256 == r2.corpus_sha256

    def test_per_source_bytes_tracked(self, tmp_path: Path) -> None:
        records = [{"text": "data", "accepted": True}]
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=records,
            source_id="src-x",
        )
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.per_source_bytes.get("src-x", 0) > 0

    def test_per_language_bytes_tracked(self, tmp_path: Path) -> None:
        records = [{"text": "data", "language": "mr", "accepted": True}]
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=records,
            language="mr",
        )
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.per_language_bytes.get("mr", 0) > 0

    def test_domain_tracking(self, tmp_path: Path) -> None:
        records = [{"text": "code text", "domain": "code", "accepted": True}]
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=records,
            domain="code",
        )
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.per_domain_records.get("code", 0) == 1
        assert result.per_domain_bytes.get("code", 0) > 0

    def test_cap_precedence_order(self, tmp_path: Path) -> None:
        root_a, ma, aa = _setup_single_release(
            tmp_path,
            shard_records=[
                {"text": f"r{i}", "language": "en", "accepted": True} for i in range(10)
            ],
            source_id="src-a",
            release_id="rel-a",
        )
        result = sample_tokenizer_corpus(
            release_roots=[root_a],
            manifest_paths=[ma],
            approval_paths=[aa],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                max_total_records=100,
                max_records_per_source={"src-a": 2},
                max_records_per_language={"en": 4},
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.per_source_cap_removed == 8
        assert result.per_language_cap_removed == 0
        assert result.total_selected == 2

    def test_missing_approval_path_raises(self, tmp_path: Path) -> None:
        root, mpath, _ = _setup_single_release(tmp_path)
        bad_apath = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="approval not found"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[bad_apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_missing_license_review_raises(self, tmp_path: Path) -> None:
        records = [{"text": "x", "accepted": True}]
        manifest = _make_manifest(records=len(records))
        approval = _make_approval(manifest, license_reviewed=False)
        release = _make_release(manifest, approval)
        root, mpath, apath = _write_release_root(
            tmp_path,
            "rel",
            manifest,
            approval,
            release,
            records,
        )
        with pytest.raises(ValueError, match="license_reviewed"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_missing_pii_review_raises(self, tmp_path: Path) -> None:
        records = [{"text": "x", "accepted": True}]
        manifest = _make_manifest(records=len(records))
        approval = _make_approval(manifest, pii_reviewed=False)
        release = _make_release(manifest, approval)
        root, mpath, apath = _write_release_root(
            tmp_path,
            "rel",
            manifest,
            approval,
            release,
            records,
        )
        with pytest.raises(ValueError, match="pii_reviewed"):
            sample_tokenizer_corpus(
                release_roots=[root],
                manifest_paths=[mpath],
                approval_paths=[apath],
                config=SamplerConfig(version="1.0", seed=0),
            )

    def test_empty_shard_handled(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=[])
        result = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "corpus.jsonl"),
                output_manifest=str(tmp_path / "manifest.json"),
            ),
        )
        assert result.total_candidates == 0
        assert result.total_selected == 0

    def test_repeated_byte_identical_corpus(self, tmp_path: Path) -> None:
        records = [{"text": "hello", "accepted": True}]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        r1 = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "c1.jsonl"),
                output_manifest=str(tmp_path / "m1.json"),
            ),
        )
        r2 = sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "c2.jsonl"),
                output_manifest=str(tmp_path / "m2.json"),
            ),
        )
        c1 = (tmp_path / "c1.jsonl").read_bytes()
        c2 = (tmp_path / "c2.jsonl").read_bytes()
        assert c1 == c2
        assert r1.corpus_sha256 == r2.corpus_sha256

    def test_repeated_byte_identical_manifest(self, tmp_path: Path) -> None:
        records = [{"text": "hello", "accepted": True}]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "c1.jsonl"),
                output_manifest=str(tmp_path / "m1.json"),
            ),
        )
        sample_tokenizer_corpus(
            release_roots=[root],
            manifest_paths=[mpath],
            approval_paths=[apath],
            config=SamplerConfig(
                version="1.0",
                seed=42,
                output_corpus=str(tmp_path / "c2.jsonl"),
                output_manifest=str(tmp_path / "m2.json"),
            ),
        )
        m1 = (tmp_path / "m1.json").read_bytes()
        m2 = (tmp_path / "m2.json").read_bytes()
        assert m1 == m2
