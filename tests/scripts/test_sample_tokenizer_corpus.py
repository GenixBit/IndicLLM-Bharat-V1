from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bharat.data.approval import DatasetApproval
from bharat.data.manifest import DatasetManifest, ShardManifest
from bharat.data.release import DatasetAuditReport, DatasetRelease


def _digest(data: bytes = b"test") -> str:
    return hashlib.sha256(data).hexdigest()


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.sample_tokenizer_corpus", *args],
        capture_output=True,
        text=True,
    )


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


class TestSampleTokenizerCorpusCLI:
    def test_help(self) -> None:
        result = run_cli(["--help"])
        assert result.returncode == 0
        assert "tokenizer corpus" in result.stdout.lower()

    def test_missing_version(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
            ]
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower()

    def test_dry_run_default(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "dry-run"
        assert data["total_candidates"] > 0
        assert "Dry-run complete" in result.stderr

    def test_output_without_execute_fails(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        corpus = tmp_path / "corpus.jsonl"
        manifest = tmp_path / "manifest.json"
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--output-corpus",
                str(corpus),
                "--output-manifest",
                str(manifest),
            ]
        )
        assert result.returncode != 0
        assert "--execute is required" in result.stderr

    def test_execute_creates_files(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        corpus = tmp_path / "corpus.jsonl"
        manifest = tmp_path / "manifest.json"
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--output-corpus",
                str(corpus),
                "--output-manifest",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["status"] == "success"
        assert corpus.exists()
        assert manifest.exists()
        assert corpus.stat().st_size > 0
        assert data["corpus_sha256"] != "0" * 64

    def test_execute_deterministic(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        c1 = tmp_path / "c1.jsonl"
        m1 = tmp_path / "m1.json"
        r1 = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--output-corpus",
                str(c1),
                "--output-manifest",
                str(m1),
                "--execute",
            ]
        )
        assert r1.returncode == 0
        c2 = tmp_path / "c2.jsonl"
        m2 = tmp_path / "m2.json"
        r2 = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--output-corpus",
                str(c2),
                "--output-manifest",
                str(m2),
                "--execute",
            ]
        )
        assert r2.returncode == 0
        assert c1.read_bytes() == c2.read_bytes()
        m1_data = json.loads(m1.read_text(encoding="utf-8"))
        m2_data = json.loads(m2.read_text(encoding="utf-8"))
        assert m1_data["total_selected"] == m2_data["total_selected"]
        assert m1_data["total_corpus_bytes"] == m2_data["total_corpus_bytes"]
        assert m1_data["corpus_sha256"] == m2_data["corpus_sha256"]

    def test_multiple_releases(self, tmp_path: Path) -> None:
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
        corpus = tmp_path / "corpus.jsonl"
        manifest = tmp_path / "manifest.json"
        result = run_cli(
            [
                "--release-root",
                str(root_a),
                "--release-root",
                str(root_b),
                "--manifest-path",
                str(ma),
                "--manifest-path",
                str(mb),
                "--approval-path",
                str(aa),
                "--approval-path",
                str(ab),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--output-corpus",
                str(corpus),
                "--output-manifest",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["total_selected"] == 2

    def test_count_mismatch_rejected(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--release-root",
                str(tmp_path / "extra"),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
            ]
        )
        assert result.returncode != 0
        assert "counts must match" in result.stderr

    def test_rejected_approval_fails(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            approval_status="rejected",
            shard_records=[{"text": "x", "accepted": True}],
        )
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
            ]
        )
        assert result.returncode != 0
        assert "approval status" in result.stderr.lower()

    def test_missing_shard_fails(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        (root / "shards" / "shard-0000").unlink()
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
            ]
        )
        assert result.returncode != 0
        assert "shard not found" in result.stderr.lower()

    def test_tampered_shard_fails(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        (root / "shards" / "shard-0000").write_bytes(b"tampered")
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
            ]
        )
        assert result.returncode != 0
        assert "sha256 mismatch" in result.stderr.lower()

    def test_global_record_cap(self, tmp_path: Path) -> None:
        records = [{"text": f"r{i}", "accepted": True} for i in range(10)]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        corpus = tmp_path / "corpus.jsonl"
        manifest = tmp_path / "manifest.json"
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--max-total-records",
                "3",
                "--output-corpus",
                str(corpus),
                "--output-manifest",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["total_selected"] == 3
        assert data["global_cap_removed"] == 7

    def test_dedup_flag(self, tmp_path: Path) -> None:
        records = [
            {"text": "unique", "accepted": True},
            {"text": "dup", "accepted": True},
            {"text": "dup", "accepted": True},
        ]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["exact_dedup_removed"] == 1

    def test_no_dedup_flag(self, tmp_path: Path) -> None:
        records = [
            {"text": "dup", "accepted": True},
            {"text": "dup", "accepted": True},
        ]
        root, mpath, apath = _setup_single_release(tmp_path, shard_records=records)
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--no-dedup",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["exact_dedup_removed"] == 0

    def test_invalid_cap_format(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--max-records-per-source",
                "badformat",
            ]
        )
        assert result.returncode != 0
        assert "Invalid cap" in result.stderr

    def test_invalid_cap_value(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--max-records-per-source",
                "src:notanint",
            ]
        )
        assert result.returncode != 0
        assert "must be an integer" in result.stderr

    def test_manifest_digest_mismatch_fails(self, tmp_path: Path) -> None:
        root, _, apath = _setup_single_release(
            tmp_path,
            shard_records=[{"text": "x", "accepted": True}],
        )
        bad_manifest = _make_manifest(dataset_id="ds-test", records=1)
        bad_mpath = root / "manifest.json"
        bad_mpath.write_text(json.dumps(bad_manifest.to_dict(), indent=2), encoding="utf-8")
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(bad_mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
            ]
        )
        assert result.returncode != 0
        assert "manifest digest mismatch" in result.stderr.lower()

    def test_json_output_fields(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "status" in data
        assert "total_candidates" in data
        assert "total_selected" in data
        assert "exact_dedup_removed" in data
        assert "total_corpus_bytes" in data
        assert "corpus_sha256" in data

    def test_execute_result_includes_manifest_sha256(self, tmp_path: Path) -> None:
        root, mpath, apath = _setup_single_release(tmp_path)
        corpus = tmp_path / "corpus.jsonl"
        manifest = tmp_path / "manifest.json"
        result = run_cli(
            [
                "--release-root",
                str(root),
                "--manifest-path",
                str(mpath),
                "--approval-path",
                str(apath),
                "--version",
                "1.0",
                "--seed",
                "42",
                "--output-corpus",
                str(corpus),
                "--output-manifest",
                str(manifest),
                "--execute",
            ]
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["manifest_sha256"] != ""
        assert len(data["manifest_sha256"]) == 64
