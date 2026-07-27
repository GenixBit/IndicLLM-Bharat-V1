from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.tokenizer.sampler import sample_tokenizer_corpus


def _write_release(root: Path, records: list[dict[str, object]]) -> None:
    root.mkdir()
    (root / "shards").mkdir()
    release = {
        "release_id": "release-demo-2026-07-27",
        "dataset_id": "demo",
        "manifest_digest": "a" * 64,
        "approval_digest": "b" * 64,
        "shard_count": 1,
        "records": len(records),
        "bytes_utf8": 0,
        "created_at": "2026-07-27T00:00:00Z",
        "package_sha256": "c" * 64,
    }
    audit = {
        "dataset_id": "demo",
        "manifest_digest": "a" * 64,
        "approval_digest": "b" * 64,
        "shard_checks_passed": True,
        "approval_checks_passed": True,
        "total_records": len(records),
        "total_bytes_utf8": 0,
        "issues": [],
    }
    (root / "dataset_release.json").write_text(json.dumps(release), encoding="utf-8")
    (root / "audit_report.json").write_text(json.dumps(audit), encoding="utf-8")
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    (root / "shards" / "part-00000.jsonl").write_text(lines, encoding="utf-8")


def test_sampling_is_deterministic(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write_release(
        release,
        [
            {"record_id": "c", "text": "বাংলা"},
            {"record_id": "a", "text": "हिन्दी"},
            {"record_id": "b", "text": "English"},
        ],
    )

    first = sample_tokenizer_corpus(release, tmp_path / "first.txt", sample_size=2, seed=7)
    second = sample_tokenizer_corpus(release, tmp_path / "second.txt", sample_size=2, seed=7)

    assert first == second
    assert (tmp_path / "first.txt").read_bytes() == (tmp_path / "second.txt").read_bytes()
    assert first.selected_records == 2
    assert len(first.corpus_sha256) == 64


def test_shard_order_does_not_change_result(tmp_path: Path) -> None:
    records = [
        {"record_id": "1", "text": "one"},
        {"record_id": "2", "text": "two"},
        {"record_id": "3", "text": "three"},
    ]
    first_release = tmp_path / "first-release"
    second_release = tmp_path / "second-release"
    _write_release(first_release, records)
    _write_release(second_release, list(reversed(records)))

    first = sample_tokenizer_corpus(first_release, tmp_path / "one.txt", sample_size=3, seed=2)
    second = sample_tokenizer_corpus(second_release, tmp_path / "two.txt", sample_size=3, seed=2)

    assert first.record_ids == second.record_ids
    assert first.corpus_sha256 == second.corpus_sha256


def test_rejects_unapproved_release(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write_release(release, [{"record_id": "1", "text": "text"}])
    audit_path = release / "audit_report.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["approval_checks_passed"] = False
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="not fully approved"):
        sample_tokenizer_corpus(release, tmp_path / "out.txt", sample_size=1)


def test_rejects_remote_paths() -> None:
    with pytest.raises(ValueError, match="remote paths"):
        sample_tokenizer_corpus("https://example.com/release", "out.txt", sample_size=1)


def test_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write_release(
        release,
        [
            {"record_id": "same", "text": "one"},
            {"record_id": "same", "text": "two"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate record_id"):
        sample_tokenizer_corpus(release, tmp_path / "out.txt", sample_size=1)


def test_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write_release(release, [{"record_id": "1", "text": "text"}])
    output = tmp_path / "out.txt"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        sample_tokenizer_corpus(release, output, sample_size=1)

    assert output.read_text(encoding="utf-8") == "existing"
