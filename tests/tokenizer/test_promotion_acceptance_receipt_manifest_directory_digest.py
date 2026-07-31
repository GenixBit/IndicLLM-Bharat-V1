from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from bharat.tokenizer import (
    promotion_acceptance,
    promotion_acceptance_directory,
    promotion_acceptance_receipt,
    promotion_acceptance_receipt_directory,
    promotion_acceptance_receipt_manifest,
    promotion_bundle_directory,
    promotion_receipt,
)
from bharat.tokenizer import promotion_acceptance_receipt_manifest_directory_digest as module
from bharat.tokenizer.production_evidence_builder import write_candidate_manifest
from bharat.tokenizer.production_evidence_readiness import write_readiness_report
from bharat.tokenizer.promotion_decision import build_promotion_decision, write_promotion_decision
from bharat.tokenizer.promotion_package import verify_promotion_package
from bharat.tokenizer.promotion_package_directory import PromotionPackageDirectoryVerification
from tests.tokenizer.evidence_fixtures import (
    build_acceptance_decision,
    build_bpe_tokenizer,
    build_input_jsonl,
    build_production_thresholds,
    canonical_bytes,
    compute_real_report,
)


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "manifest-evidence"
    receipt_directory = root / "receipt-evidence"
    acceptance_directory = receipt_directory / "accepted-promotion"
    acceptance_directory.mkdir(parents=True)
    (acceptance_directory / "decision.json").write_text("accepted", encoding="utf-8")
    (receipt_directory / "acceptance-receipt.json").write_text("receipt", encoding="utf-8")
    (root / "acceptance-receipt-manifest.json").write_text("manifest", encoding="utf-8")
    return root


def test_verifies_exact_directory_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path)
    verified = SimpleNamespace(root=root)
    calls: list[Path] = []

    def verifier(path: Path) -> object:
        calls.append(path)
        return verified

    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest_directory",
        verifier,
    )
    expected = module.promotion_acceptance_receipt_manifest_directory_sha256(root)

    result = module.verify_promotion_acceptance_receipt_manifest_directory_digest(
        root,
        expected,
    )

    assert result.directory is verified
    assert result.sha256 == expected
    assert calls == [root]


def test_digest_is_deterministic_across_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    (first / "b.txt").write_text("two", encoding="utf-8")
    (first / "a.txt").write_text("one", encoding="utf-8")
    (second / "a.txt").write_text("one", encoding="utf-8")
    (second / "b.txt").write_text("two", encoding="utf-8")

    first_digest = module.promotion_acceptance_receipt_manifest_directory_sha256(first)
    second_digest = module.promotion_acceptance_receipt_manifest_directory_sha256(second)

    assert first_digest == second_digest


def test_digest_changes_with_path_or_content(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "evidence.json"
    path.write_text("one", encoding="utf-8")
    original = module.promotion_acceptance_receipt_manifest_directory_sha256(root)

    path.write_text("two", encoding="utf-8")
    changed_content = module.promotion_acceptance_receipt_manifest_directory_sha256(root)
    path.rename(root / "renamed.json")
    changed_path = module.promotion_acceptance_receipt_manifest_directory_sha256(root)

    assert changed_content != original
    assert changed_path != changed_content


def test_rejects_missing_digest_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a regular directory"):
        module.promotion_acceptance_receipt_manifest_directory_sha256(tmp_path / "missing")


def test_rejects_symlinked_digest_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must be a regular directory"):
        module.promotion_acceptance_receipt_manifest_directory_sha256(link)


def test_rejects_regular_file_digest_root(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a regular directory"):
        module.promotion_acceptance_receipt_manifest_directory_sha256(path)


def test_verifier_rejects_missing_root_before_semantic_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verifier(path: Path) -> object:
        raise AssertionError("semantic verifier must not run for a missing root")

    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest_directory",
        verifier,
    )

    with pytest.raises(ValueError, match="must be a regular directory"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(
            tmp_path / "missing",
            "0" * 64,
        )


@pytest.mark.parametrize(
    "value",
    ["", "0" * 63, "0" * 65, "A" * 64, "g" * 64],
)
def test_rejects_invalid_expected_digest(tmp_path: Path, value: str) -> None:
    root = _fixture(tmp_path)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(root, value)


def test_rejects_digest_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest_directory",
        lambda path: SimpleNamespace(root=path),
    )

    with pytest.raises(ValueError, match="digest does not match"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(
            root,
            "0" * 64,
        )


def test_rejects_symlinked_entries(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    (root / "evidence.txt").symlink_to(target)

    with pytest.raises(ValueError, match="non-regular entry"):
        module.promotion_acceptance_receipt_manifest_directory_sha256(root)


def _make_mutating_verifier(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[Path], None],
) -> None:
    original = module.promotion_acceptance_receipt_manifest_directory_sha256
    calls: list[int] = []

    def flaky(path: Path) -> str:
        calls.append(len(calls))
        if len(calls) == 2:
            mutate(path)
        return original(path)

    monkeypatch.setattr(module, "promotion_acceptance_receipt_manifest_directory_sha256", flaky)
    monkeypatch.setattr(
        module,
        "verify_promotion_acceptance_receipt_manifest_directory",
        lambda path: SimpleNamespace(root=path),
    )


def test_rejects_content_mutation_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path)
    target = root / "receipt-evidence" / "acceptance-receipt.json"

    def mutate(path: Path) -> None:
        target.write_text("tampered", encoding="utf-8")

    _make_mutating_verifier(monkeypatch, mutate)

    with pytest.raises(ValueError, match="changed during verification"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(root, "0" * 64)


def test_rejects_rename_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path)
    target = root / "receipt-evidence" / "acceptance-receipt.json"
    renamed = root / "receipt-evidence" / "renamed.json"

    def mutate(path: Path) -> None:
        target.rename(renamed)

    _make_mutating_verifier(monkeypatch, mutate)

    with pytest.raises(ValueError, match="changed during verification"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(root, "0" * 64)


def test_rejects_new_file_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path)
    added = root / "receipt-evidence" / "unexpected.json"

    def mutate(path: Path) -> None:
        added.write_text("unexpected", encoding="utf-8")

    _make_mutating_verifier(monkeypatch, mutate)

    with pytest.raises(ValueError, match="changed during verification"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(root, "0" * 64)


def test_rejects_symlink_substitution_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path)
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    replaced = root / "receipt-evidence" / "acceptance-receipt.json"

    def mutate(path: Path) -> None:
        replaced.unlink()
        replaced.symlink_to(target)

    _make_mutating_verifier(monkeypatch, mutate)

    with pytest.raises(ValueError, match="non-regular entry"):
        module.verify_promotion_acceptance_receipt_manifest_directory_digest(root, "0" * 64)


def test_integration_real_acceptance_evidence_hierarchy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = "operator@example.com"
    reviewer = "reviewer@example.com"

    package = tmp_path / "package"
    package.mkdir()
    tokenizer_path = build_bpe_tokenizer(package, "tokenizer.json")
    input_path = build_input_jsonl(package, name="input.jsonl")
    thresholds_path = build_production_thresholds(package, name="thresholds.json")
    report_path = compute_real_report(
        package,
        tokenizer_path,
        input_path,
        "test-bpe",
        "report.json",
    )
    acceptance_decision_path = build_acceptance_decision(
        package,
        report_path,
        thresholds_path,
        "test-bpe",
        "acceptance-decision.json",
    )

    manifest_path = package / "manifest.json"
    write_candidate_manifest(
        manifest_path,
        evidence_root=package,
        repository_commit_sha="a" * 40,
        tokenizer_path=tokenizer_path,
        evaluation_input_path=input_path,
        evaluation_report_path=report_path,
        acceptance_decision_path=acceptance_decision_path,
        threshold_configuration_path=thresholds_path,
        generating_commands=["integration-test-fixture"],
    )
    readiness_path = package / "readiness.json"
    write_readiness_report(manifest_path, readiness_path)
    decision_path = package / "decision.json"
    decision = build_promotion_decision(
        manifest_path,
        readiness_path,
        decision="approve",
        operator=operator,
        rationale="integration test fixture",
    )
    write_promotion_decision(decision, decision_path)

    def package_verifier(directory: Path) -> PromotionPackageDirectoryVerification:
        for name in ("manifest.json", "readiness.json", "decision.json"):
            entry = directory / name
            if not entry.is_file():
                raise ValueError(f"required package file is missing: {name}")
        package_verification = verify_promotion_package(
            directory / "manifest.json",
            directory / "readiness.json",
            directory / "decision.json",
        )
        return PromotionPackageDirectoryVerification(
            package=package_verification,
            filenames=("manifest.json", "readiness.json", "decision.json"),
        )

    monkeypatch.setattr(promotion_receipt, "verify_promotion_package_directory", package_verifier)

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.move(str(package), str(bundle / "package"))
    (bundle / "receipt.json").write_bytes(
        canonical_bytes(
            {
                "schema_version": "tokenizer-promotion-receipt-v1",
                "manifest_sha256": digest(bundle / "package" / "manifest.json"),
                "readiness_sha256": digest(bundle / "package" / "readiness.json"),
                "decision_sha256": digest(bundle / "package" / "decision.json"),
                "operator": operator,
            }
        )
    )
    promotion_bundle_directory.verify_promotion_bundle_directory(bundle)

    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_bytes(
        canonical_bytes(
            {
                "schema_version": "tokenizer-promotion-acceptance-v1",
                "accepted": True,
                "operator": operator,
                "reviewer": reviewer,
                "receipt_sha256": digest(bundle / "receipt.json"),
            }
        )
    )
    promotion_acceptance.verify_promotion_acceptance(bundle, acceptance_path)

    accepted = tmp_path / "accepted-promotion"
    accepted.mkdir()
    shutil.move(str(bundle), str(accepted / "bundle"))
    shutil.move(str(acceptance_path), str(accepted / "acceptance.json"))
    promotion_acceptance_directory.verify_promotion_acceptance_directory(accepted)

    receipt_path = tmp_path / "acceptance-receipt.json"
    receipt_path.write_bytes(
        canonical_bytes(
            {
                "schema_version": "tokenizer-promotion-acceptance-receipt-v1",
                "operator": operator,
                "reviewer": reviewer,
                "acceptance_sha256": digest(accepted / "acceptance.json"),
            }
        )
    )
    promotion_acceptance_receipt.verify_promotion_acceptance_receipt(accepted, receipt_path)

    receipt_directory = tmp_path / "receipt-evidence"
    receipt_directory.mkdir()
    shutil.move(str(accepted), str(receipt_directory / "accepted-promotion"))
    shutil.move(str(receipt_path), str(receipt_directory / "acceptance-receipt.json"))
    promotion_acceptance_receipt_directory.verify_promotion_acceptance_receipt_directory(
        receipt_directory
    )

    top_manifest = tmp_path / "acceptance-receipt-manifest.json"
    top_manifest.write_bytes(
        canonical_bytes(
            {
                "schema_version": "tokenizer-promotion-acceptance-receipt-manifest-v1",
                "operator": operator,
                "reviewer": reviewer,
                "receipt_sha256": digest(receipt_directory / "acceptance-receipt.json"),
            }
        )
    )
    promotion_acceptance_receipt_manifest.verify_promotion_acceptance_receipt_manifest(
        receipt_directory,
        top_manifest,
    )

    root = tmp_path / "acceptance-evidence"
    root.mkdir()
    shutil.move(str(receipt_directory), str(root / "receipt-evidence"))
    shutil.move(str(top_manifest), str(root / "acceptance-receipt-manifest.json"))

    expected = module.promotion_acceptance_receipt_manifest_directory_sha256(root)
    result = module.verify_promotion_acceptance_receipt_manifest_directory_digest(root, expected)

    assert result.sha256 == expected
    assert result.directory.root == root
    assert result.directory.manifest.reviewer == reviewer
