from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bharat.data.contamination import ContaminationChecker
from bharat.data.local_reader import read_local_text
from bharat.data.manifest import (
    DatasetManifest,
    create_manifest,
    digest_processing_config,
)
from bharat.data.processing import DataProcessor, ProcessingConfig
from bharat.data.records import ProcessedRecord
from bharat.data.shard_writer import ShardWriter, ShardWriterConfig

_CONTAMINATION_REASONS = {
    "exact": "contamination:exact",
    "normalized": "contamination:normalized",
}

_CONTAMINATION_NGRAM_PREFIX = "contamination:ngram_"


@dataclass(frozen=True)
class PreparationConfig:
    source_id: str
    source_version: str
    license: str
    language: str
    split: str
    domain: str = ""
    output_dir: str = "output"
    max_records_per_shard: int = 10000
    max_bytes_per_shard: int = 64 * 1024 * 1024
    created_at: str | None = None
    dry_run: bool = False
    blocklist_path: str | None = None
    processing_config: ProcessingConfig | None = None


@dataclass(frozen=True)
class PreparationReport:
    total_records: int
    accepted_records: int
    rejected_records: int
    shard_count: int
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    language_distribution: dict[str, int] = field(default_factory=dict)
    manifest_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "shard_count": self.shard_count,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "language_distribution": dict(sorted(self.language_distribution.items())),
            "manifest_digest": self.manifest_digest,
        }


class LocalPreparer:
    def __init__(self, config: PreparationConfig) -> None:
        self._config = config
        self._processor = DataProcessor(config.processing_config)
        self._contamination: ContaminationChecker | None = None
        if config.blocklist_path:
            self._contamination = ContaminationChecker()
            self._contamination.load_blocklist(config.blocklist_path)

    def prepare(self, input_path: str | Path) -> tuple[DatasetManifest, PreparationReport]:
        raw_records = read_local_text(input_path)
        texts = [r.text for r in raw_records]
        decisions = self._processor.process_batch(texts)

        processed: list[ProcessedRecord] = []
        rejection_reasons: dict[str, int] = {}
        lang_dist: dict[str, int] = {}

        for raw, dec in zip(raw_records, decisions, strict=True):
            lang_dist[dec.language] = lang_dist.get(dec.language, 0) + 1
            for reason in dec.reasons:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

            processed.append(
                ProcessedRecord(
                    record_id=raw.record_id,
                    text=dec.normalized_text or raw.text,
                    language=dec.language,
                    quality_score=dec.quality_score,
                    source_path=raw.source_path,
                    line_number=raw.line_number,
                    processing_reasons=dec.reasons,
                    accepted=dec.accepted,
                )
            )

        if self._contamination is not None:
            updated: list[ProcessedRecord] = []
            for r in processed:
                text = r.text
                result = self._contamination.check_all(text)
                if result.is_contaminated:
                    if result.method in _CONTAMINATION_REASONS:
                        reason = _CONTAMINATION_REASONS[result.method]
                    elif result.method.startswith("ngram_"):
                        n = result.method.removeprefix("ngram_")
                        reason = f"{_CONTAMINATION_NGRAM_PREFIX}{n}"
                    else:
                        reason = f"contamination:{result.method}"
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    updated.append(
                        ProcessedRecord(
                            record_id=r.record_id,
                            text=r.text,
                            language=r.language,
                            quality_score=r.quality_score,
                            source_path=r.source_path,
                            line_number=r.line_number,
                            processing_reasons=(*r.processing_reasons, reason),
                            accepted=False,
                        )
                    )
                else:
                    updated.append(r)
            processed = updated

        total = len(processed)
        accepted_records = [r for r in processed if r.accepted]
        rejected_count = total - len(accepted_records)

        shard_count = 0
        manifest: DatasetManifest | None = None

        if self._config.dry_run:
            placeholder_sha = hashlib.sha256(b"dry_run").hexdigest()
            manifest = create_manifest(
                dataset_id=self._config.source_id,
                source_id=self._config.source_id,
                source_version=self._config.source_version,
                license=self._config.license,
                language=self._config.language,
                split=self._config.split,
                records=len(accepted_records),
                bytes_utf8=sum(len(r.text.encode("utf-8")) for r in accepted_records),
                sha256=placeholder_sha,
                processing_config_digest=digest_processing_config(
                    self._config.processing_config or ProcessingConfig()
                ),
                registry_digest=hashlib.sha256(b"local_no_registry").hexdigest(),
                policy_digest=hashlib.sha256(b"local_no_policy").hexdigest(),
                created_at=self._config.created_at,
                domain=self._config.domain,
            )
            shard_count = 0
        else:
            out_dir = Path(self._config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            writer_config = ShardWriterConfig(
                output_dir=str(out_dir),
                source_id=self._config.source_id,
                split=self._config.split,
                max_records_per_shard=self._config.max_records_per_shard,
                max_bytes_per_shard=self._config.max_bytes_per_shard,
            )
            writer = ShardWriter(writer_config)
            writer.write_shard(accepted_records)

            shards = writer.manifests
            shard_count = len(shards)

            total_bytes = sum(s.bytes_utf8 for s in shards)
            combined_sha = hashlib.sha256()
            for s in shards:
                combined_sha.update(f"{s.shard_id}:{s.sha256}".encode())
            manifest_sha = combined_sha.hexdigest()

            manifest = create_manifest(
                dataset_id=self._config.source_id,
                source_id=self._config.source_id,
                source_version=self._config.source_version,
                license=self._config.license,
                language=self._config.language,
                split=self._config.split,
                records=len(accepted_records),
                bytes_utf8=total_bytes,
                sha256=manifest_sha,
                processing_config_digest=digest_processing_config(
                    self._config.processing_config or ProcessingConfig()
                ),
                registry_digest=hashlib.sha256(b"local_no_registry").hexdigest(),
                policy_digest=hashlib.sha256(b"local_no_policy").hexdigest(),
                shards=shards,
                created_at=self._config.created_at,
                domain=self._config.domain,
            )

            manifest_path = out_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        report = PreparationReport(
            total_records=total,
            accepted_records=len(accepted_records),
            rejected_records=rejected_count,
            shard_count=shard_count,
            rejection_reasons=rejection_reasons,
            language_distribution=lang_dist,
            manifest_digest=manifest.digest(),
        )

        if not self._config.dry_run:
            report_path = out_dir / "preparation_report.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

        return manifest, report
