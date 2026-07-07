from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bharat.data.licensing import LicenseDecision, LicensePolicy, load_license_policy
from bharat.data.schema import (
    DataSourceSpec,
    SourceKind,
    SourceStatus,
    UsagePurpose,
)
from bharat.data.sources import load_source_spec


class DataRegistry:
    def __init__(
        self,
        sources: tuple[DataSourceSpec, ...],
        policy: LicensePolicy,
        policy_path: str,
    ) -> None:
        self._sources = sources
        self._policy = policy
        self._policy_path = policy_path

    @property
    def policy(self) -> LicensePolicy:
        return self._policy

    def get(self, source_id: str, version: str | None = None) -> DataSourceSpec | None:
        matching = [s for s in self._sources if s.source_id == source_id]
        if not matching:
            return None
        if version is not None:
            matching = [s for s in matching if s.version == version]
            if not matching:
                return None
            return matching[0]
        return max(matching, key=lambda s: s.version)

    def list_all(self) -> tuple[DataSourceSpec, ...]:
        return self._sources

    def filter(
        self,
        *,
        status: SourceStatus | None = None,
        purpose: UsagePurpose | None = None,
        language: str | None = None,
        domain: str | None = None,
    ) -> tuple[DataSourceSpec, ...]:
        results = self._sources
        if status is not None:
            results = tuple(s for s in results if s.status == status)
        if purpose is not None:
            results = tuple(s for s in results if purpose in s.purposes)
        if language is not None:
            lang_norm = language.strip().lower().replace("-", "_")
            results = tuple(s for s in results if lang_norm in s.languages)
        if domain is not None:
            results = tuple(s for s in results if domain.lower() in s.domains)
        return results

    def approved_for(
        self,
        purpose: UsagePurpose,
        *,
        language: str | None = None,
    ) -> tuple[DataSourceSpec, ...]:
        results = self.filter(status=SourceStatus.APPROVED, purpose=purpose)
        if language is not None:
            lang_norm = language.strip().lower().replace("-", "_")
            results = tuple(s for s in results if lang_norm in s.languages)
        return results

    def digest(self) -> str:
        """SHA-256 over canonical JSON of all records in deterministic order."""
        records = []
        for s in self._sources:
            d = s.to_dict()
            d.pop("notes", None)
            records.append(d)
        canonical = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_snapshot(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "sources": [s.to_dict() for s in self._sources],
            "digest": self.digest(),
        }

    @classmethod
    def load(
        cls,
        registry_dir: str | Path,
        policy_path: str | Path | None = None,
    ) -> DataRegistry:
        registry_dir = Path(registry_dir)
        dir_path = str(registry_dir)

        if not registry_dir.is_dir():
            raise NotADirectoryError(f"Registry directory not found: {dir_path}")

        if policy_path is None:
            policy_path = registry_dir.parent / "license_policy.yaml"

        policy_path = Path(policy_path)
        if not policy_path.exists():
            alt = registry_dir / "license_policy.yaml"
            if alt.exists():
                policy_path = alt

        policy = load_license_policy(policy_path)

        sources: list[DataSourceSpec] = []
        seen_pairs: set[tuple[str, str]] = set()
        seen_uri_revisions: set[tuple[str, str]] = set()
        source_ids: set[str] = set()

        # Collect all source IDs first for supersession checking
        yaml_files = sorted(
            p
            for p in registry_dir.iterdir()
            if p.suffix in (".yaml", ".yml") and p.name != "license_policy.yaml"
        )

        for f in yaml_files:
            spec = load_source_spec(f)
            pair = (spec.source_id, spec.version)
            if pair in seen_pairs:
                for s in sources:
                    if (s.source_id, s.version) == pair:
                        break
                raise ValueError(f"Duplicate source/version pair: {spec.source_id} v{spec.version}")
            seen_pairs.add(pair)
            sources.append(spec)
            source_ids.add(spec.source_id)

            uri_rev = (spec.uri, spec.revision)
            if spec.status == SourceStatus.APPROVED:
                if uri_rev in seen_uri_revisions:
                    raise ValueError(
                        f"Duplicate active source for URI '{spec.uri}' revision '{spec.revision}'"
                    )
                seen_uri_revisions.add(uri_rev)

            # Integrity checks for approved sources
            if spec.status == SourceStatus.APPROVED:
                if spec.integrity is None:
                    raise ValueError(
                        f"Approved source '{spec.source_id}' v{spec.version} "
                        f"at {f.name}: integrity record required"
                    )
                if spec.kind == SourceKind.HUGGINGFACE:
                    if not spec.integrity.sha256:
                        raise ValueError(
                            f"Approved Hugging Face source '{spec.source_id}' "
                            f"v{spec.version}: SHA-256 required"
                        )
                elif spec.kind == SourceKind.HTTP:
                    if not spec.integrity.sha256:
                        raise ValueError(
                            f"Approved HTTP source '{spec.source_id}' "
                            f"v{spec.version}: SHA-256 checksum required"
                        )
                elif (
                    spec.kind
                    in (
                        SourceKind.S3,
                        SourceKind.GCS,
                        SourceKind.AZURE_BLOB,
                        SourceKind.LOCAL,
                    )
                    and not spec.integrity.sha256
                    and not spec.integrity.manifest_sha256
                ):
                    raise ValueError(
                        f"Approved {spec.kind.value} source '{spec.source_id}' "
                        f"v{spec.version}: SHA-256 or manifest SHA-256 required"
                    )

            # Licence decision constraints
            decision = policy.decision_for(spec.license)
            if spec.status == SourceStatus.APPROVED and decision != LicenseDecision.ALLOW:
                raise ValueError(
                    f"Source '{spec.source_id}' v{spec.version}: "
                    f"status is approved but licence '{spec.license}' has decision "
                    f"'{decision.value}' (must be 'allow')"
                )
            if spec.status in (SourceStatus.REJECTED, SourceStatus.DEPRECATED):
                if decision == LicenseDecision.ALLOW:
                    pass
                if spec.status == SourceStatus.REJECTED and not spec.notes:
                    raise ValueError(
                        f"Rejected source '{spec.source_id}' v{spec.version}: "
                        "rejected status requires a reason in 'notes'"
                    )

        # Sort deterministically
        sources.sort(key=lambda s: (s.source_id, s.version))

        # Supersession checks
        for s in sources:
            if s.supersedes is not None:
                if s.supersedes == s.source_id:
                    raise ValueError(
                        f"Source '{s.source_id}' v{s.version}: cannot supersede itself"
                    )
                superseded = [x for x in sources if x.source_id == s.supersedes]
                if not superseded:
                    raise ValueError(
                        f"Source '{s.source_id}' v{s.version}: "
                        f"supersedes unknown source '{s.supersedes}'"
                    )

        # Check for supersession cycles
        for s in sources:
            visited: set[str] = set()
            current = s.source_id
            while current is not None:
                if current in visited:
                    raise ValueError(f"Supersession cycle detected involving '{current}'")
                visited.add(current)
                src = next((x for x in sources if x.source_id == current), None)
                if src is None or src.supersedes is None:
                    break
                current = src.supersedes

        return cls(sources=tuple(sources), policy=policy, policy_path=str(policy_path))
