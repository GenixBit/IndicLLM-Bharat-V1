from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bharat.data.licensing import (
    LicenseDecision,
    LicensePolicy,
    _validate_allow_record,
    load_license_policy,
)
from bharat.data.schema import (
    DataSourceSpec,
    SourceKind,
    SourceStatus,
    UsagePurpose,
)
from bharat.data.sources import load_source_spec
from bharat.data.version import Version


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
        return max(matching, key=lambda s: Version.parse(s.version))

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
        validated: list[DataSourceSpec] = []
        for s in results:
            lic = self._policy.resolve(s.license)
            if lic is None:
                continue
            if lic.decision != LicenseDecision.ALLOW:
                continue
            try:
                _validate_allow_record(lic, f"policy.{lic.identifier}")
            except ValueError:
                continue
            if lic.commercial_use_allowed is not True:
                continue
            if lic.model_training_allowed is not True:
                continue
            validated.append(s)
        if language is not None:
            lang_norm = language.strip().lower().replace("-", "_")
            validated = [s for s in validated if lang_norm in s.languages]
        return tuple(validated)

    def _policy_digest(self) -> str:
        pd = self._policy.to_dict()
        canonical = json.dumps(pd, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def digest(self) -> str:
        """SHA-256 over canonical JSON of all records + policy."""
        records = []
        for s in self._sources:
            d = s.to_dict()
            records.append(d)
        payload: dict[str, Any] = {
            "sources": records,
            "policy": self._policy.to_dict(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_snapshot(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "registry_digest": self.digest(),
            "policy_digest": self._policy_digest(),
            "sources": [s.to_dict() for s in self._sources],
            "policy": self._policy.to_dict(),
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
        seen_pairs: dict[str, set[str]] = {}  # source_id -> set of versions
        seen_active_uri_revisions: set[tuple[str, str]] = set()

        yaml_files = sorted(
            p
            for p in registry_dir.iterdir()
            if p.suffix in (".yaml", ".yml") and p.name != "license_policy.yaml"
        )

        for f in yaml_files:
            spec = load_source_spec(f)

            # Check duplicate source_id/version
            if spec.source_id in seen_pairs:
                if spec.version in seen_pairs[spec.source_id]:
                    raise ValueError(
                        f"Duplicate source/version pair: {spec.source_id} v{spec.version}"
                    )
                seen_pairs[spec.source_id].add(spec.version)
            else:
                seen_pairs[spec.source_id] = {spec.version}
            sources.append(spec)

            # Active-source URI/revision duplicate detection
            if spec.status in (SourceStatus.PROPOSED, SourceStatus.APPROVED):
                uri_rev = (spec.uri, spec.revision)
                if uri_rev in seen_active_uri_revisions:
                    raise ValueError(
                        f"Duplicate active source for URI '{spec.uri}' revision '{spec.revision}'"
                    )
                seen_active_uri_revisions.add(uri_rev)

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

            # Licence decision constraints for approved sources
            if spec.status == SourceStatus.APPROVED:
                lic = policy.resolve(spec.license)
                if lic is None:
                    raise ValueError(
                        f"Source '{spec.source_id}' v{spec.version}: "
                        f"licence identifier '{spec.license}' not found in policy"
                    )
                if lic.decision != LicenseDecision.ALLOW:
                    raise ValueError(
                        f"Source '{spec.source_id}' v{spec.version}: "
                        f"status is approved but licence '{spec.license}' has decision "
                        f"'{lic.decision.value}' (must be 'allow')"
                    )
                try:
                    _validate_allow_record(lic, f"policy.{lic.identifier}")
                except ValueError as e:
                    raise ValueError(
                        f"Source '{spec.source_id}' v{spec.version}: "
                        f"licence '{spec.license}' ALLOW record is incomplete: {e}"
                    )

            if spec.status == SourceStatus.REJECTED and not spec.notes:
                raise ValueError(
                    f"Rejected source '{spec.source_id}' v{spec.version}: "
                    "rejected status requires a reason in 'notes'"
                )

        # Sort deterministically by Version
        sources.sort(key=lambda s: (s.source_id, Version.parse(s.version)))

        # Supersession checks - use source_id@version format
        all_source_refs: set[str] = set()
        for s in sources:
            all_source_refs.add(f"{s.source_id}@{s.version}")

        for s in sources:
            if s.supersedes is not None:
                # Parse source_id@version
                parts = s.supersedes.split("@", 1)
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError(
                        f"Source '{s.source_id}' v{s.version}: "
                        f"supersedes must be in format 'source_id@version', "
                        f"got '{s.supersedes}'"
                    )
                target_sid, target_ver = parts[0], parts[1]
                target_ref = f"{target_sid}@{target_ver}"

                if target_ref not in all_source_refs:
                    raise ValueError(
                        f"Source '{s.source_id}' v{s.version}: "
                        f"supersedes target '{target_ref}' does not exist in registry"
                    )
                if target_sid == s.source_id and target_ver == s.version:
                    raise ValueError(
                        f"Source '{s.source_id}' v{s.version}: cannot supersede itself"
                    )

        # Check for supersession cycles (by source_id@version)
        for s in sources:
            visited: set[str] = set()
            current = f"{s.source_id}@{s.version}"
            while current is not None:
                if current in visited:
                    raise ValueError(f"Supersession cycle detected involving '{current}'")
                visited.add(current)
                parts = current.split("@", 1)
                if len(parts) != 2:
                    break
                cur_sid, cur_ver = parts[0], parts[1]
                src = next(
                    (x for x in sources if x.source_id == cur_sid and x.version == cur_ver),
                    None,
                )
                if src is None or src.supersedes is None:
                    break
                current = src.supersedes

        return cls(sources=tuple(sources), policy=policy, policy_path=str(policy_path))
