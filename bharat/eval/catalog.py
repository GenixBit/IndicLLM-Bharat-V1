from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bharat.eval.schema import SUPPORTED_TASK_TYPES

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_slug(value: str, label: str) -> list[str]:
    errors: list[str] = []
    if not value:
        errors.append(f"{label} must not be empty")
    elif not _SLUG_RE.match(value):
        errors.append(f"{label} {value!r} must match pattern {_SLUG_RE.pattern!r}")
    return errors


@dataclass(frozen=True)
class BenchmarkCategory:
    id: str
    name: str
    description: str
    supported_task_types: frozenset[str] = field(
        default_factory=lambda: frozenset(SUPPORTED_TASK_TYPES)
    )
    safety_boundary: str | None = None

    def __post_init__(self) -> None:
        issues: list[str] = []
        issues.extend(_validate_slug(self.id, "Category id"))
        if not self.name:
            issues.append("Category name must not be empty")
        if not self.description:
            issues.append("Category description must not be empty")
        for tt in self.supported_task_types:
            if tt not in SUPPORTED_TASK_TYPES:
                issues.append(f"Unsupported task type {tt!r} in category {self.id!r}")
        if issues:
            raise ValueError("; ".join(issues))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "supported_task_types": sorted(self.supported_task_types),
        }
        if self.safety_boundary is not None:
            d["safety_boundary"] = self.safety_boundary
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkCategory:
        task_types_raw = data.get("supported_task_types")
        if task_types_raw is None:
            supported_task_types = frozenset(SUPPORTED_TASK_TYPES)
        else:
            supported_task_types = frozenset(task_types_raw)
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            supported_task_types=supported_task_types,
            safety_boundary=data.get("safety_boundary"),
        )


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_id: str
    category: str
    name: str
    description: str
    task_type: str
    num_examples: int
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "category": self.category,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "num_examples": self.num_examples,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkManifest:
        return cls(
            benchmark_id=data["benchmark_id"],
            category=data["category"],
            name=data["name"],
            description=data["description"],
            task_type=data["task_type"],
            num_examples=int(data["num_examples"]),
            version=data["version"],
        )


def validate_manifest(manifest: BenchmarkManifest) -> list[str]:
    issues: list[str] = []
    issues.extend(_validate_slug(manifest.benchmark_id, "Benchmark id"))
    issues.extend(_validate_slug(manifest.category, "Category"))
    if not manifest.name:
        issues.append("Benchmark name must not be empty")
    if not manifest.description:
        issues.append("Benchmark description must not be empty")
    if manifest.task_type not in SUPPORTED_TASK_TYPES:
        issues.append(f"Unsupported task type {manifest.task_type!r}")
    if manifest.num_examples < 0:
        issues.append(f"num_examples must be >= 0, got {manifest.num_examples}")
    if not manifest.version:
        issues.append("Version must not be empty")
    return issues


def validate_benchmark_registration(
    catalog: BenchmarkCatalog,
    manifest: BenchmarkManifest,
    examples_path: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    issues.extend(validate_manifest(manifest))

    if manifest.category not in catalog._categories:
        issues.append(f"Category {manifest.category!r} is not registered in the catalog")
    else:
        cat = catalog._categories[manifest.category]
        if manifest.task_type not in cat.supported_task_types:
            issues.append(
                f"Category {manifest.category!r} does not support task type "
                f"{manifest.task_type!r}; supported: {sorted(cat.supported_task_types)}"
            )

    if manifest.benchmark_id in catalog._benchmarks:
        issues.append(f"Benchmark {manifest.benchmark_id!r} is already registered")

    if examples_path is not None:
        if not examples_path.exists():
            issues.append(f"Examples file not found: {examples_path}")
        else:
            count = 0
            for line in examples_path.read_text().splitlines():
                if line.strip():
                    count += 1
            if count != manifest.num_examples:
                issues.append(
                    f"Expected {manifest.num_examples} examples, "
                    f"found {count} in {examples_path}"
                )

    return issues


class BenchmarkCatalog:
    def __init__(self) -> None:
        self._categories: dict[str, BenchmarkCategory] = {}
        self._benchmarks: dict[str, BenchmarkManifest] = {}

    def register_category(self, category: BenchmarkCategory) -> None:
        if category.id in self._categories:
            raise ValueError(f"Category {category.id!r} is already registered")
        self._categories[category.id] = category

    def register_benchmark(self, manifest: BenchmarkManifest) -> None:
        issues = validate_benchmark_registration(self, manifest)
        if issues:
            raise ValueError("; ".join(issues))
        self._benchmarks[manifest.benchmark_id] = manifest

    def get_category(self, category_id: str) -> BenchmarkCategory:
        if category_id not in self._categories:
            raise KeyError(f"Category {category_id!r} not found")
        return self._categories[category_id]

    def get_benchmark(self, benchmark_id: str) -> BenchmarkManifest:
        if benchmark_id not in self._benchmarks:
            raise KeyError(f"Benchmark {benchmark_id!r} not found")
        return self._benchmarks[benchmark_id]

    def list_categories(self) -> Sequence[BenchmarkCategory]:
        return list(self._categories.values())

    def list_benchmarks(self, category_id: str | None = None) -> Sequence[BenchmarkManifest]:
        if category_id is None:
            return list(self._benchmarks.values())
        return [m for m in self._benchmarks.values() if m.category == category_id]

    @property
    def category_count(self) -> int:
        return len(self._categories)

    @property
    def benchmark_count(self) -> int:
        return len(self._benchmarks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": [
                c.to_dict() for c in sorted(self._categories.values(), key=lambda x: x.id)
            ],
            "benchmarks": [
                m.to_dict() for m in sorted(self._benchmarks.values(), key=lambda x: x.benchmark_id)
            ],
        }


_BUILTIN_CATEGORIES = [
    BenchmarkCategory(
        id="language",
        name="Language Understanding",
        description=(
            "Tests language comprehension, grammar, translation, and "
            "linguistic knowledge across Indian languages."
        ),
        supported_task_types=frozenset({"qa", "classification", "generation"}),
        safety_boundary=(
            "Must respect grammatical correctness. Must not produce "
            "gibberish or incomprehensible output."
        ),
    ),
    BenchmarkCategory(
        id="reasoning",
        name="Logical & Mathematical Reasoning",
        description=(
            "Tests step-by-step logical deduction, mathematical problem "
            "solving, and commonsense reasoning."
        ),
        supported_task_types=frozenset({"qa", "classification", "generation"}),
        safety_boundary=(
            "Must provide logically consistent reasoning. Must not "
            "contain contradictions or circular arguments."
        ),
    ),
    BenchmarkCategory(
        id="coding",
        name="Code Generation & Understanding",
        description=(
            "Tests ability to generate correct, idiomatic code from "
            "natural language descriptions and to understand existing code."
        ),
        supported_task_types=frozenset({"generation"}),
        safety_boundary=(
            "Must not generate malicious code, security vulnerabilities, "
            "or exploitative software."
        ),
    ),
    BenchmarkCategory(
        id="knowledge",
        name="Factual Knowledge",
        description=(
            "Tests factual accuracy across domains including science, "
            "history, geography, and culture."
        ),
        supported_task_types=frozenset({"qa", "classification"}),
        safety_boundary=(
            "Must not hallucinate facts. Must acknowledge uncertainty " "when appropriate."
        ),
    ),
    BenchmarkCategory(
        id="safety",
        name="Safety & Alignment",
        description=(
            "Tests refusal of harmful requests, avoidance of biased or "
            "toxic output, and alignment with ethical guidelines."
        ),
        supported_task_types=frozenset({"classification", "generation"}),
        safety_boundary=(
            "Must refuse to execute harmful, illegal, or unethical "
            "requests. Must not generate offensive, discriminatory, or "
            "dangerous content."
        ),
    ),
]


def create_builtin_catalog() -> BenchmarkCatalog:
    catalog = BenchmarkCatalog()
    for cat in _BUILTIN_CATEGORIES:
        catalog.register_category(cat)
    return catalog


def discover_benchmarks(catalog: BenchmarkCatalog, root: Path) -> list[BenchmarkManifest]:
    manifests: list[BenchmarkManifest] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            manifest_path = child / "manifest.json"
            examples_path = child / "examples.jsonl"
            if manifest_path.exists() and examples_path.exists():
                data = json.loads(manifest_path.read_text())
                manifest = BenchmarkManifest.from_dict(data)
                catalog.register_benchmark(manifest)
                manifests.append(manifest)
    return manifests
