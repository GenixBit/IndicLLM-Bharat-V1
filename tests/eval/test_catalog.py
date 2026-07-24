from __future__ import annotations

import json
from pathlib import Path

import pytest

from bharat.eval.catalog import (
    _BUILTIN_CATEGORIES,
    BenchmarkCatalog,
    BenchmarkCategory,
    BenchmarkManifest,
    create_builtin_catalog,
    discover_benchmarks,
    validate_benchmark_registration,
    validate_manifest,
)
from bharat.eval.schema import SUPPORTED_TASK_TYPES, EvalExample


class TestBenchmarkCategory:
    def test_valid(self) -> None:
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa"}),
            safety_boundary="Must not produce gibberish.",
        )
        assert cat.id == "language"
        assert cat.safety_boundary == "Must not produce gibberish."

    def test_default_supported_types(self) -> None:
        cat = BenchmarkCategory(id="custom", name="Custom", description="A custom category.")
        assert cat.supported_task_types == SUPPORTED_TASK_TYPES

    def test_default_safety_boundary_none(self) -> None:
        cat = BenchmarkCategory(id="custom", name="Custom", description="A custom category.")
        assert cat.safety_boundary is None

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Category id must not be empty"):
            BenchmarkCategory(
                id="",
                name="Custom",
                description="A custom category.",
            )

    def test_invalid_id_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="must match pattern"):
            BenchmarkCategory(
                id="my-category",
                name="Custom",
                description="A custom category.",
            )

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Category name must not be empty"):
            BenchmarkCategory(
                id="custom",
                name="",
                description="A custom category.",
            )

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="Category description must not be empty"):
            BenchmarkCategory(
                id="custom",
                name="Custom",
                description="",
            )

    def test_unsupported_task_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported task type"):
            BenchmarkCategory(
                id="custom",
                name="Custom",
                description="A custom category.",
                supported_task_types=frozenset({"invalid_type"}),
            )

    def test_to_dict_roundtrip(self) -> None:
        cat1 = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa", "classification"}),
            safety_boundary="Must not produce gibberish.",
        )
        d = cat1.to_dict()
        cat2 = BenchmarkCategory.from_dict(d)
        assert cat1 == cat2

    def test_to_dict_without_safety_boundary(self) -> None:
        cat = BenchmarkCategory(id="custom", name="Custom", description="A custom category.")
        d = cat.to_dict()
        assert "safety_boundary" not in d


class TestBenchmarkManifest:
    def test_valid(self) -> None:
        m = BenchmarkManifest(
            benchmark_id="language_qa",
            category="language",
            name="Language QA",
            description="QA tests for language.",
            task_type="qa",
            num_examples=3,
            version="0.1.0",
        )
        assert m.benchmark_id == "language_qa"
        assert m.task_type == "qa"

    def test_validate_manifest_valid(self) -> None:
        m = BenchmarkManifest(
            benchmark_id="language_qa",
            category="language",
            name="Language QA",
            description="QA tests for language.",
            task_type="qa",
            num_examples=3,
            version="0.1.0",
        )
        assert validate_manifest(m) == []

    def test_validate_manifest_invalid_task_type(self) -> None:
        m = BenchmarkManifest(
            benchmark_id="test",
            category="test",
            name="Test",
            description="Test.",
            task_type="invalid",
            num_examples=1,
            version="0.1.0",
        )
        issues = validate_manifest(m)
        assert any("Unsupported task type" in i for i in issues)

    def test_validate_manifest_empty_name(self) -> None:
        m = BenchmarkManifest(
            benchmark_id="test",
            category="test",
            name="",
            description="Test.",
            task_type="qa",
            num_examples=1,
            version="0.1.0",
        )
        issues = validate_manifest(m)
        assert any("Benchmark name must not be empty" in i for i in issues)

    def test_validate_manifest_negative_examples(self) -> None:
        m = BenchmarkManifest(
            benchmark_id="test",
            category="test",
            name="Test",
            description="Test.",
            task_type="qa",
            num_examples=-1,
            version="0.1.0",
        )
        issues = validate_manifest(m)
        assert any("num_examples must be >= 0" in i for i in issues)

    def test_to_dict_roundtrip(self) -> None:
        m1 = BenchmarkManifest(
            benchmark_id="language_qa",
            category="language",
            name="Language QA",
            description="QA tests for language.",
            task_type="qa",
            num_examples=3,
            version="0.1.0",
        )
        d = m1.to_dict()
        m2 = BenchmarkManifest.from_dict(d)
        assert m1 == m2

    def test_from_dict(self) -> None:
        d = {
            "benchmark_id": "test_benchmark",
            "category": "language",
            "name": "Test Benchmark",
            "description": "A test.",
            "task_type": "qa",
            "num_examples": 5,
            "version": "1.0.0",
        }
        m = BenchmarkManifest.from_dict(d)
        assert m.benchmark_id == "test_benchmark"
        assert m.num_examples == 5


class TestBenchmarkCatalog:
    def test_empty_catalog(self) -> None:
        catalog = BenchmarkCatalog()
        assert catalog.category_count == 0
        assert catalog.benchmark_count == 0

    def test_register_category(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
        )
        catalog.register_category(cat)
        assert catalog.category_count == 1
        assert catalog.get_category("language") == cat

    def test_register_duplicate_category_raises(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
        )
        catalog.register_category(cat)
        with pytest.raises(ValueError, match="already registered"):
            catalog.register_category(cat)

    def test_get_missing_category_raises(self) -> None:
        catalog = BenchmarkCatalog()
        with pytest.raises(KeyError, match="not found"):
            catalog.get_category("nonexistent")

    def test_register_benchmark(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="language_qa",
            category="language",
            name="Language QA",
            description="QA tests.",
            task_type="qa",
            num_examples=3,
            version="0.1.0",
        )
        catalog.register_benchmark(manifest)
        assert catalog.benchmark_count == 1
        assert catalog.get_benchmark("language_qa") == manifest

    def test_register_benchmark_unregistered_category_raises(self) -> None:
        catalog = BenchmarkCatalog()
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="nonexistent",
            name="Test",
            description="Test.",
            task_type="qa",
            num_examples=1,
            version="0.1.0",
        )
        with pytest.raises(ValueError, match="not registered"):
            catalog.register_benchmark(manifest)

    def test_register_benchmark_unsupported_task_type_raises(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="knowledge",
            name="Knowledge",
            description="Tests factual knowledge.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="knowledge",
            name="Test",
            description="Test.",
            task_type="generation",
            num_examples=1,
            version="0.1.0",
        )
        with pytest.raises(ValueError, match="does not support"):
            catalog.register_benchmark(manifest)

    def test_register_duplicate_benchmark_raises(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="language_qa",
            category="language",
            name="Language QA",
            description="QA tests.",
            task_type="qa",
            num_examples=3,
            version="0.1.0",
        )
        catalog.register_benchmark(manifest)
        with pytest.raises(ValueError, match="already registered"):
            catalog.register_benchmark(manifest)

    def test_get_missing_benchmark_raises(self) -> None:
        catalog = BenchmarkCatalog()
        with pytest.raises(KeyError, match="not found"):
            catalog.get_benchmark("nonexistent")

    def test_list_categories(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
        )
        catalog.register_category(cat)
        cats = catalog.list_categories()
        assert len(cats) == 1
        assert cats[0] == cat

    def test_list_benchmarks(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa", "generation"}),
        )
        catalog.register_category(cat)
        m1 = BenchmarkManifest(
            benchmark_id="b1",
            category="language",
            name="B1",
            description="B1.",
            task_type="qa",
            num_examples=1,
            version="0.1.0",
        )
        m2 = BenchmarkManifest(
            benchmark_id="b2",
            category="language",
            name="B2",
            description="B2.",
            task_type="generation",
            num_examples=1,
            version="0.1.0",
        )
        catalog.register_benchmark(m1)
        catalog.register_benchmark(m2)
        all_b = catalog.list_benchmarks()
        assert len(all_b) == 2
        lang_b = catalog.list_benchmarks("language")
        assert len(lang_b) == 2
        other_b = catalog.list_benchmarks("nonexistent")
        assert len(other_b) == 0

    def test_to_dict(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
        )
        catalog.register_category(cat)
        d = catalog.to_dict()
        assert "categories" in d
        assert "benchmarks" in d
        assert len(d["categories"]) == 1
        assert d["categories"][0]["id"] == "language"


class TestBuiltinCatalog:
    def test_all_categories_present(self) -> None:
        catalog = create_builtin_catalog()
        assert catalog.category_count == 5

    def test_has_language_category(self) -> None:
        catalog = create_builtin_catalog()
        cat = catalog.get_category("language")
        assert cat.name == "Language Understanding"

    def test_has_reasoning_category(self) -> None:
        catalog = create_builtin_catalog()
        cat = catalog.get_category("reasoning")
        assert cat.name == "Logical & Mathematical Reasoning"

    def test_has_coding_category(self) -> None:
        catalog = create_builtin_catalog()
        cat = catalog.get_category("coding")
        assert cat.name == "Code Generation & Understanding"

    def test_has_knowledge_category(self) -> None:
        catalog = create_builtin_catalog()
        cat = catalog.get_category("knowledge")
        assert cat.name == "Factual Knowledge"

    def test_has_safety_category(self) -> None:
        catalog = create_builtin_catalog()
        cat = catalog.get_category("safety")
        assert cat.name == "Safety & Alignment"

    def test_all_categories_have_safety_boundary(self) -> None:
        catalog = create_builtin_catalog()
        for cat in catalog.list_categories():
            assert cat.safety_boundary is not None, f"{cat.id} missing safety_boundary"

    def test_all_categories_have_supported_task_types(self) -> None:
        catalog = create_builtin_catalog()
        for cat in catalog.list_categories():
            assert len(cat.supported_task_types) >= 1

    def test_builtin_constants_matches_create(self) -> None:
        catalog = create_builtin_catalog()
        assert len(_BUILTIN_CATEGORIES) == 5
        for bcat in _BUILTIN_CATEGORIES:
            assert bcat.id in [c.id for c in catalog.list_categories()]


class TestValidateRegistration:
    def test_validate_without_examples(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="language",
            name="Test",
            description="Test.",
            task_type="qa",
            num_examples=3,
            version="0.1.0",
        )
        issues = validate_benchmark_registration(catalog, manifest)
        assert issues == []

    def test_validate_unregistered_category(self) -> None:
        catalog = BenchmarkCatalog()
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="missing",
            name="Test",
            description="Test.",
            task_type="qa",
            num_examples=1,
            version="0.1.0",
        )
        issues = validate_benchmark_registration(catalog, manifest)
        assert any("not registered" in i for i in issues)

    def test_validate_unsupported_task_type(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="knowledge",
            name="Knowledge",
            description="Tests factual knowledge.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="knowledge",
            name="Test",
            description="Test.",
            task_type="generation",
            num_examples=1,
            version="0.1.0",
        )
        issues = validate_benchmark_registration(catalog, manifest)
        assert any("does not support" in i for i in issues)

    def test_validate_duplicate_benchmark(self) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="language",
            name="Test",
            description="Test.",
            task_type="qa",
            num_examples=1,
            version="0.1.0",
        )
        catalog._benchmarks["test"] = manifest
        issues = validate_benchmark_registration(catalog, manifest)
        assert any("already registered" in i for i in issues)

    def test_validate_with_examples_path(self, tmp_path: Path) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="language",
            name="Test",
            description="Test.",
            task_type="qa",
            num_examples=2,
            version="0.1.0",
        )
        examples = tmp_path / "examples.jsonl"
        examples.write_text(
            '{"example_id": "a", "task_type": "qa", "prompt": "Q1", "expected": "A1"}\n'
            '{"example_id": "b", "task_type": "qa", "prompt": "Q2", "expected": "A2"}\n'
        )
        issues = validate_benchmark_registration(catalog, manifest, examples)
        assert issues == []

    def test_validate_with_examples_wrong_count(self, tmp_path: Path) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="language",
            name="Test",
            description="Test.",
            task_type="qa",
            num_examples=5,
            version="0.1.0",
        )
        examples = tmp_path / "examples.jsonl"
        examples.write_text(
            '{"example_id": "a", "task_type": "qa", "prompt": "Q1", "expected": "A1"}\n'
        )
        issues = validate_benchmark_registration(catalog, manifest, examples)
        assert any("Expected 5 examples, found 1" in i for i in issues)

    def test_validate_with_missing_examples_path(self, tmp_path: Path) -> None:
        catalog = BenchmarkCatalog()
        cat = BenchmarkCategory(
            id="language",
            name="Language Understanding",
            description="Tests language comprehension.",
            supported_task_types=frozenset({"qa"}),
        )
        catalog.register_category(cat)
        manifest = BenchmarkManifest(
            benchmark_id="test",
            category="language",
            name="Test",
            description="Test.",
            task_type="qa",
            num_examples=1,
            version="0.1.0",
        )
        issues = validate_benchmark_registration(catalog, manifest, tmp_path / "nonexistent.jsonl")
        assert any("Examples file not found" in i for i in issues)


class TestDiscoverBenchmarks:
    def test_discover_all_fixtures(self) -> None:
        fixtures_root = Path("eval_fixtures/benchmarks")
        if not fixtures_root.exists():
            pytest.skip("Fixtures not found at eval_fixtures/benchmarks")
        catalog = create_builtin_catalog()
        discovered = discover_benchmarks(catalog, fixtures_root)
        assert len(discovered) == 5
        benchmark_ids = {m.benchmark_id for m in discovered}
        assert benchmark_ids == {
            "language_qa",
            "reasoning_cls",
            "coding_gen",
            "knowledge_qa",
            "safety_cls",
        }

    def test_discovered_benchmarks_have_valid_examples(self) -> None:
        fixtures_root = Path("eval_fixtures/benchmarks")
        if not fixtures_root.exists():
            pytest.skip("Fixtures not found at eval_fixtures/benchmarks")
        catalog = create_builtin_catalog()
        discovered = discover_benchmarks(catalog, fixtures_root)
        for m in discovered:
            child = fixtures_root / m.benchmark_id
            examples_path = child / "examples.jsonl"
            assert examples_path.exists()
            for line in examples_path.read_text().splitlines():
                if line.strip():
                    data = json.loads(line)
                    ex = EvalExample.from_dict(data)
                    assert ex.example_id
                    assert ex.task_type == m.task_type

    def test_discover_maintains_correct_counts(self) -> None:
        fixtures_root = Path("eval_fixtures/benchmarks")
        if not fixtures_root.exists():
            pytest.skip("Fixtures not found at eval_fixtures/benchmarks")
        catalog = create_builtin_catalog()
        discover_benchmarks(catalog, fixtures_root)
        assert catalog.category_count == 5
        assert catalog.benchmark_count == 5

    def test_discover_with_no_fixtures(self, tmp_path: Path) -> None:
        catalog = create_builtin_catalog()
        discovered = discover_benchmarks(catalog, tmp_path)
        assert discovered == []
        assert catalog.benchmark_count == 0


class TestBenchmarkCategoryFromDict:
    def test_minimal_from_dict(self) -> None:
        d = {
            "id": "test",
            "name": "Test",
            "description": "A test category.",
        }
        cat = BenchmarkCategory.from_dict(d)
        assert cat.id == "test"
        assert cat.supported_task_types == SUPPORTED_TASK_TYPES
        assert cat.safety_boundary is None

    def test_full_from_dict(self) -> None:
        d = {
            "id": "safety",
            "name": "Safety & Alignment",
            "description": "Tests safety.",
            "supported_task_types": ["classification", "generation"],
            "safety_boundary": "Must refuse harmful requests.",
        }
        cat = BenchmarkCategory.from_dict(d)
        assert cat.safety_boundary == "Must refuse harmful requests."
        assert cat.supported_task_types == frozenset({"classification", "generation"})
