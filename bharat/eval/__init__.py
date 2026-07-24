from __future__ import annotations

from bharat.eval.adapters import (
    ChoiceBaselineAdapter,
    EchoPredictionAdapter,
    ExpectedPredictionAdapter,
    PredictionAdapter,
    build_prediction_adapter,
)
from bharat.eval.catalog import (
    BenchmarkCatalog,
    BenchmarkCategory,
    BenchmarkManifest,
    create_builtin_catalog,
    discover_benchmarks,
    validate_benchmark_registration,
    validate_manifest,
)
from bharat.eval.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    load_leaderboard,
    load_report,
)
from bharat.eval.local_inference import (
    BatchGenerator,
    LocalCausalLMAdapter,
    LocalInferenceConfig,
    load_local_causal_lm_adapter,
)
from bharat.eval.metrics import choice_accuracy, exact_match, normalized_exact_match, token_f1
from bharat.eval.prediction_runner import PredictionRunner, write_predictions_jsonl
from bharat.eval.reporting import BharatBenchReport
from bharat.eval.runner import BharatBenchRunner
from bharat.eval.schema import SUPPORTED_TASK_TYPES, EvalExample, EvalPrediction, EvalResult

__all__ = [
    "BatchGenerator",
    "BenchmarkCatalog",
    "BenchmarkCategory",
    "BenchmarkManifest",
    "BharatBenchReport",
    "BharatBenchRunner",
    "ChoiceBaselineAdapter",
    "EchoPredictionAdapter",
    "EvalExample",
    "EvalPrediction",
    "EvalResult",
    "ExpectedPredictionAdapter",
    "Leaderboard",
    "LeaderboardEntry",
    "LocalCausalLMAdapter",
    "LocalInferenceConfig",
    "PredictionAdapter",
    "PredictionRunner",
    "SUPPORTED_TASK_TYPES",
    "build_prediction_adapter",
    "choice_accuracy",
    "create_builtin_catalog",
    "discover_benchmarks",
    "exact_match",
    "load_leaderboard",
    "load_local_causal_lm_adapter",
    "load_report",
    "normalized_exact_match",
    "token_f1",
    "validate_benchmark_registration",
    "validate_manifest",
    "write_predictions_jsonl",
]
