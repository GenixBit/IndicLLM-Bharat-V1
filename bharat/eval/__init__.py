from __future__ import annotations

from bharat.eval.adapters import (
    ChoiceBaselineAdapter,
    EchoPredictionAdapter,
    ExpectedPredictionAdapter,
    PredictionAdapter,
    build_prediction_adapter,
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
from bharat.eval.schema import EvalExample, EvalPrediction, EvalResult

__all__ = [
    "BatchGenerator",
    "BharatBenchReport",
    "BharatBenchRunner",
    "ChoiceBaselineAdapter",
    "EchoPredictionAdapter",
    "EvalExample",
    "EvalPrediction",
    "EvalResult",
    "ExpectedPredictionAdapter",
    "LocalCausalLMAdapter",
    "LocalInferenceConfig",
    "PredictionAdapter",
    "PredictionRunner",
    "build_prediction_adapter",
    "choice_accuracy",
    "exact_match",
    "load_local_causal_lm_adapter",
    "normalized_exact_match",
    "token_f1",
    "write_predictions_jsonl",
]
