from __future__ import annotations

import re


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def exact_match(expected: str, prediction: str) -> float:
    return 1.0 if expected == prediction else 0.0


def normalized_exact_match(expected: str, prediction: str) -> float:
    return 1.0 if _normalize(expected) == _normalize(prediction) else 0.0


def token_f1(expected: str, prediction: str) -> float:
    if not expected and not prediction:
        return 1.0
    if not prediction:
        return 0.0
    exp_tokens = _normalize(expected).split()
    pred_tokens = _normalize(prediction).split()
    if not exp_tokens and not pred_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0
    exp_set = set(exp_tokens)
    pred_set = set(pred_tokens)
    common = exp_set & pred_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall = len(common) / len(exp_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def choice_accuracy(expected: str, prediction: str, choices: tuple[str, ...]) -> float:
    if not prediction:
        return 0.0
    norm_pred = _normalize(prediction)
    norm_expected = _normalize(expected)
    norm_choices = tuple(_normalize(c) for c in choices)
    if norm_pred not in norm_choices:
        return 0.0
    return 1.0 if norm_pred == norm_expected else 0.0
