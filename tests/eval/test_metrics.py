from __future__ import annotations

from bharat.eval.metrics import choice_accuracy, exact_match, normalized_exact_match, token_f1


class TestExactMatch:
    def test_exact_match_identical(self) -> None:
        assert exact_match("New Delhi", "New Delhi") == 1.0

    def test_exact_match_different(self) -> None:
        assert exact_match("New Delhi", "Mumbai") == 0.0

    def test_exact_match_case_sensitive(self) -> None:
        assert exact_match("New Delhi", "new delhi") == 0.0


class TestNormalizedExactMatch:
    def test_identical(self) -> None:
        assert normalized_exact_match("New Delhi", "New Delhi") == 1.0

    def test_case_insensitive(self) -> None:
        assert normalized_exact_match("New Delhi", "new delhi") == 1.0

    def test_whitespace_normalized(self) -> None:
        assert normalized_exact_match("New Delhi", "  new  delhi  ") == 1.0

    def test_different(self) -> None:
        assert normalized_exact_match("New Delhi", "Mumbai") == 0.0


class TestTokenF1:
    def test_identical(self) -> None:
        assert token_f1("New Delhi", "New Delhi") == 1.0

    def test_no_common_tokens(self) -> None:
        assert token_f1("New Delhi", "Mumbai City") == 0.0

    def test_partial_overlap(self) -> None:
        f1 = token_f1("New Delhi India", "New York India")
        assert 0.0 < f1 < 1.0

    def test_empty_prediction(self) -> None:
        assert token_f1("New Delhi", "") == 0.0

    def test_both_empty(self) -> None:
        assert token_f1("", "") == 1.0


class TestChoiceAccuracy:
    def test_correct_choice(self) -> None:
        assert choice_accuracy("hindi", "hindi", ("hindi", "tamil", "bengali")) == 1.0

    def test_wrong_choice(self) -> None:
        assert choice_accuracy("hindi", "tamil", ("hindi", "tamil", "bengali")) == 0.0

    def test_not_in_choices(self) -> None:
        assert choice_accuracy("hindi", "malayalam", ("hindi", "tamil", "bengali")) == 0.0

    def test_empty_prediction(self) -> None:
        assert choice_accuracy("hindi", "", ("hindi", "tamil")) == 0.0

    def test_case_insensitive(self) -> None:
        assert choice_accuracy("Hindi", "hindi", ("hindi", "tamil")) == 1.0
