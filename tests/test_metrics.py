from src.eval.metrics import (
    compute_generation_metrics,
    compute_gsm8k_pass_at_1,
    extract_gsm8k_answer,
)


def test_extract_answer_from_hashed_line():
    text = "Alice has 3 apples.\nShe eats 1.\n#### 2"
    assert extract_gsm8k_answer(text) == "2"


def test_extract_answer_strips_commas_and_normalizes_integers():
    assert extract_gsm8k_answer("#### 12,345") == "12345"
    assert extract_gsm8k_answer("#### 42.0") == "42"


def test_extract_answer_handles_negative_and_decimal():
    assert extract_gsm8k_answer("#### -5") == "-5"
    assert extract_gsm8k_answer("#### 3.14") == "3.14"


def test_extract_answer_uses_last_hashed_match_when_multiple_present():
    text = "First guess:\n#### 5\nCorrected:\n#### 7"
    assert extract_gsm8k_answer(text) == "7"


def test_extract_answer_falls_back_to_last_number_by_default():
    assert extract_gsm8k_answer("The answer is 42.") == "42"


def test_extract_answer_fallback_disabled_returns_none():
    assert extract_gsm8k_answer("The answer is 42.", fallback_to_last_number=False) is None


def test_extract_answer_none_when_no_number_found():
    assert extract_gsm8k_answer("no digits at all") is None
    assert extract_gsm8k_answer(None) is None


def test_pass_at_1_scoring():
    preds = ["reasoning...\n#### 7", "text\n#### 10", "unfinished", "wrong path\n#### 5"]
    refs = ["#### 7", "#### 9", "#### 3", "#### 5"]
    result = compute_gsm8k_pass_at_1(preds, refs)
    assert result["gsm8k_pass_at_1"] == 0.5  # first and last correct
    assert result["gsm8k_extract_rate"] == 0.75  # third has no number to extract


def test_generation_metrics_includes_pass_at_1_when_task_is_gsm8k():
    preds = ["reasoning\n#### 4"]
    refs = ["work\n#### 4"]
    result = compute_generation_metrics(preds, refs, task="gsm8k")
    assert "gsm8k_pass_at_1" in result
    assert result["gsm8k_pass_at_1"] == 1.0


def test_generation_metrics_omits_pass_at_1_when_task_is_none():
    result = compute_generation_metrics(["a"], ["a"], task=None)
    assert "gsm8k_pass_at_1" not in result
