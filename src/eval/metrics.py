import re

from rouge_score import rouge_scorer

# Matches the final "#### <answer>" segment in a GSM8K-style response.
_GSM8K_ANSWER_RE = re.compile(r"####\s*([\-\+]?[\d,]*\.?\d+)")
# Fallback: last standalone number anywhere in the string.
_TRAILING_NUMBER_RE = re.compile(r"([\-\+]?[\d,]*\.?\d+)")


def _normalize_number(text: str) -> str | None:
    cleaned = text.strip().replace(",", "").rstrip(".")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return format(int(value), "d") if value.is_integer() else format(value, "g")


def extract_gsm8k_answer(text: str, *, fallback_to_last_number: bool = True) -> str | None:
    """Return the normalized final numeric answer from a GSM8K-style response, or None if none is found."""
    if text is None:
        return None
    matches = _GSM8K_ANSWER_RE.findall(text)
    if matches:
        return _normalize_number(matches[-1])
    if not fallback_to_last_number:
        return None
    fallback = _TRAILING_NUMBER_RE.findall(text)
    return _normalize_number(fallback[-1]) if fallback else None


def compute_gsm8k_pass_at_1(predictions: list[str], references: list[str]) -> dict[str, float]:
    total = max(len(predictions), 1)
    correct = 0
    extracted = 0
    for prediction, reference in zip(predictions, references):
        pred_answer = extract_gsm8k_answer(prediction)
        gold_answer = extract_gsm8k_answer(reference, fallback_to_last_number=False)
        if pred_answer is not None:
            extracted += 1
        if pred_answer is not None and gold_answer is not None and pred_answer == gold_answer:
            correct += 1
    return {
        "gsm8k_pass_at_1": correct / total,
        "gsm8k_extract_rate": extracted / total,
    }


def compute_generation_metrics(
    predictions: list[str], references: list[str], task: str | None = None
) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_scores = []
    rouge_l_scores = []
    exact_matches = []
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge_l_scores.append(scores["rougeL"].fmeasure)
        exact_matches.append(float(prediction.strip() == reference.strip()))
    result: dict[str, float] = {
        "rouge1": sum(rouge1_scores) / max(len(rouge1_scores), 1),
        "rougeL": sum(rouge_l_scores) / max(len(rouge_l_scores), 1),
        "exact_match": sum(exact_matches) / max(len(exact_matches), 1),
    }
    if task == "gsm8k":
        result.update(compute_gsm8k_pass_at_1(predictions, references))
    return result
