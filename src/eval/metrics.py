from rouge_score import rouge_scorer


def compute_generation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_scores = []
    rouge_l_scores = []
    exact_matches = []
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge_l_scores.append(scores["rougeL"].fmeasure)
        exact_matches.append(float(prediction.strip() == reference.strip()))
    return {
        "rouge1": sum(rouge1_scores) / max(len(rouge1_scores), 1),
        "rougeL": sum(rouge_l_scores) / max(len(rouge_l_scores), 1),
        "exact_match": sum(exact_matches) / max(len(exact_matches), 1),
    }
