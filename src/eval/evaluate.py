from torch.utils.data import DataLoader

from src.eval.generation import generate_predictions
from src.eval.metrics import compute_generation_metrics


def evaluate_model(model, tokenizer, dataset, collator, config: dict, device: str):
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )
    loss_values = []
    all_predictions = []
    all_references = []
    all_prompt_texts = []

    model.eval()
    for batch in loader:
        tensor_batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        outputs = model(
            input_ids=tensor_batch["input_ids"],
            attention_mask=tensor_batch["attention_mask"],
            labels=tensor_batch["labels"],
        )
        loss_values.append(float(outputs.loss.item()))
        predictions = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=batch["prompt_texts"],
            generation_config=config["evaluation"],
            device=device,
        )
        all_predictions.extend(predictions)
        all_references.extend(batch["target_texts"])
        all_prompt_texts.extend(batch["prompt_texts"])

    metrics = compute_generation_metrics(all_predictions, all_references)
    metrics["loss"] = sum(loss_values) / max(len(loss_values), 1)
    return metrics, {
        "predictions": all_predictions,
        "references": all_references,
        "prompts": all_prompt_texts,
    }
