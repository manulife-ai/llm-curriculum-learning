from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
    progress_bar = tqdm(
        loader,
        desc="Evaluating",
        total=len(loader),
        dynamic_ncols=True,
    )
    for batch_idx, batch in enumerate(progress_bar, start=1):
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
        running_loss = sum(loss_values) / max(len(loss_values), 1)
        progress_bar.set_postfix(batch=batch_idx, loss=f"{running_loss:.4f}")
    progress_bar.close()

    metrics = compute_generation_metrics(all_predictions, all_references)
    metrics["loss"] = sum(loss_values) / max(len(loss_values), 1)
    return metrics, {
        "predictions": all_predictions,
        "references": all_references,
        "prompts": all_prompt_texts,
    }
