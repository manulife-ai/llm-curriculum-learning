import torch


def _build_replace_mask(batch: dict[str, torch.Tensor], p_ar: float) -> torch.Tensor:
    target_mask = batch["labels"] != -100
    replace_mask = torch.zeros_like(target_mask, dtype=torch.bool)
    if p_ar <= 0.0:
        return replace_mask
    replace_mask[:, 1:] = target_mask[:, 1:] & (torch.rand_like(batch["input_ids"][:, 1:].float()) < p_ar)
    return replace_mask


def _build_mix_stats(replace_mask: torch.Tensor, labels: torch.Tensor, mixer_type: str) -> dict[str, float | int | str]:
    replaced_tokens = int(replace_mask.sum().item())
    total_target_tokens = int((labels != -100).sum().item())
    replacement_rate = replaced_tokens / max(total_target_tokens, 1)
    return {
        "mixer_type": mixer_type,
        "replacement_rate": replacement_rate,
        "replaced_tokens": replaced_tokens,
    }


class TokenLevelGreedyMixer:
    mixer_type = "single_pass"

    def mix(self, model, batch: dict[str, torch.Tensor], p_ar: float):
        replace_mask = _build_replace_mask(batch, p_ar)
        if not replace_mask.any():
            return batch["input_ids"], _build_mix_stats(replace_mask, batch["labels"], self.mixer_type)

        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            greedy_predictions = outputs.logits[:, :-1, :].argmax(dim=-1)

        mixed_input_ids = batch["input_ids"].clone()
        shifted_predictions = torch.zeros_like(mixed_input_ids)
        shifted_predictions[:, 1:] = greedy_predictions
        mixed_input_ids[replace_mask] = shifted_predictions[replace_mask]
        return mixed_input_ids, _build_mix_stats(replace_mask, batch["labels"], self.mixer_type)


class SequentialGreedyRollInMixer:
    mixer_type = "sequential"

    def mix(self, model, batch: dict[str, torch.Tensor], p_ar: float):
        replace_mask = _build_replace_mask(batch, p_ar)
        if not replace_mask.any():
            return batch["input_ids"], _build_mix_stats(replace_mask, batch["labels"], self.mixer_type)

        mixed_input_ids = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]
        with torch.no_grad():
            for position in range(mixed_input_ids.shape[1] - 1):
                rows_to_replace = replace_mask[:, position + 1]
                if not rows_to_replace.any():
                    continue
                outputs = model(
                    input_ids=mixed_input_ids,
                    attention_mask=attention_mask,
                )
                next_token_predictions = outputs.logits[:, position, :].argmax(dim=-1)
                mixed_input_ids[rows_to_replace, position + 1] = next_token_predictions[rows_to_replace]

        return mixed_input_ids, _build_mix_stats(replace_mask, batch["labels"], self.mixer_type)


def build_mixer(config: dict):
    mixer_type = config.get("curriculum", {}).get("mixer_type", "single_pass")
    if mixer_type == "single_pass":
        return TokenLevelGreedyMixer()
    if mixer_type == "sequential":
        return SequentialGreedyRollInMixer()
    raise ValueError(f"Unsupported mixer type: {mixer_type}")
