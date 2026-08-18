"""Δ(k) exposure-bias diagnostic.

For each example we compute the per-token negative log-likelihood of the gold
continuation conditioned on (a) the gold prefix and (b) the model's own greedy
prefix. Δ(k) = NLL_own(k) − NLL_gold(k), averaged across examples, is a
prefix-source gap that quantifies how much the model's own generations drift
from the training distribution as horizon k grows.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

# A prompt token has no label; loss uses this sentinel to skip it.
IGNORE_INDEX = -100


def _greedy_generate_prefix(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> torch.Tensor:
    """Return the model's greedy continuation as a 1-D tensor of new token ids."""
    with torch.no_grad():
        output = model.generate(
            input_ids=prompt_ids.unsqueeze(0).to(device),
            attention_mask=torch.ones_like(prompt_ids).unsqueeze(0).to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output[0, prompt_ids.shape[0]:]
    return generated.detach().cpu()


def _mean_continuation_nll(
    model,
    prompt_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    device: str,
) -> float:
    """Return the mean NLL of `continuation_ids` given `prompt_ids + prefix_ids` as context."""
    input_ids = torch.cat([prompt_ids, prefix_ids, continuation_ids]).unsqueeze(0).to(device)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    continuation_start = prompt_ids.shape[0] + prefix_ids.shape[0]
    labels[0, continuation_start:] = input_ids[0, continuation_start:]
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels)
    # Hugging Face CausalLM losses are already mean cross-entropy over unmasked positions.
    return float(output.loss.item())


def _iter_examples(examples: Iterable[dict[str, str]], max_examples: int | None):
    for i, ex in enumerate(examples):
        if max_examples is not None and i >= max_examples:
            break
        yield ex


def compute_delta_k(
    model,
    tokenizer,
    examples: Iterable[dict[str, str]],
    k_values: list[int],
    device: str,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Compute Δ(k) over a batch of {prompt_text, target_text} examples.

    Returns a dict with per-k mean NLLs, per-k Δ, per-k example counts (some may be
    dropped when the gold target is shorter than k tokens plus at least one continuation
    token), and the k values used.
    """
    if not k_values:
        raise ValueError("k_values must be non-empty")
    k_values = sorted(set(int(k) for k in k_values))
    max_new_tokens = max(k_values)

    per_k_nll_own: dict[int, list[float]] = {k: [] for k in k_values}
    per_k_nll_gold: dict[int, list[float]] = {k: [] for k in k_values}

    model.eval()
    for example in _iter_examples(examples, max_examples):
        prompt_text = example["prompt_text"]
        target_text = example["target_text"]
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt_text, add_special_tokens=False), dtype=torch.long
        )
        gold_ids = torch.tensor(
            tokenizer.encode(target_text, add_special_tokens=False), dtype=torch.long
        )
        if gold_ids.numel() < 2:
            continue

        own_ids = _greedy_generate_prefix(
            model, tokenizer, prompt_ids, max_new_tokens=max_new_tokens, device=device
        )

        for k in k_values:
            if gold_ids.numel() <= k or own_ids.numel() < k:
                continue
            continuation_ids = gold_ids[k:]
            gold_prefix = gold_ids[:k]
            own_prefix = own_ids[:k]
            per_k_nll_gold[k].append(
                _mean_continuation_nll(model, prompt_ids, gold_prefix, continuation_ids, device)
            )
            per_k_nll_own[k].append(
                _mean_continuation_nll(model, prompt_ids, own_prefix, continuation_ids, device)
            )

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    nll_own = {k: _mean(per_k_nll_own[k]) for k in k_values}
    nll_gold = {k: _mean(per_k_nll_gold[k]) for k in k_values}
    delta_k = {k: nll_own[k] - nll_gold[k] for k in k_values}
    counts = {k: len(per_k_nll_own[k]) for k in k_values}

    return {
        "k_values": k_values,
        "nll_own": nll_own,
        "nll_gold": nll_gold,
        "delta_k": delta_k,
        "counts": counts,
    }
