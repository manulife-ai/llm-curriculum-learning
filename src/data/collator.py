from typing import Any

import torch


class CausalLMCollator:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def _encode_example(self, prompt_text: str, target_text: str) -> dict[str, Any]:
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        bos_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        input_ids = bos_ids + prompt_ids + target_ids + eos_ids
        prompt_length = len(bos_ids) + len(prompt_ids)
        labels = [-100] * prompt_length + target_ids + eos_ids
        input_ids = input_ids[: self.max_seq_length]
        labels = labels[: self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": min(prompt_length, len(input_ids)),
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = [self._encode_example(item["prompt_text"], item["target_text"]) for item in batch]
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(len(item["input_ids"]) for item in encoded)

        def pad(values: list[int], pad_value: int) -> list[int]:
            return values + [pad_value] * (max_len - len(values))

        input_ids = torch.tensor([pad(item["input_ids"], pad_token_id) for item in encoded], dtype=torch.long)
        attention_mask = torch.tensor([pad(item["attention_mask"], 0) for item in encoded], dtype=torch.long)
        labels = torch.tensor([pad(item["labels"], -100) for item in encoded], dtype=torch.long)
        prompt_lengths = torch.tensor([item["prompt_length"] for item in encoded], dtype=torch.long)
        return {
            "example_ids": [item["example_id"] for item in batch],
            "prompt_texts": [item["prompt_text"] for item in batch],
            "target_texts": [item["target_text"] for item in batch],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": prompt_lengths,
        }
