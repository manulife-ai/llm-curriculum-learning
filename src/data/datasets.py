import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.utils.io import resolve_path


@dataclass
class IncidentExample:
    example_id: str
    input_text: str
    target_text: str


class IncidentSummaryDataset(Dataset):
    def __init__(self, examples: list[IncidentExample], prompt_template: str):
        self.examples = examples
        self.prompt_template = prompt_template

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "example_id": example.example_id,
            "prompt_text": self.prompt_template.format(input_text=example.input_text),
            "target_text": example.target_text,
        }


def load_jsonl_dataset(path_str: str) -> list[IncidentExample]:
    path = resolve_path(path_str)
    examples: list[IncidentExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            examples.append(
                IncidentExample(
                    example_id=record["id"],
                    input_text=record["input_text"],
                    target_text=record["target_text"],
                )
            )
    return examples


def build_datasets(config: dict[str, Any]) -> dict[str, IncidentSummaryDataset]:
    prompt_template = config["data"]["prompt_template"]
    data_cfg = config["data"]
    return {
        "train": IncidentSummaryDataset(load_jsonl_dataset(data_cfg["train_path"]), prompt_template),
        "val": IncidentSummaryDataset(load_jsonl_dataset(data_cfg["val_path"]), prompt_template),
        "test": IncidentSummaryDataset(load_jsonl_dataset(data_cfg["test_path"]), prompt_template),
    }
