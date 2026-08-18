from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_CONFIG_NAME = "main"


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _hf_row_to_example(index: int, split: str, row: dict[str, Any]) -> dict[str, Any]:
    # Canonical GSM8K target retains the full chain of thought ending in "#### <answer>".
    return {
        "id": f"gsm8k_{split}_{index:06d}",
        "input_text": row["question"].strip(),
        "target_text": row["answer"].strip(),
    }


def split_train_val(
    train_rows: list[dict[str, Any]], val_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_size <= 0 or val_size >= len(train_rows):
        raise ValueError("val_size must be between 1 and len(train_rows) - 1")
    indices = list(range(len(train_rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_indices = sorted(indices[:val_size])
    train_indices = sorted(indices[val_size:])
    return (
        [train_rows[i] for i in train_indices],
        [train_rows[i] for i in val_indices],
    )


def prepare_gsm8k(
    output_dir: str | Path,
    manifest_path: str | Path,
    val_size: int = 500,
    seed: int = 42,
    dataset_loader=None,
) -> dict[str, Any]:
    """Materialize the GSM8K splits as project-standard JSONL and write a checksummed manifest.

    `dataset_loader` is injected for testing; when None it falls back to `datasets.load_dataset`.
    """
    if dataset_loader is None:
        from datasets import load_dataset  # deferred so tests can run without the datasets package

        dataset_loader = load_dataset

    hf_train = list(dataset_loader(GSM8K_DATASET_ID, GSM8K_CONFIG_NAME, split="train"))
    hf_test = list(dataset_loader(GSM8K_DATASET_ID, GSM8K_CONFIG_NAME, split="test"))

    train_rows, val_rows = split_train_val(hf_train, val_size=val_size, seed=seed)
    train_examples = [_hf_row_to_example(i, "train", row) for i, row in enumerate(train_rows)]
    val_examples = [_hf_row_to_example(i, "val", row) for i, row in enumerate(val_rows)]
    test_examples = [_hf_row_to_example(i, "test", row) for i, row in enumerate(hf_test)]

    output_dir = Path(output_dir)
    paths = {
        "train": output_dir / "gsm8k_train.jsonl",
        "val": output_dir / "gsm8k_val.jsonl",
        "test": output_dir / "gsm8k_test.jsonl",
    }
    counts = {
        "train": _write_jsonl(paths["train"], train_examples),
        "val": _write_jsonl(paths["val"], val_examples),
        "test": _write_jsonl(paths["test"], test_examples),
    }
    manifest = {
        "dataset_id": GSM8K_DATASET_ID,
        "config_name": GSM8K_CONFIG_NAME,
        "seed": seed,
        "val_size": val_size,
        "counts": counts,
        "sha256": {split: _hash_file(path) for split, path in paths.items()},
        "paths": {split: str(path) for split, path in paths.items()},
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
