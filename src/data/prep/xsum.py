from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

XSUM_DATASET_ID = "EdinburghNLP/xsum"


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
    return {
        "id": f"xsum_{split}_{index:06d}",
        "input_text": row["document"].strip(),
        "target_text": row["summary"].strip(),
    }


def deterministic_subsample(
    rows: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    if size <= 0 or size > len(rows):
        raise ValueError("size must be between 1 and len(rows)")
    indices = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = sorted(indices[:size])
    return [rows[i] for i in selected]


def prepare_xsum(
    output_dir: str | Path,
    manifest_path: str | Path,
    train_size: int = 20000,
    val_size: int = 1000,
    test_size: int = 1000,
    seed: int = 42,
    dataset_loader=None,
) -> dict[str, Any]:
    """Materialize a deterministic sub-sample of XSum as project-standard JSONL with a checksummed manifest."""
    if dataset_loader is None:
        from datasets import load_dataset  # deferred so tests can run without the datasets package

        dataset_loader = load_dataset

    # XSum's validation split is used as-is when large enough; sub-sampled otherwise.
    hf_train = list(dataset_loader(XSUM_DATASET_ID, split="train"))
    hf_val = list(dataset_loader(XSUM_DATASET_ID, split="validation"))
    hf_test = list(dataset_loader(XSUM_DATASET_ID, split="test"))

    train_rows = deterministic_subsample(hf_train, size=train_size, seed=seed)
    val_rows = deterministic_subsample(hf_val, size=val_size, seed=seed + 1)
    test_rows = deterministic_subsample(hf_test, size=test_size, seed=seed + 2)

    train_examples = [_hf_row_to_example(i, "train", row) for i, row in enumerate(train_rows)]
    val_examples = [_hf_row_to_example(i, "val", row) for i, row in enumerate(val_rows)]
    test_examples = [_hf_row_to_example(i, "test", row) for i, row in enumerate(test_rows)]

    output_dir = Path(output_dir)
    paths = {
        "train": output_dir / "xsum_train.jsonl",
        "val": output_dir / "xsum_val.jsonl",
        "test": output_dir / "xsum_test.jsonl",
    }
    counts = {
        "train": _write_jsonl(paths["train"], train_examples),
        "val": _write_jsonl(paths["val"], val_examples),
        "test": _write_jsonl(paths["test"], test_examples),
    }
    manifest = {
        "dataset_id": XSUM_DATASET_ID,
        "seed": seed,
        "sizes": {"train": train_size, "val": val_size, "test": test_size},
        "counts": counts,
        "sha256": {split: _hash_file(path) for split, path in paths.items()},
        "paths": {split: str(path) for split, path in paths.items()},
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
