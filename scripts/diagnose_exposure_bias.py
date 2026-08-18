"""CLI entrypoint for running the Δ(k) exposure-bias diagnostic on a completed run."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import build_datasets
from src.eval.exposure_bias import compute_delta_k
from src.utils.io import load_yaml, save_json
from src.utils.logging import configure_logging


def run_diagnostic(
    run_dir: str,
    split: str = "val",
    k_values: list[int] | None = None,
    max_examples: int | None = None,
    output_name: str | None = None,
) -> dict:
    logger = configure_logging()
    run_path = Path(run_dir)
    config = load_yaml(str(run_path / "resolved_config.yaml"))
    checkpoint_dir = run_path / "best_checkpoint"
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = config["runtime"].get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    model.to(device)

    datasets = build_datasets(config)
    if split not in datasets:
        raise ValueError(f"Unknown split '{split}'. Available: {list(datasets)}")
    dataset = datasets[split]
    examples = (dataset[i] for i in range(len(dataset)))

    k_values = k_values or [8, 16, 32, 64]
    result = compute_delta_k(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        k_values=k_values,
        device=device,
        max_examples=max_examples,
    )
    result["split"] = split
    result["max_examples"] = max_examples
    result["checkpoint"] = str(checkpoint_dir)

    output_name = output_name or f"exposure_bias_{split}.json"
    output_path = run_path / output_name
    save_json(str(output_path), result)
    logger.info("Exposure-bias diagnostic written to %s: Δ(k)=%s", output_path, result["delta_k"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run directory containing best_checkpoint/ and resolved_config.yaml")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate on")
    parser.add_argument("--k", nargs="*", type=int, default=None, help="Prefix horizons; defaults to 8 16 32 64")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap the number of examples for a fast diagnostic")
    parser.add_argument("--output-name", default=None, help="Filename to write inside the run dir")
    args = parser.parse_args()

    result = run_diagnostic(
        run_dir=args.run_dir,
        split=args.split,
        k_values=args.k,
        max_examples=args.max_examples,
        output_name=args.output_name,
    )
    print(result["delta_k"])


if __name__ == "__main__":
    main()
