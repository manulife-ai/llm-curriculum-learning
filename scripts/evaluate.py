import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collator import CausalLMCollator
from src.data.datasets import build_datasets
from src.eval.evaluate import evaluate_model
from src.utils.io import load_yaml, save_json
from src.utils.logging import configure_logging


def run_evaluation(run_dir: str):
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
    collator = CausalLMCollator(tokenizer, config["training"]["max_seq_length"])
    datasets = build_datasets(config)
    val_metrics, val_outputs = evaluate_model(model, tokenizer, datasets["val"], collator, config, device)
    test_metrics, test_outputs = evaluate_model(model, tokenizer, datasets["test"], collator, config, device)
    payload = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(str(run_path / "evaluation_summary.json"), payload)
    save_json(str(run_path / "evaluation_outputs_val.json"), val_outputs)
    save_json(str(run_path / "evaluation_outputs_test.json"), test_outputs)
    logger.info("Evaluation complete: %s", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    print(run_evaluation(args.run_dir))


if __name__ == "__main__":
    main()
