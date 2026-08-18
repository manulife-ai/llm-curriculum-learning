import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.curriculum.mixer import build_mixer
from src.curriculum.schedules import build_schedule
from src.data.collator import CausalLMCollator
from src.data.datasets import build_datasets
from src.modeling.wrapper import load_model
from src.train.trainer import CurriculumTrainer
from src.utils.io import deep_merge, load_experiment_config, load_yaml
from src.utils.logging import configure_logging, configure_runtime_noise
from src.utils.seed import set_seed


def run_training(experiment_config: str, runtime_config: str | None = None, seed: int | None = None):
    config = load_experiment_config(experiment_config)
    if runtime_config:
        config = deep_merge(config, load_yaml(runtime_config))
    if seed is not None:
        config["seed"] = seed

    configure_runtime_noise(config)
    logger = configure_logging()
    set_seed(config["seed"])
    datasets = build_datasets(config)
    device = config["runtime"].get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
        config["runtime"]["device"] = device
    model, tokenizer, metadata = load_model(
        uri=config["model"]["base_model_uri"],
        flavor=config["model"]["load_flavor"],
        tokenizer_name=config["model"].get("tokenizer_name"),
        device=device,
    )
    config["model_loading"] = metadata
    collator = CausalLMCollator(tokenizer, config["training"]["max_seq_length"])
    trainer = CurriculumTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        collator=collator,
        schedule=build_schedule(config),
        mixer=build_mixer(config),
        logger=logger,
        device=device,
    )
    return trainer.train(datasets["train"], datasets["val"], datasets["test"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment yaml")
    parser.add_argument("--runtime-config", default=None, help="Optional runtime yaml override")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    args = parser.parse_args()
    result = run_training(args.config, args.runtime_config, seed=args.seed)
    print(result)


if __name__ == "__main__":
    main()
