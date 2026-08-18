from pathlib import Path

from src.utils.io import load_experiment_config, load_yaml


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_default_best_metric_is_rougeL():
    config = load_yaml(str(CONFIGS / "default.yaml"))
    assert config["evaluation"]["best_metric"] == "rougeL"


def test_gsm8k_overrides_best_metric_to_pass_at_1():
    config = load_yaml(str(CONFIGS / "data" / "gsm8k.yaml"))
    assert config["evaluation"]["best_metric"] == "gsm8k_pass_at_1"


def test_gsm8k_experiment_config_resolves_best_metric():
    resolved = load_experiment_config(str(CONFIGS / "experiment" / "k_p1_gsm8k_sft.yaml"))
    assert resolved["evaluation"]["best_metric"] == "gsm8k_pass_at_1"
    assert resolved["evaluation"]["task"] == "gsm8k"


def test_xsum_experiment_config_keeps_default_best_metric():
    resolved = load_experiment_config(str(CONFIGS / "experiment" / "k3_xsum_sft.yaml"))
    assert resolved["evaluation"]["best_metric"] == "rougeL"
