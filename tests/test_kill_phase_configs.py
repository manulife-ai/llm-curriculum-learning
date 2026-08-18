"""Ensure the frozen kill-phase experiment configs resolve to a well-formed merged config."""

from pathlib import Path

import pytest

from src.utils.io import load_experiment_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiment"

KILL_PHASE_CONFIGS = [
    "k_p1_gsm8k_sft.yaml",
    "k1_gsm8k_sft.yaml",
    "k2_gsm8k_ss_linear.yaml",
    "k3_xsum_sft.yaml",
    "k4_xsum_ss_linear.yaml",
    "k5_gsm8k_ss_low.yaml",
]

REQUIRED_TOP_LEVEL = {
    "seed",
    "experiment_name",
    "training",
    "evaluation",
    "data",
    "model",
    "curriculum",
    "runtime",
    "tracking",
}


@pytest.mark.parametrize("config_name", KILL_PHASE_CONFIGS)
def test_kill_phase_config_resolves(config_name):
    resolved = load_experiment_config(str(CONFIG_DIR / config_name))
    missing = REQUIRED_TOP_LEVEL - set(resolved.keys())
    assert not missing, f"{config_name} is missing keys: {missing}"
    assert resolved["experiment_name"], f"{config_name} has no experiment_name"
    assert resolved["training"]["num_epochs"] >= 1


def test_gsm8k_configs_enable_pass_at_1_task():
    for name in [
        "k_p1_gsm8k_sft.yaml",
        "k1_gsm8k_sft.yaml",
        "k2_gsm8k_ss_linear.yaml",
        "k5_gsm8k_ss_low.yaml",
    ]:
        resolved = load_experiment_config(str(CONFIG_DIR / name))
        assert resolved["evaluation"].get("task") == "gsm8k", name


def test_xsum_configs_do_not_set_gsm8k_task():
    for name in ["k3_xsum_sft.yaml", "k4_xsum_ss_linear.yaml"]:
        resolved = load_experiment_config(str(CONFIG_DIR / name))
        assert resolved["evaluation"].get("task") != "gsm8k", name


def test_curriculum_settings_match_expected_schedule():
    sft_names = ["k_p1_gsm8k_sft.yaml", "k1_gsm8k_sft.yaml", "k3_xsum_sft.yaml"]
    for name in sft_names:
        resolved = load_experiment_config(str(CONFIG_DIR / name))
        assert resolved["curriculum"]["schedule_type"] == "teacher_forcing", name
        assert resolved["curriculum"]["p_max"] == 0.0, name

    for name in ["k2_gsm8k_ss_linear.yaml", "k4_xsum_ss_linear.yaml"]:
        resolved = load_experiment_config(str(CONFIG_DIR / name))
        assert resolved["curriculum"]["schedule_type"] == "linear", name
        assert resolved["curriculum"]["p_max"] == 0.5, name

    low = load_experiment_config(str(CONFIG_DIR / "k5_gsm8k_ss_low.yaml"))
    assert low["curriculum"]["schedule_type"] == "linear"
    assert low["curriculum"]["p_max"] == 0.3
