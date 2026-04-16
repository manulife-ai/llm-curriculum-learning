import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path_str: str) -> dict[str, Any]:
    with resolve_path(path_str).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def save_yaml(path_str: str, payload: dict[str, Any]) -> None:
    path = resolve_path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_json(path_str: str, payload: dict[str, Any]) -> None:
    path = resolve_path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_experiment_config(experiment_config_path: str) -> dict[str, Any]:
    experiment = load_yaml(experiment_config_path)
    defaults = experiment.pop("defaults", [])
    resolved: dict[str, Any] = {}
    for default_path in defaults:
        resolved = deep_merge(resolved, load_yaml(default_path))
    resolved = deep_merge(resolved, experiment)
    resolved["config_path"] = experiment_config_path
    return resolved


def append_row_to_csv(path_str: str, row: dict[str, Any]) -> None:
    path = resolve_path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
