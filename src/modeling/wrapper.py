from typing import Any

import mlflow
import torch
from transformers import AutoTokenizer


def _unwrap_loaded_model(loaded: Any):
    if hasattr(loaded, "model"):
        return loaded.model
    return loaded


def _extract_tokenizer(loaded: Any, tokenizer_name: str | None):
    if hasattr(loaded, "tokenizer") and loaded.tokenizer is not None:
        return loaded.tokenizer
    if tokenizer_name:
        return AutoTokenizer.from_pretrained(tokenizer_name)
    model_hint = getattr(getattr(loaded, "model", loaded), "name_or_path", None)
    if model_hint and not str(model_hint).startswith("models:/"):
        return AutoTokenizer.from_pretrained(model_hint)
    raise ValueError("Unable to resolve tokenizer. Set model.tokenizer_name in config.")


def load_model(uri: str, flavor: str = "transformers", tokenizer_name: str | None = None, device: str | None = None):
    if flavor == "transformers":
        loaded = mlflow.transformers.load_model(uri)
    elif flavor == "pyfunc":
        loaded = mlflow.pyfunc.load_model(uri)
    else:
        raise ValueError(f"Unsupported flavor: {flavor}")

    model = _unwrap_loaded_model(loaded)
    tokenizer = _extract_tokenizer(loaded, tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device and device.startswith("cuda") and torch.cuda.is_available():
        model.to(device)
    metadata = {
        "source_uri": uri,
        "flavor": flavor,
        "resolved_model_name": getattr(getattr(model, "config", None), "name_or_path", None),
    }
    return model, tokenizer, metadata
