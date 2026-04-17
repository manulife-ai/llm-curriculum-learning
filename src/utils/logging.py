import logging
import importlib
import os
import sys
import warnings
from pathlib import Path
from typing import Any


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)
    logger = logging.getLogger("curriculum_learning")
    logger.setLevel(level)
    for noisy in (
        "azure",
        "azure.core",
        "azure.core.pipeline",
        "azure.core.pipeline.policies.http_logging_policy",
        "azure.identity",
        "azure.storage",
        "azure.storage.blob",
        "httpx",
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "py4j",
        "absl",
        "mlflow.transformers.model_io",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    return logger


def configure_runtime_noise(config: dict[str, Any] | None = None) -> None:
    runtime_cfg = (config or {}).get("runtime", {})
    environment = runtime_cfg.get("environment", "").lower()

    # Prefer persistent HF dataset cache on Databricks.
    if environment == "databricks":
        cache_dir = runtime_cfg.get("hf_datasets_cache", "/dbfs/tmp/hf_datasets_cache")
        os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
        Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    # Disable datasets progress widgets/bars if the package is available.
    try:
        datasets_logging = importlib.import_module("datasets.utils.logging")
        datasets_logging.disable_progress_bar()
    except Exception:
        pass

    # Disable Transformers/Hugging Face progress bars where supported.
    try:
        transformers_logging = importlib.import_module("transformers.utils.logging")
        transformers_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        hf_hub_utils = importlib.import_module("huggingface_hub.utils")
        hf_hub_utils.disable_progress_bars()
    except Exception:
        pass

    # Force tqdm.auto to use plain text tqdm instead of notebook widgets.
    try:
        tqdm_std = importlib.import_module("tqdm")
        tqdm_auto = importlib.import_module("tqdm.auto")
        tqdm_auto.tqdm = tqdm_std.tqdm
        tqdm_auto.trange = tqdm_std.trange
    except Exception:
        pass

    warnings.filterwarnings(
        "ignore",
        message=r"The cache_dir for this dataset is /tmp/.hf.data.cache.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"During large dataset downloads, there could be multiple progress bar widgets.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch_dtype` is deprecated! Use `dtype` instead!",
    )
