from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    import mlflow
except ImportError:  # mlflow is a runtime dep; guard so tests without it still import this module.
    mlflow = None


def get_git_revision(cwd: str | os.PathLike | None = None) -> dict[str, str]:
    """Return the current git commit hash and dirty flag, or empty strings if unavailable."""
    cwd = str(cwd) if cwd is not None else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": "", "git_dirty": ""}
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode()
        dirty = "true" if status.strip() else "false"
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""
    return {"git_sha": sha, "git_dirty": dirty}


def _flatten_params(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            _flatten_params(child_key, value, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = ",".join(str(v) for v in obj)
    else:
        out[prefix] = obj


class MLflowTracker:
    """Thin wrapper around MLflow that no-ops when tracking is disabled or unavailable."""

    def __init__(
        self,
        enabled: bool,
        experiment_name: str | None,
        run_name: str | None,
        tracking_uri: str | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("curriculum_learning")
        self.enabled = enabled and mlflow is not None
        self._active_run = None
        if not self.enabled:
            if enabled and mlflow is None:
                self.logger.warning("mlflow requested but not installed; tracking disabled")
            return
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        self._run_name = run_name

    @classmethod
    def from_config(cls, config: dict[str, Any], logger: logging.Logger | None = None) -> "MLflowTracker":
        tracking_cfg = config.get("tracking", {}) or {}
        run_name = tracking_cfg.get("run_name") or config.get("experiment_name")
        return cls(
            enabled=bool(tracking_cfg.get("enabled", False)),
            experiment_name=tracking_cfg.get("experiment_name"),
            run_name=run_name,
            tracking_uri=tracking_cfg.get("tracking_uri"),
            logger=logger,
        )

    def start(self, tags: dict[str, str] | None = None) -> None:
        if not self.enabled:
            return
        self._active_run = mlflow.start_run(run_name=self._run_name)
        if tags:
            mlflow.set_tags(tags)

    def log_params(self, config: dict[str, Any]) -> None:
        if not self.enabled:
            return
        flat: dict[str, Any] = {}
        _flatten_params("", config, flat)
        # MLflow caps param length at 500 chars and rejects None; coerce defensively.
        safe = {k: str(v)[:500] for k, v in flat.items() if v is not None}
        mlflow.log_params(safe)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if clean:
            mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | os.PathLike) -> None:
        if not self.enabled:
            return
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))

    def end(self) -> None:
        if not self.enabled or self._active_run is None:
            return
        mlflow.end_run()
        self._active_run = None

    def __enter__(self) -> "MLflowTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end()
