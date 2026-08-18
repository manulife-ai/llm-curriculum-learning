from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.eval.evaluate import evaluate_model
from src.train.loop import run_train_step
from src.utils.checkpoint import save_checkpoint
from src.utils.io import append_row_to_csv, resolve_path, save_json, save_yaml
from src.utils.seed import build_generator
from src.utils.tracking import MLflowTracker, build_run_metadata, get_git_revision


class CurriculumTrainer:
    def __init__(self, config, model, tokenizer, collator, schedule, mixer, logger, device: str):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.schedule = schedule
        self.mixer = mixer
        self.logger = logger
        self.device = device
        self.training_history: list[dict] = []

    def _build_run_dir(self) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config['experiment_name']}_{timestamp}"
        run_dir = resolve_path(self.config["output_root"]) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def train(self, train_dataset, val_dataset, test_dataset):
        run_dir = self._build_run_dir()
        save_yaml(str(run_dir / "resolved_config.yaml"), self.config)
        run_metadata = build_run_metadata(self.config, cwd=resolve_path("."))
        save_json(str(run_dir / "run_metadata.json"), run_metadata)

        tracker = MLflowTracker.from_config(self.config, logger=self.logger)
        git_rev = get_git_revision(cwd=resolve_path("."))
        tracker.start(tags={
            "experiment_name": self.config.get("experiment_name", ""),
            "seed": str(self.config.get("seed", "")),
            **git_rev,
        })
        tracker.log_params(self.config)
        tracker.log_artifact(run_dir / "resolved_config.yaml")
        tracker.log_artifact(run_dir / "run_metadata.json")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            collate_fn=self.collator,
            generator=build_generator(self.config["seed"]),
        )
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        total_steps = max(
            1,
            (len(train_loader) * self.config["training"]["num_epochs"])
            // self.config["training"]["gradient_accumulation_steps"],
        )
        warmup_steps = int(total_steps * self.config["training"]["warmup_ratio"])
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.model.train()
        self.model.to(self.device)
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        best_val_rouge = -1.0
        best_checkpoint = None

        for epoch in range(self.config["training"]["num_epochs"]):
            epoch_losses = []
            epoch_p_ar = []
            epoch_replacement_rate = []
            skipped_batches = 0
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}",
                total=len(train_loader),
                dynamic_ncols=True,
            )
            for batch_idx, batch in enumerate(progress_bar, start=1):
                tensor_batch = {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in batch.items()
                }
                step_metrics = run_train_step(
                    model=self.model,
                    batch=tensor_batch,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    mixer=self.mixer,
                    schedule=self.schedule,
                    global_step=global_step,
                    grad_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
                    max_grad_norm=self.config["training"]["max_grad_norm"],
                    precision=self.config["training"]["precision"],
                    device=self.device,
                )
                if math.isfinite(step_metrics["loss"]):
                    epoch_losses.append(step_metrics["loss"])
                else:
                    skipped_batches += 1
                epoch_p_ar.append(step_metrics["p_ar"])
                epoch_replacement_rate.append(step_metrics["replacement_rate"])
                if (
                    batch_idx == 1
                    or batch_idx % self.config["training"].get("log_every_steps", 1) == 0
                    or batch_idx == len(train_loader)
                ):
                    progress_bar.set_postfix(
                        loss=f"{step_metrics['loss']:.4f}",
                        p_ar=f"{step_metrics['p_ar']:.3f}",
                        repl=f"{step_metrics['replacement_rate']:.3f}",
                        skipped=skipped_batches,
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                global_step += 1
            progress_bar.close()

            val_metrics, val_outputs = evaluate_model(
                self.model,
                self.tokenizer,
                val_dataset,
                self.collator,
                self.config,
                self.device,
            )
            history_row = {
                "epoch": epoch + 1,
                "train_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                "avg_p_ar": sum(epoch_p_ar) / max(len(epoch_p_ar), 1),
                "avg_replacement_rate": sum(epoch_replacement_rate) / max(len(epoch_replacement_rate), 1),
                "skipped_batches": skipped_batches,
                "val_loss": val_metrics["loss"],
                "val_rouge1": val_metrics["rouge1"],
                "val_rougeL": val_metrics["rougeL"],
                "val_exact_match": val_metrics["exact_match"],
            }
            self.training_history.append(history_row)
            self.logger.info("Epoch %s metrics: %s", epoch + 1, history_row)
            tracker.log_metrics(
                {k: v for k, v in history_row.items() if k != "epoch"},
                step=epoch + 1,
            )

            if val_metrics["rougeL"] > best_val_rouge:
                best_val_rouge = val_metrics["rougeL"]
                best_checkpoint = save_checkpoint(
                    self.model,
                    self.tokenizer,
                    str(run_dir),
                    "best_checkpoint",
                    val_metrics,
                )
                save_json(str(run_dir / "best_val_predictions.json"), val_outputs)

        save_json(str(run_dir / "training_history.json"), {"history": self.training_history})
        test_metrics, test_outputs = evaluate_model(
            self.model,
            self.tokenizer,
            test_dataset,
            self.collator,
            self.config,
            self.device,
        )
        save_json(str(run_dir / "test_predictions.json"), test_outputs)
        run_summary = {
            "experiment_name": self.config["experiment_name"],
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
            "best_val_rougeL": best_val_rouge,
            "test_loss": test_metrics["loss"],
            "test_rouge1": test_metrics["rouge1"],
            "test_rougeL": test_metrics["rougeL"],
            "test_exact_match": test_metrics["exact_match"],
        }
        save_json(str(run_dir / "run_summary.json"), run_summary)
        append_row_to_csv(self.config["summary_path"], run_summary)
        tracker.log_metrics(
            {k: v for k, v in run_summary.items() if isinstance(v, (int, float))}
        )
        tracker.log_artifact(run_dir / "training_history.json")
        tracker.log_artifact(run_dir / "run_summary.json")
        tracker.end()
        return run_summary
