
# Sequence-level Curriculum Learning for Language Models

This project explores sequence-level curriculum learning techniques to improve language model performance by reducing exposure bias.

## Structure

- `data/`: synthetic train/val/test JSONL files
- `configs/`: experiment, model, runtime, and curriculum configs
- `src/`: data pipeline, model loading, training, and evaluation code
- `scripts/`: CLI entrypoints used by notebooks and terminal execution
- `notebooks/`: Databricks notebook entrypoints

## Quick Start

Baseline:

```bash
python scripts/train.py --config configs/experiment/baseline_tf.yaml
```

Single-pass linear curriculum:

```bash
python scripts/train.py --config configs/experiment/curriculum_linear.yaml
```

Evaluate a saved run:

```bash
python scripts/evaluate.py --run_dir experiments/runs/<run_id>
```