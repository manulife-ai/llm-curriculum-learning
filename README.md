
# Sequence-level Curriculum Learning for Autoregressive LLMs

This project investigates sequence-level curriculum learning strategies to enhance the generation quality of autoregressive large language models (LLMs) by mitigating exposure bias during training.

## Project Tree


```
llm-curriculum-learning/
├── data/
├── configs/
│   ├── experiment/
│   ├── model/
│   ├── runtime/
│   ├── curriculum/
│   └── data/
├── src/
│   ├── curriculum/
│   │   ├── mixer.py
│   │   ├── schedules.py
│   │   └── __init__.py
│   ├── train/
│   │   ├── loop.py
│   │   ├── trainer.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── checkpoint.py
│   │   ├── io.py
│   │   ├── logging.py
│   │   ├── seed.py
│   │   └── __init__.py
│   └── data/
│       ├── collator.py
│       ├── datasets.py
│       └── __init__.py
├── scripts/
│   ├── evaluate.py
│   ├── run_experiments.py
│   └── train.py
├── notebooks/
├── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Structure

- `data/`: Train/val/test JSON or JSONL files
- `configs/`: Experiment, model, runtime, and curriculum configs
- `src/`: Data pipeline, model loading, training, and evaluation code
- `scripts/`: CLI entrypoints used by notebooks and terminal execution
- `notebooks/`: Notebook entrypoints