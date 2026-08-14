
# Token-level Curriculum Learning for Autoregressive LLMs

This project explores token-level curriculum learning for the supervised fine-tuning of autoregressive large language models (LLMs). Training gradually mixes the model's own predictions into teacher-forced inputs at the token level, with the aim of narrowing the train/inference distribution gap and improving generation quality.

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