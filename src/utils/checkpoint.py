from src.utils.io import save_json
from src.utils.io import resolve_path


def save_checkpoint(model, tokenizer, output_dir: str, tag: str, metrics: dict):
    checkpoint_dir = resolve_path(output_dir) / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # model.save_pretrained(checkpoint_dir)
    # tokenizer.save_pretrained(checkpoint_dir)
    save_json(str(checkpoint_dir / "metrics.json"), metrics)
    return checkpoint_dir
