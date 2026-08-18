import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prep.xsum import prepare_xsum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data", help="Directory to write JSONL splits")
    parser.add_argument("--manifest", default="data/manifests/xsum.json", help="Manifest path")
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = prepare_xsum(
        output_dir=PROJECT_ROOT / args.output_dir,
        manifest_path=PROJECT_ROOT / args.manifest,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"XSum prepared: {manifest['counts']}")
    print(f"Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
