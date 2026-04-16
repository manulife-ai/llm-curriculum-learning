import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, help="Directory containing experiment yaml files")
    parser.add_argument("--seeds", nargs="*", default=["42"])
    args = parser.parse_args()

    matrix_dir = PROJECT_ROOT / args.matrix
    experiment_files = sorted(matrix_dir.glob("*.yaml"))
    for config_path in experiment_files:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train.py"),
                "--config",
                str(config_path),
            ]
            env = None
            print(f"Running {config_path.name} with seed {seed}")
            subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
