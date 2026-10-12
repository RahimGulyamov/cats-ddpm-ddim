"""Redraw the training dashboard from measured CSV data without modifying logs."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.journal import Journal


def read_rows(path):
    integer_keys = {"step", "examples_seen", "skipped_updates"}
    with path.open(newline="", encoding="utf8") as handle:
        return [{key: int(value) if key in integer_keys else float(value)
                 for key, value in row.items()} for row in csv.DictReader(handle)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("experiments/cats128-gpu"))
    args = parser.parse_args()
    training = read_rows(args.run / "train.csv")
    validation = read_rows(args.run / "validation.csv")
    if not training or not validation:
        raise SystemExit("Need completed training updates and at least one validation row")
    Journal.plot_history(args.run, training, validation)
    print(f"Dashboard updated: {args.run / 'training-dashboard.png'}; CSV data unchanged")
