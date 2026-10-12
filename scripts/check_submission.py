"""Check that measured comparison, video and README links are ready to publish."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    required = ["results/comparison/timings.json", "results/comparison/timings.csv",
                "results/comparison/demo.mp4", "results/comparison/demo.gif", "results/checkpoint.json",
                "results/training/events.jsonl", "results/training/environment.json",
                "results/training/split.json", "checkpoints/cats128-step10000.pt"]
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f"Missing {name}")
    text = (ROOT / "README.md").read_text()
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if "://" not in target and not (ROOT / target.split("#")[0]).exists():
            errors.append(f"Broken README link: {target}")
    if errors:
        raise SystemExit("Submission not ready:\n" + "\n".join(errors))
    metadata = json.loads((ROOT / "results/checkpoint.json").read_text())
    digest = hashlib.sha256()
    with (ROOT / "checkpoints/cats128-step10000.pt").open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != metadata["sha256"]:
        raise SystemExit("Checkpoint hash mismatch")
    with (ROOT / "results/training/train.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10000 or int(rows[-1]["step"]) != 10000:
        raise SystemExit("Expected 10,000 actual training updates")
    print("PASS: training logs, measured timings, video, weights and README links")


if __name__ == "__main__":
    main()
