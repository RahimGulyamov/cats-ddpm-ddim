"""Check that measured comparison, video and README links are ready to publish."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-assets", action="store_true")
    args = parser.parse_args()
    errors = []
    af_hq = (ROOT / "results/afhq/assets.json").is_file()
    required = ["results/comparison/timings.json", "results/comparison/timings.csv",
                "results/comparison/demo.mp4", "results/comparison/demo.gif", "results/checkpoint.json",
                "results/training/events.jsonl", "results/training/environment.json",
                "results/training/split.json"]
    if af_hq:
        required.extend(["results/afhq/comparison/timings.json", "results/afhq/comparison/timings.csv",
                         "results/afhq/comparison/demo.mp4", "results/afhq/comparison/demo.gif",
                         "results/afhq/training/train.csv", "results/afhq/training/validation.csv",
                         "results/afhq/training/loss.png", "results/afhq/training/split.json",
                         "checkpoints/afhq128-step10000.pt"])
        if args.release_assets:
            required.append("checkpoints/diffusion-afhq-data.zip")
    else:
        required.append("checkpoints/cats128-step10000.pt")
    for name in required:
        if not (ROOT / name).is_file():
            errors.append(f"Missing {name}")
    for document in (ROOT / "README.md", ROOT / "docs/few-shot.md", ROOT / "docs/theory.md"):
        if not document.exists():
            continue
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if "://" not in target and not (document.parent / target.split("#")[0]).exists():
                errors.append(f"Broken link in {document.name}: {target}")
    if errors:
        raise SystemExit("Submission not ready:\n" + "\n".join(errors))
    metadata = json.loads((ROOT / "results/checkpoint.json").read_text())
    def check_asset(path, expected):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise SystemExit(f"Asset hash mismatch: {path.name}")
        if "bytes" in expected and path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"Asset size mismatch: {path.name}")

    old_weights = ROOT / "checkpoints/cats128-step10000.pt"
    if old_weights.exists():
        check_asset(old_weights, metadata)
    with (ROOT / "results/training/train.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 10000 or int(rows[-1]["step"]) != 10000:
        raise SystemExit("Expected 10,000 actual training updates")
    if af_hq:
        assets = json.loads((ROOT / "results/afhq/assets.json").read_text())
        check_asset(ROOT / "checkpoints/afhq128-step10000.pt", assets["checkpoint"])
        if args.release_assets:
            check_asset(ROOT / "checkpoints/diffusion-afhq-data.zip", assets["dataset"])
        timings = json.loads((ROOT / "results/afhq/comparison/timings.json").read_text())
        if timings["checkpoint_sha256"] != assets["checkpoint"]["sha256"]:
            raise SystemExit("AFHQ comparison checkpoint differs from the release asset")
        if timings["dataset_sha256"] != assets["dataset"]["images_sha256"]:
            raise SystemExit("AFHQ comparison dataset differs from the release asset")
        with (ROOT / "results/afhq/training/train.csv").open() as handle:
            history = list(csv.DictReader(handle))
        if [int(row["step"]) for row in history] != list(range(1, 10001)):
            raise SystemExit("Incomplete AFHQ training history")
    print("PASS: training logs, measured timings, video, weights and README links")


if __name__ == "__main__":
    main()
