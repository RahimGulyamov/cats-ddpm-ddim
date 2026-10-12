"""Import the measured AFHQ experiment without replacing the few-shot baseline."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from import_results import ROOT, sha256, validate


def extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        if sum(info.file_size for info in source.infolist()) > 2 * 1024**3:
            raise ValueError("Unexpected archive size")
        for info in source.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe archive path")
        source.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=50000)
    parser.add_argument("--release-tag", default="submission-v3")
    args = parser.parse_args()
    if args.updates < 1 or args.updates % 2000:
        parser.error("--updates must be a positive multiple of the validation interval (2000)")
    comparison_name = f"afhq-comparison-{args.updates}"
    with tempfile.TemporaryDirectory(prefix="afhq-results-") as temporary:
        incoming = Path(temporary)
        extract(args.archive, incoming)
        timings, checkpoint_sha = validate(
            incoming, ROOT / "configs/afhq128-gpu.yaml",
            ROOT / "results/afhq/dataset-manifest.json", comparison_name, 2000, args.updates)
        manifest = json.loads((ROOT / "results/afhq/dataset-manifest.json").read_text())
        split = json.loads((incoming / "split.json").read_text())
        if (split["total"], split["training"], split["validation"]) != (5065, 4559, 506):
            raise ValueError("Unexpected AFHQ train/validation split")
        with zipfile.ZipFile(args.data_archive) as source:
            included = json.loads(source.read("data/afhq-cat/manifest.json"))
            if included != manifest:
                raise ValueError("Prepared dataset manifest differs from the training dataset")
            digest = hashlib.sha256()
            for record in manifest["records"]:
                image_sha = hashlib.sha256(source.read("data/afhq-cat/" + record["file"])).digest()
                if image_sha.hex() != record["sha256"]:
                    raise ValueError("Prepared image checksum mismatch")
                digest.update(image_sha)
            if digest.hexdigest() != split["sha256"]:
                raise ValueError("Prepared dataset checksum mismatch")
        comparison = incoming / comparison_name
        if not (comparison / "demo.mp4").exists():
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise ValueError("ffmpeg is needed to encode the recorded GIF as MP4")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(comparison / "demo.gif"),
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(comparison / "demo.mp4")], check=True)
        output = ROOT / "results/afhq"
        training = output / "training"
        training.mkdir(parents=True, exist_ok=True)
        for name in ("train.csv", "validation.csv", "summary.json", "config.yaml", "environment.json",
                     "split.json", "events.jsonl", "training-dashboard.png"):
            shutil.copy2(incoming / name, training / name)
        for name in ("previews", "tensorboard"):
            if (training / name).exists():
                shutil.rmtree(training / name)
            shutil.copytree(incoming / name, training / name)
        if (output / "comparison").exists():
            shutil.rmtree(output / "comparison")
        shutil.copytree(comparison, output / "comparison")
        weights = ROOT / f"checkpoints/afhq128-step{args.updates}.pt"
        weights.parent.mkdir(exist_ok=True)
        shutil.copy2(incoming / f"step-{args.updates:07d}.pt", weights)
        data_asset = weights.parent / "diffusion-afhq-data.zip"
        shutil.copy2(args.data_archive, data_asset)
        assets = {
            "checkpoint": {"file": weights.name, "sha256": checkpoint_sha, "bytes": weights.stat().st_size,
                           "optimizer_updates": args.updates, "weights": "EMA", "dataset_sha256": split["sha256"]},
            "dataset": {"file": data_asset.name, "sha256": sha256(data_asset), "bytes": data_asset.stat().st_size,
                        "images": manifest["count"], "images_sha256": split["sha256"]},
            "release_tag": args.release_tag,
            "distribution": f"GitHub Release {args.release_tag}",
        }
        (output / "assets.json").write_text(json.dumps(assets, indent=2))
        with (training / "train.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        with (training / "validation.csv").open() as handle:
            validation = list(csv.DictReader(handle))
        analysis = {
            "optimizer_updates": len(rows), "train_mse_last": float(rows[-1]["train_mse"]),
            "train_mse_last1000_mean": sum(float(row["train_mse"]) for row in rows[-1000:]) / 1000,
            "examples_seen": int(rows[-1]["examples_seen"]),
            "epoch_equivalent": float(rows[-1]["epoch_equivalent"]),
            "validation_curve": [{"step": int(row["step"]), "noise_mse": float(row["val_noise_mse"])}
                                 for row in validation],
            "note": "Train: raw weights/random t; validation: EMA/fixed t. MSE values across datasets are not directly comparable.",
        }
        (training / "analysis.json").write_text(json.dumps(analysis, indent=2))
        subprocess.run([sys.executable, str(ROOT / "scripts/plot_results.py"), "--folder", str(training)], check=True)
        print(f"Imported AFHQ weights, validated dataset, measured comparison and demo to {output}")


if __name__ == "__main__":
    main()
