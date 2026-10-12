"""Import pod logs, benchmark and EMA weights; update the measured results table."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(directory, config_path=None, manifest_path=None,
             comparison_name="submission-comparison", validation_every=1000, updates=10000):
    config_path = config_path or ROOT / "configs/train.yaml"
    manifest_path = manifest_path or ROOT / "data/cats/manifest.json"
    expected = ["train.csv", "validation.csv", "summary.json", "config.yaml", "environment.json",
                "split.json", "events.jsonl", "training-dashboard.png", f"step-{updates:07d}.pt",
                f"{comparison_name}/timings.json", f"{comparison_name}/timings.csv",
                f"{comparison_name}/contact-sheet.png", f"{comparison_name}/speed.png",
                f"{comparison_name}/demo.gif"]
    for name in expected:
        if not (directory / name).is_file():
            raise ValueError(f"Missing pod artifact: {name}")
    summary = json.loads((directory / "summary.json").read_text())
    config = yaml.safe_load((directory / "config.yaml").read_text())
    split = json.loads((directory / "split.json").read_text())
    timings = json.loads((directory / comparison_name / "timings.json").read_text())
    if config != yaml.safe_load(config_path.read_text()):
        raise ValueError("Pod training configuration differs from the submitted configuration")
    if timings["training_config"] != config or timings["dataset_sha256"] != split["sha256"]:
        raise ValueError("Benchmark belongs to a different configuration or dataset")
    manifest = json.loads(manifest_path.read_text())
    if split["sha256"] != manifest["images_sha256"]:
        raise ValueError("Pod dataset differs from the images included in the repository")
    if timings["precision"] != "float32" or timings["ddim_eta"] != 0 or timings["seed"] != 2026:
        raise ValueError("Benchmark precision, DDIM eta or seed differs from the reported setup")
    if (summary["updates"], timings["optimizer_updates"], summary["parameters"]) != (updates, updates, 78642819):
        raise ValueError(f"Expected the 78.6M-parameter checkpoint at {updates:,} optimizer updates")
    if updates == config["training"]["max_updates"] and summary["status"] != "complete":
        raise ValueError("Expected completed training at the configured update limit")
    checkpoint_sha = sha256(directory / f"step-{updates:07d}.pt")
    if checkpoint_sha != timings["checkpoint_sha256"]:
        raise ValueError("Benchmark checkpoint SHA256 does not match the provided weights")
    with (directory / "train.csv").open(newline="") as handle:
        training = list(csv.DictReader(handle))
    with (directory / "validation.csv").open(newline="") as handle:
        validation = list(csv.DictReader(handle))
    if [int(r["step"]) for r in training] != list(range(1, updates + 1)):
        raise ValueError(f"Expected complete training history from updates 1 through {updates:,}")
    if [int(r["step"]) for r in validation] != list(range(validation_every, updates + 1, validation_every)):
        raise ValueError("Validation history does not match the experiment")
    for rows in (training, validation):
        if any(not math.isfinite(float(value)) for row in rows for value in row.values()):
            raise ValueError("Nonfinite values in the imported CSV")
    for name in ("previews", "tensorboard"):
        if not (directory / name).is_dir():
            raise ValueError(f"Missing {name} directory")
    settings = {(row["method"], row["steps"]) for row in timings["results"]}
    if settings != {("ddpm", 1000), *(('ddim', s) for s in (50, 100, 200, 500, 1000))}:
        raise ValueError("Expected DDPM1000 and DDIM50/100/200/500/1000")
    for row in timings["results"]:
        durations = timings["durations"][f"{row['method']}-{row['steps']}"]
        if row["images"] != 8 or row["repeats"] != 3 or len(durations) != 3:
            raise ValueError("Expected batch 8 and three timing repeats")
        if any(not math.isfinite(d) or d <= 0 for d in durations):
            raise ValueError("Invalid generation durations")
        if not math.isclose(row["mean_seconds"], statistics.mean(durations), rel_tol=1e-9):
            raise ValueError("Timing mean differs from its measured repeats")
        if not math.isclose(row["std_seconds"], statistics.stdev(durations), rel_tol=1e-9):
            raise ValueError("Timing standard deviation differs from its measured repeats")
        if not math.isclose(row["speedup_vs_ddpm"], timings["results"][0]["mean_seconds"] / row["mean_seconds"], rel_tol=1e-9):
            raise ValueError("Reported speedup differs from measured generation times")
        if not (directory / comparison_name / f"{row['method']}-{row['steps']}.png").is_file():
            raise ValueError("Missing a benchmark sample grid")
    return timings, checkpoint_sha


def update_readme(timings):
    rows = timings["results"]
    table = ["| Метод | Шаги | Batch 8, с (mean ± std) | Ускорение к DDPM |",
             "|:--|--:|--:|--:|"]
    for row in rows:
        table.append(f"| {row['method'].upper()} | {row['steps']} | "
                     f"{row['mean_seconds']:.3f} ± {row['std_seconds']:.3f} | "
                     f"{row['speedup_vs_ddpm']:.2f}× |")
    hardware = timings["hardware"]
    section = "\n".join([
        "## Сравнение DDPM и DDIM", "",
        f"Один EMA checkpoint шага {timings['optimizer_updates']:,}, "
        f"batch 8, seed {timings['seed']}, FP32, DDIM eta=0. "
        f"Устройство: **{hardware['gpu'] or hardware['device']}**, PyTorch {hardware['torch']}. "
        "Для каждого метода проведён прогрев и три измерения с синхронизацией устройства. "
        "Загрузка весов и запись файлов исключены; сэмплинг на одной GPU.", "",
        *table, "",
        "![Время генерации](results/comparison/speed.png)", "",
        "![DDPM и DDIM при разных бюджетах шагов](results/comparison/contact-sheet.png)", "",
        "[Исходные замеры JSON](results/comparison/timings.json), "
        "[CSV](results/comparison/timings.csv), [все сетки](results/comparison/). "
        "Изображения показываются без отбора. Число вызовов сети равно числу шагов в таблице.", "",
        "```bash",
        "python benchmark.py --checkpoint checkpoints/cats128-step10000.pt --ddim-steps 50 100 200 500 1000 --count 8 --repeats 3 --device cuda --demo --output results/comparison",
        "```", "", ""])
    path = ROOT / "README.md"
    if (ROOT / "results/afhq/assets.json").exists():
        path = ROOT / "docs/few-shot.md"
        section = section.replace("](results/", "](../results/")
    text = path.read_text()
    start = text.index("## Сравнение DDPM и DDIM")
    end = text.index("## Анализ качества и выводы", start)
    text = text[:start] + section + text[end:]
    # Idempotent insertion of the video preview before verification instructions.
    if "![Обратный процесс DDIM]" not in text:
        text = text.replace("## Проверки", "![Обратный процесс DDIM](results/comparison/demo.gif)\n\n## Проверки")
    path.write_text(text, encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cats-pod-results-") as temporary:
        incoming = Path(temporary)
        with zipfile.ZipFile(args.archive) as archive:
            if sum(info.file_size for info in archive.infolist()) > 2 * 1024**3:
                raise ValueError("Unexpected result archive size")
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe path in result archive")
            archive.extractall(incoming)
        timings, checkpoint_sha = validate(incoming)
        comparison = incoming / "submission-comparison"
        if not (comparison / "demo.mp4").is_file():
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise ValueError("GIF imported, but ffmpeg is needed on this computer to create MP4")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(comparison / "demo.gif"),
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(comparison / "demo.mp4")], check=True)
        training = ROOT / "results/training"
        for name in ("train.csv", "validation.csv", "summary.json", "config.yaml", "environment.json",
                     "split.json", "events.jsonl", "training-dashboard.png"):
            shutil.copy2(incoming / name, training / name)
        for name in ("previews", "tensorboard"):
            if not (incoming / name).is_dir():
                raise ValueError(f"Missing {name} directory")
            shutil.copytree(incoming / name, training / name, dirs_exist_ok=True)
        shutil.copytree(comparison, ROOT / "results/comparison", dirs_exist_ok=True)
        weights = ROOT / "checkpoints/cats128-step10000.pt"
        weights.parent.mkdir(exist_ok=True)
        shutil.copy2(incoming / "step-0010000.pt", weights)
        (ROOT / "results/checkpoint.json").write_text(json.dumps({
            "file": "cats128-step10000.pt", "weights": "EMA", "optimizer_updates": 10000,
            "sha256": checkpoint_sha, "dataset_sha256": timings["dataset_sha256"],
            "distribution": "GitHub Release submission-v1"}, indent=2), encoding="utf8")
        update_readme(timings)
        print(f"Imported validated pod results and video to {ROOT}")
        print(f"EMA weights: {weights} (excluded from Git, intended for Release)")


if __name__ == "__main__":
    main()
