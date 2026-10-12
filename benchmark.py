"""Compare full ancestral DDPM with DDIM step budgets on identical initial noise."""
import argparse
import hashlib
import statistics
from pathlib import Path

import torch
from diffusion.artifacts import (
    device_for,
    grid,
    hardware,
    load_checkpoint,
    save_demo,
    write_csv,
    write_json,
)
from PIL import Image
from sample import positive

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/afhq128-step10000.pt"))
    parser.add_argument("--ddim-steps", nargs="+", type=positive, default=[50, 100, 200, 500, 1000])
    parser.add_argument("--count", type=positive, default=8)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparison"))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    device = device_for(args.device)
    torch.set_num_threads(4)
    model, process, state = load_checkpoint(args.checkpoint, device)
    if any(steps > process.timesteps for steps in args.ddim_steps):
        parser.error("DDIM steps cannot exceed training timesteps")
    args.output.mkdir(parents=True, exist_ok=True)
    size = state["config"]["image_size"]
    shape = (args.count, 3, size, size)
    rows, times, tiles = [], {}, []
    settings = [("ddpm", process.timesteps)] + [("ddim", s) for s in args.ddim_steps]
    for method, steps in settings:
        # Warm up kernels with the same batch shape; warmup is not measured.
        process.sample(model, shape, "ddim", 2, args.seed)
        durations = []
        for repeat in range(args.repeats):
            images, duration, _ = process.sample(model, shape, method, steps, args.seed)
            durations.append(duration)
            print(f"{method.upper()} {steps}: repeat {repeat + 1}/{args.repeats}, {duration:.3f}s", flush=True)
        tile = grid(images, title=f"{method.upper()} | {steps} steps")
        tile.save(args.output / f"{method}-{steps}.png")
        tiles.append(tile)
        mean = statistics.mean(durations)
        rows.append({"method": method, "steps": steps, "images": args.count,
                     "repeats": args.repeats, "mean_seconds": mean,
                     "std_seconds": statistics.stdev(durations) if len(durations) > 1 else 0.,
                     "seconds_per_image": mean / args.count,
                     "speedup_vs_ddpm": rows[0]["mean_seconds"] / mean if rows else 1.})
        times[f"{method}-{steps}"] = durations
    columns = 3
    panel = Image.new("RGB", (columns * tiles[0].width,
                              ((len(tiles) + columns - 1) // columns) * tiles[0].height), "#f2eee6")
    for i, tile in enumerate(tiles):
        panel.paste(tile, ((i % columns) * tile.width, (i // columns) * tile.height))
    panel.save(args.output / "contact-sheet.png")
    write_csv(args.output / "timings.csv", rows)
    write_json(args.output / "timings.json", {"checkpoint": str(args.checkpoint),
               "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
               "epoch": state["epoch"], "optimizer_updates": state["updates"],
               "seed": args.seed, "weights": "ema", "precision": "float32",
               "training_config": state["config"], "dataset_sha256": state["dataset"]["sha256"],
               "resolution": size, "hardware": hardware(device), "ddim_eta": 0.,
               "measurement": "Wall time after warmup; synchronized device; identical initial CPU noise; "
                              "includes model calls, scheduler and DDPM reverse-noise transfers; "
                              "excludes checkpoint loading, final CPU transfer and artifact encoding.",
               "durations": times, "results": rows})
    lines = ["| Метод | Шаги | Время batch, с (mean ± std) | Ускорение |",
             "|:--|--:|--:|--:|"]
    for row in rows:
        lines.append(f"| {row['method'].upper()} | {row['steps']} | "
                     f"{row['mean_seconds']:.3f} ± {row['std_seconds']:.3f} | "
                     f"{row['speedup_vs_ddpm']:.2f}× |")
    (args.output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"figure.facecolor": "#f2eee6", "axes.facecolor": "#f2eee6"}):
        figure, axis = plt.subplots(figsize=(8, 4.2))
        axis.bar([f"{r['method'].upper()} {r['steps']}" for r in rows],
                 [r["mean_seconds"] for r in rows],
                 yerr=[r["std_seconds"] for r in rows], color="#247b7b", capsize=4)
        axis.set(ylabel="Batch generation time (seconds)", title=f"EMA | {args.count} images | {size}px | {args.repeats} repeats")
        axis.tick_params(axis="x", labelrotation=20)
        figure.tight_layout()
        figure.savefig(args.output / "speed.png", dpi=150)
        plt.close(figure)
    if args.demo:
        demo_steps = min(200, process.timesteps)
        _, _, frames = process.sample(model, (min(args.count, 4), 3, size, size),
                                       "ddim", demo_steps, args.seed, record=True)
        save_demo(frames, args.output / "demo.mp4", "ddim", demo_steps)
