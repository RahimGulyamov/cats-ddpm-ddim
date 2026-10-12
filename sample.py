"""Generate a grid using the EMA weights of a checkpoint; optionally make a demo."""
import argparse
from pathlib import Path

import torch
from diffusion.artifacts import device_for, grid, hardware, load_checkpoint, save_demo, write_json


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/afhq128-step10000.pt"))
    parser.add_argument("--method", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--steps", type=positive, default=100)
    parser.add_argument("--count", type=positive, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eta", type=float, default=0.)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sample"))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.method == "ddpm" and args.eta != 0:
        parser.error("--eta is only meaningful for DDIM")
    device = device_for(args.device)
    torch.set_num_threads(4)
    model, process, state = load_checkpoint(args.checkpoint, device)
    size = state["config"]["image_size"]
    args.output.mkdir(parents=True, exist_ok=True)
    images, elapsed, _ = process.sample(model, (args.count, 3, size, size), args.method,
                                        args.steps, args.seed, args.eta)
    grid(images, title=f"{args.method.upper()} | {args.steps} steps | seed {args.seed}").save(args.output / "grid.png")
    write_json(args.output / "generation.json", {"checkpoint": str(args.checkpoint),
               "epoch": state["epoch"], "optimizer_updates": state["updates"],
               "weights": "ema", "method": args.method,
               "steps": args.steps, "count": args.count, "seed": args.seed, "eta": args.eta,
               "seconds": elapsed, "hardware": hardware(device)})
    print(f"{args.method.upper()}: {args.count} images, {args.steps} steps, {elapsed:.3f}s")
    if args.demo:
        # Capturing frames transfers data to CPU; run separately from timing.
        _, _, frames = process.sample(model, (min(4, args.count), 3, size, size), args.method,
                                       args.steps, args.seed, args.eta, record=True)
        save_demo(frames, args.output / "demo.mp4", args.method, args.steps)
