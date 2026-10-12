"""Pair each generated preview tile with its closest train image (RGB MSE).

This is a visual memorization diagnostic, not FID or a calibrated copy detector.
Only grids produced by diffusion.artifacts.grid with title and scale=2 are read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.artifacts import grid
from diffusion.data import load_images


def read_preview(path, size, count, columns=4):
    columns = min(columns, count)
    tile, gap, margin, header = size * 2, 6, 12, 38
    rows = (count + columns - 1) // columns
    expected = (columns * (tile + gap) + 2 * margin - gap,
                rows * (tile + gap) + header + margin - gap)
    tiles = []
    with Image.open(path) as image:
        if image.size != expected:
            raise ValueError(f"Expected native titled grid {expected}, got {image.size}")
        for index in range(count):
            x = margin + index % columns * (tile + gap)
            y = header + index // columns * (tile + gap)
            pixels = np.asarray(image.convert("RGB").crop((x, y, x + tile, y + tile))
                                .resize((size, size), Image.Resampling.NEAREST), dtype=np.float32)
            tiles.append(torch.from_numpy(pixels.copy()).permute(2, 0, 1) / 127.5 - 1)
    return torch.stack(tiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cats128-gpu.yaml"))
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    torch.set_num_threads(4)
    config = yaml.safe_load(args.config.read_text())
    training, _, split = load_images(config["data_dir"], config["image_size"], config["seed"],
                                     config.get("val_fraction", 0.1))
    generated = read_preview(args.grid, config["image_size"], args.count)
    # Chunk the train bank and use squared-distance matrix products to avoid
    # materializing [generated, train, channels, height, width] differences.
    queries = generated.flatten(1)
    query_norm = queries.square().sum(1, keepdim=True)
    best_distance = torch.full((len(generated),), float("inf"))
    best_neighbor = torch.zeros(len(generated), dtype=torch.long)
    for flipped in (False, True):
        for start in range(0, len(training), 128):
            block = training[start:start + 128]
            if flipped:
                block = block.flip(-1)
            flat = block.flatten(1)
            distance = (query_norm + flat.square().sum(1)[None] - 2 * queries @ flat.T)
            distance = (distance / queries.shape[1]).clamp_min(0)
            values, indices = distance.min(1)
            improved = values < best_distance
            best_distance[improved] = values[improved]
            best_neighbor[improved] = indices[improved] + start + int(flipped) * len(training)
    pairs, rows = [], []
    for index, image in enumerate(generated):
        neighbor = int(best_neighbor[index])
        nearest = training[neighbor % len(training)]
        if neighbor >= len(training):
            nearest = nearest.flip(-1)
        pairs.extend((image, nearest))
        rows.append({"tile": index + 1, "train_file": split["train_files"][neighbor % len(training)],
                     "horizontal_flip": neighbor >= len(training),
                     "rgb_mse": float((image - nearest).square().mean())})
    args.output.mkdir(parents=True, exist_ok=True)
    grid(torch.stack(pairs), columns=4, title="Generated / nearest TRAIN (including flips) | RGB MSE diagnostic")\
        .save(args.output / "neighbors.png")
    report = {"source_grid": str(args.grid), "dataset_sha256": split["sha256"],
              "train_count": len(training), "generated_count": len(generated),
              "normalization": "RGB in [-1,1]; native preview quantized to uint8",
              "median_nearest_train_rgb_mse": float(np.median([r["rgb_mse"] for r in rows])),
              "limitations": "Pixel proximity supports visual inspection; no copy threshold or FID. "
                             "Colors and alignment affect distances. Both original and flipped train images checked.",
              "pairs": rows}
    (args.output / "neighbors.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(f"Saved {len(rows)} pairs to {args.output}")


if __name__ == "__main__":
    main()
