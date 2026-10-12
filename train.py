"""Train an epsilon-prediction U-Net on one device or with torchrun/DDP."""
import argparse
from pathlib import Path

import yaml
from train_distributed import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.output:
        config["output_dir"] = str(args.output)
    if args.device:
        config["device"] = args.device
    run(config, args.resume, args.stop_after_updates)
