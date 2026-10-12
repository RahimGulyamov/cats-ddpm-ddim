"""Check the offline runtime, model shape and GPUs before torchrun."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--require-gpus", type=int, default=4)
    args = parser.parse_args()
    import torch
    import yaml
    from diffusion.data import load_images
    from diffusion.factory import create_model
    from train_distributed import validate
    config = yaml.safe_load(args.config.read_text())
    validate(config)
    errors = []
    versions = {}
    for name in ("numpy", "Pillow", "matplotlib", "PyYAML", "tensorboard"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing {name}")
    cuda_probe = None
    if args.require_gpus and torch.cuda.device_count() >= args.require_gpus:
        from scripts.check_cuda import check_gpus
        cuda_probe = check_gpus(args.require_gpus)
        errors.extend(cuda_probe["errors"])
    if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 4):
        errors.append("Torch >=2.4 is required; this code was tested with 2.9.0")
    if torch.cuda.device_count() < args.require_gpus:
        errors.append(f"Need {args.require_gpus} CUDA GPUs, found {torch.cuda.device_count()}")
    if args.require_gpus and not torch.distributed.is_nccl_available():
        errors.append("Torch has no NCCL backend")
    if args.require_gpus and config["training"]["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        errors.append("BF16 is unavailable")
    training, validation, dataset = load_images(config["data_dir"], config["image_size"], config["seed"], config["val_fraction"])
    with torch.device("meta"):
        model = create_model(config["model"])
    result = {"python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "dependencies": versions, "parameters": sum(p.numel() for p in model.parameters()),
              "train_images": len(training), "validation_images": len(validation),
              "dataset_sha256": dataset["sha256"], "errors": errors,
              "cuda_probe": cuda_probe,
              "note": "No download; primary model built on meta; small real kernel updates on each GPU when requested. NCCL collectives are checked by the DDP smoke run."}
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if errors else 0)
