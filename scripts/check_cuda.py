"""Execute representative training kernels on each GPU, without downloads or DDP.

This standalone file only needs the pod's installed PyTorch. It checks runtime
execution, rather than inferring compatibility from cuda.is_available().
"""
import argparse
import copy
import json
import time

import torch
from torch import nn
from torch.nn import functional as F


def probe(device, precision="bf16"):
    """Conv/GroupNorm, attention, backward, AdamW and EMA in one small update."""
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    enabled = precision == "bf16"
    model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32),
                          nn.SiLU(), nn.Dropout(0.1), nn.Conv2d(32, 3, 3, padding=1)).to(device)
    before = model[-1].weight.detach().clone()
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    clean = torch.randn(2, 3, 16, 16, device=device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        predicted = model(clean)
        q = torch.randn(2, 4, 64, 16, device=device, requires_grad=True)
        # Linear/matmul also exercises the autocast path used by timestep MLP.
        projected = q @ torch.eye(16, device=device)
        attended = F.scaled_dot_product_attention(projected, projected, projected)
        loss = (predicted.float() - clean).square().mean() + attended.float().square().mean()
    loss.backward()
    tensors = [loss, q.grad, *(p.grad for p in model.parameters())]
    if any(t is None or not torch.isfinite(t).all() for t in tensors):
        raise RuntimeError("Nonfinite loss or gradient in runtime probe")
    optimizer.step()
    with torch.no_grad():
        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 0.1)
    if torch.equal(before, model[-1].weight):
        raise RuntimeError("Optimizer did not update weights")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"loss": float(loss.detach().cpu()), "precision": precision,
            "operations": "conv, GroupNorm, matmul, SDPA, backward, AdamW foreach, EMA"}


def check_gpus(required=4):
    result = {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "required_gpus": required, "visible_gpus": torch.cuda.device_count(),
              "nccl_available": torch.distributed.is_nccl_available(), "gpus": [], "errors": []}
    if not torch.cuda.is_available() or result["visible_gpus"] < required:
        result["errors"].append(f"Need {required} functional CUDA GPUs; visible={result['visible_gpus']}")
    for index in range(min(required, result["visible_gpus"])):
        tick = time.perf_counter()
        record = {"index": index}
        try:
            torch.cuda.set_device(index)
            device = torch.device("cuda", index)
            record["name"] = torch.cuda.get_device_name(index)
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 unavailable on this GPU")
            record.update(probe(device))
            record["status"] = "PASS"
        except Exception as error:
            record["status"] = "FAIL"
            record["error"] = f"{type(error).__name__}: {error}"
            result["errors"].append(f"GPU {index}: {record['error']}")
        record["seconds"] = round(time.perf_counter() - tick, 3)
        result["gpus"].append(record)
    result["status"] = "FAIL" if result["errors"] else "PASS"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpus", type=int, default=4)
    args = parser.parse_args()
    if args.require_gpus < 1:
        parser.error("--require-gpus must be positive")
    result = check_gpus(args.require_gpus)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["errors"]))
