"""Test step-based checkpoint/resume on CPU including dropout and accumulation."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def run(*args):
    subprocess.run([sys.executable, "train_distributed.py", "--config", "configs/ddp-smoke.yaml", *args], check=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="quality-resume-") as directory:
        direct, resumed = Path(directory) / "direct", Path(directory) / "resumed"
        run("--output", str(direct))
        run("--output", str(resumed), "--stop-after-updates", "3")
        run("--output", str(resumed), "--resume", str(resumed / "last.pt"))
        first = torch.load(direct / "last.pt", map_location="cpu", weights_only=True)
        second = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        for name in ("model", "ema"):
            for key in first[name]:
                torch.testing.assert_close(first[name][key], second[name][key], atol=0, rtol=0)
        assert first["updates"] == second["updates"]
        assert first["stream"] == second["stream"]
        assert first["validation_history"] == second["validation_history"]
        assert [r["train_mse"] for r in first["train_history"]] == [r["train_mse"] for r in second["train_history"]]
        print(json.dumps({"resume": "PASS", "weights": "exact match", "updates": second["updates"],
                          "checkpointed_dropout": True, "gradient_accumulation": True}))
