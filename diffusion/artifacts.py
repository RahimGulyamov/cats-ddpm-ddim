"""Portable artifact helpers; visuals are created from actual model tensors."""
import csv
import json
import math
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .factory import create_model
from .process import Diffusion


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    return torch.device(name)


def hardware(device):
    return {"device": str(device), "processor": platform.processor(), "system": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__), "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model(state["config"]["model"])
    model.load_state_dict(state["ema"])
    process = Diffusion(**state["config"]["diffusion"], device=device)
    return model.to(device).eval(), process, state


def grid(images, columns=4, scale=2, title=""):
    pixels = ((images.detach().cpu().float().clamp(-1, 1) + 1) * 127.5).round().byte()
    pixels = pixels.permute(0, 2, 3, 1).numpy()
    columns = min(columns, len(pixels))
    height, width = pixels.shape[1:3]
    header, margin, gap = (38 if title else 12), 12, 6
    canvas = Image.new("RGB", (columns * (width * scale + gap) + margin * 2 - gap,
                               math.ceil(len(pixels) / columns) * (height * scale + gap) + header + margin - gap),
                       "#f2eee6")
    if title:
        ImageDraw.Draw(canvas).text((margin, 12), title, fill="#273c40")
    for i, pixel in enumerate(pixels):
        tile = Image.fromarray(pixel).resize((width * scale, height * scale), Image.Resampling.NEAREST)
        canvas.paste(tile, (margin + (i % columns) * (width * scale + gap),
                            header + (i // columns) * (height * scale + gap)))
    return canvas


def loss_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"figure.facecolor": "#f2eee6", "axes.facecolor": "#f2eee6"}):
        figure, axis = plt.subplots(figsize=(8, 4.2))
        axis.plot([r["epoch"] for r in rows], [r["train_mse"] for r in rows], color="#247b7b", label="Train MSE (raw)")
        axis.plot([r["epoch"] for r in rows], [r["val_mse"] for r in rows], color="#c16b42", label="Holdout MSE (EMA)")
        axis.set(xlabel="Epoch", ylabel="Noise prediction MSE", yscale="log")
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)


def save_demo(frames, path, method, steps):
    """Write a 12s accelerated denoising video and a GIF without extra Python deps."""
    pictures = [grid(tensor, title=f"{method.upper()} | reverse step {t} | {steps} evaluations")
                for t, tensor in frames]
    pictures[0].save(Path(path).with_suffix(".gif"), save_all=True,
                     append_images=pictures[1:] + [pictures[-1]] * 10,
                     duration=max(40, 10000 // len(pictures)), loop=0)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            pass
    if ffmpeg is None:
        print("GIF created; install ffmpeg to also create MP4", flush=True)
        return False
    width, height = pictures[0].size
    command = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", "10", "-i", "-", "-an", "-c:v", "libx264",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for i in range(120):
            index = min(len(pictures) - 1, round(i / 99 * (len(pictures) - 1)))
            encoder.stdin.write(np.asarray(pictures[index]).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg video encoding failed")
    return True
