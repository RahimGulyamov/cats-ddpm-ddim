"""RGB image loading with a deterministic holdout, shared across experiments."""
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

DATASET_URL = "https://huggingface.co/datasets/huggan/few-shot-cat"
PARQUET_URL = DATASET_URL + "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"


def prepare(destination, arrow_cache=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if all((destination / name).is_file() for name in manifest["files"]):
            print(f"Dataset already prepared: {len(manifest['files'])} images")
            return
    if any(destination.glob("*.png")):
        raise ValueError("Incomplete image folder: use a new destination")
    if arrow_cache:
        paths = sorted(Path(arrow_cache).rglob("few-shot-cat-train.arrow"))
        if len(paths) != 1:
            raise ValueError("Expected exactly one few-shot-cat-train.arrow in --arrow-cache")
        raw = paths[0].read_bytes()
        table = pa.ipc.open_stream(pa.BufferReader(raw)).read_all()
        origin = "local Hugging Face Arrow cache"
    else:
        try:
            with urllib.request.urlopen(PARQUET_URL, timeout=90) as response:
                raw = response.read()
        except OSError as error:
            raise RuntimeError("Dataset download failed. Check internet or use --arrow-cache PATH") from error
        table = pq.read_table(pa.BufferReader(raw))
        origin = PARQUET_URL
    records, digest = [], hashlib.sha256()
    for index, record in enumerate(table["image"].to_pylist()):
        payload = record.get("bytes")
        if payload is None:
            raise ValueError("Dataset row has no embedded image bytes")
        name = f"cat_{index:04d}.png"
        with Image.open(io.BytesIO(payload)) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(destination / name)
        digest.update(hashlib.sha256((destination / name).read_bytes()).digest())
        records.append(name)
    if len(records) < 2:
        raise ValueError("Need at least two images for training/validation")
    manifest = {"dataset": "huggan/few-shot-cat", "url": DATASET_URL,
                "origin": origin, "count": len(records), "files": records,
                "archive_sha256": hashlib.sha256(raw).hexdigest(),
                "images_sha256": digest.hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(f"Prepared {len(records)} images in {destination}")


def load_images(directory, size, seed, val_fraction=0.1):
    files = sorted(p for p in Path(directory).rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if len(files) < 2:
        raise ValueError("No training images. Run python scripts/prepare_data.py first")
    images, digest = [], hashlib.sha256()
    for path in files:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        with Image.open(path) as image:
            image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"),
                                 (size, size), method=Image.Resampling.LANCZOS)
            pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1
            images.append(torch.from_numpy(pixels.copy()).permute(2, 0, 1))
    images = torch.stack(images)
    order = torch.randperm(len(files), generator=torch.Generator().manual_seed(seed))
    heldout = max(1, min(len(files) - 1, round(len(files) * val_fraction)))
    validation, training = order[:heldout], order[heldout:]
    metadata = {"total": len(files), "training": len(training), "validation": heldout,
                "sha256": digest.hexdigest(),
                "train_files": [str(files[i].relative_to(directory)) for i in training.tolist()],
                "val_files": [str(files[i].relative_to(directory)) for i in validation.tolist()]}
    return images[training], images[validation], metadata
