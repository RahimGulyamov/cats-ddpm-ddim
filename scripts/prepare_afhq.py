"""On the online Mac: prepare AFHQv2 train cats and package data for the offline pod.

Uses only Pillow and the standard library. Training code has no new dependencies.
Official val cats are excluded; existing trainer holds out 10% of official train cats.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
SOURCE = "https://github.com/clovaai/stargan-v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading official AFHQv2 archive to {path} (all animal classes)", flush=True)
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "diffusion-lab/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            total, reported = 0, 0
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                output.write(block)
                total += len(block)
                if total - reported >= 64 * 1024 * 1024:
                    print(f"Downloaded {total / 1024**2:.0f} MiB", flush=True)
                    reported = total
        if not zipfile.is_zipfile(partial):
            raise ValueError("Download is not a ZIP. Download the official archive in a browser and use --archive PATH.")
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def prepare(archive, destination, size):
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}. Use a fresh --output or reuse an existing bundle.")
    # Never extract archive paths directly, and exclude metadata files and other domains.
    with zipfile.ZipFile(archive) as source:
        selected = []
        for info in source.infolist():
            parts = PurePosixPath(info.filename).parts
            if "__MACOSX" in parts or info.is_dir():
                continue
            if len(parts) >= 3 and parts[-3:-1] == ("train", "cat") and \
                    PurePosixPath(info.filename).suffix.lower() in {".png", ".jpg", ".jpeg"}:
                selected.append(info)
        selected.sort(key=lambda item: item.filename)
        if len(selected) < 2:
            raise ValueError("Archive has no usable train/cat image set; expected official AFHQv2 ZIP structure")
        destination.mkdir(parents=True)
        records, duplicates, seen = [], [], set()
        for index, info in enumerate(selected):
            with source.open(info) as handle, Image.open(handle) as image:
                image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (size, size),
                                     method=Image.Resampling.LANCZOS)
                pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
                if pixel_sha in seen:
                    duplicates.append(info.filename)
                    continue
                seen.add(pixel_sha)
                name = f"cat_{len(records):05d}.png"
                image.save(destination / name)
            records.append({"file": name, "source_member": info.filename,
                            "sha256": sha256(destination / name)})
            if (index + 1) % 500 == 0:
                print(f"Prepared {index + 1}/{len(selected)} source images", flush=True)
    if len(records) < 2:
        raise ValueError("Too few unique cats")
    digest = hashlib.sha256()
    for row in records:
        digest.update(bytes.fromhex(row["sha256"]))
    manifest = {"dataset": "AFHQv2, official train/cat", "url": SOURCE, "download_url": URL,
                "license": "CC BY-NC 4.0", "archive_sha256": sha256(archive),
                "image_size": size, "source_count": len(selected), "count": len(records),
                "images_sha256": digest.hexdigest(), "duplicates_excluded": duplicates,
                "split_policy": "seed 2026: 10% of prepared official train cats held out by trainer; "
                                "official val and other classes are excluded",
                "files": [r["file"] for r in records], "records": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "downloads/afhq_v2.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "data/afhq-cat")
    parser.add_argument("--bundle", type=Path, default=ROOT.parent / "diffusion-afhq-data.zip")
    args = parser.parse_args()
    if args.bundle.exists():
        parser.error("Bundle already exists; choose another --bundle to preserve it")
    if not args.archive.exists():
        download(args.archive)
    manifest = prepare(args.archive, args.output, 128)
    config = ROOT / "configs/afhq128-gpu.yaml"
    args.bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.bundle.with_suffix(".zip.part")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as target:
            for path in sorted(args.output.iterdir()):
                target.write(path, "data/afhq-cat/" + path.name)
            target.write(config, "configs/afhq128-gpu.yaml")
        with zipfile.ZipFile(temporary) as check:
            bad = check.testzip()
            if bad:
                raise ValueError(f"Bundle CRC failed for {bad}")
        os.replace(temporary, args.bundle)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"prepared_cats": manifest["count"], "dataset_sha256": manifest["images_sha256"],
                      "bundle": str(args.bundle), "bundle_sha256": sha256(args.bundle),
                      "bundle_mib": round(args.bundle.stat().st_size / 1024**2, 1)}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
