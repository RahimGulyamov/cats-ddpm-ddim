"""Download the course dataset, or extract a previously downloaded Arrow cache."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion.data import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/cats"))
    parser.add_argument("--arrow-cache", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.arrow_cache)
