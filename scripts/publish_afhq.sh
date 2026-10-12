#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${PYTHON_BIN:-python3}"
"$task_python" scripts/check_submission.py --release-assets
account_name="$(gh api user --jq .login)"
repo_name="cats-ddpm-ddim"
release_tag="submission-v2"
git config --local --replace-all credential.https://github.com.helper ''
git config --local --add credential.https://github.com.helper '!gh auth git-credential'
"$task_python" - "$account_name" <<'PY'
from pathlib import Path
import re
import sys
for path in (Path("README.md"), Path("docs/few-shot.md")):
    text = re.sub(r"https://github\.com/[^/]+/cats-ddpm-ddim/",
                  f"https://github.com/{sys.argv[1]}/cats-ddpm-ddim/", path.read_text())
    path.write_text(text)
PY
git add .
if ! git diff --cached --quiet; then
  git commit -m "Add AFHQv2 training logs, sampling comparison and generation demo"
fi
git push -u origin main
if gh release view "$release_tag" --repo "$account_name/$repo_name" >/dev/null 2>&1; then
  gh release upload "$release_tag" checkpoints/afhq128-step10000.pt checkpoints/diffusion-afhq-data.zip \
    --repo "$account_name/$repo_name" --clobber
else
  gh release create "$release_tag" checkpoints/afhq128-step10000.pt checkpoints/diffusion-afhq-data.zip \
    --repo "$account_name/$repo_name" \
    --title "AFHQv2 cats: EMA checkpoint at 10,000 updates" \
    --notes "78.6M-parameter 128px U-Net trained from scratch on 4559 AFHQv2 cats, 506 held out. Includes EMA inference weights and the reproducible prepared dataset (AFHQv2, CC BY-NC 4.0, source: clovaai/stargan-v2). SHA256 and asset sizes are recorded in results/afhq/assets.json; sampling timings, complete loss logs and MP4 demo are in the repository."
fi
gh api "repos/$account_name/$repo_name/releases/tags/$release_tag" \
  --jq '.assets | map({name, size})' > checkpoints/afhq-uploaded-assets.json
"$task_python" - "$account_name" <<'PY'
from pathlib import Path
import json
import subprocess
import sys
expected = json.loads(Path("results/afhq/assets.json").read_text())
uploaded = {row["name"]: row["size"] for row in json.loads(Path("checkpoints/afhq-uploaded-assets.json").read_text())}
for kind in ("checkpoint", "dataset"):
    asset = expected[kind]
    if uploaded.get(asset["file"]) != asset["bytes"]:
        raise SystemExit(f"Release asset size mismatch: {asset['file']}")
record = {
    "repository": f"https://github.com/{sys.argv[1]}/cats-ddpm-ddim",
    "release": f"https://github.com/{sys.argv[1]}/cats-ddpm-ddim/releases/tag/submission-v2",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "checkpoint_sha256": expected["checkpoint"]["sha256"], "assets": uploaded,
}
Path("checkpoints/afhq-publication.json").write_text(json.dumps(record, indent=2))
PY
printf 'Published AFHQ results: https://github.com/%s/%s\n' "$account_name" "$repo_name"
