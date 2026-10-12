#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
repo_name="cats-ddpm-ddim"
release_tag="submission-v1"
task_python="${PYTHON_BIN:-python3}"
"$task_python" scripts/check_submission.py
account_name="$(gh api user --jq .login)"

# Use the authenticated GitHub CLI for HTTPS, only in this repository.
git config --local --replace-all credential.https://github.com.helper ''
git config --local --add credential.https://github.com.helper '!gh auth git-credential'

# Resolve the account before committing the public download URLs.
"$task_python" - "$account_name" "$repo_name" <<'PY'
from pathlib import Path
import re
import sys
path = Path("README.md")
text = re.sub(r"https://github\.com/[^/]+/cats-ddpm-ddim/", f"https://github.com/{sys.argv[1]}/{sys.argv[2]}/", path.read_text())
path.write_text(text)
PY

git add .
if ! git diff --cached --quiet; then
  git commit -m "Add diffusion training, sampling experiments and results"
fi

if git remote get-url origin >/dev/null 2>&1; then
  gh repo view "$account_name/$repo_name" --json nameWithOwner >/dev/null
  git push -u origin main
else
  gh repo create "$account_name/$repo_name" --public --source=. --remote=origin --push \
    --description "DDPM training on cat images, DDIM sampling, reproducible logs and generation demo"
fi

if gh release view "$release_tag" --repo "$account_name/$repo_name" >/dev/null 2>&1; then
  gh release upload "$release_tag" checkpoints/cats128-step10000.pt --repo "$account_name/$repo_name" --clobber
else
  gh release create "$release_tag" checkpoints/cats128-step10000.pt \
    --repo "$account_name/$repo_name" \
    --title "Cats diffusion: EMA checkpoint at 10,000 updates" \
    --notes "EMA weights for the 128px U-Net used in the README sampling comparison. SHA256 is recorded in results/checkpoint.json."
fi

asset_bytes="$(gh api "repos/$account_name/$repo_name/releases/tags/$release_tag" \
  --jq '.assets[] | select(.name == "cats128-step10000.pt") | .size')"
"$task_python" - "$account_name" "$repo_name" "$asset_bytes" <<'PY'
from pathlib import Path
import json
import subprocess
import sys
weights = Path("checkpoints/cats128-step10000.pt")
if int(sys.argv[3]) != weights.stat().st_size:
    raise SystemExit("Release asset size does not match the local checkpoint")
metadata = json.loads(Path("results/checkpoint.json").read_text())
record = {
    "repository": f"https://github.com/{sys.argv[1]}/{sys.argv[2]}",
    "release": f"https://github.com/{sys.argv[1]}/{sys.argv[2]}/releases/tag/submission-v1",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "checkpoint_sha256": metadata["sha256"],
    "asset_bytes": int(sys.argv[3]),
}
Path("checkpoints/publication.json").write_text(json.dumps(record, indent=2))
PY

printf 'Published: https://github.com/%s/%s\n' "$account_name" "$repo_name"
