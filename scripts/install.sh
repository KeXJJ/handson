#!/usr/bin/env bash
set -euo pipefail
HANDSON_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HANDSON_REPO_ROOT"
if ! command -v uv >/dev/null 2>&1; then
    echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12.15 .venv
fi
uv pip install --python .venv/bin/python -r requirements-lock.txt
uv pip check --python .venv/bin/python
.venv/bin/python scripts/check_vendor.py
printf 'Installed. Next: source scripts/env.sh\n'
