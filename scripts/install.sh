#!/bin/bash
# Linux uv installation; dependency versions: pyproject.toml + uv.lock.
# Usage: bash scripts/install.sh [none|baidu|aliyun]
# GVHMR_ENV_DIR optionally selects an existing/custom environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_DIR="${GVHMR_ENV_DIR:-$REPO_ROOT/.venv}"
PROXY="${1:-none}"
case "$PROXY" in
    none) ;;
    baidu)
        export https_proxy=http://agent.baidu.com:8188
        export http_proxy="$https_proxy"
        ;;
    aliyun)
        export https_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
        export http_proxy="$https_proxy"
        ;;
    *) echo "Usage: bash scripts/install.sh [none|baidu|aliyun]" >&2; exit 2 ;;
esac
command -v uv >/dev/null 2>&1 || { echo 'Install uv: https://docs.astral.sh/uv/' >&2; exit 1; }
if [ "$(uname -s)" != Linux ]; then
    echo 'On Windows use: powershell -File tools/setup_uv.ps1' >&2
    exit 1
fi
cd "$REPO_ROOT"
export UV_PROJECT_ENVIRONMENT="$ENV_DIR"
echo "[uv] repo=$REPO_ROOT environment=$ENV_DIR"
uv python install 3.10
# Build inputs precede chumpy's legacy no-isolation build. Linux PyTorch3D
# uses the pinned CUDA wheel; Windows uses the local compiler setup.
uv sync --locked --extra compat --extra serve --no-install-package pytorch3d --no-install-package chumpy
uv sync --locked --extra compat --extra serve
TORCHLIB="$ENV_DIR/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCHLIB:${LD_LIBRARY_PATH:-}"
"$ENV_DIR/bin/python" tools/check_env.py
echo '[done] Download model assets with scripts/download_models.sh; start with scripts/run_service.sh.'
