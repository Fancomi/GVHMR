#!/bin/bash
# 在 gvhmr 环境补装 GVHMR 推理服务的额外依赖 (仅 flask, 其余推理依赖已由 install.sh 装好)
# 用法: bash scripts/install_service.sh [proxy]
#   proxy: baidu (默认) | aliyun
set -e

ENV_DIR="/root/paddlejob/workspace/env_run/penghaotian/envs/gvhmr"
PROXY="${1:-baidu}"

if [ "$PROXY" = "aliyun" ]; then
    export https_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
    export http_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
    PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"
else
    export https_proxy=http://agent.baidu.com:8188
    export http_proxy=http://agent.baidu.com:8188
    PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple/"
fi
echo "[proxy] $PROXY  PIP_INDEX=$PIP_INDEX"

PYTHON="$ENV_DIR/bin/python"
echo "[install] flask 到 $ENV_DIR"
uv pip install --python "$PYTHON" --link-mode=copy -i "$PIP_INDEX" "flask>=3.0,<4.0"

echo
echo "[done] 服务依赖就绪. 启动:"
echo "  bash scripts/run_service.sh"
