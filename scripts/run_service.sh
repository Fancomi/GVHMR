#!/bin/bash
# 启动 GVHMR 内网单图推理服务 (Flask + 有界队列 + 单 GPU worker)
# 自包含: 仅依赖本仓库自身, 不引用任何外部仓库。
#
# 用法: bash scripts/run_service.sh [GPU_ID] [PORT] [MAXSIZE]
#   GPU_ID:  CUDA 设备号 (默认 0, 使用单卡)
#   PORT:    监听端口    (默认 8666)
#   MAXSIZE: 队列上限    (默认 64, 满则返回 503 busy)
#
# 可选环境变量:
#   GVHMR_ENV_DIR  虚拟环境路径 (默认见下方; 与 install.sh 保持一致, 换机器请改)
#
# 注: 绑定 0.0.0.0 以便内网经 IP:PORT 访问; 服务无鉴权, 仅限内网。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# 默认指向本机现有环境; 别人换机器改这里或设 GVHMR_ENV_DIR。
ENV_DIR="${GVHMR_ENV_DIR:-/root/paddlejob/workspace/env_run/penghaotian/envs/gvhmr}"

GPU_ID="${1:-0}"
PORT="${2:-8666}"
MAXSIZE="${3:-64}"

ACTIVATE="$ENV_DIR/bin/activate"
if [ ! -f "$ACTIVATE" ]; then
    echo "[ERROR] 虚拟环境不存在: $ENV_DIR"
    echo "        先运行 bash scripts/install.sh, 或设置 GVHMR_ENV_DIR 指向已有环境:"
    echo "        GVHMR_ENV_DIR=/path/to/env bash scripts/run_service.sh"
    exit 1
fi

source "$ACTIVATE"
cd "$REPO_ROOT"

echo "[run] GPU=$GPU_ID  bind=0.0.0.0:$PORT  maxsize=$MAXSIZE  ENV_DIR=$ENV_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" python tools/serve/app.py \
    --host 0.0.0.0 --port "$PORT" --maxsize "$MAXSIZE"
