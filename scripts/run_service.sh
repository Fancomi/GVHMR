#!/bin/bash
# 启动 GVHMR 内网推理服务 (Flask + 有界队列 + 单 GPU worker)
# 用法: bash scripts/run_service.sh [GPU_ID] [PORT] [MAXSIZE]
#   GPU_ID:  CUDA 设备号 (默认 4, 本机 4-7 空闲)
#   PORT:    监听端口   (默认 8666, 内网可达端口)
#   MAXSIZE: 队列上限   (默认 64, 满则返回 503 busy)
# 注: 绑定 0.0.0.0 以便内网(Mac)经 IP:PORT 访问; 服务无鉴权, 仅限内网。
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # 仓库根目录
ENV_DIR="/root/paddlejob/workspace/env_run/penghaotian/envs/gvhmr"

GPU_ID="${1:-4}"
PORT="${2:-8666}"
MAXSIZE="${3:-64}"

source "$ENV_DIR/bin/activate"
cd "$SCRIPT_DIR"

echo "[run] GPU=$GPU_ID  bind=0.0.0.0:$PORT  maxsize=$MAXSIZE  (内网可达, 不经 frp)"
CUDA_VISIBLE_DEVICES="$GPU_ID" python tools/serve/app.py \
    --host 0.0.0.0 --port "$PORT" --maxsize "$MAXSIZE"
