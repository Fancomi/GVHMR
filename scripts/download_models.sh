#!/bin/bash
# 准备 GVHMR demo 所需 checkpoints
# 策略: 能软链复用 DuoMo 本地权重的就软链, GVHMR/HMR2 专有 ckpt 从 HuggingFace (camenduru/GVHMR) 下载
# 已存在则跳过. 关闭 SLAM/DPVO, 不下载 dpvo.pth
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # 仓库根目录
CKPT="$SCRIPT_DIR/inputs/checkpoints"
DUOMO="/root/paddlejob/workspace/env_run/penghaotian/sport_project/DuoMo"

HF_REPO="camenduru/GVHMR"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"

# aria2c 用于多线程加速 HF 大权重下载 (hmr2 ckpt 达 2.5G)
if ! command -v aria2c >/dev/null 2>&1; then
    echo "[ERROR] 未找到 aria2c, 请先安装 (apt install aria2 / conda install aria2)"; exit 1
fi

mkdir -p "$CKPT/gvhmr" "$CKPT/hmr2" "$CKPT/vitpose" "$CKPT/yolo" \
         "$CKPT/body_models/smpl" "$CKPT/body_models/smplx"

# ---- 工具函数 ----
link_if_missing() {  # $1=src $2=dst
    if [ -e "$2" ] || [ -L "$2" ]; then
        echo "[skip] $2 已存在"
    elif [ -e "$1" ]; then
        ln -sf "$1" "$2"
        echo "[link] $2 -> $1"
    else
        echo "[WARN] 源不存在, 无法软链: $1"
    fi
}

dl_if_missing() {  # $1=relpath_on_hf $2=dst
    if [ -s "$2" ]; then
        echo "[skip] $2 已存在 ($(du -h "$2" | cut -f1))"
        return
    fi
    echo "[down] $HF_BASE/$1 -> $2 (aria2c 16线程)"
    if ! aria2c -x 16 -s 16 -k 1M --file-allocation=none --console-log-level=warn \
                --summary-interval=10 -d "$(dirname "$2")" -o "$(basename "$2")" \
                "$HF_BASE/$1"; then
        echo "[ERROR] 下载失败: $1 (检查代理/网络)"; rm -f "$2" "$2.aria2"; exit 1
    fi
}

# ---- 1. 软链复用 DuoMo 本地权重 ----
# vitpose / yolo
link_if_missing "$DUOMO/data/third_party/vitpose-h-multi-coco.pth" "$CKPT/vitpose/vitpose-h-multi-coco.pth"
link_if_missing "$DUOMO/data/third_party/yolo/yolov8x.pt"          "$CKPT/yolo/yolov8x.pt"
# body models (SMPL .pkl / SMPLX .npz, 跟随符号链接到真实文件)
for g in NEUTRAL MALE FEMALE; do
    link_if_missing "$DUOMO/data/body_models/smpl/SMPL_${g}.pkl" "$CKPT/body_models/smpl/SMPL_${g}.pkl"
done
link_if_missing "$DUOMO/data/body_models/smplx/SMPLX_NEUTRAL.npz" "$CKPT/body_models/smplx/SMPLX_NEUTRAL.npz"

# ---- 2. 从 HF 下载 GVHMR / HMR2 专有 ckpt ----
dl_if_missing "gvhmr/gvhmr_siga24_release.ckpt" "$CKPT/gvhmr/gvhmr_siga24_release.ckpt"
dl_if_missing "hmr2/epoch=10-step=25000.ckpt"   "$CKPT/hmr2/epoch=10-step=25000.ckpt"

# vitpose / yolo 若本地软链失败则回退 HF 下载
[ -e "$CKPT/vitpose/vitpose-h-multi-coco.pth" ] || dl_if_missing "vitpose/vitpose-h-multi-coco.pth" "$CKPT/vitpose/vitpose-h-multi-coco.pth"
[ -e "$CKPT/yolo/yolov8x.pt" ]                  || dl_if_missing "yolo/yolov8x.pt"                  "$CKPT/yolo/yolov8x.pt"

echo
echo "[ckpt] 完成. 目录结构:"
find "$CKPT" -maxdepth 2 \( -type f -o -type l \) -printf '  %p\n' 2>/dev/null | sort
