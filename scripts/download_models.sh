#!/bin/bash
# 下载 GVHMR 单图服务所需权重 (纯 HuggingFace, 不引用任何外部仓库)。
# 自包含: 权重落到本仓库 inputs/checkpoints/ 下。关闭 SLAM/DPVO, 不下 dpvo.pth。
#
# 用法: bash scripts/download_models.sh [proxy]
#   proxy: baidu (默认) | aliyun (HF 快)
#
# 注意 (body model 需自备):
#   SMPL / SMPLX 有 license, 不在 HF camenduru/GVHMR 里, 本脚本不下载, 只检查。
#   需到官网注册下载 (见结尾提示):
#     [服务必需] inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
#     [可选,仅评测/渲染] inputs/checkpoints/body_models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl
#   单图推理服务全链路只用 SMPLX_NEUTRAL.npz (make_smplx supermotion / v437coco17),
#   不读 SMPL .pkl; 那 3 个 .pkl 仅 3DPW/EMDB/RICH 评测与渲染用到。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CKPT="$REPO_ROOT/inputs/checkpoints"
PROXY="${1:-baidu}"

if [ "$PROXY" = "aliyun" ]; then
    export https_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
    export http_proxy=http://njxg-banqian20230721-sousuo00230.njxg:3231/
else
    export https_proxy=http://agent.baidu.com:8188
    export http_proxy=http://agent.baidu.com:8188
fi

HF_REPO="camenduru/GVHMR"
HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"
echo "[proxy] $PROXY   HF=$HF_BASE"
echo "[paths] CKPT=$CKPT"

command -v aria2c >/dev/null 2>&1 || { echo "[ERROR] 需要 aria2c (apt install aria2 / conda install aria2)"; exit 1; }

mkdir -p "$CKPT/gvhmr" "$CKPT/hmr2" "$CKPT/vitpose" "$CKPT/yolo" \
         "$CKPT/body_models/smpl" "$CKPT/body_models/smplx"

dl_if_missing() {  # $1=relpath_on_hf  $2=dst
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

# ---- 1. HF 下载: GVHMR / HMR2 / ViTPose / YOLO (无 license, 直接拉) ----
dl_if_missing "gvhmr/gvhmr_siga24_release.ckpt" "$CKPT/gvhmr/gvhmr_siga24_release.ckpt"
dl_if_missing "hmr2/epoch=10-step=25000.ckpt"   "$CKPT/hmr2/epoch=10-step=25000.ckpt"
dl_if_missing "vitpose/vitpose-h-multi-coco.pth" "$CKPT/vitpose/vitpose-h-multi-coco.pth"
dl_if_missing "yolo/yolov8x.pt"                  "$CKPT/yolo/yolov8x.pt"

# ---- 2. body model: 有 license, 不自动下, 仅检查 ----
#   SMPLX_NEUTRAL.npz : 服务必需 (缺则 make_smplx 报错, 服务起不来)
#   SMPL_*.pkl        : 可选, 仅评测/渲染用; 单图服务不读
SMPLX_OK=1
[ -e "$CKPT/body_models/smplx/SMPLX_NEUTRAL.npz" ] || SMPLX_OK=0

SMPL_OK=1
for g in NEUTRAL MALE FEMALE; do
    [ -e "$CKPT/body_models/smpl/SMPL_${g}.pkl" ] || SMPL_OK=0
done

echo
if [ "$SMPLX_OK" = "1" ]; then
    echo "[body model] SMPLX_NEUTRAL.npz 已就位 ✓ (服务必需)"
else
    echo "============================================================"
    echo " [!] 缺少 SMPLX body model (服务必需, 有 license 需手动获取)"
    echo
    echo "   SMPLX : https://smpl-x.is.tue.mpg.de/   (注册后下 SMPLX_NEUTRAL)"
    echo "   放到 (文件名需完全一致):"
    echo "     $CKPT/body_models/smplx/SMPLX_NEUTRAL.npz"
    echo
    echo "   缺它, 服务启动时 make_smplx(\"supermotion\") 会失败。"
    echo "============================================================"
fi

if [ "$SMPL_OK" = "1" ]; then
    echo "[body model] SMPL_*.pkl 已就位 ✓ (可选: 仅评测/渲染)"
else
    echo "[body model] SMPL_*.pkl 缺失 (可选: 仅 3DPW/EMDB/RICH 评测与渲染用,"
    echo "             单图推理服务不需要; 需要评测再去 https://smpl.is.tue.mpg.de/ 下 v1.1.0)"
fi

echo
echo "[ckpt] 当前目录结构:"
find "$CKPT" -maxdepth 2 \( -type f -o -type l \) -printf '  %p\n' 2>/dev/null | sort
