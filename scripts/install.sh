#!/bin/bash
# GVHMR 一键安装 (虚拟环境 + 推理依赖 + 服务依赖 flask, SLAM/DPVO 关闭)
# 自包含: 仅依赖本仓库自身, 不引用任何外部仓库 (如 DuoMo)。
#
# 用法: bash scripts/install.sh [proxy]
#   proxy: baidu (默认, GIT/PIP 国内/torch 快) | aliyun (HF 快)
#
# 可选环境变量 (不设则用默认):
#   GVHMR_ENV_DIR  虚拟环境路径   (默认见下方; 换机器/换人请改这里或设此变量)
#   CUDA_HOME      CUDA 安装路径   (默认 /usr/local/cuda)
#
# 装完后还需: (1) bash scripts/download_models.sh 下权重
#             (2) bash scripts/run_service.sh      起服务
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
# 默认指向本机现有环境; 别人换机器改这里或设 GVHMR_ENV_DIR。
ENV_DIR="${GVHMR_ENV_DIR:-/root/paddlejob/workspace/env_run/penghaotian/envs/gvhmr}"
PROXY="${1:-baidu}"

# 代理 (本环境专用; 换网络环境请改这里或自行 export http(s)_proxy)
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
echo "[paths] REPO_ROOT=$REPO_ROOT  ENV_DIR=$ENV_DIR"

# CUDA (pytorch3d 预编译 wheel 用 cu121, 这里仅保证 nvcc 可见)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[cuda] CUDA_HOME=$CUDA_HOME"

command -v uv >/dev/null 2>&1 || { echo "[ERROR] 需要 uv, 先装: https://docs.astral.sh/uv/"; exit 1; }

echo "[1/8] 创建虚拟环境 (python 3.10, pytorch3d wheel 要求 cp310)"
uv venv "$ENV_DIR" --python 3.10 2>/dev/null || true
PYTHON="$ENV_DIR/bin/python"
UV_INSTALL="uv pip install --python $PYTHON --link-mode=copy"

echo "[2/8] 基础构建工具 pip/wheel/setuptools"
$UV_INSTALL pip wheel "setuptools>=68.0" -i "$PIP_INDEX"

echo "[3/8] PyTorch 2.3.0 + torchvision 0.18.0 (cu121, 含 sm_90/Hopper)"
$UV_INSTALL torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

echo "[4/8] 精简依赖子集 (剔除 jupyter/black 等开发工具, 关闭 DPVO)"
$UV_INSTALL -i "$PIP_INDEX" \
    "timm==0.9.12" \
    "lightning==2.3.0" \
    "hydra-core==1.3" \
    hydra-zen hydra_colorlog rich \
    "numpy==1.23.5" \
    matplotlib tensorboardX termcolor einops joblib \
    opencv-python ffmpeg-python scikit-image \
    "imageio==2.34.1" "av==13.0.0" \
    trimesh smplx wis3d pycolmap \
    "ultralytics==8.2.42" cython_bbox lapx

echo "[5/8] pytorch3d 0.7.6 预编译 wheel (py310 + cu121 + pyt230)"
$UV_INSTALL "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl"

echo "[6/8] chumpy (--no-build-isolation, 兼容 numpy<1.24)"
$UV_INSTALL "git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17" --no-build-isolation

echo "[7/8] editable 安装 gvhmr 本仓库"
cd "$REPO_ROOT"
$UV_INSTALL -e . --no-build-isolation

echo "[8/8] 服务依赖 flask (合并自原 install_service.sh)"
$UV_INSTALL -i "$PIP_INDEX" "flask>=3.0,<4.0"

# pytorch3d._C 需要 libc10.so (torch/lib); uv venv 不像 conda 自动设 LD_LIBRARY_PATH
TORCHLIB="$ENV_DIR/lib/python3.10/site-packages/torch/lib"
if ! grep -q "GVHMR torch lib" "$ENV_DIR/bin/activate"; then
    {
        echo ""
        echo "# GVHMR torch lib (pytorch3d._C 需要 libc10.so)"
        echo "export LD_LIBRARY_PATH=\"$TORCHLIB:\${LD_LIBRARY_PATH:-}\""
    } >> "$ENV_DIR/bin/activate"
fi

echo
echo "============================================================"
echo " 依赖安装完成 (SLAM/DPVO 已关闭, 仅静态相机 -s / 单图服务)"
echo " 虚拟环境: $ENV_DIR"
echo
echo " 下一步:"
echo "   1) 下载权重:  bash scripts/download_models.sh $PROXY"
echo "   2) 启动服务:  bash scripts/run_service.sh"
echo "      (默认 GPU 0, 端口 8666; 改用法见 scripts/README.md)"
echo "============================================================"
