#!/bin/bash
# GVHMR 一键安装脚本 (精简版, SLAM/DPVO 关闭)
# 参考 DuoMo/install.sh 风格, 仅为单独跑通 GVHMR demo (静态相机 -s)
# 用法: bash install.sh [proxy]
#   proxy: baidu (默认, GIT/PIP国内/torch 快) | aliyun (HF 快)
set -e

ENV_DIR="/root/paddlejob/workspace/env_run/penghaotian/envs/gvhmr"
PROXY="${1:-baidu}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 代理配置
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

# CUDA 环境 (pytorch3d 预编译 wheel 用 cu121, 这里仅保证 nvcc 可见)
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[cuda] CUDA_HOME=$CUDA_HOME"

echo "[1/7] 创建虚拟环境: $ENV_DIR (python 3.10, pytorch3d 预编译 wheel 要求 cp310)"
uv venv "$ENV_DIR" --python 3.10 2>/dev/null || true

PYTHON="$ENV_DIR/bin/python"
UV_INSTALL="uv pip install --python $PYTHON --link-mode=copy"

echo "[2/7] 安装基础构建工具 pip/wheel/setuptools"
$UV_INSTALL pip wheel "setuptools>=68.0" -i "$PIP_INDEX"

echo "[3/7] 安装 PyTorch 2.3.0 + torchvision 0.18.0 (cu121, 含 sm_90/Hopper 支持)"
$UV_INSTALL torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

echo "[4/7] 安装精简依赖子集 (剔除 torch/jupyter/black 等开发工具, 关闭 DPVO)"
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

echo "[5/7] 安装 pytorch3d 0.7.6 预编译 wheel (py310 + cu121 + pyt230)"
$UV_INSTALL "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl"

echo "[6/7] 安装 chumpy (--no-build-isolation, 兼容 numpy<1.24)"
$UV_INSTALL "git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17" --no-build-isolation

echo "[7/7] 以 editable 方式安装 gvhmr 本仓库"
cd "$SCRIPT_DIR"
$UV_INSTALL -e . --no-build-isolation

# pytorch3d._C 需要 libc10.so (torch/lib), uv venv 不像 conda 自动设置 LD_LIBRARY_PATH
TORCHLIB="$ENV_DIR/lib/python3.10/site-packages/torch/lib"
if ! grep -q "GVHMR torch lib" "$ENV_DIR/bin/activate"; then
    {
        echo ""
        echo "# GVHMR torch lib (pytorch3d._C 需要 libc10.so)"
        echo "export LD_LIBRARY_PATH=\"$TORCHLIB:\${LD_LIBRARY_PATH:-}\""
    } >> "$ENV_DIR/bin/activate"
fi

# 准备 checkpoints (软链复用 DuoMo 本地权重 + 从 HF 下载 gvhmr/hmr2)
echo "[ckpt] 准备模型权重 (软链复用 + HF 下载)"
export PATH="$ENV_DIR/bin:$PATH"
bash "$SCRIPT_DIR/scripts/download_models.sh"

echo
echo "============================================================"
echo " 安装完成 (SLAM/DPVO 已关闭, 仅支持静态相机 -s)"
echo " 激活: source $ENV_DIR/bin/activate"
echo
echo " 跑通示例 (本机需显式指定 GPU):"
echo "   source $ENV_DIR/bin/activate"
echo "   CUDA_VISIBLE_DEVICES=0 python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s"
echo "============================================================"
