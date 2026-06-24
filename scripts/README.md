# GVHMR 单图推理服务 — 脚本与使用说明 (scripts/)

本目录提供 GVHMR **单图推理服务**的全部脚本:安装、下载权重、启动服务。
三个脚本**自包含**(只依赖本仓库自身,不引用任何外部仓库),路径从脚本位置自动推导。

服务能力:输入图像 → 人体检测 + SMPL;或 图像 + bbox → SMPL。输出对齐下游
`label_mocap` 的 COCO 标注格式。Flask + 有界队列 + 单 GPU worker,队列满返回 503 繁忙。

---

## 本机环境快照 (已验证可跑)

| 项 | 值 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS (kernel 5.15.0) |
| GPU | NVIDIA H800 80GB ×8 (服务默认用单卡) |
| Python | 3.10.12 (pytorch3d 预编译 wheel 要求 cp310) |
| PyTorch | 2.3.0+cu121 (强依赖, 换 CUDA 版本预测会严重出错) |
| nvcc | 12.9 (仅需可见; pytorch3d 用 cu121 wheel, 不本地编译) |
| uv | 0.11.7 |
| 虚拟环境 | `/root/paddlejob/workspace/env_run/penghaotian/envs/gvhmr` |
| 服务地址 | `http://10.52.104.78:8666` (内网, 无鉴权) |

> 本机复用现有环境的启动命令(脚本默认就指向上面这个 env, 直接跑即可):
> ```bash
> bash scripts/run_service.sh        # GPU 0, 端口 8666
> ```

## 前置条件 (新人必读)

- **机器**: Linux + NVIDIA GPU, CUDA 12.1。
- **uv**: 建虚拟环境与装包用。先装: https://docs.astral.sh/uv/ (脚本会检查, 缺则报错)。
- **aria2c**: 多线程下大权重 (hmr2 ckpt ~2.5G)。`apt install aria2` 或 `conda install aria2` (脚本会检查, 缺则报错)。
- **SMPL / SMPLX body model (有 license, 需手动获取)**: 不在 HF 自动下载范围, 需到官网注册下载:
  - SMPL: https://smpl.is.tue.mpg.de/ (下 v1.1.0)
  - SMPLX: https://smpl-x.is.tue.mpg.de/ (下 SMPLX_NEUTRAL)
  - 放到 `inputs/checkpoints/body_models/` 下 (文件名需完全一致):
    ```
    body_models/smpl/SMPL_NEUTRAL.pkl
    body_models/smpl/SMPL_MALE.pkl
    body_models/smpl/SMPL_FEMALE.pkl
    body_models/smplx/SMPLX_NEUTRAL.npz
    ```
  - 缺这些, 服务启动时 `make_smplx("supermotion")` 会失败。`download_models.sh` 会检查并打印提示。

---

## 三步流程

```bash
# (1) 安装: 建 env + torch/pytorch3d/依赖 + editable 装 gvhmr + flask
#     proxy 可选 baidu(默认, 国内 PIP/torch 快) | aliyun(HF 快)
bash scripts/install.sh baidu

# (2) 下权重: gvhmr/hmr2/vitpose/yolo 从 HF 拉; SMPL/SMPLX 需自备(见上)
bash scripts/download_models.sh baidu

# (3) 起服务: 默认 GPU 0, 端口 8666, 队列上限 64
bash scripts/run_service.sh [GPU_ID] [PORT] [MAXSIZE]
```

启动后健康检查:
```bash
curl http://<服务IP>:8666/health
# {"status":"ok","queue":0,"maxsize":64}
```

---

## 换机器/换人要改的地方

脚本路径自动推导, 但有几处**本机专属默认值**, 别人复用时改这里(或用环境变量覆盖):

| 文件 | 项 | 当前默认值 | 覆盖方式 |
| --- | --- | --- | --- |
| `install.sh` / `run_service.sh` | `ENV_DIR` | `/root/paddlejob/.../envs/gvhmr` | 设 `GVHMR_ENV_DIR=/your/env` |
| `install.sh` / `download_models.sh` | 代理 URL | 百度/阿里内网代理 | 改脚本内 `http(s)_proxy` 段 |
| `run_service.sh` | `GPU_ID` 默认 | `0` | 命令行第 1 个参数 |
| `../tools/serve/call_with_bbox.py` | `URL` | `http://10.52.104.78:8666/...` | 改成你的内网 IP:端口 |

---

## API 调用

`POST /gvhmr/infer`  Content-Type: application/json

**请求体**:
```jsonc
{
  "image_b64": "<base64 jpg/png>",   // 必填
  "bbox": [x, y, w, h],              // 可选: 不传则 YOLO 自动检测人体
  "cam_K": {"fx":2203,"fy":2203,"cx":960,"cy":540},  // 可选: 真实内参(或 3x3 矩阵), 不传则按图像尺寸估计
  "file_name": "0001.jpg"            // 可选
}
```

**响应 200** — 完整 COCO 文档, 可直接落成 `player_0.json` 喂给 label_mocap:
```jsonc
{
  "images": [{"id":0,"file_name":"...","width":1920,"height":1080,
              "cam_K":{"fx":...,"fy":...,"cx":...,"cy":...}}],   // 回显实际所用内参
  "annotations": [{
    "bbox":[x,y,w,h], "root_pos":[3], "root_rota":[3],
    "body_pose":[63], "betas":[10], "keypoints":[156],          // 156 = 52 slot ×3, 前 24 有效
    "right_hand_pose":[45], "left_hand_pose":[45], "occlution_joint":[52],
    "id":0, "image_id":0, "category_id":1
  }],
  "categories": []
}
```

**其他响应**: `503 {"error":"busy"}` (队列满) | `400 {"error":"bad_request"}` (入参错) | `500` (推理异常)。

**最小客户端**(纯标准库, Mac 直接跑; 完整示例见 `../tools/serve/call_with_bbox.py`):
```bash
python3 tools/serve/call_with_bbox.py <图片路径>            # 链路1: 自动检测
python3 tools/serve/call_with_bbox.py <图片路径> 633 0 991 991  # 链路2: 带 bbox
```

---

## 服务行为要点

- **单图 + 静态相机**: 服务对单图按静态相机推理 (GVHMR 本是时序模型, 单图质量弱于视频, 但够用)。
- **坐标系**: 输出已从 GVHMR 的 CV 相机系转到 label_mocap 的 GL 系 (绕 X 轴 180°)。`root_pos` 取 posed 骨盆世界位置, 精确落在 `keypoints[0]`。
- **bbox 回显**: 即使你传了 bbox, 响应里的 `bbox` 是"推理实际所用的框"(转正方形 + 放大 1.2 倍), 非原始输入。
- **并发**: Flask 并发收, 单 worker 串行跑 GPU; 队列(默认 64)满则 503。提速可加 worker 绑空闲卡(改 `app.py`)。

---

## 目录内容

| 文件 | 作用 |
| --- | --- |
| `install.sh` | 一键装环境 + 推理依赖 + flask (合并了原根目录 install.sh 与 install_service.sh) |
| `download_models.sh` | 从 HF 下 gvhmr/hmr2/vitpose/yolo; 检查 SMPL/SMPLX 是否就位 |
| `run_service.sh` | 启动 Flask 推理服务 (默认 GPU 0 / 端口 8666 / 队列 64) |

相关代码(在仓库其他位置, 非本目录): `tools/serve/` (服务实现 + 客户端示例 + 对齐验证脚本)。
