# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## 项目概述
GVHMR - 基于重力视图坐标的世界定位人体运动恢复系统，使用 PyTorch Lightning + Hydra 配置框架。

## 关键命令

### 训练
```bash
# 训练模型（需要 2x4090 GPU，420 epochs）
python tools/train.py exp=gvhmr/mixed/mixed

# 从检查点恢复训练
python tools/train.py exp=gvhmr/mixed/mixed resume_mode=latest
```

### 测试
```bash
# 测试所有数据集（3DPW + EMDB + RICH）
python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt

# 测试单个数据集
python tools/train.py global/task=gvhmr/test_3dpw exp=gvhmr/mixed/mixed ckpt_path=<path>
python tools/train.py global/task=gvhmr/test_rich exp=gvhmr/mixed/mixed ckpt_path=<path>
python tools/train.py global/task=gvhmr/test_emdb exp=gvhmr/mixed/mixed ckpt_path=<path>
```

### 演示推理
```bash
# 静态相机（使用 -s 跳过视觉里程计）
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s

# 动态相机（默认使用 SimpleVO，不使用 DPVO）
python tools/demo/demo.py --video=<path>

# 指定焦距（iPhone 15p: 13/24/48/77mm）
python tools/demo/demo.py --video=<path> --f_mm=24

# 批量处理文件夹
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### 数据集单元测试
```bash
# 测试数据集加载
python tools/unitest/run_dataset.py  # 修改脚本中的 DATA_TYPE
```

## 非显而易见的关键点

### Hydra 配置系统
- **配置入口**: [`tools/train.py`](tools/train.py:79) 使用 `@hydra.main(config_path="../hmr4d/configs", config_name="train")`
- **必须注册**: 在 `main()` 之前调用 [`register_store_gvhmr()`](hmr4d/configs/store_gvhmr.py) 注册所有配置组
- **覆盖顺序**: [`hmr4d/configs/train.yaml`](hmr4d/configs/train.yaml:5-29) 中的 `defaults` 列表决定配置覆盖顺序
- **命令行覆盖**: 使用 `key=value` 语法，例如 `exp=gvhmr/mixed/mixed` 会加载 [`hmr4d/configs/exp/gvhmr/mixed/mixed.yaml`](hmr4d/configs/exp/gvhmr/mixed/mixed.yaml)

### 自定义日志系统
- **不使用标准 logging**: 使用 [`hmr4d.utils.pylogger.Log`](hmr4d/utils/pylogger.py:12) 而非 Python 标准 `logging`
- **GPU 同步计时**: 使用 `Log.sync_time()` 而非 `Log.time()` 进行 GPU 操作计时
- **彩色输出**: 自动使用 colorlog 格式化

### Demo 管道缓存机制
- **预处理缓存**: Demo 会将中间结果缓存到 `outputs/demo/<video_name>/preprocess/` 目录
- **缓存文件**: `bbx.pt`, `vitpose.pt`, `vit_features.pt`, `slam_results.pt`
- **跳过已存在**: 如果缓存文件存在，会直接加载而不重新计算
- **调试时清理**: 修改预处理代码后需要手动删除对应缓存文件

### PyTorch Lightning 特殊处理
- **检查点保存**: [`gvhmr_pl.py`](hmr4d/model/gvhmr/gvhmr_pl.py:292-297) 的 `on_save_checkpoint` 会自动移除 `ignored_weights_prefix` 中的权重（默认忽略 `smplx` 和 `pipeline.endecoder`）
- **测试精度**: 测试模式强制使用 `precision=32`（见 [`tools/train.py`](tools/train.py:54-55)）
- **验证=测试**: `test_step = predict_step = validation_step`（见 [`gvhmr_pl.py`](hmr4d/model/gvhmr/gvhmr_pl.py:46)）

### 数据加载器
- **自定义 collate**: 使用 [`collate_fn`](hmr4d/datamodule/mocap_trainX_testY.py:17-30) 处理 `meta` 前缀的键（不批处理）
- **文件描述符限制**: 自动设置 `RLIMIT_NOFILE` 为 4096（见 [`mocap_trainX_testY.py`](hmr4d/datamodule/mocap_trainX_testY.py:13-14)）
- **训练数据**: 使用 `ConcatDataset` 合并多个数据集
- **验证/测试数据**: 使用 `CombinedLoader` 顺序模式

### 代码风格
- **Black 格式化**: 行长度 120（见 [`pyproject.toml`](pyproject.toml:2)）
- **导入顺序**: 标准库 → 第三方库 → 本地模块
- **类型提示**: 使用 `from typing import Any, Dict` 等

### 依赖关键点
- **PyTorch**: 必须使用 CUDA 12.1 版本（`torch==2.3.0+cu121`）
- **PyTorch3D**: 使用预编译 wheel，特定于 Python 3.10 + CUDA 12.1
- **imageio**: 使用 `av==13.0.0` 后端以提高性能（而非 ffmpeg）
