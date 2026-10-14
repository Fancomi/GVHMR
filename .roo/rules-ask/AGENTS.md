# 问答模式规则（仅非显而易见内容）

## 项目架构理解

### Hydra 配置系统
- **配置组织**: 配置文件在 `hmr4d/configs/` 下，按功能分组（`data/`, `model/`, `exp/`, `global/`）
- **覆盖机制**: `defaults` 列表从上到下覆盖，`_self_` 决定当前文件的优先级
- **动态注册**: 使用 `hydra-zen` 的 `builds()` 和 `MainStore.store()` 动态注册配置
- **命令行语法**: `key=value` 覆盖配置，`group/key=value` 选择配置组

### Demo 管道流程
- **预处理阶段**: 跟踪 → VitPose → ViT 特征 → 视觉里程计（可选）
- **推理阶段**: 加载模型 → 预测 → 后处理
- **渲染阶段**: 相机内渲染 + 全局渲染 → 合并视频
- **缓存策略**: 每个阶段的结果都会缓存，避免重复计算

### 数据加载器设计
- **训练**: 使用 `ConcatDataset` 合并多个数据集，随机打乱
- **验证/测试**: 使用 `CombinedLoader` 顺序模式，依次处理每个数据集
- **元数据处理**: `meta` 前缀的键不会被批处理，保持为列表

### PyTorch Lightning 集成
- **回调系统**: 使用自定义回调计算指标（`metric_3dpw.py`, `metric_rich.py`, `metric_emdb.py`）
- **检查点策略**: 保存时自动移除大型预训练权重（`smplx`, `pipeline.endecoder`）
- **多阶段复用**: `test_step = predict_step = validation_step` 避免重复代码

## 常见问题

### 为什么不使用标准 logging？
- 项目使用自定义 `Log` 对象，集成了 colorlog 和 GPU 同步计时功能
- `Log.sync_time()` 确保 GPU 操作完成后再计时，避免异步执行导致的计时不准

### 为什么 Demo 需要手动删除缓存？
- 预处理步骤（跟踪、姿态估计、特征提取）耗时较长，缓存可以加速重复运行
- 但修改预处理代码后，缓存会导致使用旧结果，必须手动清理

### 为什么测试批次大小必须为 1？
- 评估时需要处理完整序列，不同序列长度不同，无法批处理
- 代码中有断言检查：`assert batch["B"] == 1`（见 [`gvhmr_pl.py:163`](hmr4d/model/gvhmr/gvhmr_pl.py:163)）

### 为什么需要 register_store_gvhmr()？
- Hydra 需要在解析配置前注册所有配置组
- `register_store_gvhmr()` 使用 `hydra-zen` 动态注册模型、数据集等配置
- 必须在 `@hydra.main` 装饰的函数调用前执行
