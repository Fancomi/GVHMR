# 架构模式规则（仅非显而易见内容）

## 核心架构约束

### Hydra 配置架构
- **配置存储**: 使用 `hydra-zen` 的 `MainStore` 全局存储，而非直接写 YAML
- **构建模式**: `builds()` 创建配置节点，`populate_full_signature=True` 暴露所有参数
- **组注册**: 配置必须注册到特定组（`model/`, `data/`, `exp/` 等），通过 `group=` 参数指定
- **延迟实例化**: 使用 `hydra.utils.instantiate()` 延迟实例化，支持 `_recursive_=False` 控制递归

### PyTorch Lightning 架构
- **模块复用**: `test_step = predict_step = validation_step` 避免重复实现
- **检查点策略**: `on_save_checkpoint` 钩子移除大型权重，`ignored_weights_prefix` 控制忽略列表
- **回调解耦**: 指标计算通过回调实现，而非在模型中硬编码
- **精度控制**: Trainer 控制精度，测试时强制 `precision=32`

### 数据管道架构
- **训练**: `ConcatDataset` 合并 → `DataLoader` 随机打乱 → `collate_fn` 批处理
- **验证/测试**: 多个 `DataLoader` → `CombinedLoader(mode="sequential")` 顺序处理
- **元数据隔离**: `meta` 前缀的键不批处理，保持为列表，避免张量化失败
- **文件描述符**: DataModule 初始化时设置 `RLIMIT_NOFILE=4096`，避免多进程加载时文件句柄耗尽

### Demo 管道架构
- **阶段分离**: 预处理 → 推理 → 渲染，每个阶段独立缓存
- **缓存策略**: 检查文件存在性 → 加载或计算 → 保存到磁盘
- **视觉里程计**: 静态相机跳过 VO，动态相机默认使用 SimpleVO（不使用 DPVO）
- **渲染双轨**: 相机内渲染（原始视角）+ 全局渲染（第三人称视角）

### SMPL/SMPLX 模型架构
- **工厂模式**: `make_smplx(model_type)` 统一创建接口，支持多种模型类型
- **稀疏转换**: SMPLX → SMPL 使用预加载的稀疏矩阵，避免重复计算
- **批处理**: 模型支持批处理，输出 `vertices` 和 `joints`

## 架构决策记录

### 为什么使用 hydra-zen 而非纯 YAML？
- 支持动态注册配置，避免手动维护大量 YAML 文件
- `builds()` 提供类型检查和自动补全
- 可以在 Python 代码中定义配置，与模型定义放在一起

### 为什么 Demo 使用缓存而非流式处理？
- 预处理步骤（跟踪、姿态估计）耗时较长，缓存可以加速调试
- 每个阶段独立缓存，方便单独重新运行某个步骤
- 磁盘空间换时间，适合研究场景

### 为什么检查点保存时移除权重？
- SMPLX 模型和 endecoder 是固定的预训练权重，不需要保存
- 减小检查点文件大小，加快保存和加载速度
- 加载时会自动重新初始化这些权重

### 为什么测试和验证使用相同的步骤？
- 评估逻辑完全相同，只是触发时机不同
- 避免代码重复，减少维护成本
- 通过 `self.trainer.state.stage` 区分是否应用后处理
