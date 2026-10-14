# 代码模式规则（仅非显而易见内容）

## 必须遵循的编码模式

### Hydra 配置注册
- **必须先注册**: 任何使用 `@hydra.main` 的脚本都必须在 `main()` 之前调用 `register_store_gvhmr()`
- **配置构建**: 使用 `builds()` 而非直接实例化，例如 [`gvhmr_pl.py:317-323`](hmr4d/model/gvhmr/gvhmr_pl.py:317-323)
- **存储位置**: 使用 `MainStore.store(name=..., node=..., group=...)` 注册到配置组

### 自定义日志
- **强制使用**: 必须使用 `from hmr4d.utils.pylogger import Log` 而非 `import logging`
- **GPU 计时**: GPU 相关操作必须使用 `Log.sync_time()` 而非 `Log.time()`
- **日志级别**: `Log.info()`, `Log.warn()`, `Log.error()` 已配置 colorlog

### PyTorch Lightning 模块
- **检查点忽略**: 如果添加新的大型预训练模型，需要在 `ignored_weights_prefix` 中添加前缀（见 [`gvhmr_pl.py:35`](hmr4d/model/gvhmr/gvhmr_pl.py:35)）
- **测试=验证**: 不要重复实现 `test_step`，直接赋值 `self.test_step = self.validation_step`（见 [`gvhmr_pl.py:46`](hmr4d/model/gvhmr/gvhmr_pl.py:46)）
- **精度控制**: 测试时会自动强制 `precision=32`，不要在模型中硬编码精度

### 数据集实现
- **collate_fn**: 如果数据包含 `meta` 前缀的键，必须使用 [`hmr4d/datamodule/mocap_trainX_testY.py`](hmr4d/datamodule/mocap_trainX_testY.py:17-30) 中的 `collate_fn`
- **返回字典**: 数据集必须返回字典，collate 后会自动添加 `"B"` 键表示批次大小
- **文件描述符**: 如果遇到 "too many open files" 错误，已在 datamodule 中设置 `RLIMIT_NOFILE=4096`

### 路径处理
- **相对路径**: 所有路径使用 `Path` 对象，相对于项目根目录
- **Hydra 输出**: Hydra 会自动创建 `outputs/` 目录，配置中使用 `${output_dir}` 变量

### SMPL/SMPLX 模型
- **模型创建**: 使用 `make_smplx(model_type)` 而非直接实例化
- **稀疏转换**: SMPLX 到 SMPL 使用预加载的稀疏矩阵 `smplx2smpl_sparse.pt`（见 [`demo.py:211`](tools/demo/demo.py:211)）
