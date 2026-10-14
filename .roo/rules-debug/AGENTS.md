# 调试模式规则（仅非显而易见内容）

## Demo 管道调试

### 预处理缓存机制
- **缓存位置**: `outputs/demo/<video_name>/preprocess/` 目录
- **缓存文件**: `bbx.pt`, `vitpose.pt`, `vit_features.pt`, `slam_results.pt`
- **调试陷阱**: 修改预处理代码后，必须手动删除对应缓存文件，否则会使用旧的缓存结果
- **检查方法**: 在 [`demo.py`](tools/demo/demo.py:108-170) 中，每个预处理步骤都会先检查 `Path(paths.xxx).exists()`

### 日志和计时
- **GPU 计时**: 必须使用 `Log.sync_time()` 而非 `Log.time()` 进行 GPU 操作计时（见 [`demo.py:322`](tools/demo/demo.py:322)）
- **日志颜色**: 使用 `Log.info()`, `Log.warn()`, `Log.error()` 会自动彩色输出
- **计时装饰器**: 可以使用 `@timer(sync_cuda=True, mem=True)` 装饰器（见 [`pylogger.py:29-65`](hmr4d/utils/pylogger.py:29-65)）

### Hydra 输出目录
- **自动创建**: Hydra 会在 `outputs/` 下创建子目录，结构为 `outputs/${data_name}/${exp_name}`
- **日志文件**: 训练日志自动保存到 `${output_dir}/${hydra.job.name}.log`（见 [`hydra/default.yaml:16`](hmr4d/configs/hydra/default.yaml:16)）

### 测试模式强制精度
- **精度覆盖**: 测试模式会强制使用 `precision=32`（见 [`train.py:54-55`](tools/train.py:54-55)）
- **不要硬编码**: 不要在模型中硬编码精度设置，让 Trainer 控制

### 数据集调试
- **单元测试**: 使用 [`tools/unitest/run_dataset.py`](tools/unitest/run_dataset.py) 测试数据集加载
- **修改数据类型**: 在脚本中修改 `DATA_TYPE` 变量（第 18 行）
- **批次大小**: 评估时批次大小必须为 1（见 [`gvhmr_pl.py:163`](hmr4d/model/gvhmr/gvhmr_pl.py:163)）
