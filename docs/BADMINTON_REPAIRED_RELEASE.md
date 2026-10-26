# 双机训练发布与消费契约

当前累计发布为
`D:/PROJECT/LIDAR/datas/lidar2smpl_badminton/releases/20261024_expanded_228059/manifest.json`：
train/val/test **170,815 / 23,749 / 33,495**，总计 **228,059** 人帧。
发布链和验收见 [原生生产](BADMINTON_FULL_NATIVE.md)。
这里的 GT 是多视角/点云拟合伪标签，不是独立光学动捕真值。

## 发布版本与角色

| 独立完整版本 | train | val | test | 合计 |
| --- | ---: | ---: | ---: | ---: |
| `20261017_repaired_62414` 固定对照视图 | 62,414 | 10,782 | 6,583 | 79,779 |
| `20261019_expanded_142567` | 98,011 | 19,213 | 25,343 | 142,567 |
| `20261024_expanded_228059` | 170,815 | 23,749 | 33,495 | 228,059 |

所有版本位于同一归档根目录，彼此有重叠，不能相加或混用 split。
同一版本按整场来源隔离 train/val/test；跨场次真实运动员身份未证明互斥。
原始固定测试场为 `20261010_190709`（6,583 人帧）；后续完整测试集合扩大，
不改变这组原场景样本。固定视图源自 80,385 条完整合格旧池，
完整旧池 train 为 63,020；另 606 条不属于固定 62,414 训练前缀。

`manifest.json` 是归档入口，`dataset_manifest.json`、`training_manifest.json`
及各 split tensor/metadata 的实际绑定以该版本清单为准。每次发布保存独立字节副本、
SHA、角色/来源校准与验证证据。旧绝对路径可经兼容 Junction 指向同一份归档，
历史字符串/清单/SHA 不改写，见 [DATA_RELOCATIONS.md](DATA_RELOCATIONS.md)。
默认 `seal_badminton_expansion.py --data-root` 为
`D:/PROJECT/LIDAR/datas/lidar2smpl_badminton`。

## 修复版基线入口

旧七场发布从 `outputs/badminton_body_repair_v2/final_labels` 的 80,973
人体 QA 候选出发，按 `bilateral_detection` 规则重新准入。
原生检测、完整网格及一对一关联均须真实存在；人体 QA 不等于双机准入。

```powershell
uv run --no-sync python tools/publish_badminton_repaired_pairs.py observe
uv run --no-sync python tools/publish_badminton_repaired_pairs.py publish
uv run --no-sync python tools/seal_badminton_repaired_release.py --train-count 62414
```

默认独立输出 `outputs/badminton_body_repair_v2_paired/` 已封存，不能当新输出覆盖。
核心产物：

- `datasets/dataset_manifest.json`：完整双机发布，只有 `status=complete` 可消费。
- `training_manifest.json`：固定视图，指向真实 tensor 与逐行 metadata。
- `release_receipt.json`、`aggregate_source_contract.json`：发布与原生窗口等值证明。
- `identity_point_label_mapping.jsonl`：canonical/原始 detector 身份、teacher/fit SHA、
  原点成员、pc 和 SMPL 参数摘要。
- `train_sequences.jsonl`、`val_sequences.jsonl`、`test_sequences.jsonl`：对应 split 行索引。
- `provenance/source/`：实际执行源码快照；原生观测、像素补证及其证据保持 SHA 绑定。

固定训练视图选择真实唯一准入样本的确定性前缀；数量不足则终止，不复制凑数。
序列按完整 `motion_id` 分组，仅在扫描索引相邻、真实时间差 ≤150 ms 时连续；
缺失扫描、窗口或 track segment 改变都断开，不合成中间标签。
消费者十二帧窗口采用同一算法，必须重新验证确切索引。

## 原生检测和双机投影

检测可与旧身份分离复用，但须校验实际原视频、大小/时间、标定、native index、
PTS/common clock、原像素 SHA 与 journal 完成状态。把修复后的**完整同帧人物集合**
（包括人体 QA 拒绝者）一起参加一对一框关联；不以旧 motion ID 或旧网格匹配
决定新身份。缺失有效证据则重新解码/检测。发布值须逐项等于来源 tensor 的真实子集。

split metadata 的 `session_id`、`lidar_index`、`scan_mid_us` 指向实际扫描；
`cameras.cam1/cam2` 分别包含 `record_index`、`media_pts_us`、
`common_time_us`、`time_error_us`。同一扫描所有人共用这两组原生图像地址。
模型按人帧计数；播放器按唯一扫描组合全部人物。

```python
from pathlib import Path
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.rally_pipeline import load_run_session
from hmr4d.utils.body_model.contracts import forward_smpl_at_pelvis

source = Path(training_manifest["source_run"])
calibration = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
session = load_run_session(source, calibration, row["session_id"])
vertices, joints = forward_smpl_at_pelvis(model, root_pose, body_pose, shape, transl)
for cam in ("cam1", "cam2"):
    address = row["cameras"][cam]
    raw_bgr = session.read_image(cam, address["record_index"])
    uv, depth = calibration.project(vertices[0].detach().cpu().numpy(), cam, distort=True)
```

展开的累计数据使用每个来源自己的 run/calibration；不能用一个场景的
`source_run` 或 snapshot 投影所有来源。顺序播放用
`session.iter_video_frames(cam, sorted_record_indices)` 避免逐帧随机 seek。

原图为 raw Brown 像素域，投影用 `distort=True`；rectified 2D 观测采用保留 K 的
去畸变域，不能混用。`transl` 是形态骨盆，不是标准 SMPL 加性平移；统一使用
`forward_smpl_at_pelvis`。参数、坐标和形态转换见
[neutral SMPL 契约](INSTALL.md#neutral-smpl-and-the-single-frame-consumer)。

浏览器复用完整网格和共享投影。先准备同一扫描两张原生图、所有人的标签/预测网格和
真实点云，再一次提交全部视图。缺少准入或预测则明确显示缺失，不跨扫描补图、
插值预测或使用 MP4 播放时间代替 native frame 地址。

## 原像素补证

原生像素审计和拟合验证独立。历史 cam1 的 scan 5221–5224、5382–5390
共 13 帧旧 BGR 摘要与当前原片不同；现存摘要不足以重建历史异常原因。
`audit_badminton_native_pixels.py` 的原始独立调查保留在
`outputs/badminton_native_pixel_audit_20261018/`，完整旧测试检查了
4,043 扫描/8,086 图像，受影响 26 人帧的双机准入不变。

补证 `native_pixel_corrections.json`（`native-pixel-corrections/v1`）严格绑定
原训练/发布清单、metadata、观测报告及每个证据 SHA；更正只限具体场景、camera、
record、精确 PTS、原摘要和复核摘要。消费者只有在完整状态、
`all_person_admissions_unchanged=true` 且所有绑定通过后才应用列出的更正。
其他帧继续严格要求原像素一致，任何准入改变都终止补证。

扩量阶段的 `recheck_badminton_expanded_pixels.py` 保存 lossless native-PTS PNG，
同时绑定原片、PNG、BGR 和原生地址，并按完整同帧人物集合复验两台相机。
消费者优先验证和回放这些确切捕获；不能据此声称将来实时解码逐字节可复现。
旧清单、标签和样本集合不因补证改写，相关 journal/PNG/teacher/fit 仍是发布依赖。
