# 修复版双机训练发布

本流程消费 `outputs/badminton_body_repair_v2/final_labels`，将 track / 腿部修复后的
80,973 个人体 QA 候选按既有 `bilateral_detection` 规则重新准入。
原始 `outputs/badminton_full_plus_new_v1` 及其 79,776 条数据不覆盖。
这里的测试 GT 指修复后的多视角 / 点云拟合标签，不是独立光学动捕真值。

本次封存的真实数量：

| 视图 | train | val（161544） | test（190709） | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 完整双机准入 | 63,020 | 10,782 | 6,583 | 80,385 |
| 固定训练视图 | 62,414 | 10,782 | 6,583 | 79,779 |

训练前缀保留五场各 14,662 / 34,464 / 1,367 / 5,996 / 5,925 条，
最后一场 160054 末尾的 606 条不进入固定训练视图。8 个腿部后修正案例全部通过准入，
其中 7 条在固定训练集、1 条在测试集。测试共 4,043 个唯一扫描；验证共 5,521 个。
train / val / test 分别有 1,459 / 55 / 250 个按实测扫描分开的序列片段。

## 发布与固定训练视图

```powershell
uv run --no-sync python tools/publish_badminton_repaired_pairs.py observe
uv run --no-sync python tools/publish_badminton_repaired_pairs.py publish
uv run --no-sync python tools/seal_badminton_repaired_release.py --train-count 62414
```

输出为独立的 `outputs/badminton_body_repair_v2_paired/`：

- `datasets/dataset_manifest.json`：完整双机准入后的发布，`status=complete` 才可消费。
- `training_manifest.json`：固定训练视图；`suggested_session_splits` 的 train 指向
  `datasets/train62414.pth` 及其逐行 metadata，val / test 保留各自全部新准入数据。
- `release_receipt.json`：完整发布、固定训练 manifest 的路径、SHA 与真实计数。
- `aggregate_source_contract.json`：所有聚合候选的点云 / SMPL 参数及原点成员、身份和时间元数据
  与已经通过原生重建验证的 75 个窗口逐条相等。
- `identity_point_label_mapping.jsonl`：每条发布样本对应的 canonical track、teacher 实测帧的原始 detector track、
  teacher / fit SHA、原点成员摘要、pc 摘要和 SMPL 参数摘要。
- `provenance/source/`：与观测 / 发布收据摘要一致的实际执行源码快照，保留生成版本。
- `train_sequences.jsonl`、`val_sequences.jsonl`、`test_sequences.jsonl`：索引指向对应 split tensor / metadata 的行。

训练集使用原场次顺序的确定性前缀，严格选择 62,414 个唯一的真实准入样本。
如果不足，写 `insufficient_training_count.json` 并终止，不补帧或复制样本凑数。
验证场为 `20261012_161544`，测试场为 `20261010_190709`；其余五场训练。
序列按完整 `motion_id` 分组，仅在扫描索引相邻且真实时间差 ≤150 ms 时连续；
缺失扫描、窗口或 track segment 变化都断开，不合成中间观测。

## 原生检测复用的边界

独立人体检测与旧身份分离。只有已完成、摘要未变的原生观测 journal 才能复用；
原视频路径 / 大小 / 修改时间、标定、native index / PTS / common time、原像素 SHA 必须一致。
然后读取**修复后的完整网格**，把同帧全部拟合人物（包括人体 QA 拒绝者）一起参加一对一框关联。
旧 motion ID、旧网格投影和旧匹配关系均不用于决定新身份。缺少原生检测的帧重新解码并检测。
最终仍由通用 `dual_view.py` 审核、导出并验证：两台相机都要有唯一对应的真实人物检测，
发布值必须逐项等于修复版来源 tensor 的真实子集。

使用新的根目录修改策略；已发布 manifest 和固定训练视图不覆盖。
SHA 绑定的旧原生检测 journal、teacher、源窗口和修复结果仍是本发布的依赖，不能作为实验缓存删除。

## 两台相机与 3D 同步显示的入口

每条 split metadata 的 `session_id`、`lidar_index`、`scan_mid_us` 指向实际扫描。
`cameras.cam1` / `cam2` 分别提供 `record_index`、`media_pts_us`、`common_time_us` 和 `time_error_us`。
同一扫描的所有人物共用这组原生图像索引。模型推理和评价按人帧计数，播放器按唯一扫描组合人物。

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
    uv, camera_depth = calibration.project(vertices[0].detach().cpu().numpy(), cam, distort=True)
```

顺序播放用 `session.iter_video_frames(cam, sorted_record_indices)`，避免每帧随机 seek。
原生寻址由 `sessions.py` / `preview_decoder.py` 校验 PTS，正式 dualcam metadata 还绑定对应原始 BGR SHA。
`native_projection_smoke_verification.json` 抽查测试场首 / 中 / 尾及 scan 40，共四个扫描、八张原生图像，
BGR 摘要均与发布观测一致，重建网格的 Brown 投影与存储网格最大偏差小于 0.000183 px。
在原生图像上投影须使用 `distort=True`；rectified 关键点观测与 raw Brown 像素空间不能混用。
`transl` 是形态骨盆位置，不能直接当作 SMPL 模型平移量；重建统一调用 `forward_smpl_at_pelvis`。

浏览器复用 `review_assets/projection.js`、完整网格 `review.js` 和现有同步提交机制。
先准备同一扫描的两台原生图像、各人的 GT 与两个模型网格、真实点云，再一次性提交所有视图。
完整测试播放器应覆盖正式 test metadata 的全部唯一扫描，并显示当前 native scan / PTS；
未准入或没有预测的帧明确缺失，不跨扫描补图或插值模型输出。
