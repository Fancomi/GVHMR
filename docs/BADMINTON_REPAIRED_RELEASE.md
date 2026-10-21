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

## 历史原像素摘要的独立补证

全量测试导出发现旧 `badminton_rallies_v1_paired` 的 cam1 journal 有两段共 13 帧的
raw BGR 摘要与当前原片不一致：scan 5221–5224、5382–5390。生产 PyAV 13、消费端、
首帧开始的连续解码和 18 个 seek 起点均得到一致的当前像素，未复现旧摘要。
历史像素只留有摘要，无法单独从 SHA 还原当时生成异常的确切触发原因。

执行完整测试像素审计与受影响扫描的双机全人物重验：

```powershell
uv run --no-sync python tools/audit_badminton_native_pixels.py
```

工具在独立 `outputs/badminton_native_pixel_audit_20261018/` 中输出完整审计及证据。
本次精确检查 4,043 扫描 × 两台相机 = 8,086 张原生图像，cam2 全部匹配；
对 13 个受影响扫描重新独立检测两台相机、按全部原生网格做唯一人物关联，
26 个人帧的准入结果全部不变。该目录的 `investigation.json` 保留调查结论，
`evidence/` 保留逐扫描检测 / 关联与新旧准入，`source_snapshots/` 保留实际执行源码。

完整补充凭据已原子发布到
`outputs/badminton_body_repair_v2_paired/native_pixel_corrections.json`，
schema 为 `native-pixel-corrections/v1`。它绑定原 `training_manifest`、test metadata、
原观测报告及每项重验文件的 SHA；每条更正明确限定 scene、camera、record index、
精确 PTS、原摘要和复核摘要。原训练 / 发布清单、标签和样本集合全部保留。

消费者仅在 `status=complete`、`all_person_admissions_unchanged=true` 且全部绑定验证通过时，
对列出的具体帧应用新摘要；其他帧继续要求原像素严格一致。任何人物准入改变都终止补证，
写独立变化报告；不能静默改集合、放宽像素比较或关闭原生图像审计。
补充文件发布时使用同目录临时文件写完整 payload 后原子 rename，最终文件名不出现半成品。

## 2026-10-19 扩量发布

新版本保留完整旧合格池 80,385 人帧，并加入 1015 新数据 62,182 人帧，
总计 **142,567**：train **98,011**、val **19,213**、test **25,343**。
此前已训练的固定子集仍是 79,779，归档在 `20261017_repaired_62414`；
本次另外保留旧池中未进入该固定训练子集的 606 条 train 样本。
两次发布独立保存，旧版本不被替换。

正式入口为 `outputs/badminton_expanded_v1/training_manifest.json`，
SHA256 `559ba09126f57a41b433b91adfbc21261247a9d3b5a61838a657e3545aa5d61f`。
该清单各 split 的张量和 metadata 统一存放在：
`D:/PROJECT/LIDAR/datas/lidar2smpl_badminton/releases/20261019_expanded_142567`。
2026-10-21 按用户要求迁移至 `D:/PROJECT/LIDAR/datas/lidar2smpl_badminton`；
后续发布的 `seal_badminton_expansion.py --data-root` 默认使用该根目录。
旧 `D:/PROJECT/LABEL/datas/lidar2smpl_badminton` 仅保留兼容 Junction，
供已冻结清单、收据中的旧绝对路径继续读取同一份数据；不改写历史清单、SHA 绑定或张量。
归档入口 `manifest.json` SHA256 为
`62b12c725b1a42a6501bce4b2593d6e250f616b3590431acaae3d9322e673c1f`。
归档使用独立字节副本；公共校验器与消费侧均通过实际全量验收。
目录索引追加此版本，原 `default_release` 保留；当前训练显式选用新版本。

新数据的 62,895 是人体拟合 QA 候选数，62,182 才是最终双机准入数。
新源 164748 没有合格回合，继续显式保留为零。全量原生检查后，
双机原图复核发现七处跨人身份边界，独立 canonical 发布切断这些连接、
调整 329 条身份键；所有点云、SMPL 数值和准入集合逐项保持不变。
全量连续性复核不再有未处理的跨人切换，保留一处已确认同人的运动学复核标记。
这不代表关节具有独立动作捕捉真值。九处有原生证据支持的腿部修正被保留。

实际 Chrome 验收覆盖十段不少于五秒的连续播放及九处身份边界，
检查两台原生图像、完整 SMPL 网格和真实点云的同一扫描同步。
最终新增数据回放是 `http://127.0.0.1:8774/`；
固定生产与训练看板是 `http://127.0.0.1:8780/`。
点到表面均值约 17.80 mm，逐人帧 P90 的均值约 36.52 mm，重投影均值约
5.56 px，均为新数据候选的内部拟合一致性，不能当作独立测试误差。

全量测试审计检查 17,896 张相机图像。对新增摘要差异与历史补证涉及的
130 个扫描保留 260 张双机无损 PNG，原生地址、原片 SHA、PNG SHA 和
原始 BGR SHA 都被绑定；327 次全人物比较的准入结果保持不变。
归档 `provenance/native_pixel_corrections.json` 同时提供更正和原摘要未变的
配对图像，消费侧严格校验并优先使用这些捕获。未声称未来实时解码逐字节可复现。

消费侧实际验收证明旧样本逐项不变、所有划分隔离，并重算相同的 12 帧窗口：
train 21,243 个、val 1,490 个；不跨人物、扫描缺口或超过 150 ms 的时间间隔。
正式 CPU 回执通过后已交还 GPU，Sonata 冻结骨干流程已开始从统一归档提取
训练特征，之后按队列运行 head 100 轮和 rollout 微调至多 30 轮。
数据封存时的 `release_receipt.json` 保留原状态，后续 GPU 交接单独记录于
`outputs/badminton_expanded_v1/gpu_handoff.json`。
