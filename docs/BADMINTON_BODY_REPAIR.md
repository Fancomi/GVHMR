# 羽毛球人体标签修复

保留的 v2 策略用于固定相机下的人体身份、点云归属和足踝拟合。
它不优化标定，不提供跨场次运动员识别或独立动捕真值。
新源运行顺序见 [原生生产](BADMINTON_FULL_NATIVE.md#new-source-body-production)，
安装与模型约定见 [INSTALL.md](INSTALL.md)。

## 保留策略

1. 颜色按稳定 track 分配，不能用当前可见人物的数组下标。换色、断轨和混人是不同问题。
2. 保留原始 tracklet，使用对侧相机连续性、唯一极线对应和原生 RGB 审查边界。
   单视角位置、尺度、速度只产生候选；未知关系分开保留，不自动合并。
3. 先定身份和点云归属，再拟合。两台相机独立观察原生姿态；用正深度、唯一性、
   双视角框和骨架限制点云归属。同一主相机射线上的另一人不能成为当前人的点云。
4. 先将 SMPL-X 网格转为 neutral SMPL，再使用 `distal_pose_prior_weight=0.03`
   和 `lower_limb_reprojection_weight=2`。不同模型的局部旋转和形态参数不能直接复制。
   同一身份共享形态；缺失第二视角时按实际参与视角归一化。矛盾射线降低有效置信度，
   保留原始坐标。
5. 只有明确左右腿对应证据才做局部后处理：从存储的原生 SMPL 出发，固定 root、
   骨盆、形态和上身，只优化八个腿关节，比较两种相机腿标签及全量/半量修正。
   单关节变化 ≤40°，共同点表面 mean/P90 退化 ≤1/2 mm；
   双机平均无序膝踝集合残差改善 ≥`max(1 px,10%)`，任一相机恶化 ≤1 px，
   穿透增量 ≤1 cm，原人体 QA 仍通过。按真实扫描时间选连续修正，
   相邻变化 ≤20.5°，超过 150 ms 的缺口断序列。
6. 身份或观测改变后重建所有受影响的 teacher、分割、拟合、导出、验证和评审页。
   标定改变必须新开版本，重建投影和全部下游结果。

点云残差比较须使用两版共同的实际点集；各自分割点残差不能直接衡量改进。
完整模型和两台相机都参与拟合时，表面残差和重投影是内部一致性，不是独立准确率。

## 运行入口

在仓库根目录使用 `uv run --no-sync python tools/<entry> ...`。
每个新配置使用新的输出根目录，同一根目录只由一个拟合驱动拥有。

| 阶段 | 入口 / 关键输入 |
| --- | --- |
| 冻结旧交付的修复范围与连续对比 | `prepare_badminton_repair_cases.py --output-root <repair>`；原 union、审计和 tracking replay |
| 新源范围 | `prepare_badminton_new_source_repair.py --source-root <prepared-run> --root <repair>` |
| 原生双机观测 | `observe_badminton_repair_poses.py --source-root <prepared-run> --root <repair>` |
| 身份证据与决策 | `prepare_badminton_identity_review.py` → `review_badminton_identity_views.py` → `build_badminton_identity_decisions.py`，显式 `--root <repair>` |
| 准备拟合 | `repair_badminton_body_labels.py --stage prepare --output-root <repair>/final_labels --native-poses-root <repair> --joint-ray-filter --distal-pose-weight 0.03 --lower-limb-reprojection-weight 2 --identity-decisions <repair>/identity_evidence/identity_decisions.json`；新源还需显式 union/source-root |
| 隔离窗口拟合与续跑 | `run_badminton_repair_windows.py --root <repair>` |
| 原生验证和共同评价 | `verify_badminton_native_repair.py`、`report_badminton_native_repair.py`、`render_badminton_repair_comparison.py`，均传 `--root <repair>` |
| 腿候选、时序选择、应用、重验与完成 | `finish_badminton_body_repair.py --root <repair>`；所有基线窗口必须完整 |

历史七场的人工身份例外不得套到新源。新源使用真实边界库存和证据，
或显式保守 abstention 策略保留未知 tracklet，见原生生产指南。
`final_labels/post_refinement.json` 封存已修正基线；普通基线入口拒绝覆盖它。
正式已接纳运行使用单独的 postleg CPU 收据接口，不能自动重启腿部 CUDA。

## 原生验证、缓存和评审

Teacher 只有在完整实测时间上下文、源帧、模型和设置相同后才复用。
裁剪/关键点/特征须通过来源指纹；分割缓存要求完整可见集合和几何相同，
且没有双视角覆盖。每窗口独立进程，最多三次重试，续跑重新校验收据。
永久计时写入运行内的 `execution_timing.json`，不依赖临时日志时间。

```powershell
uv run --no-sync python tools/verify_badminton_native_repair.py --root <repair>
uv run --no-sync python tools/verify_badminton_dataset.py --output-root <repair>/final_labels --samples-per-session 64
uv run --no-sync python tools/render_badminton_repair_comparison.py --root <repair>
```

旧/新比较复用完整 6,890 顶点渲染和共同投影代码；两版 RGB、mesh、cloud 都就绪后
一次提交六视图。实际浏览器检查覆盖连续 native scan、原生 RGB、seek 和完整网格。
不要通过覆盖封存收据宣称复验；对不可变运行使用新报告路径或新验收目录。

## 已完成的修复基线与保留证据

`outputs/badminton_body_repair_v2/` 是旧七场、75 窗口的完整修复基线：
85,216 拟合人帧、80,973 人体 QA 候选。经独立双机准入得到 80,385 条，
train/val/test 为 63,020/10,782/6,583；固定训练对照视图为 62,414 条。
这组历史计数不代表 [当前累计发布](BADMINTON_FULL_NATIVE.md#current-accepted-data)。

可复核结果在 `completion_receipt.json`、`end_to_end_receipt.json`、
`comparison/index.html`（十段各五秒）、`comparison/report.html` 和
`evaluation/metrics.json`。实际身份边界及腿部修正证据在 `identity_evidence/`、
`leg_refinement_proposals/`、选择/应用收据和 `leg_refinement_before/`。
原生 pose、teacher、最终 fits、原点成员、对比页及其依赖不能作为普通缓存删除。
v1 的 `labels/` 和 `tracking_final/` 仍提供已验证 teacher 与 replay 来源。
正式发布和投影契约见 [修复版训练发布](BADMINTON_REPAIRED_RELEASE.md)。

未采用的做法及原因：仅凭单视角距离合并 ID 会混人；删除矛盾腿约束不能修复已错三维腿；
从 teacher 反复转换重拟合可能回到同一错误局部解；直接应用单帧最优腿会新增跳变；
按同名缓存认定原生图像相同会漏掉 PTS/像素差异。这些边界由现有身份、连续选择、
原生像素和共同观测回归覆盖，不再维护替代实验入口。
