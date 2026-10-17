# 羽毛球人体标签修复：保留策略与运行约定

适用范围：本次冻结的七场录像、75 个原始窗口及其原生 LiDAR 扫描。
这是对已交付数据的修复候选，不是跨场次运动员识别方案，也不是新录像的自动身份真值。

## 当前有效结果

| 内容 | 位置 / 结果 |
| --- | --- |
| 最终运行 | `outputs/badminton_body_repair_v2/` |
| 原生 SMPL / 人体点云 | `final_labels/`：85,216 拟合人帧，80,973 通过人体 QA 的候选 |
| 完成收据 | `completion_receipt.json`、`end_to_end_receipt.json` |
| 连续对比 | `comparison/index.html`：10 段 × 5 秒，每段 50 个真实 10 Hz 扫描 |
| 质量报告 | `comparison/report.html`、`evaluation/metrics.json` |
| 身份证据 | `identity_evidence/`：64 个实际 ID 边界，18 个混人观测及 47 个重复观测移除 |
| 腿部修正 | `leg_refinement_selection.json`、`leg_refinement_receipt.json`：7 窗口、8 人帧 |
| 效率 | 完整拟合、后处理、验证及渲染约 4.27 小时；复用已校验的原生像素 / teacher 缓存 |

在 6,294 个共同人帧上，用同一组实际分配点比较两版网格，平均表面残差
24.04 → 20.10 mm；cam1 / cam2 踝点残差分别为 13.74 → 9.04 px、6.81 → 5.45 px。
两台相机都参与新版本拟合，这些是拟合一致性指标，不是独立三维真值。
脚尖方向仍有遮挡和观测歧义，个别帧的点云残差有小幅退化；报告保留这些案例。

原交付 union 仍为 **79,776** 人帧。80,973 是候选人体 QA 数量，须另行经过双机准入、
导出及验证后才能形成新的训练发布。原队列和 session 划分不因本次修复改变。

## 保留的修复顺序

1. **先区分换色与换人。** 颜色按 canonical track 分配，不按当前可见人物的数组下标。
   1009 回合 1 的开头换色包含一个显示问题；同时存在真实断轨和混人，不能仅靠改颜色处理。
2. **保留连续原始 tracklet，再审断点。** 单视角位置、尺度和速度只产生候选。
   对侧相机连续性、唯一极线对应和原生 RGB 用来决定连接、禁止连接或移除污染片段。
   50/90 的早期统计含 26 个原本就相同的 ID，只有 24 个是新几何连接；应采用最终 64 个实际边界的复核结果。
3. **先定身份和点云归属，再拟合。** 两台相机的原生姿态独立观测；结合正深度和唯一性约束做对应，
   双视角框 / 骨架限制人体点云归属。相同主相机射线上的另一个人不能成为当前人物的点云。
4. **在统一的 neutral SMPL 解剖基下约束足踝。** 先完成 SMPL-X → SMPL 几何转换，
   再施加 `distal_pose_prior_weight=0.03`、`lower_limb_reprojection_weight=2`。
   不直接复制两种模型的局部旋转数值。同一身份共享形态；缺失第二视角时按实际参与视角归一化，
   不能把主视角损失减半。矛盾关节射线降低有效置信度，保留原始观测坐标。
5. **有左右腿对应证据时，才做局部后处理。** 从已存原生 SMPL 出发，固定 root、骨盆、形态及上身参数，
   只优化八个腿部关节；比较两种相机腿标签假设及全量 / 半量修正。
   单关节修正不超过 40°；共同点表面 mean / P90 相对基线最多退化 1 / 2 mm；
   双机平均无序膝踝集合残差改善至少 `max(1 px, 10%)`，任一相机不能恶化超过 1 px，
   地面穿透增量不超过 1 cm，原人体 QA 门限仍需通过。
   时序选择按真实扫描时间限制相邻修正变化 ≤20.5°；超过 150 ms 的缺口开启新序列。
6. **重建所有受影响的下游产物。** 身份或观测约束改变后，重出 teacher 上下文、分割、拟合、导出、
   验证、数值报告和 HTML。相机外参改变则另开版本，重新建立投影与全部下游结果。
   本项目消费 LiDAR 项目给定的标定；本次人体修复不调相机、球场、点云坐标或显示裁剪。

## 已淘汰的方案

| 方案 | 淘汰原因 / 替代做法 |
| --- | --- |
| 以可见人物下标确定颜色 | 人物进出画面会换色；使用 canonical track 调色板 |
| 仅凭单视角相近位置合并 ID | 遮挡、跳跃、交叉可误合并；使用双机证据、cannot-link 和污染片段剔除 |
| 遇到矛盾腿射线只删除约束 | 可避免坏约束，但不能修正已错的三维腿；增加有证据且有质量门限的局部修正 |
| 从 GVHMR teacher 反复转换并重拟合问题脚 | 个案试验改善不稳定，可能回到错误局部解；从现存原生 SMPL 做受限腿部优化 |
| 把单帧最优腿修正直接写回 | 会引入新跳变；以真实时间轴做连续选择并约束序列端点 |
| 只用各自分割点的残差比较新旧质量 | 点归属变化使结果不可比；两版统一使用相同实际点集和原生关键点 |
| 按同名缓存认定原生帧相同 | 解码 / 帧对应可能不同；校验 PTS、raw BGR SHA、权重和配置 |
| Windows 单进程连续跑所有窗口 | 曾发生原生库崩溃；每窗口独立进程、最多三次重试，完成窗口按收据续跑 |
| 另写一个简化比较渲染器 | 易造成网格 / 投影不一致；复用完整 6,890 顶点渲染器和共享投影代码 |
| 用临时日志文件的时间戳作为永久运行依赖 | 日志可清理；计时保存在运行目录的 `execution_timing.json` |

## 运行入口

在仓库根目录使用本地 uv 环境：`uv run --no-sync python ...`。
查看现有结果不需要重跑拟合：

```powershell
uv run --no-sync python tools/serve_badminton_body_repair.py --display-crop D:/PROJECT/LIDAR/lidar_project/config/calibration/badminton-20261012/display-crop.json
uv run --no-sync python -m http.server 8772 --bind 127.0.0.1 --directory outputs/badminton_body_repair_v2/comparison
```

两个命令分别在两个终端运行，打开 `http://127.0.0.1:8771/` 和 `http://127.0.0.1:8772/`。
已有服务时直接访问；离线分享则保留整个 `comparison/` 文件夹。

下表为完整流程的生产入口。对新版本统一显式传入自己的根目录；当前已封存的 v2 不应再次作为基线输出。

| 阶段 | 入口及关键输入 |
| --- | --- |
| 冻结窗口、十段对比与问题扫描 | `prepare_badminton_repair_cases.py --output-root <run>`；原 union、全量审计及 tracking replay |
| 原生双机观测 | `observe_badminton_repair_poses.py --root <run>`；冻结 case plan、原生录像、标定 |
| 实际 ID 边界证据 | `prepare_badminton_identity_review.py --root <run>`；随后对 `<run>/identity_evidence` 运行原生观测及 `review_badminton_identity_views.py --root <run>` |
| 审核并封存身份决策 | `build_badminton_identity_decisions.py --root <run>`；保留原生 RGB 人工复核图片、对应 SHA 和 `manual_review_receipt.json` |
| 准备全量拟合 | `repair_badminton_body_labels.py --stage prepare --output-root <run>/final_labels --native-poses-root <run> --joint-ray-filter --lower-limb-reprojection-weight 2 --identity-decisions <run>/identity_evidence/identity_decisions.json` |
| 隔离进程拟合 / 断点续跑 | `run_badminton_repair_windows.py --root <run>`；可选 `--seed-teacher-root outputs/badminton_body_repair_v1/labels` 复用已验证 teacher |
| 基线检查与报告 | `verify_badminton_native_repair.py`、`report_badminton_native_repair.py`、`render_badminton_repair_comparison.py`，均传 `--root <run>` |
| 生成受限腿部候选 | `propose_badminton_leg_refinement.py --root <run>`；仅在所有基线窗口完成后执行 |
| 连续选择、回滚快照、应用、重验、重绘、封存 | `finish_badminton_body_repair.py --root <run>`；要求前述基线收据均完整，且比较页在 8772 服务 |

身份决策脚本包含本次七场录像的人工例外，不能直接套到新录像。新数据需要重新审阅边界和混人片段。
同一输出根目录只由一个拟合驱动拥有；不再保留针对特定场次的临时并行进程终止逻辑。
`final_labels/post_refinement.json` 表示基线拟合已封存，基线入口会拒绝覆盖这些腿部修正。

## 缓存、验证和清理边界

Teacher 仅在完整实测时间上下文相同时复用推理；改变上下文就重推 GVHMR。
原生裁剪 / 关键点 / 特征须满足来源和参数指纹；分割缓存仅在完整可见集合与几何都相同、
且没有双视角覆盖时复用。记录覆盖范围、配置、代码、源文件及产物 SHA，不能用“文件已存在”代替续跑验证。

已有每帧原点成员与 SMPL 重建验证收据。代码或数据实际改变后，运行：

```powershell
uv run --no-sync python tools/verify_badminton_native_repair.py --root outputs/badminton_body_repair_v2
uv run --no-sync python tools/verify_badminton_dataset.py --output-root outputs/badminton_body_repair_v2/final_labels --samples-per-session 64
uv run --no-sync python tools/verify_badminton_repair_comparison.py --url http://127.0.0.1:8772/ --output outputs/badminton_body_repair_v2/comparison/browser_verification_recheck.json
```

仅清理未改变的旁支实验时，校验封存收据的 SHA 链和页面资源即可。
不要直接覆盖封存收据；正式重验若要写回，需重新验证并封存完整关联收据。
比较页必须等待两版 RGB / mesh / cloud 全部就绪后原子提交六视图；
HTTP 与离线浏览器检查均覆盖十段全部 50 帧、原生 RGB、同步 seek 和完整网格。

| 应保留 | 原因 |
| --- | --- |
| v2 `final_labels/`、`comparison/`、`evaluation/` | 最终数据、连续 HTML 及共同观测评价 |
| v2 `native_poses/`、`identity_evidence/` | 拟合和身份收据绑定的原生证据 |
| v2 `leg_refinement_proposals/`、选择 / 应用收据、`leg_refinement_before/` | 候选、质量门限、连续选择和回滚的可追溯证据；未选中候选也属于封存策略证据 |
| v1 `labels/`、`tracking_final/` | 已验证 teacher 缓存与最终原生 tracking replay；仍用于复核 / 复现 |
| 原始交付、源录像、标定、原始审计与转换几何 | 正式输入、共同评价参照与训练发布边界 |
| 正在运行的 HTTP 服务日志 | 当前服务仍持有文件句柄；停止对应服务后才可清理 |

已删除早期 case / pilot 拟合、重复关联缓存、旧 tracking 候选、旧静态报告、
临时脚部优化及调试脚本。`maintenance_receipt.json` 记录本次清理类别与最终收据检查。
有效代码保留在 `hmr4d/lidar_mocap/`，正式命令保留在 `tools/`；`.local/` 不作为运行输入。
