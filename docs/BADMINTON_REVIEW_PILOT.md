# R11 原生回合评审试跑

输入合同是 `badminton-review-pilot-v1`，绑定 R11 五件原文件、真实评审授权、
源 index 和原生双机确认的回合范围。所有输出保持 `review_only=true`、
`training_published=false`；不修改相机正式接纳状态或已发布训练集。

| 来源 | 范围 | 输出根目录 | 当前已完成阶段 |
| --- | --- | --- | --- |
| 20261019_152542 | 5 回合，525 扫描，52.333915 秒 | `outputs/badminton_1019_r11_pilot_v1` | 双机独立 teacher、原生 pose、人工 identity、拟合、成员/SMPL 验证及播放器 |
| 20261019_152750 | 9 回合，1,260 扫描，125.701709 秒 | `outputs/badminton_1019_r11_pilot_second_v1` | 双机独立 teacher、原生 pose、人工 identity、拟合及成员/SMPL 验证完成；独立播放器 8782 |

两段合计 178.035624 秒。第二段用独立输出，不修改第一段封存输入。
第二段 CPU 收据是 `cpu_prepare_receipt.json`，SHA256
`2a9b482a9f1857dea91fa7e6996ffd40f92946e477fedac5bb96be0c327fa59e`。

两台相机分别执行真实 GVHMR。既有 v2 拟合使用 cam2 prior、双机独立原生
2D 观测及真实 LiDAR；独立 cam1 prior 保留用于一致性评审，不做未经 identity
证实的跨相机 3D 平均。所有轨迹保留局部编号；未知关系分开保留，identity
库存必须经过实际原生图评审后才能进入拟合。

第一段完成 1,050 / 1,050 个 body-QA 候选。实测阶段耗时：cam2 teacher
433.68 秒、cam1 teacher 合计 279.23 秒、双机原生 pose 102.98 秒、拟合
347.74 秒、成员/原生 SMPL 重建验证约 62.08 秒。阶段 wall time 包含相应
CPU、GPU 和 IO，不等同于纯 GPU kernel 时间，也不含全部人工评审等待。

原始可见点到 SMPL 表面均值为 24.12 mm；原始未按拟合过滤的高置信度
2D 观测重投影均值是 cam1 7.98 px、cam2 4.40 px。该点到表面指标是单侧
可见表面残差，不是人体真值误差。21 / 1,050 个候选存在大于 20 mm 的
地面穿透；scan 443 约 110.39 mm，作为明确坏例保留评审，不能据整体均值
声称所有姿态最优。详见第一段 `quality_report.json` 和 `outlier_review.json`。

播放器 `http://127.0.0.1:8781/` 展示第一段完整 6,890 顶点 SMPL、真实点云及
双机原生图。五个连续片段、严格同后端 RGBA、三维/投影同扫描提交、原生 PTS
及一屏视野已经验收；迁移删除后另有完整复验。

第二段 manifest 冻结了 [历史来源路径映射](DATA_RELOCATIONS.md)。每条命令从
原始 SHA 绑定的 pilot manifest 显式加载一次 resolver，核对 registry、snapshot、
plan 和 pilot_input 的相同绑定；不使用全局自动映射，也不回退到旧目录。
旧无映射输入保持原行为，运行期路径参数不能替换已冻结的新绑定。

```powershell
uv run --no-sync python tools/run_badminton_pilot.py `
  --root outputs/badminton_1019_r11_pilot_second_v1 --stage teacher `
  --gpu-lease D:/PROJECT/LIDAR/datas/badminton/collections/gvhmr-1019-r11-pilot-20261021/handoff/gpu-lease-second-source.json `
  --threads 2
```

只有 Sports360 交出已保存的完整 epoch/CAL、实际 CUDA worker 退出，并把新的
独立 lease 记录为 held 后才能运行该命令。两次 lease 现在均已 released，不能复用。
第二段于 2026-10-21 20:32:06 UTC 归还 GPU；实际 CUDA worker 及拟合进程树
全部退出，Sports360 随后已从保存的 epoch 3 恢复。后续验证和播放器只做 CPU 数据处理。

第二段实测 cam2 teacher 1,045.99 秒、cam1 合计 828.24 秒，双机原生 pose
332.73 秒、原生 v2 拟合 910.99 秒、成员/SMPL 验证 151.94 秒。CPU 准备 CLI
36.21 秒。它们是实际阶段 wall time，包含 CPU、GPU、IO，不是纯 kernel 时间。

`coverage_audit.json` 及 SHA 绑定的 `coverage_by_scan_track.jsonl` 逐一记录
18 条保留轨迹和 2,520 个参考 track-scan 单元。实际拟合 2,503、QA 通过 2,502、
QA 拒收 1、未拟合 17。这个参考分母来自每窗两条保留轨迹，不能当作独立人工
占场真值；2,520 张原生 pose 图像则来自 1,260 扫描乘两台相机，是另一种计数。
17 个未拟合单元都在原回合 32 的棕发 raw track 2、scan 5278–5294；最近原生
teacher 时间超过既有 18 ms 采样门限，不外推补标签。
全部 34 张双机原生画面已实际评审：棕发运动员在 cam2 里仍然可见，cam1 后段才
逐步离开视野。因此缺口不等于无人或视频缺帧；现有冻结记录不足以进一步区分
primary teacher 的 ROI 排除和检测置信遗漏。证据见 `coverage_native_review/`。

第二段全部已拟合人帧的表面残差均值为 23.96 mm，原始高置信度观测重投影均值
cam1 / cam2 为 8.15 / 4.49 px。62 / 2,503 行地面穿透超过 20 mm，14 行超过
50 mm；scan 1198 穿透 170.37 mm，被既有 QA 拒收。QA 通过的 scan 5277 仍有
98.41 mm 穿透，原始 cam1 / cam2 重投影 41.01 / 23.52 px，是明确坏例，不能用
整体均值掩盖。scan 2576→2577 的骨盆位移为 0.599 m / 99.984 ms，触发评审筛选；
原生双机证据保留在 `interior_review/`。评审不改写原 QA，也不构成人体真值认证。

合并入口 `http://127.0.0.1:8783/` 展示十段各五秒的原生连续评审。
`tools/serve_badminton_pilot_review.py` 核对两段五件相机原文件、拓扑、显式路径映射和
已验证的数据集绑定，复用 8781 / 8782 原生几何，直接读取校验过的原始 MP4。
它不复制人体标签。两段 snapshot digest 各自保留，不能拿第一段 digest 代表第二段。
真实 Chrome 十段均通过：每段实际超过 5 秒、51–52 个扫描，双机严格 RGBA 零差异、
完整 6,890 顶点/人及三维/投影同扫描提交。另有四种屏幕尺寸的真实控件回归，
以及拒收/大位移/QA 通过坏例的原生图与完整网格检查。收据位于
`outputs/badminton_1019_r11_review/`。这些候选未进入正式训练集。
