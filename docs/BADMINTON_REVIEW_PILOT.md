# R11 原生回合评审试跑

输入合同是 `badminton-review-pilot-v1`，绑定 R11 五件原文件、真实评审授权、
源 index 和原生双机确认的回合范围。所有输出保持 `review_only=true`、
`training_published=false`；不修改相机正式接纳状态或已发布训练集。

| 来源 | 范围 | 输出根目录 | 当前已完成阶段 |
| --- | --- | --- | --- |
| 20261019_152542 | 5 回合，525 扫描，52.333915 秒 | `outputs/badminton_1019_r11_pilot_v1` | 双机独立 teacher、原生 pose、人工 identity、拟合、成员/SMPL 验证及播放器 |
| 20261019_152750 | 9 回合，1,260 扫描，125.701709 秒 | `outputs/badminton_1019_r11_pilot_second_v1` | CPU 准备，36.21 秒 CLI wall；GPU 等待新的安全交接 |

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
独立 lease 记录为 held 后才能运行该命令。旧第一段 lease 已 released，不能复用。
后续阶段为 poses → identity 库存及人工原生评审 → fit → verify → viewer。
第二段 viewer 使用 8782，避免覆盖第一段服务。
