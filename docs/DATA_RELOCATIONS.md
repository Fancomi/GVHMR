# 数据目录与历史来源映射

正式训练归档根目录是 `E:/Datas/lidar2smpl_badminton`。
旧 `D:/PROJECT/LIDAR/datas/lidar2smpl_badminton` 是指向该 E 盘根的兼容 Junction，
更早的 `D:/PROJECT/LABEL/datas/lidar2smpl_badminton` 经它继续解析。录制旧根
`D:/PROJECT/LIDAR/datas/badminton` 同样只保留到 `E:/Datas/badminton-recoding` 的 Junction。
这些入口保留封存清单的绝对路径，实际数据都在 E 盘；不要当成重复实体清理。

当前录制根为 `E:/Datas/badminton-recoding`，正式素材及其元数据在 `sessions/`。
从 `sessions/_meta/canonical-ready.json` 选择已发布的 release 和 SHA，再显式使用
`prepare_badminton_source.py --final-current-release`；参数见
[Source and time contracts](INSTALL.md#source-and-time-contracts)。该 CPU 入口读取当前
PNG/时钟/相机信息，保留 v2 已退役逐点映射的限制，不授予旧训练或人体生产资格。
`gvhmr_temp` 仅为临时元数据和稀疏点证据，不能替代原生录制或上述独立训练归档。

以下是历史显式映射合同，使用时必须另行提供仍有完整兼容输入的映射。
当前录制根已经不保留 `data-relocations.json`、旧 `collections/` 等输入树；
不要把下列旧文件名、SHA 或历史输出当作当前素材入口。历史映射的原身份为：

```text
data-relocations.json (historical; not present in the current recording root)
SHA256 c5e86b2ba0cbb297b41e9d765997bceee1df3da46c942496b57a367fbc89119b
```

该历史清单将旧 `upstream` 和指定 `device-time-sources` 文件逐项映射到同一
source ID 的 `sessions` 文件，并绑定目标字节数和 SHA256。每条命令加载一次
固定映射；命中映射后只读取对应目标，失败即报错，不搜索其他文件，也不回退旧目录。
后续扩充映射使用新文件名及新 SHA，保留已被封存输入绑定的原清单。
来源 index、clock、已发布训练清单及 SMPL 数值不改写；clock 中相对 raw-index
仍按原逻辑父目录解析，再应用映射。

仅对明确保留的完整历史素材及其封存输入，启动时显式传入该素材实际映射和 SHA：

```powershell
uv run --no-sync python tools/serve_badminton_canonical_tracks.py `
  --manifest outputs/badminton_1015_v2_repair/temporal_identity/paired/datasets/dataset_manifest.json `
  --port 8774 `
  --data-relocations <retained-compatible-relocations.json> `
  --data-relocations-sha256 <that-file-sha256> `
  --viewer-output outputs/badminton_relocation_runtime_acceptance_20261021/canonical_8774/viewer
```

`prepare_badminton_source.py`、`serve_badminton_rallies.py` 和
`serve_badminton_body_repair.py` 支持同一对映射参数。
三个看板启动工具的 `--viewer-output` 将运行期页面和映射收据放到独立目录；
修复候选看板的运行期状态文件也放在该目录。
无 live build status 的 canonical 发布看板从 SHA 绑定的正式发布清单读取接纳数，
从真实 fit reports 读取拟合统计。

历史 8767/8771 各有七个来源，其中三个使用旧 SessionIndex，另外四个使用显式
source index。当时仅审核三个 legacy 来源不足以豁免整个端口；192753 的原生视频、
clock 和 9,050 个 PNG 依赖待清理的 device 目录，需带对应映射重启服务。
未传参数时保留原行为，不自动启用全局映射，也不改变相机接纳门禁。

当时的迁移验收输出在 `outputs/badminton_relocation_runtime_acceptance_20261021/`。
检查包括封存输入 SHA 不变、拒绝旧域打开的 guard、实际点云/原生 RGB/容器 PTS，
以及真实 Chrome 的完整网格、双相机投影和严格 RGBA 一致性。
对仍保留完整输入的历史看板，可用 `tools/verify_badminton_viewer.py --url <viewer-url>` 检查
原生双相机播放、扫描同步、完整网格、投影和严格 RGBA 一致性；
显式来源映射与封存输入不变由 `tests/test_lidar_runtime_relocations.py` 回归保护。
