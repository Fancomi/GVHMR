# 数据目录与历史来源映射

正式训练归档根目录是 `D:/PROJECT/LIDAR/datas/lidar2smpl_badminton`。
旧 `D:/PROJECT/LABEL/datas/lidar2smpl_badminton` 是指向它的兼容 Junction，
用于保留历史封存清单的绝对路径。不要把该 Junction 当成重复源目录清理。

原始羽毛球数据采用另一套显式映射：

```text
D:/PROJECT/LIDAR/datas/badminton/data-relocations.json
SHA256 c5e86b2ba0cbb297b41e9d765997bceee1df3da46c942496b57a367fbc89119b
```

该清单将旧 `upstream` 和指定 `device-time-sources` 文件逐项映射到同一
source ID 的 `sessions` 文件，并绑定目标字节数和 SHA256。每条命令加载一次
固定映射；命中映射后只读取对应目标，失败即报错，不搜索其他文件，也不回退旧目录。
后续扩充映射使用新文件名及新 SHA，保留已被封存输入绑定的原清单。
来源 index、clock、已发布训练清单及 SMPL 数值不改写；clock 中相对 raw-index
仍按原逻辑父目录解析，再应用映射。

准备来源或启动历史看板时显式传入两个参数：

```powershell
uv run --no-sync python tools/serve_badminton_canonical_tracks.py `
  --manifest outputs/badminton_1015_v2_repair/temporal_identity/paired/datasets/dataset_manifest.json `
  --port 8774 `
  --data-relocations D:/PROJECT/LIDAR/datas/badminton/data-relocations.json `
  --data-relocations-sha256 c5e86b2ba0cbb297b41e9d765997bceee1df3da46c942496b57a367fbc89119b `
  --viewer-output outputs/badminton_relocation_runtime_acceptance_20261021/canonical_8774/viewer
```

`prepare_badminton_source.py`、`serve_badminton_rallies.py` 和
`serve_badminton_body_repair.py` 支持同一对映射参数。
三个看板启动工具的 `--viewer-output` 将运行期页面和映射收据放到独立目录；
修复候选看板的运行期状态文件也放在该目录。
无 live build status 的 canonical 发布看板从 SHA 绑定的正式发布清单读取接纳数，
从真实 fit reports 读取拟合统计。

8767/8771 各有七个来源，其中三个使用旧 SessionIndex，另外四个使用显式
source index。仅审核三个 legacy 来源不足以豁免整个端口；192753 的原生视频、
clock 和 9,050 个 PNG 原来依赖待清理的 device 目录，需要带映射重启服务。
未传参数时保留原行为，不自动启用全局映射，也不改变相机接纳门禁。

迁移验收输出在 `outputs/badminton_relocation_runtime_acceptance_20261021/`。
检查包括封存输入 SHA 不变、拒绝旧域打开的 guard、实际点云/原生 RGB/容器 PTS，
以及真实 Chrome 的完整网格、双相机投影和严格 RGBA 一致性。
`tools/verify_badminton_relocated_viewer.py` 可在物理删除后对相同 URL 再次验收；
8767/8771 额外传 `--without-canonical-publication`。
