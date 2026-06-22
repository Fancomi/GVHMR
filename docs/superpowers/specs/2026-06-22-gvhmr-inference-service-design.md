# GVHMR 单图推理服务设计

日期: 2026-06-22
状态: 已批准, 待实现

## 1. 目标

把 GVHMR 包成一个常驻 HTTP 服务, **仅在内网环境调用 (不使用 frp)**, 支持两条链路:

- **链路 1**: 输入图像 → YOLO 检测人体 bbox + GVHMR 出 SMPL
- **链路 2**: 输入图像 + bbox → 跳过检测, 直接 GVHMR 出 SMPL

输出格式与下游 `label_mocap` 的 COCO 标注模式严格对齐, 可直接落盘成 `player_0.json` 被标注器加载。

性能目标: 1 秒内可接收 30+ 并发调用, 内部排队串行处理; 队列满立即返回"繁忙"(HTTP 503)。

测试图像: `/root/paddlejob/workspace/env_run/penghaotian/datas/Test/0001.jpg` (1920x1080)。

## 2. 关键约束 (来自代码勘查)

### 2.1 下游 label_mocap 数据格式
`label_mocap/smpl_edit/coco_document.js` 的 `defaultAnnotation` 定义单条标注字段:
```
bbox:[x,y,w,h]   root_pos:[3]   root_rota:[3](轴角)   body_pose:[63]
betas:[10]       keypoints:[156](52 slot ×3)          right/left_hand_pose:[45]
occlution_joint:[52]  p3d:[]  segmentation:[]  category_id:1  iscrowd:0  area:0
```
顶层 COCO 文档: `{images:[{id,file_name}], annotations:[...], categories:[]}`。
单人写到 `json_results/player_0/player_0.json` (见 `dataset_paths.js: DATA_JSON_PATH`)。

### 2.2 SMPLX 逐字段对应
GVHMR `pred_smpl_params_incam` = `{global_orient:3, body_pose:63, betas:10, transl:3}`
(见 `hmr4d/model/gvhmr/pipeline/gvhmr_pipeline.py:78-83`)。
与 label_mocap 字段 1:1: `global_orient→root_rota`, `body_pose→body_pose`,
`betas→betas`, `transl→root_pos`(需坐标转换)。

### 2.3 坐标系转换 (唯一的非平凡换算)
- GVHMR: CV 相机系 (Y 下, Z 前), `perspective_projection` 用 `points/points[...,-1]` (`hmr_cam.py:169`)
- label_mocap: GL 系 (Y 上, -Z 前), `projection.js`: `u=fx*x/(-z)+cx, v=fy*(-y)/(-z)+cy`
- 换算 = 绕 X 轴 180°, `M = diag(1,-1,-1)`:
  - `root_pos = (tx, -ty, -tz)`
  - `root_rota = log_so3(M @ exp_so3(global_orient))`  (轴角→矩阵→左乘 M→回轴角)
  - `body_pose`, `betas` 不变 (关节相对旋转与坐标系无关)
- 因 SMPL 模板相同, `P_gl = M @ P_cv` 严格成立, 投影像素一致。

### 2.4 相机内参
GVHMR `estimate_K` (hmr_cam.py:10): `focal = sqrt(W²+H²)`, `cx=W/2, cy=H/2`。
1920x1080 → focal≈2202.9, cx=960, cy=540。
**注意**: label_mocap `app.js:344` 当前写死 `fx=1850`。服务在输出 json 的 image 条目里附
`cam_K:{fx,fy,cx,cy}`, 需 label_mocap 改读它, 否则 mesh 叠加错位。这是已知的下游改动项。

### 2.5 单图推理风险
GVHMR 是时序模型 (relative_transformer + cam_angvel)。单图 length=1 会退化, 姿态质量可能不如视频。
静态相机: `R_w2c = eye(3)`, `cam_angvel` 用零序列。实现后用 0001.jpg 实测确认可用。

### 2.6 预处理模块单图能力
- `vitfeat_extractor.py: get_batch` 支持 `path_type="image"` 与 `"np"`(传 ndarray)。
- `VitPoseExtractor.extract` / `Extractor.extract_video_features` 接受 `torch.Tensor` 直接跳过读视频。
- `Tracker` (YOLO) 面向视频; 单图改用 `YOLO.predict` 取 person bbox, 不用 track。

## 3. 架构: Flask + 有界队列 + 单 GPU 常驻 worker

```
            HTTP (并发)              线程安全队列              GPU 串行
 client ──────────────▶ Flask handler ──put(job)──▶ Queue(maxsize=N) ──▶ worker thread
                            │                            │                    │ (常驻模型)
                            │◀──── future.result() ──────┘                    │
                            ▼                          队列满→拒绝            推理
                     200 / 503 busy                  (return 503)         infer_core
```

- **启动加载一次**: YOLO, ViTPose, HMR2 Extractor, GVHMR DemoPL → 常驻显存。
- **Flask** `threaded=True` 接并发连接; 每请求构造 job 放入 `Queue`。
- **队列满** (`queue.Full`) → 立刻 `503 {"error":"busy"}`, 不阻塞。
- **单 worker 线程** FIFO 取 job, 串行跑 GPU, 用 `concurrent.futures.Future` 把结果回传对应请求。
- worker 数量是参数 (默认 1)。后续提速可加 worker 绑空闲 GPU (机器 GPU4-7 空闲), 升级为多 worker 并行, 不改 API。

### 3.1 为何串行
GPU 推理本身串行最高效; 单 worker 避免显存竞争与 OOM。队列吸收突发并发, 满则快速失败 (繁忙), 符合需求。

## 4. 推理核 `infer_core.py`

单一职责: `infer(image_bgr, bbox_xywh=None) -> coco_annotation_dict`。

```
1. 准备 K_fullimg = estimate_K(W,H)
2. 取 bbox:
   - bbox_xywh is None → YOLO.predict(image, classes=0) → 选最佳框(面积×中心) → xyxy
   - 否则 → 用传入 [x,y,w,h] 转 xyxy
3. bbx_xys = get_bbx_xys_from_xyxy(xyxy[None], base_enlarge=1.2)  # (1,3)
4. ViTPose.extract(img_tensor) → kp2d (1,17,3)         # 经 get_batch(path_type=np)
5. HMR2.extract_video_features(img_tensor) → (1,1024)
6. data = {length=1, bbx_xys, kp2d, K_fullimg, cam_angvel=zeros(1,6), f_imgseq}
7. DemoPL.predict(data, static_cam=True) → pred_smpl_params_incam
8. 坐标转换 (§2.3): incam → label_mocap 系
9. 投影 24 关节 → keypoints[156] (复用 demo export_json 的 SMPL 前向+perspective_projection)
10. 组 COCO annotation dict
```

公用预处理 batch 构造 (get_batch) 抽一个 helper, 两条链路共用 (区别只在第 2 步 bbox 来源)。

## 5. API

`POST /gvhmr/infer`  Content-Type: application/json
```json
请求: { "image_b64": "<base64 jpg/png>", "bbox": [x,y,w,h] }   // bbox 可选
响应 200: {
  "images":[{"id":0,"file_name":"input.jpg","width":1920,"height":1080,
             "cam_K":{"fx":2202.9,"fy":2202.9,"cx":960,"cy":540}}],
  "annotations":[{ bbox, root_pos, root_rota, body_pose, betas, keypoints,
                   right_hand_pose, left_hand_pose, occlution_joint,
                   id:0, image_id:0, category_id:1, ... }],
  "categories":[]
}
响应 503: {"error":"busy","detail":"queue full"}
响应 400: {"error":"bad_request","detail":"..."}
```
`GET /health` → `{"status":"ok","queue":<size>,"maxsize":N}`。

传输用 base64 JSON: 与之前 flask demo 风格一致, 纯文本好调试, 单张 1080p jpg ~600KB 可接受。

## 6. 文件结构

```
tools/serve/infer_core.py     # 推理核 (§4), 模型类常驻封装 + infer()
tools/serve/verify_single.py  # 阶段1: 不起服务, 跑通 0001.jpg → 写 player_0.json, 肉眼验证
tools/serve/app.py            # 阶段2: Flask + Queue + worker (§3,§5)
scripts/install_service.sh    # 在 gvhmr 环境补装 flask (用户要求: install 放进 script)
scripts/run_service.sh        # 启动服务 (用户要求: 服务脚本同)
```

## 7. 实现顺序 (层层递进)

1. **infer_core + verify_single**: 跑通 0001.jpg, 产出 player_0.json, 确认 SMPL 投影对齐 (肉眼/数值)。这是正确性地基。
2. **app.py 起服务**: 包 Flask + 有界队列 + worker; 本地 curl /health 与 /infer 验证两条链路 + 503 繁忙。
3. **内网验证**: 同机/内网另一进程 client 调用 `127.0.0.1:port` 验证全链路 (不使用 frp)。

## 8. 测试与验证

- 阶段1: `verify_single.py` 输出 json 字段齐全、维度正确 (root_pos 3, body_pose 63, betas 10, keypoints 156); 投影 keypoints 落在人体上。
- 阶段2: 单请求 200 正确; 并发压测 (30+ 并发) 验证排队 + 队列满 503; 两条链路 (有/无 bbox) 都正确。
- 阶段3: 内网另一进程/另一台机器 client 调用拿到与本地一致的结果。

## 9. 安全说明

服务无鉴权 (内网场景, 与现有 flask demo 一致)。base64 图像解码有大小上限保护, 防止超大 body 打爆内存。生产化鉴权与公网暴露不在本期范围。

## 10. 不做 (YAGNI)

- 不做 global (世界系) 轨迹: 单图无意义, 只出 incam。
- 不做视频/多帧: 本期单图。
- 不做多人: 下游 v1 单人 (player_0), 只出最佳单框。
- 不做异步 job/轮询: 同步 + 队列已满足需求。
- 不做渲染 overlay 返回: 下游标注器自己渲染。
