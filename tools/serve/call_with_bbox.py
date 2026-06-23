#!/usr/bin/env python3
"""链路2 示例: 输入 图像 + bbox -> SMPL (跳过 YOLO 检测, 用你给的框)。

Mac 上直接运行 (只用标准库 urllib, 无需 requests):
    python3 call_with_bbox.py <图片路径> <x> <y> <w> <h>
不传 bbox 则退化为链路1 (服务端自动检测):
    python3 call_with_bbox.py <图片路径>
"""
import base64
import json
import sys
import urllib.request

URL = "http://10.52.104.78:8666/gvhmr/infer"


def main():
    if len(sys.argv) < 2:
        print("用法: python3 call_with_bbox.py <图片路径> [x y w h]")
        sys.exit(1)
    img_path = sys.argv[1]
    img_b64 = base64.b64encode(open(img_path, "rb").read()).decode()

    payload = {"image_b64": img_b64, "file_name": img_path.split("/")[-1]}
    if len(sys.argv) >= 6:
        x, y, w, h = (float(v) for v in sys.argv[2:6])
        payload["bbox"] = [x, y, w, h]  # 链路2: [x, y, w, h]
        print(f"[链路2] 带 bbox = {payload['bbox']}")
    else:
        print("[链路1] 不带 bbox, 服务端 YOLO 自动检测")

    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode()}")  # 503 busy / 400 bad_request
        sys.exit(1)

    doc = json.loads(resp.read())
    ann = doc["annotations"][0]
    print("HTTP", resp.status)
    print("bbox     :", [round(v, 1) for v in ann["bbox"]])
    print("root_pos :", [round(v, 3) for v in ann["root_pos"]])
    print("root_rota:", [round(v, 3) for v in ann["root_rota"]])
    print("dims     : body_pose=%d betas=%d keypoints=%d" % (
        len(ann["body_pose"]), len(ann["betas"]), len(ann["keypoints"])))
    print("cam_K    :", doc["images"][0]["cam_K"])

    # 落盘成 label_mocap 可加载的 player_0.json
    out = "player_0.json"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
