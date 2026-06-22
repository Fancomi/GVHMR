"""Intranet client test for the GVHMR service: both paths + concurrency/503.

Usage:
  python tools/serve/test_client.py --url http://127.0.0.1:8090 \
      --image /root/.../0001.jpg [--concurrency 40]
"""
import argparse
import base64
import json
import threading
import time

import requests


def encode(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def check_doc(doc):
    ann = doc["annotations"][0]
    dims = {"root_pos": 3, "root_rota": 3, "body_pose": 63, "betas": 10,
            "keypoints": 156, "bbox": 4}
    for k, n in dims.items():
        assert len(ann[k]) == n, f"{k}: {len(ann[k])} != {n}"
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8090")
    p.add_argument("--image", required=True)
    p.add_argument("--concurrency", type=int, default=40)
    args = p.parse_args()
    img_b64 = encode(args.image)
    base = args.url.rstrip("/")

    print("== health ==")
    print(requests.get(f"{base}/health").json())

    print("\n== path 1: image only (auto-detect) ==")
    t = time.time()
    r = requests.post(f"{base}/gvhmr/infer", json={"image_b64": img_b64})
    print("status", r.status_code, "elapsed %.2fs" % (time.time() - t))
    assert r.status_code == 200, r.text
    doc = r.json()
    check_doc(doc)
    print("bbox", [round(v, 1) for v in doc["annotations"][0]["bbox"]],
          "cam_K", doc["images"][0]["cam_K"])

    print("\n== path 2: image + bbox ==")
    det_bbox = doc["annotations"][0]["bbox"]
    r2 = requests.post(f"{base}/gvhmr/infer", json={"image_b64": img_b64, "bbox": det_bbox})
    print("status", r2.status_code)
    assert r2.status_code == 200, r2.text
    check_doc(r2.json())
    print("path2 root_pos", [round(v, 3) for v in r2.json()["annotations"][0]["root_pos"]])

    print(f"\n== concurrency: {args.concurrency} simultaneous posts ==")
    codes = {}
    lock = threading.Lock()

    def fire():
        rr = requests.post(f"{base}/gvhmr/infer", json={"image_b64": img_b64})
        with lock:
            codes[rr.status_code] = codes.get(rr.status_code, 0) + 1

    threads = [threading.Thread(target=fire) for _ in range(args.concurrency)]
    t = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("elapsed %.2fs" % (time.time() - t), "status counts:", codes)
    print("200 (served):", codes.get(200, 0), " 503 (busy):", codes.get(503, 0))
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
