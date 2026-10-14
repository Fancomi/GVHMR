"""Stage-1 verification: run GVHMR single-image inference on one image, write a
label_mocap-style player_0.json, and sanity-check field dims + keypoint overlay.

Usage:
  python tools/serve/verify_single.py --image /path/0001.jpg [--bbox x y w h] \
      --out outputs/serve_verify

Produces:
  <out>/json_results/player_0/player_0.json   (loadable by label_mocap)
  <out>/overlay.jpg                            (keypoints drawn for eyeball check)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from infer_core import GVHMRInfer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("X", "Y", "W", "H"))
    p.add_argument("--out", default="outputs/serve_verify")
    return p.parse_args()


def check_dims(ann):
    """Assert label_mocap schema field dimensions; raise on mismatch."""
    expect = {"root_pos": 3, "root_rota": 3, "body_pose": 63, "betas": 10,
              "keypoints": 156, "right_hand_pose": 45, "left_hand_pose": 45,
              "occlution_joint": 52, "bbox": 4}
    for k, n in expect.items():
        got = len(ann[k])
        assert got == n, f"field {k}: expected {n}, got {got}"
    print("[check] all field dims OK:", ", ".join(f"{k}={v}" for k, v in expect.items()))


def draw_overlay(image_bgr, ann, out_path):
    """Draw projected keypoints (conf>0) and bbox to verify they land on the body."""
    img = image_bgr.copy()
    kp = ann["keypoints"]
    for j in range(24):
        x, y, c = kp[j * 3], kp[j * 3 + 1], kp[j * 3 + 2]
        if c > 0:
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
    x, y, w, h = ann["bbox"]
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img)
    print(f"[overlay] wrote {out_path}")


def main():
    args = parse_args()
    image_bgr = cv2.imread(args.image)
    assert image_bgr is not None, f"failed to read {args.image}"
    print(f"[input] {args.image}  HxW={image_bgr.shape[0]}x{image_bgr.shape[1]}")

    infer = GVHMRInfer()
    file_name = Path(args.image).name
    doc = infer.infer(image_bgr, bbox_xywh=args.bbox, file_name=file_name)

    ann = doc["annotations"][0]
    check_dims(ann)
    print("[cam_K]", doc["images"][0]["cam_K"])
    print("[root_pos]", [round(v, 3) for v in ann["root_pos"]])
    print("[root_rota]", [round(v, 3) for v in ann["root_rota"]])

    out_dir = Path(args.out)
    json_path = out_dir / "json_results" / "player_0" / "player_0.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc, indent=2))
    print(f"[json] wrote {json_path}")

    draw_overlay(image_bgr, ann, out_dir / "overlay.jpg")


if __name__ == "__main__":
    main()
