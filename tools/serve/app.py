"""GVHMR inference HTTP service (intranet only, no frp).

Architecture (see design spec 3): Flask accepts concurrent connections; each
request is put on a bounded Queue. A single resident GPU worker thread pulls
jobs FIFO and runs inference serially. When the queue is full, the request is
rejected immediately with 503 busy (backpressure), so the service can absorb
bursts of 30+ concurrent calls and stay responsive.

Endpoints:
  POST /gvhmr/infer  { image_b64, bbox?:[x,y,w,h] } -> COCO doc | 503 busy | 400
  GET  /health       -> { status, queue, maxsize }
"""
import argparse
import base64
import os
import queue
import threading
from concurrent.futures import Future

import cv2
import numpy as np
from flask import Flask, request, jsonify

# Bound request body so a giant base64 payload can't exhaust memory.
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_job_queue: "queue.Queue" = None  # set in main()
_infer = None  # GVHMRInfer, lazily created inside the worker thread (CUDA ctx)
_ready = threading.Event()


def _worker():
    """Single GPU worker: build models once, then serve jobs FIFO and serially."""
    global _infer
    from infer_core import GVHMRInfer

    _infer = GVHMRInfer()
    _ready.set()
    while True:
        future, image_bgr, bbox, file_name = _job_queue.get()
        if future.set_running_or_notify_cancel():
            try:
                doc = _infer.infer(image_bgr, bbox_xywh=bbox, file_name=file_name)
                future.set_result(doc)
            except Exception as e:  # noqa: BLE001 - report any infer error back to caller
                future.set_exception(e)
        _job_queue.task_done()


def _decode_image(image_b64):
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("failed to decode image")
    return img


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if _ready.is_set() else "loading",
        "queue": _job_queue.qsize(),
        "maxsize": _job_queue.maxsize,
    })


@app.route("/gvhmr/infer", methods=["POST"])
def infer():
    if not _ready.is_set():
        return jsonify({"error": "loading", "detail": "models not ready"}), 503

    body = request.get_json(silent=True)
    if not body or "image_b64" not in body:
        return jsonify({"error": "bad_request", "detail": "missing image_b64"}), 400
    try:
        image_bgr = _decode_image(body["image_b64"])
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "bad_request", "detail": str(e)}), 400

    bbox = body.get("bbox")
    if bbox is not None and (not isinstance(bbox, (list, tuple)) or len(bbox) != 4):
        return jsonify({"error": "bad_request", "detail": "bbox must be [x,y,w,h]"}), 400
    file_name = body.get("file_name", "input.jpg")

    future = Future()
    try:
        _job_queue.put_nowait((future, image_bgr, bbox, file_name))
    except queue.Full:
        return jsonify({"error": "busy", "detail": "queue full"}), 503

    try:
        doc = future.result()
    except ValueError as e:
        return jsonify({"error": "bad_request", "detail": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "infer_error", "detail": str(e)}), 500
    return jsonify(doc), 200


def main():
    global _job_queue
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--maxsize", type=int, default=64, help="bounded queue size; full -> 503 busy")
    args = p.parse_args()

    _job_queue = queue.Queue(maxsize=args.maxsize)
    threading.Thread(target=_worker, daemon=True).start()
    # Flask threaded=True so concurrent connections can enqueue while the single
    # worker drains the queue; processes=1 keeps one CUDA context.
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
