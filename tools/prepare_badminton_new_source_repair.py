"""Prepare a new-source intake for existing native v2 repair; teacher is explicit and bounded."""
import argparse
from pathlib import Path
from hmr4d.lidar_mocap.new_source_repair import prepare_new_sources, initial_teacher_window


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("prepare", "teacher"), default="prepare")
    p.add_argument("--source")
    p.add_argument("--window")
    p.add_argument('--camera',choices=('cam1','cam2'),help='Independent secondary camera teacher; native primary stays unchanged')
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    if a.stage == "prepare":
        result = prepare_new_sources(a.source_root, a.root)
        print(dict(windows=len(result["jobs"]), training_published=False))
    else:
        if not a.source or not a.window:
            p.error("Teacher execution requires one --source and --window")
        import torch
        import cv2
        torch.set_num_threads(a.threads)
        cv2.setNumThreads(a.threads)
        result = initial_teacher_window(a.root, a.source, a.window,camera=a.camera)
        print(dict(tracks=len(result["tracks"]), training_published=False))


if __name__ == "__main__":
    main()
