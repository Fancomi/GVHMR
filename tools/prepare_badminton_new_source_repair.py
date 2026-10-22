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
    p.add_argument("--teacher-sampling-input", type=Path)
    p.add_argument("--formal-input-equivalence", type=Path)
    p.add_argument("--formal-input-equivalence-sha256")
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    if (a.formal_input_equivalence is None) != (a.formal_input_equivalence_sha256 is None):
        p.error("Formal equivalence path and SHA256 must be supplied together")
    if a.stage == "prepare":
        proof = (dict(path=str(a.formal_input_equivalence.resolve()), sha256=a.formal_input_equivalence_sha256)
                 if a.formal_input_equivalence else None)
        result = prepare_new_sources(a.source_root, a.root, formal_equivalence=proof)
        print(dict(windows=len(result["jobs"]), training_published=False))
    else:
        if not a.source or not a.window:
            p.error("Teacher execution requires one --source and --window")
        import torch
        import cv2
        torch.set_num_threads(a.threads)
        cv2.setNumThreads(a.threads)
        result = initial_teacher_window(a.root, a.source, a.window,camera=a.camera,
                                        sampling_input=a.teacher_sampling_input)
        print(dict(tracks=len(result["tracks"]), training_published=False))


if __name__ == "__main__":
    main()
