"""Save a few real source video frames without changing the recordings."""
from pathlib import Path
import cv2
import json
import numpy as np


def main():
    source = Path('D:/PROJECT/LIDAR/datas/badminton/sessions')
    output = Path('outputs/badminton/audit')
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for session in sorted(source.iterdir()):
        if not session.is_dir():
            continue
        for camera in ('cam1', 'cam2'):
            path = session / 'output-zcam' / f'zcam_{session.name}' / f'{camera}.mp4'
            capture = cv2.VideoCapture(str(path))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for index in (120, count // 2):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f'Failed to decode {path} frame {index}')
                target = output / f'{session.name}_{camera}_{index}.jpg'
                cv2.imwrite(str(target), image)
                report.append({'session': session.name, 'camera': camera, 'record_index': index,
                               'image': str(target), 'video': str(path), 'frame_count': count})
            capture.release()
    (output / 'frames.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
