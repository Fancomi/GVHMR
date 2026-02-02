from ultralytics import YOLO
from hmr4d import PROJ_ROOT

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path, center_priority=True):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        
        # 可选：按中心距离和面积综合排序
        if center_priority:
            id_score = {}
            cx_img, cy_img = w / 2, h / 2
            for k, v in id_to_bbx_xyxys.items():
                # 计算平均中心距离（归一化）
                cx = (v[:, 0] + v[:, 2]) / 2
                cy = (v[:, 1] + v[:, 3]) / 2
                dist = np.sqrt((cx - cx_img)**2 + (cy - cy_img)**2) / np.sqrt(w**2 + h**2)
                avg_dist = dist.mean()
                
                # 综合得分：面积权重 0.6，中心距离权重 0.4（距离越小越好）
                area_norm = id_area_sum[k] / max(id_area_sum.values())
                score = 0.6 * area_norm + 0.4 * (1 - avg_dist)
                id_score[k] = score
            id_sorted = sorted(id_score.keys(), key=lambda k: id_score[k], reverse=True)
        else:
            id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def get_one_track(self, video_path, center_priority=True):
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path, center_priority)
        track_id = id_sorted[0]
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)

        return bbx_xyxy_one_track
    
    def get_n_tracks(self, video_path, n_people=1, center_priority=True):
        """获取前 N 个人的轨迹"""
        track_history = self.track(video_path)
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path, center_priority)
        
        # 取前 N 个
        selected_ids = id_sorted[:n_people]
        tracks = []
        for track_id in selected_ids:
            frame_ids = torch.tensor(id_to_frame_ids[track_id])
            bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])
            
            # 插值缺失帧
            mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
            bbx_xyxy_track = rearrange_by_mask(bbx_xyxys, mask)
            missing_frame_id_list = get_frame_id_list_from_mask(~mask)
            bbx_xyxy_track = linear_interpolate_frame_ids(bbx_xyxy_track, missing_frame_id_list)
            
            # 平滑
            bbx_xyxy_track = moving_average_smooth(bbx_xyxy_track, window_size=5, dim=0)
            bbx_xyxy_track = moving_average_smooth(bbx_xyxy_track, window_size=5, dim=0)
            
            tracks.append(bbx_xyxy_track)
        
        return tracks
