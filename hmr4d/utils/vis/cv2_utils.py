import torch
import cv2
import numpy as np
from hmr4d.utils.wis3d_utils import get_colors_by_conf


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    elif isinstance(x, list):
        return np.array(x)
    return x.clone().cpu().numpy()


def draw_bbx_xys_on_image(bbx_xys, image, conf=True):
    assert isinstance(bbx_xys, np.ndarray)
    assert isinstance(image, np.ndarray)
    image = image.copy()
    lu_point = (bbx_xys[:2] - bbx_xys[2:] / 2).astype(int)
    rd_point = (bbx_xys[:2] + bbx_xys[2:] / 2).astype(int)
    color = (255, 178, 102) if conf == True else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, lu_point, rd_point, color, 2)
    return image


def draw_bbx_xys_on_image_batch(bbx_xys_batch, image_batch, conf=None):
    """conf: if provided, list of bool"""
    use_conf = conf is not None
    bbx_xys_batch = to_numpy(bbx_xys_batch)
    assert len(bbx_xys_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xys_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i], conf[i]))
        else:
            image_batch_out.append(draw_bbx_xys_on_image(bbx_xys_batch[i], image_batch[i]))
    return image_batch_out


def draw_bbx_xyxy_on_image(bbx_xys, image, conf=True):
    bbx_xys = to_numpy(bbx_xys)
    image = to_numpy(image)
    color = (255, 178, 102) if conf == True else (128, 128, 128)  # orange or gray
    image = cv2.rectangle(image, (int(bbx_xys[0]), int(bbx_xys[1])), (int(bbx_xys[2]), int(bbx_xys[3])), color, 2)
    return image


def draw_bbx_xyxy_on_image_batch(bbx_xyxy_batch, image_batch, mask=None, conf=None):
    """
    Args:
        conf: if provided, list of bool, mutually exclusive with mask
        mask: whether to draw, historically used
    """
    if mask is not None:
        assert conf is None
    if conf is not None:
        assert mask is None
    use_conf = conf is not None
    bbx_xyxy_batch = to_numpy(bbx_xyxy_batch)
    image_batch = to_numpy(image_batch)
    assert len(bbx_xyxy_batch) == len(image_batch)
    image_batch_out = []
    for i in range(len(bbx_xyxy_batch)):
        if use_conf:
            image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i], conf[i]))
        else:
            if mask is None or mask[i]:
                image_batch_out.append(draw_bbx_xyxy_on_image(bbx_xyxy_batch[i], image_batch[i]))
            else:
                image_batch_out.append(image_batch[i])
    return image_batch_out


def draw_kpts(frame, keypoints, color=(0, 255, 0), thickness=2):
    frame_ = frame.copy()
    for x, y in keypoints:
        cv2.circle(frame_, (int(x), int(y)), thickness, color, -1)
    return frame_


def draw_kpts_with_conf(frame, kp2d, conf, thickness=2):
    """
    Args:
        kp2d: (J, 2),
        conf: (J,)
    """
    frame_ = frame.copy()
    conf = conf.reshape(-1)
    colors = get_colors_by_conf(conf)  # (J, 3)
    colors = colors[:, [2, 1, 0]].int().numpy().tolist()
    for j in range(kp2d.shape[0]):
        x, y = kp2d[j, :2]
        c = colors[j]
        cv2.circle(frame_, (int(x), int(y)), thickness, c, -1)
    return frame_


def draw_kpts_with_conf_batch(frames, kp2d_batch, conf_batch, thickness=2):
    """
    Args:
        kp2d_batch: (B, J, 2),
        conf_batch: (B, J)
    """
    assert len(frames) == len(kp2d_batch)
    assert len(frames) == len(conf_batch)
    frames_ = []
    for i in range(len(frames)):
        frames_.append(draw_kpts_with_conf(frames[i], kp2d_batch[i], conf_batch[i], thickness))
    return frames_


def draw_skeleton(img, keypoints, skeleton_def, conf_thr=0, color=(0, 255, 0), thickness=4):
    """Draw skeleton on image
    Args:
        img: image array
        keypoints: (J, 2) or (J, 3) with confidence
        skeleton_def: list of [idx1, idx2] bone connections
        conf_thr: confidence threshold
        color: BGR color tuple
        thickness: line thickness
    """
    use_conf_thr = keypoints.shape[1] == 3
    img = img.copy()
    for bone in skeleton_def:
        if use_conf_thr:
            kp1, kp2 = keypoints[bone[0]][:2].astype(int), keypoints[bone[1]][:2].astype(int)
            kp1_c, kp2_c = keypoints[bone[0]][2], keypoints[bone[1]][2]
            if kp1_c > conf_thr and kp2_c > conf_thr:
                img = cv2.line(img, tuple(kp1), tuple(kp2), color, thickness)
            if kp1_c > conf_thr:
                img = cv2.circle(img, tuple(kp1), 6, color, -1)
            if kp2_c > conf_thr:
                img = cv2.circle(img, tuple(kp2), 6, color, -1)
        else:
            kp1, kp2 = keypoints[bone[0]][:2].astype(int), keypoints[bone[1]][:2].astype(int)
            img = cv2.line(img, tuple(kp1), tuple(kp2), color, thickness)
    return img


# COCO17 skeleton definition
COCO17_SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
                   [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

# SMPL skeleton definition (24 joints)
SMPL_SKELETON = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9],
                 [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17],
                 [16, 18], [17, 19], [18, 20], [19, 21]]


def draw_coco17_skeleton(img, keypoints, conf_thr=0, color=(0, 255, 0), thickness=4):
    """Draw COCO17 skeleton on image"""
    return draw_skeleton(img, keypoints, COCO17_SKELETON, conf_thr, color, thickness)


def draw_smpl_skeleton(img, keypoints, color=(0, 0, 255), thickness=4):
    """Draw SMPL skeleton on image"""
    return draw_skeleton(img, keypoints, SMPL_SKELETON, 0, color, thickness)


def draw_coco17_skeleton_batch(imgs, keypoints_batch, conf_thr=0, color=(0, 255, 0)):
    """Draw COCO17 skeleton on batch of images"""
    assert len(imgs) == len(keypoints_batch)
    keypoints_batch = to_numpy(keypoints_batch)
    return [draw_coco17_skeleton(imgs[i], keypoints_batch[i], conf_thr, color) for i in range(len(imgs))]


def draw_dual_skeletons_batch(imgs, vitpose_batch, smpl_kp2d_batch, conf_thr=0):
    """Draw both ViTPose (green) and SMPL (red) skeletons
    Args:
        imgs: list of images
        vitpose_batch: (B, 17, 3) ViTPose COCO17 keypoints with confidence
        smpl_kp2d_batch: (B, 24, 2) SMPL projected keypoints
        conf_thr: confidence threshold for ViTPose
    """
    assert len(imgs) == len(vitpose_batch) == len(smpl_kp2d_batch)
    vitpose_batch = to_numpy(vitpose_batch)
    smpl_kp2d_batch = to_numpy(smpl_kp2d_batch)
    
    imgs_out = []
    for i in range(len(imgs)):
        img = draw_coco17_skeleton(imgs[i], vitpose_batch[i], conf_thr, color=(0, 255, 0), thickness=3)
        img = draw_smpl_skeleton(img, smpl_kp2d_batch[i], color=(0, 0, 255), thickness=3)
        imgs_out.append(img)
    return imgs_out
