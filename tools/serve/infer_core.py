"""GVHMR single-image inference core.

Wraps detection + ViTPose + HMR2 features + GVHMR into one reusable object that
loads all models once and exposes infer(image_bgr, bbox_xywh=None) -> COCO
annotation dict aligned to the label_mocap schema.

Two paths share the same body:
  - bbox_xywh is None : YOLO detects the best person box, then SMPL.
  - bbox_xywh given   : skip detection, run SMPL on that box.

Coordinate conversion (see design spec 2.3): GVHMR is CV-camera frame (Y down,
Z fwd); label_mocap is GL frame (Y up, -Z fwd). M = diag(1,-1,-1):
  root_pos  = (tx, -ty, -tz)
  root_rota = log( M @ exp(global_orient) )
  body_pose / betas unchanged.
"""
import numpy as np
import torch
from hydra import initialize_config_module, compose
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from einops import einsum

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.preproc import Extractor, VitPoseExtractor
from hmr4d.utils.preproc.vitfeat_extractor import get_batch
from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy, perspective_projection
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.net_utils import to_cuda
from ultralytics import YOLO
from hmr4d import PROJ_ROOT

# CV->GL frame flip (rotate 180 about X). Persisted on CPU; moved per-use.
_M = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
# Single static frame: identity camera angular velocity as rotation-6d.
_STATIC_ANGVEL = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])


def _resolve_K(cam_K, w, h):
    """Build a (3,3) float K from caller input, or estimate from image size.

    cam_K accepts: None -> estimate_K(w,h); dict {fx,fy,cx,cy}; or a 3x3
    nested list / ndarray. Raises ValueError on malformed input.
    """
    if cam_K is None:
        return estimate_K(w, h)
    if isinstance(cam_K, dict):
        try:
            fx, fy, cx, cy = (float(cam_K[k]) for k in ("fx", "fy", "cx", "cy"))
        except (KeyError, TypeError, ValueError):
            raise ValueError("cam_K dict must have numeric fx, fy, cx, cy")
        K = torch.eye(3)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
        return K.float()
    arr = np.asarray(cam_K, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError("cam_K matrix must be 3x3 or a {fx,fy,cx,cy} dict")
    return torch.from_numpy(arr).float()


def _load_cfg():
    """Compose the demo Hydra config without any video-specific overrides."""
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=["video_name=__serve__", "static_cam=True"])
    return cfg


class GVHMRInfer:
    """Holds all models resident on one CUDA device; serial inference per call."""

    def __init__(self):
        cfg = _load_cfg()
        # Detection: YOLO predict (not track) for single images.
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")
        self.vitpose = VitPoseExtractor()
        self.extractor = Extractor()
        # GVHMR predictor.
        import hydra

        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        self.model = model.eval().cuda()
        # SMPL(X) forward for keypoint projection.
        self.smplx = make_smplx("supermotion").cuda()
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    def _detect_bbox_xyxy(self, image_bgr):
        """Return best person bbox (x1,y1,x2,y2) by area*center score, or None."""
        res = self.yolo.predict(image_bgr, classes=0, conf=0.5, verbose=False, device="cuda")[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()  # (N, 4)
        h, w = image_bgr.shape[:2]
        cx_img, cy_img = w / 2.0, h / 2.0
        diag = (w**2 + h**2) ** 0.5
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        dist = np.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2) / diag
        score = 0.6 * (areas / areas.max()) + 0.4 * (1 - dist)
        return xyxy[int(score.argmax())]

    @torch.no_grad()
    def infer(self, image_bgr, bbox_xywh=None, file_name="input.jpg", cam_K=None):
        """image_bgr: HxWx3 uint8 (BGR, as cv2 reads). Returns a COCO doc dict.

        bbox_xywh: optional [x, y, w, h]. If None, YOLO detects the person.
        cam_K: optional real intrinsics. Accepts {fx,fy,cx,cy} or a 3x3 matrix
               (list/ndarray). If None, falls back to estimate_K (diagonal-FOV
               guess). Real K improves depth (tz = 2f/(s*b)) and the CLIFF
               position prior the network consumes.
        """
        h, w = image_bgr.shape[:2]
        K_fullimg = _resolve_K(cam_K, w, h)  # (3, 3)

        # --- bbox (xyxy) ---
        if bbox_xywh is None:
            xyxy = self._detect_bbox_xyxy(image_bgr)
            if xyxy is None:
                raise ValueError("no person detected")
        else:
            x, y, bw, bh = bbox_xywh
            xyxy = np.array([x, y, x + bw, y + bh], dtype=np.float32)
        bbx_xyxy = torch.from_numpy(np.asarray(xyxy, dtype=np.float32))[None]  # (1, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (1, 3)

        # --- crop batch shared by ViTPose & HMR2 (RGB, single image as np path) ---
        # get_batch's "np" path iterates the leading axis, so pass (1,H,W,3).
        image_rgb = image_bgr[..., ::-1].copy()
        imgs, _ = get_batch(image_rgb[None], bbx_xys, img_ds=1.0, path_type="np")  # (1,3,256,256)

        kp2d = self.vitpose.extract(imgs, bbx_xys)  # (1, 17, 3); bbx_xys unused on tensor path
        f_imgseq = self.extractor.extract_video_features(imgs, bbx_xys)  # (1, 1024)

        data = {
            "length": torch.tensor(1),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg[None],  # (1, 3, 3)
            "cam_angvel": _STATIC_ANGVEL.clone(),  # (1, 6)
            "f_imgseq": f_imgseq,
        }
        pred = self.model.predict(data, static_cam=True)
        params = pred["smpl_params_incam"]  # dict of (1, D)
        return self._to_coco(params, K_fullimg, bbx_xys, file_name, w, h)

    def _to_coco(self, params, K_fullimg, bbx_xys, file_name, w, h):
        """Convert incam SMPL params (CV frame) to a label_mocap COCO doc (GL frame)."""
        # --- SMPL params: 1:1 fields, with CV->GL frame flip on root ---
        global_orient = params["global_orient"].reshape(1, 3).cpu()  # (1,3)
        body_pose = params["body_pose"].reshape(63).cpu().tolist()  # unchanged
        betas = params["betas"].reshape(-1)[:10].cpu().tolist()  # unchanged
        transl = params["transl"].reshape(3).cpu()  # (3,)

        R_cv = axis_angle_to_matrix(global_orient)  # (1,3,3)
        R_gl = _M @ R_cv  # left-multiply frame flip
        root_rota = matrix_to_axis_angle(R_gl).reshape(3).tolist()
        root_pos = [float(transl[0]), float(-transl[1]), float(-transl[2])]

        # --- keypoints: project 24 SMPL joints (incam, pixel space) ---
        smplx_out = self.smplx(**to_cuda({k: v for k, v in params.items()}))
        verts = torch.matmul(self.smplx2smpl, smplx_out.vertices[0])  # (6890,3)
        joints3d = einsum(self.J_regressor, verts, "j v, v i -> j i")  # (24,3)
        kp2d = perspective_projection(joints3d[None].cuda(), K_fullimg[None].cuda())[0].cpu()  # (24,2)
        keypoints = [0.0] * (52 * 3)
        for j in range(24):
            keypoints[j * 3 + 0] = float(kp2d[j, 0])
            keypoints[j * 3 + 1] = float(kp2d[j, 1])
            keypoints[j * 3 + 2] = 2.0

        # --- bbox [x,y,w,h] from the (enlarged) xys used for inference ---
        cx, cy, size = [float(v) for v in bbx_xys[0]]
        bx, by = cx - size / 2.0, cy - size / 2.0
        bbox = [bx, by, size, size]

        K = K_fullimg
        annotation = {
            "id": 0,
            "image_id": 0,
            "category_id": 1,
            "bbox": bbox,
            "root_pos": root_pos,
            "root_rota": root_rota,
            "body_pose": body_pose,
            "betas": betas,
            "keypoints": keypoints,
            "right_hand_pose": [0.0] * 45,
            "left_hand_pose": [0.0] * 45,
            "occlution_joint": [0] * 52,
            "p3d": [],
            "segmentation": [],
            "iscrowd": 0,
            "area": float(bbox[2] * bbox[3]),
        }
        image = {
            "id": 0,
            "file_name": file_name,
            "width": w,
            "height": h,
            "cam_K": {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                      "cx": float(K[0, 2]), "cy": float(K[1, 2])},
        }
        return {"images": [image], "annotations": [annotation], "categories": []}
