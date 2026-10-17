"""Associate actual foreground LiDAR points with calibrated athlete tracks."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .calibration import bbox_rotation_matrix


def court_polygon(calibration, camera):
    """Project the adopted court footprint onto the teacher's undistorted image."""
    bbox = calibration.data['bbox']
    regions = bbox.get('regions', bbox.get('boxes', [])) if isinstance(bbox, dict) else bbox
    if isinstance(regions, dict):
        regions = list(regions.values())
    ground = next((r for r in regions if str(r.get('role', r.get('name', r.get('id', '')))).lower() == 'ground'), None)
    if ground is None:
        # Snapshot keeps original bbox schema; find an explicit ground entry only.
        ground = bbox.get('ground') if isinstance(bbox, dict) else None
    if ground is None:
        raise ValueError('Adopted bbox has no ground footprint')
    center = np.asarray(ground['center'], dtype=float)
    size = np.asarray(ground['size'], dtype=float)
    frozen_rotation = calibration.data.get('bbox_geometry',{}).get('ground_rotation_local_to_viewer')
    rotation = np.asarray(frozen_rotation) if frozen_rotation is not None else bbox_rotation_matrix(
        ground.get('rotation_deg',[0,0,ground.get('yaw_deg',ground.get('yaw',0.0))]))
    corners = np.column_stack([np.array([[-1,-1],[1,-1],[1,1],[-1,1]],dtype=float)*size[:2]/2,np.zeros(4)])
    xy = (corners @ rotation.T + center)[:,:2]
    plane = calibration.reference_plane
    origin = np.asarray(plane['origin_viewer_m'])
    normal = np.asarray(plane['normal_viewer'])
    z = origin[2] - ((xy - origin[:2]) @ normal[:2]) / normal[2]
    uv, depth = calibration.project(np.column_stack([xy, z]), camera, distort=False)
    if not np.isfinite(uv).all() or (depth <= 0).any():
        raise ValueError('Court footprint is not in front of teacher camera')
    return uv.tolist()


def farthest_sample(points, count=256):
    """Deterministic FPS from real returns; repeats only when fewer than count."""
    points = np.asarray(points, dtype=np.float32)
    if not len(points):
        raise ValueError('Cannot sample an empty human cloud')
    indices = np.empty(min(count, len(points)), dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)
    current = int(np.argmax(np.sum((points - points.mean(0)) ** 2, axis=1)))
    for i in range(len(indices)):
        indices[i] = current
        distances = np.minimum(distances, np.sum((points - points[current]) ** 2, axis=1))
        distances[indices[:i+1]] = -1
        current = int(np.argmax(distances))
    if len(indices) < count:
        indices = np.resize(indices, count)
    return points[indices], indices


def _skeleton_distance(uv, keypoints, scale):
    edges = ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
             (11,13),(13,15),(12,14),(14,16),(0,5),(0,6))
    distances = np.full(len(uv), np.inf)
    for i, j in edges:
        if min(keypoints[i,2], keypoints[j,2]) < .25:
            continue
        start, end = keypoints[i,:2], keypoints[j,:2]
        delta = end - start
        t = np.clip(((uv-start) @ delta) / max(float(delta@delta), 1e-8), 0, 1)
        distances = np.minimum(distances, np.linalg.norm(uv - start - t[:,None]*delta, axis=1))
    return distances / max(scale, 1)


def segment_people(points, tracks, calibration, camera='cam2', *, radius=.22, min_points=80):
    """Return per-track clouds and original indices; no synthetic surface points.

    ``tracks`` holds track_id, bbox (undistorted xyxy), kp2d (17,3).
    Component votes preserve separated people; ambiguous overlaps are excluded.
    """
    points = np.asarray(points, dtype=np.float32)
    result = {str(t['track_id']): {'points': np.empty((0,3), np.float32),
                                  'indices': np.empty(0, np.int64), 'status': 'no_cloud'} for t in tracks}
    if not len(points) or not tracks:
        return result
    uv, depth = calibration.project(points, camera, distort=False)
    origin = np.asarray(calibration.reference_plane['origin_viewer_m'])
    normal = np.asarray(calibration.reference_plane['normal_viewer'])
    height = (points - origin) @ normal
    valid = np.isfinite(uv).all(1) & (depth > 0) & (height > .04) & (height < 2.8)
    scores = np.full((len(points),len(tracks)), np.inf)
    other_projections = {}
    for track in tracks:
        for name in track.get('views',{}):
            if name!=camera and name not in other_projections:
                other_projections[name] = calibration.project(points,name,distort=False)
    for k, track in enumerate(tracks):
        box = np.asarray(track['bbox'], dtype=float)
        wh = np.maximum(box[2:] - box[:2], 1)
        margin = wh * .06
        inside = valid & (uv >= box[:2]-margin).all(1) & (uv <= box[2:]+margin).all(1)
        # Scores outside this person's box remain infinite and are never used.
        # Evaluate the identical per-point formula only on possible members;
        # full-room fixed24 exports otherwise repeat 14 bone distances over
        # hundreds of thousands of irrelevant background returns per athlete.
        candidates = uv[inside]
        center_score = np.linalg.norm((candidates - (box[:2]+box[2:])/2) / wh, axis=1)
        skeleton = _skeleton_distance(candidates, np.asarray(track['kp2d']), wh[1])
        score = np.where(np.isfinite(skeleton), skeleton + center_score*.08, center_score)
        scores[inside,k] = score
        for name,view in track.get('views',{}).items():
            if name==camera:continue
            other_uv,other_depth=other_projections[name]
            other_box=np.asarray(view['bbox'],float)
            other_wh=np.maximum(other_box[2:]-other_box[:2],1)
            supported=(np.isfinite(other_uv).all(1)&(other_depth>0)
                       &(other_uv>=other_box[:2]-other_wh*.06).all(1)
                       &(other_uv<=other_box[2:]+other_wh*.06).all(1))
            scores[~supported,k]=np.inf
            selected=inside&supported
            candidate_uv=other_uv[selected]
            center_distance=np.linalg.norm((candidate_uv-(other_box[:2]+other_box[2:])/2)/other_wh,axis=1)
            bone_distance=_skeleton_distance(candidate_uv,np.asarray(view['kp2d']),other_wh[1])
            other_score=np.where(np.isfinite(bone_distance),bone_distance+center_distance*.08,center_distance)
            scores[selected,k]=(scores[selected,k]+other_score)/2
    active = np.flatnonzero(np.isfinite(scores).any(1))
    if not len(active):
        return result
    neighbor_pairs = cKDTree(points[active]).query_pairs(radius, output_type='ndarray')
    graph = coo_matrix((np.ones(len(neighbor_pairs)), (neighbor_pairs[:,0], neighbor_pairs[:,1])),
                       shape=(len(active),len(active)))
    _, labels = connected_components(graph, directed=False)
    nearest = scores[active].argmin(1)
    for k, track in enumerate(tracks):
        selected = []
        for label in np.unique(labels):
            local = np.flatnonzero(labels == label)
            vote = np.bincount(nearest[local], minlength=len(tracks))
            winner = int(vote.argmax())
            # Touching athletes cannot become a single pseudo-labelled person.
            multiview=len(track.get('views',{}))>=2
            if not multiview and (winner != k or vote[k] / len(local) < .90):
                continue
            candidate = active[local[nearest[local] == k]]
            # Only include pixels actually supported by this athlete's box.
            candidate = candidate[np.isfinite(scores[candidate,k])]
            if len(tracks) > 1 and len(candidate):
                ordered = np.sort(scores[candidate],axis=1)
                unique = ~np.isfinite(ordered[:,1]) | ((ordered[:,1]-ordered[:,0]) > .025)
                candidate = candidate[unique]
            if len(candidate) >= 12:
                selected.extend(candidate.tolist())
        selected = np.array(selected, dtype=np.int64)
        if not len(selected):
            continue
        cloud = points[selected]
        # Trim isolated rackets/shuttle returns without imposing body symmetry.
        tree = cKDTree(cloud)
        support = tree.query_ball_point(cloud, .12, return_length=True) >= 3
        selected, cloud = selected[support], cloud[support]
        span = np.ptp((cloud-origin) @ normal) if len(cloud) else 0
        status = 'segmented' if len(cloud) >= min_points and span >= .55 else 'insufficient_human_cloud'
        result[str(track['track_id'])] = {'points': cloud, 'indices': selected, 'status': status,
                                         'point_count': len(cloud), 'height_span_m': float(span)}
    return result
