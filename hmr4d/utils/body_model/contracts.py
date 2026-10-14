"""Shared body topology and pelvis placement for demo, service and LiDAR fitting."""
import torch


def map_smplx_vertices(vertices, mapping):
    """Map geometry to 6890 vertices; this does not convert SMPL-X parameters."""
    if vertices.shape[-2:] != (10475,3) or mapping.shape != (6890,10475):
        raise ValueError('Expected SMPL-X vertices (...,10475,3) and the 6890x10475 mapping')
    leading = vertices.shape[:-2]
    flat = vertices.reshape(-1,10475,3)
    packed = flat.permute(1,0,2).reshape(10475,-1)
    mapped = mapping @ packed
    return mapped.reshape(6890,-1,3).permute(1,0,2).reshape(*leading,6890,3)


def mapped_smpl_geometry(vertices, mapping, joint_regressor):
    mapped = map_smplx_vertices(vertices,mapping)
    if joint_regressor.shape != (24,6890):
        raise ValueError('Expected the 24x6890 SMPL joint regressor')
    return mapped, joint_regressor @ mapped


def forward_smpl_at_pelvis(model, root_pose, body_pose, shape, pelvis):
    """Neutral SMPL with transl equal to shaped pelvis position in world metres."""
    count = len(root_pose)
    shape = shape.reshape(-1,10).expand(count,-1)
    output = model(betas=shape,body_pose=body_pose.reshape(count,69),
                   global_orient=root_pose.reshape(count,3),transl=root_pose.new_zeros((count,3)),
                   return_verts=True)
    origin = output.joints[:,:1]
    return output.vertices-origin+pelvis[:,None],output.joints[:,:24]-origin+pelvis[:,None]
