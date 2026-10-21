"""Verify derived fullraw members against emitted PNG cells, never packet returns."""

import cv2
import numpy as np


def verify_fullraw_members(frame, child_path, data_root, grid, bound_file, *, relocations=None):
    mapping = frame["member_mapping"]
    options = dict(relocations=relocations) if relocations is not None else {}
    path = bound_file(data_root, mapping, **options)
    parent_path = bound_file(data_root, frame["parent_png"], **options)
    width, height = grid["point_width"], grid["point_height"]
    images = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in (parent_path, child_path)]
    if any(image is None or image.dtype != np.uint8 or image.shape != (height * 3, width, 3) for image in images):
        raise ValueError("Fullraw member verification requires the original RGB8 fixed24 grids")
    parent, child = [image.reshape(3, width * height, 3) for image in images]
    parent_grid, child_grid = [np.flatnonzero(np.any(image != 0, axis=-1).all(axis=0)) for image in (parent, child)]
    with np.load(path, allow_pickle=False) as data:
        expected = dict(
            schema="badminton-native-roi-member-map-v1",
            dtype="int64",
            parent_scope="fullraw_exported_clouds",
            point_index_space="decoded_fixed24_valid_returns_before_raw_to_viewer",
            grid_index_space="zero-based row-major cell in one coordinate band",
            parent_png_sha256=frame["parent_png"]["sha256"],
            child_png_sha256=frame["png_sha256"],
            crop_binding_sha256=frame["crop_binding_sha256"],
        )
        if any(data[name].shape != () or data[name].item() != value for name, value in expected.items()):
            raise ValueError("Fullraw member map scope or PNG/crop binding differs")
        parent_to_child = data["parent_valid_to_child"]
        selected = data["child_valid_to_parent"]
        original_grid = data["parent_valid_to_grid"]
        if (
            any(value.dtype != np.int64 or value.ndim != 1 for value in (parent_to_child, selected, original_grid))
            or len(parent_to_child) != len(parent_grid)
            or len(original_grid) != len(parent_grid)
            or len(selected) != len(child_grid)
            or len(child_grid) != frame["point_count"]
            or len(parent_grid) != frame["parent_png"]["valid_points"]
            or not np.array_equal(original_grid, parent_grid)
            or np.any((selected < 0) | (selected >= len(parent_grid)))
            or np.any(np.diff(selected) <= 0)
        ):
            raise ValueError("Fullraw member map does not cover actual emitted-cloud valid returns")
        inverse = np.full(len(parent_grid), -1, dtype=np.int64)
        inverse[selected] = np.arange(len(selected), dtype=np.int64)
        if (
            not np.array_equal(parent_to_child, inverse)
            or not np.array_equal(child_grid, parent_grid[selected])
            or not np.array_equal(child[:, child_grid], parent[:, parent_grid[selected]])
        ):
            raise ValueError("Derived crop is not an exact encoded-cell subset of its emitted fullraw PNG")
    return dict(
        member_mapping=mapping,
        parent_png=frame["parent_png"],
        parent_valid_points=len(parent_grid),
        child_valid_points=len(child_grid),
        encoded_coordinates_bit_exact=True,
        parent_scope="fullraw_exported_clouds",
        maps_to_full_pcap_returns=False,
    )
