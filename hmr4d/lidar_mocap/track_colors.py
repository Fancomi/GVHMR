"""Stable display colors for local detector tracks; never edits or stitches IDs."""

import colorsys

PALETTE = [(255, 157, 48), (79, 163, 255), (128, 224, 134), (216, 132, 226)]


def display_track_key(motion_id):
    return motion_id.rsplit(":", 1)[0]


def track_color_registry(frames):
    intervals = {}
    for scan, rows in sorted(frames.items()):
        for row in rows:
            key = display_track_key(row["motion_id"])
            if key not in intervals:
                intervals[key] = [scan, scan, len(intervals)]
            intervals[key][1] = max(intervals[key][1], scan)
    assigned = {}
    active = []
    for key, (start, end, order) in sorted(intervals.items(), key=lambda item: (item[1][0], item[1][2])):
        active = [entry for entry in active if entry[0] >= start]
        occupied = {entry[1] for entry in active}
        slot = next(i for i in range(len(occupied) + 1) if i not in occupied)
        if slot < len(PALETTE):
            color = PALETTE[slot]
        else:
            color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb((slot * 0.61803398875) % 1, 0.6, 0.95))
        assigned[key] = color
        active.append((end, slot))
    return {row["motion_id"]: assigned[display_track_key(row["motion_id"])] for rows in frames.values() for row in rows}
