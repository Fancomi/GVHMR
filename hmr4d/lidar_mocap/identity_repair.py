"""Apply hash-bound native-camera identity decisions to observed detections only."""

import hashlib
import json

import numpy as np


def decision_digest(decision):
    return hashlib.sha256(
        json.dumps(decision, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def apply_identity_decisions(tracks, decision):
    """Keep source crops, split contaminated spans, join confirmed people and deduplicate.

    Tracklets are decomposed to their original detector identities before edits.
    Missing detections remain missing; duplicate observations never enter GVHMR.
    """
    pieces = {}
    groups = []
    for key, observations in tracks.items():
        ids = set()
        for observation in observations:
            old = int(observation.get("original_track_id", key))
            pieces.setdefault(old, []).append(dict(observation, original_track_id=old))
            ids.add(old)
        groups.append(sorted(ids))
    parent = {key: key for key in pieces}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(a, b):
        if a not in parent or b not in parent:
            raise ValueError(f"Identity decision refers to absent detector track {a}/{b}")
        aa, bb = root(a), root(b)
        parent[max(aa, bb)] = min(aa, bb)

    reset=set(decision.get('reset_tracklets',[]))
    for group in groups:
        group=[key for key in group if key not in reset]
        for key in group[1:]:
            join(group[0], key)
    for action in decision.get("links", []):
        join(*map(int, action["tracks"]))
    for a, b in decision.get("cannot_link", []):
        if root(a) == root(b):
            raise ValueError(f"Identity decision joins two witnessed people: {a}/{b}")
    drops = decision.get("drop_spans", [])
    by_frame = {}
    removed = []
    for old, observations in pieces.items():
        for obs in observations:
            pos = int(obs["position"])
            discard = next((d for d in drops if old == d["track_id"] and d["begin"] <= pos <= d["end"]), None)
            if discard:
                removed.append(dict(track_id=old, position=pos, reason=discard["reason"]))
                continue
            key = root(old), pos
            by_frame.setdefault(key, []).append(obs)
    result = {}
    duplicates = []
    for (identity, pos), candidates in sorted(by_frame.items()):
        # Prefer an explicit observed source crop. Otherwise retain the larger
        # complete crop; torso-only duplicates are common under occlusion.
        preference = decision.get("duplicate_preference", {}).get(str(identity))

        def rank(obs):
            box = np.asarray(obs["bbx_xyxy"])
            return (
                obs["original_track_id"] == preference,
                float(box[3] - box[1]),
                float(obs.get("detection_confidence", 0)),
                -obs["original_track_id"],
            )

        chosen = max(candidates, key=rank)
        chosen["association_recovered"] = (
            chosen.get("association_recovered", False) or chosen["original_track_id"] != identity
        )
        result.setdefault(identity, []).append(chosen)
        for obs in candidates:
            if obs is not chosen:
                duplicates.append(
                    dict(
                        position=pos,
                        canonical_track=identity,
                        retained_original=chosen["original_track_id"],
                        dropped_original=obs["original_track_id"],
                        reason="same_person_duplicate_native_detection",
                    )
                )
    receipt = dict(
        decision_sha256=decision_digest(decision),
        links=decision.get("links", []),
        cannot_link=decision.get("cannot_link", []),
        removed_observations=removed,
        duplicate_observations=duplicates,
        geometry_interpolated=False,
        observations_before=sum(map(len, tracks.values())),
        observations_after=sum(map(len, result.values())),
    )
    return result, receipt
