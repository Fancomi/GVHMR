"""Select bounded leg corrections through observed native scan sequences."""

import numpy as np


def rotation_separation(a, b):
    relative = np.asarray(a) @ np.swapaxes(np.asarray(b), -1, -2)
    return np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) * 0.5, -1, 1))


def select_correction_sequence(frames, maximum_step_degrees=20.5, smoothness_weight=25.0):
    """Viterbi path; preserve a zero-correction option at every recorded frame.

    An unobserved gap starts a new sequence. End corrections must also stay
    near zero, so an isolated repaired frame cannot introduce a large jerk.
    """
    if not frames:
        return []
    identity = np.broadcast_to(np.eye(3), (8, 3, 3))
    bound = np.deg2rad(maximum_step_degrees)
    cuts = (
        [0]
        + [i for i in range(1, len(frames)) if frames[i]["time_us"] - frames[i - 1]["time_us"] > 150000]
        + [len(frames)]
    )
    choices = []
    for left, right in zip(cuts, cuts[1:]):
        costs = []
        parents = []
        for pos in range(left, right):
            states = frames[pos]["states"]
            current = np.full(len(states), np.inf)
            previous_indices = np.full(len(states), -1, int)
            for j, state in enumerate(states):
                if pos == left:
                    if rotation_separation(state["delta_rotations"], identity).max() <= bound:
                        current[j] = -state["gain_px"]
                    continue
                for k, old in enumerate(frames[pos - 1]["states"]):
                    delta = rotation_separation(state["delta_rotations"], old["delta_rotations"])
                    if delta.max() > bound:
                        continue
                    value = costs[-1][k] - state["gain_px"] + smoothness_weight * float(np.mean(delta**2))
                    if value < current[j]:
                        current[j] = value
                        previous_indices[j] = k
            costs.append(current)
            parents.append(previous_indices)
        end = costs[-1].copy()
        for j, state in enumerate(frames[right - 1]["states"]):
            if rotation_separation(state["delta_rotations"], identity).max() > bound:
                end[j] = np.inf
        best = int(np.argmin(end))
        if not np.isfinite(end[best]):
            raise ValueError("Correction path has no valid baseline state")
        path = [best]
        for i in range(len(parents) - 1, 0, -1):
            best = int(parents[i][best])
            path.append(best)
        choices.extend(reversed(path))
    return choices
