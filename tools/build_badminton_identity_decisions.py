"""Seal native-view review decisions without modifying any original labels."""

import argparse
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    args = p.parse_args()
    evidence = args.root / "identity_evidence"
    source_plan = read_json(args.root / "case_plan.json")
    if read_json(source_plan["source_union"]).get("schema") == "badminton-new-source-repair-input-v1":
        from hmr4d.lidar_mocap.new_source_identity import seal_identity
        print(seal_identity(args.root))
        return
    review = read_json(evidence / "native_identity_review.json")
    plan = read_json(args.root / "case_plan.json")
    if review.get('partial') or len(review['cases'])!=64 or read_json(evidence/'pose_receipt.json')['state']!='complete':
        raise ValueError('Identity review is not the complete frozen 64-boundary inventory')
    if file_sha(plan['source_union'])!=plan['source_union_sha256']:
        raise ValueError('Frozen delivered union changed')
    expected={0:('20261012_154726','000120000000_000180000000',[3,4]),
              9:('20261012_163452','000120000000_000180000000',[17,36]),
              27:('20261012_163452','000300000000_000360000000',[23,24]),
              28:('20261012_163452','000300000000_000360000000',[24,25]),
              45:('20261012_163452','000720000000_000780000000',[6,7]),
              58:('20261012_163452','000960000000_000986217360',[12,16])}
    for index,(sid,window,ids) in expected.items():
        case=review['cases'][index]
        if case['event']['source_id']!=sid or Path(case['source_window']).name!=window or [v['original_track_id'] for v in case['votes']]!=ids:
            raise ValueError('Manual native-review decision refers to another event inventory')
    windows = {}
    statuses = {}
    # The camera-1 tracker fragments when a jumper leaves its detection box;
    # native RGB on both sides confirms these two and the partial-view bystander.
    visually_same = {
        0: "white_shirt_jump_partial_detection",
        9: "black_shirt_jump_partial_detection",
        45: "same_foreground_red_shorts_cam2_continuity_cam1_partial_fov",
    }
    for c in review["cases"]:
        sid = c["event"]["source_id"]
        source = Path(c["source_window"])
        key = sid + ":" + source.name
        w = windows.setdefault(
            key,
            dict(
                source_window=str(source),
                source_teacher_manifest_sha256=file_sha(source / "teacher/manifest.json"),
                links=[],
                cannot_link=[],
                drop_spans=[],
                review_cases=[],
            ),
        )
        ids = [v["original_track_id"] for v in c["votes"]]
        status = c["status"]
        method = "native_opposite_camera_continuity_and_epipolar_pose"
        if c["index"] in visually_same:
            status = "same_native_person"
            method = "native_RGB_visual_review:" + visually_same[c["index"]]
        if status == "same_native_person":
            w["links"].append(dict(tracks=ids, case=c["index"], method=method))
        elif status == "different_native_people":
            w["cannot_link"].append(ids)
        w["review_cases"].append(c["index"])
        statuses[str(c["index"])] = status
    # 300-360s: original 24 covers two people in a single crop. Preserve the
    # two actual identities through the occlusion, omit the contaminated crop.
    key = "20261012_163452:000300000000_000360000000"
    w = windows[key]
    w["links"] += [
        dict(tracks=[11, 25], method="native_RGB_red_shorts_before_after_occlusion"),
        dict(tracks=[23, 27], method="native_RGB_navy_shirt_before_after_occlusion"),
    ]
    w["cannot_link"].append([11, 23])
    w['reset_tracklets']=[23,25,29]
    w['links'] += [dict(tracks=[25,29],method='native_RGB_red_shorts_end_of_window'),
                   dict(tracks=[27,28],method='native_RGB_navy_shirt_end_of_window')]
    w["drop_spans"] += [
        dict(track_id=k, begin=0, end=999999, reason="merged_crop_spans_two_native_people") for k in [24, 26]
    ]
    statuses["27"] = statuses["28"] = "mixed_person_crop_removed_true_identities_preserved"
    # Last window: 12 switches from the black shirt to the white shirt; 4 has
    # one ambiguous crossing frame. Keep actual measured 16/17/18 identities.
    w = windows["20261012_163452:000960000000_000986217360"]
    w["links"] += [
        dict(tracks=[12, 16], case=58, method="native_RGB_black_shirt_continuity_with_occluded_torso"),
        dict(tracks=[15, 18], method="native_RGB_white_shirt_continuity"),
    ]
    w["cannot_link"] += [[4, 12], [4, 15], [12, 15]]
    w["drop_spans"] += [
        dict(track_id=12, begin=538, end=999999, reason="old_track_switches_to_another_person"),
        dict(track_id=4, begin=538, end=538, reason="ambiguous_crossing_crop"),
    ]
    statuses["58"] = "mixed_track_tail_removed_same_black_shirt_linked"
    for w in windows.values():
        w["duplicate_policy"] = "one_original_observed_crop_per_native_frame_larger_complete_body_first"
    native_paths = sorted((evidence / "native_poses").glob("*/*/*.json"))
    result = dict(
        schema="badminton-native-identity-decisions-v1",
        source_union_sha256=plan["source_union_sha256"],
        calibration_digest=read_json(evidence / "pose_receipt.json")["config"]["calibration_digest"],
        windows=windows,
        statuses=statuses,
        evidence_files={
            str(path): file_sha(path)
            for path in [evidence / "native_identity_review.json", evidence / "pose_receipt.json", *native_paths]
        },
        limits=[
            "Local person identity across reviewed detector boundaries, not cross-session athlete recognition.",
            "Ambiguous mixed crops are omitted; missing observations are not synthesized.",
        ],
        training_published=False,
    )
    write_json(evidence / "identity_decisions.json", result)
    print(
        dict(
            windows=len(windows),
            links=sum(len(w["links"]) for w in windows.values()),
            cannot_link=sum(len(w["cannot_link"]) for w in windows.values()),
            native_images=len(native_paths),
        )
    )


if __name__ == "__main__":
    main()
