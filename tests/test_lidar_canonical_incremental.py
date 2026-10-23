"""Tiny real CPU publications: explicit v2 cuts never manufacture empty roles."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import joblib
import pytest
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap import canonical_tracks as canonical
from hmr4d.lidar_mocap.export import FIELDS
from test_lidar_canonical_no_change import bind, exported, make_case, samples
from test_lidar_incremental_release import write


@pytest.fixture
def case(tmp_path, monkeypatch):
    value = make_case(tmp_path, monkeypatch, step=1.0)
    previous = json.loads(value["review_path"].read_text())
    cases = deepcopy(previous["cases"])
    for index, item in enumerate(cases):
        item.update(verdict="different_person", new_track_id=1000001 + index)
    value["review_path"] = value["repair"] / "reviewed_cuts.json"
    write(
        value["review_path"],
        dict(
            schema="badminton-reviewed-canonical-track-boundaries-v1",
            review_status="reviewed",
            audit=previous["continuity_audit"],
            cases=cases,
        ),
    )
    value["output"] = value["repair"] / "canonical_cuts"
    return value


def pair(case):
    return dict(release_policy=case["selected"]["path"], release_policy_sha256=case["selected"]["sha256"])


def publish(case, **kwargs):
    return canonical.publish_canonical_view(case["source_path"], case["review_path"], case["output"], **kwargs)


def source_change(case, change):
    source = json.loads(case["source_path"].read_text())
    change(source)
    write(case["source_path"], source)
    completion = json.loads(case["completion_path"].read_text())
    completion["evidence"][str(case["source_path"])] = bind(case["source_path"])["sha256"]
    write(case["completion_path"], completion)


def test_real_cut_train_only_preserves_empty_original_bytes_and_all_numeric_values(case):
    original = json.loads(case["source_path"].read_text())
    paths = [case["source_path"], case["review_path"], case["completion_path"]]
    paths += [
        Path(entry[key]) for entry in original["suggested_session_splits"].values() for key in ("path", "metadata")
    ]
    before = {str(path): path.read_bytes() for path in paths}
    manifest = publish(case, **pair(case))
    proof = canonical.verify_canonical_publication(manifest)
    assert manifest["schema"] == "badminton-canonical-bilateral-publication-v1"
    assert manifest["release_policy"] == case["selected"]
    assert proof["accepted"] == 3 and proof["reviewed_boundary_count"] == 3
    assert proof["sessions"]["fresh"]["changed_canonical_keys"] == 2
    assert json.loads(case["audit_path"].read_text())["counts"]["person_frames"] == 4
    a = joblib.load(original["exports"]["fresh"]["path"])
    b = joblib.load(manifest["exports"]["fresh"]["path"])
    assert all(torch.equal(a[name], b[name]) for name in FIELDS)
    for role in ("val", "test"):
        old, new = original["suggested_session_splits"][role], manifest["suggested_session_splits"][role]
        assert new["accepted"] == 0 and new["sessions"] == []
        for name, sha in (("path", "sha256"), ("metadata", "metadata_sha256")):
            assert Path(new[name]).read_bytes() == Path(old[name]).read_bytes()
            assert new[sha] == old[sha] and not Path(old[name]).samefile(new[name])
    assert all(Path(path).read_bytes() == payload for path, payload in before.items())


def test_legacy_nonempty_cut_publication_keeps_original_default(case):
    source = json.loads(case["source_path"].read_text())
    body_path = Path(source["source_dataset_manifest"])
    body = json.loads(body_path.read_text())
    for role in ("val", "test"):
        sid = role + "_source"
        full = samples(0.1)
        for sample in full:
            motion = sid + ":window:3:0"
            sample["motion_id"] = motion
            sample["metadata"].update(session_id=sid, motion_id=motion)
        body["exports"][sid] = exported(body_path.parent / (sid + ".pth"), full)
        chosen = deepcopy(full[:1])
        chosen[0]["metadata"].update(
            source_tensor_index=0,
            dualcam=json.loads(Path(source["exports"]["fresh"]["metadata"]).read_text().splitlines()[0])["dualcam"],
        )
        source["exports"][sid] = exported(case["source_path"].parent / (sid + ".pth"), chosen)
        source["suggested_session_splits"][role] = dict(source["exports"][sid], sessions=[sid])
    body["accepted"] = 12
    write(body_path, body)
    source.update(accepted=5, source_dataset_manifest_sha256=bind(body_path)["sha256"])
    write(case["source_path"], source)
    result = publish(case)
    assert "release_policy" not in result
    assert canonical.verify_canonical_publication(result)["accepted"] == 5
    assert [value["accepted"] for value in result["suggested_session_splits"].values()] == [3, 1, 1]


@pytest.mark.parametrize("missing", ["release_policy", "release_policy_sha256"])
def test_half_policy_pair_fails_before_output(case, missing):
    arguments = pair(case)
    arguments.pop(missing)
    with pytest.raises(ValueError, match="together"):
        publish(case, **arguments)
    assert not case["output"].exists()


def test_wrong_policy_sha_fails_before_output(case):
    arguments = pair(case)
    arguments["release_policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dependency changed"):
        publish(case, **arguments)
    assert not case["output"].exists()


def test_legacy_empty_roles_do_not_silently_enable_v2(case):
    with pytest.raises(RuntimeError, match="non-empty"):
        publish(case)
    assert not (case["output"] / "datasets/dataset_manifest.json").exists()


def test_declared_v2_source_requires_explicit_policy_pair(case):
    source_change(case, lambda source: source.update(release_policy=case["selected"]))
    with pytest.raises(ValueError, match="explicit release-policy"):
        publish(case)
    assert not case["output"].exists()


@pytest.mark.parametrize("field", ["roles", "admission", "policy", "completion", "audit"])
def test_rebound_foreign_current_source_scope_is_rejected(case, field):
    if field == "roles":
        source_change(case, lambda source: source["suggested_session_splits"]["train"].update(sessions=[]))
    elif field in ("admission", "policy"):
        path = case["root"] / "source/source_registry.json"
        registry = json.loads(path.read_text())
        if field == "admission":
            registry["sources"]["fresh"]["source_index"] = bind(case["audit_path"])
        else:
            registry["release_policy"] = bind(case["audit_path"])
        write(path, registry)
        input_path = case["repair"] / "new_source_input.json"
        source_input = json.loads(input_path.read_text())
        source_input["source_registry_sha256"] = bind(path)["sha256"]
        write(input_path, source_input)
    elif field == "completion":
        completion = json.loads(case["completion_path"].read_text())
        completion["state"] = "pending"
        write(case["completion_path"], completion)
    else:
        audit = json.loads(case["audit_path"].read_text())
        audit["body_manifest"] = bind(case["source_path"])
        write(case["audit_path"], audit)
        review = json.loads(case["review_path"].read_text())
        review["audit"] = bind(case["audit_path"])
        write(case["review_path"], review)
    with pytest.raises(ValueError):
        publish(case, **pair(case))
    assert not (case["output"] / "datasets/dataset_manifest.json").exists()


@pytest.mark.parametrize("defect", ["bool_count", "rows", "shape", "missing", "sha"])
def test_actual_empty_payload_is_required_not_constructed(case, defect):
    def change(source):
        entry = source["suggested_session_splits"]["val"]
        if defect == "bool_count":
            entry["accepted"] = False
        elif defect == "rows":
            Path(entry["metadata"]).write_text("{}\n", encoding="utf-8")
            entry["metadata_sha256"] = bind(entry["metadata"])["sha256"]
        elif defect == "shape":
            data = joblib.load(entry["path"])
            data["transl"] = torch.zeros((0, 4), dtype=torch.float32)
            joblib.dump(data, entry["path"])
            entry["sha256"] = bind(entry["path"])["sha256"]
        elif defect == "missing":
            Path(entry["path"]).unlink()
        else:
            entry["sha256"] = "0" * 64

    source_change(case, change)
    with pytest.raises((ValueError, FileNotFoundError)):
        publish(case, **pair(case))
    assert not (case["output"] / "datasets/dataset_manifest.json").exists()


@pytest.mark.parametrize("location", ["manifest", "marker"])
@pytest.mark.parametrize("mode", ["reviewed_no_changes", "unknown", True])
def test_cut_verifier_rejects_nochange_or_unknown_mode(case, location, mode):
    manifest = publish(case, **pair(case))
    target = manifest if location == "manifest" else manifest["canonical_identity_revision"]
    target["mode"] = mode
    with pytest.raises(ValueError, match="mode"):
        canonical.verify_canonical_publication(manifest)


def test_readonly_verifier_rejects_original_empty_alias(case):
    manifest = publish(case, **pair(case))
    source = json.loads(case["source_path"].read_text())
    manifest["suggested_session_splits"]["val"] = source["suggested_session_splits"]["val"]
    with pytest.raises(ValueError, match="independently"):
        canonical.verify_canonical_publication(manifest)


def test_readonly_verifier_cannot_downgrade_empty_role_policy_by_removing_binding(case):
    manifest = publish(case, **pair(case))
    del manifest["release_policy"]
    with pytest.raises(ValueError, match="policy-bound"):
        canonical.verify_canonical_publication(manifest)


def test_readonly_verifier_rejects_empty_hardlink_alias(case):
    manifest = publish(case, **pair(case))
    original = json.loads(case["source_path"].read_text())["suggested_session_splits"]["val"]
    revised = Path(manifest["suggested_session_splits"]["val"]["path"])
    payload = Path(original["path"]).read_bytes()
    revised.unlink()
    revised.hardlink_to(original["path"])
    with pytest.raises(ValueError, match="independently"):
        canonical.verify_canonical_publication(manifest)
    assert Path(original["path"]).read_bytes() == payload


@pytest.mark.parametrize("field,value", [("calibration_digest", "0" * 64), ("opaque_original_field", None)])
def test_readonly_verifier_preserves_original_nonidentity_fields(case, field, value):
    manifest = publish(case, **pair(case))
    manifest[field] = value
    with pytest.raises(ValueError, match="non-identity fields"):
        canonical.verify_canonical_publication(manifest)


def test_empty_copy_exclusive_destination_preserves_existing_file(case):
    source = json.loads(case["source_path"].read_text())
    path = case["root"] / "private-existing.pth"
    path.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        canonical._empty_role(source["suggested_session_splits"]["val"], path)
    assert path.read_bytes() == b"do not overwrite"


def test_real_cut_remains_mandatory(case):
    review = json.loads(case["review_path"].read_text())
    for item in review["cases"]:
        item["verdict"] = "same_person"
    write(case["review_path"], review)
    with pytest.raises(ValueError, match="no reviewed cross-person"):
        publish(case, **pair(case))
    assert not (case["output"] / "datasets/dataset_manifest.json").exists()


def test_mutation_after_verifier_cannot_publish_completion_marker(case, monkeypatch):
    original = canonical.verify_canonical_publication

    def mutate(manifest):
        proof = original(manifest)
        with case["completion_path"].open("ab") as stream:
            stream.write(b"\n")
        return proof

    monkeypatch.setattr(canonical, "verify_canonical_publication", mutate)
    with pytest.raises(ValueError, match="dependency changed"):
        publish(case, **pair(case))
    assert not (case["output"] / "datasets/dataset_manifest.json").exists()


def test_cli_half_pair_rejected_before_reading_source(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PROJ_ROOT / "tools/publish_badminton_canonical_tracks.py"),
            "--source-manifest",
            str(tmp_path / "missing.json"),
            "--reviewed-boundaries",
            str(tmp_path / "missing2.json"),
            "--output-root",
            str(PROJ_ROOT / "outputs/not-created"),
            "--release-policy",
            str(tmp_path / "missing-policy"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2 and "together" in result.stderr
