import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hmr4d import PROJ_ROOT


spec = importlib.util.spec_from_file_location("full_sequence", PROJ_ROOT / "tools/run_badminton_full_sequence.py")
sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequence)


def test_transient_windows_read_lock_retries_without_synthesizing_progress():
    path = Mock()
    path.read_bytes.side_effect = [PermissionError("atomic replacement"), b'{"state":"running","completed":101}']
    with patch.object(sequence.time, "sleep") as pause:
        assert sequence.read_state_json(path) == {"state": "running", "completed": 101}
    assert path.read_bytes.call_count == 2
    pause.assert_called_once_with(.05)


def test_persistent_lock_is_bounded_and_invalid_json_is_not_retried():
    path = Mock()
    path.read_bytes.side_effect = PermissionError("persistent lock")
    with patch.object(sequence.time, "sleep") as pause, pytest.raises(PermissionError):
        sequence.read_state_json(path, attempts=3)
    assert path.read_bytes.call_count == 3 and pause.call_count == 2
    path = Mock()
    path.read_bytes.return_value = b'invalid'
    with patch.object(sequence.time, "sleep") as pause, pytest.raises(json.JSONDecodeError):
        sequence.read_state_json(path)
    pause.assert_not_called()


def test_completed_review_resume_requires_native_binding_and_every_actual_window(tmp_path):
    def put(name, value):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value), encoding="utf-8")
        return sequence.binding(p)

    manifest = put("input.json", {})
    put("pilot_input.json", dict(manifest=manifest))
    native = put("repair/final_labels/native_verification.json",
                 dict(passed=True, state="complete", verified_windows=2, total_windows=2, accepted_checked=5))
    receipt = dict(state="review_complete", review_only=True, training_published=False, manifest=manifest,
                   evidence={"repair/final_labels/native_verification.json": native}, counts=dict(accepted=5))
    put("full_native_review_completion.json", receipt)
    assert sequence.completed_review_ready(tmp_path) == tmp_path / "full_native_review_completion.json"
    receipt["counts"]["accepted"] = 6
    put("full_native_review_completion.json", receipt)
    with pytest.raises(ValueError, match="full native verification"):
        sequence.completed_review_ready(tmp_path)
