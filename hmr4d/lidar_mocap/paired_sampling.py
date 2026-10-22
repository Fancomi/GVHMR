"""Explicit full-run sampling contract shared by initial and repaired teachers."""

from pathlib import Path

from hmr4d import PROJ_ROOT
from .pilot import binding, check_binding
from .rally_pipeline import read_json
from .repair_execution import write_progress_json
from .sparse_teacher import normalize_sparse_rows


POLICY = "paired_native_rgb10_latent_v1"


def freeze_sampling_input(root, backend_comparison):
    root, backend_comparison = Path(root).resolve(), Path(backend_comparison).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use one prepared full-native output root")
    frozen = read_json(root / "pilot_input.json")
    comparison = read_json(backend_comparison)
    check_binding(comparison["progress"])
    check_binding(comparison["release"])
    progress = read_json(comparison["progress"]["path"])
    if progress["state"] != "complete" or read_json(comparison["release"]["path"])["status"] != "released":
        raise ValueError("Real completed and released native backend validation is required")
    for phase in ("baseline30", "paired10"):
        result = progress["phases"][phase]
        verification = result["native_verification"]
        if (result["fitted"] != 100 or result["accepted"] != 100 or result["unfitted_two_person_cells"]
                or verification["accepted_checked"] != 100 or verification["maximums"]["source_point_error_m"] > 1e-6
                or comparison["phases"][phase]["bilateral_detection_rule_pass"] != 100):
            raise ValueError("The bounded native backend acceptance is incomplete")
    value = dict(schema="badminton-paired-native-rgb-sampling-v1", policy=POLICY, root=str(root),
        manifest=frozen["manifest"], source_input=binding(root / "repair/new_source_input.json"),
        backend_comparison=binding(backend_comparison), native_rgb_selection="actual_matched_scan_mid_pairs_with_original_records_PTS_common_clock",
        temporal_model_grid="piecewise_near_30hz_with_exact_native_anchors_and_missing_virtual_2D",
        origin="first_core_selected_scan_mid_us_not_first_return_or_each_camera_first_frame",
        context_margin_us=1_000_000, maximum_real_RGB_query_error_us=8_500,
        output_scans="unchanged_original_selected_source_pairs", existing_completed_teachers="reuse_exact_original_artifacts",
        numerical_camera_parameters_changed=False, training_published=False)
    path = root / "teacher_sampling_input.json"
    if path.exists() and read_json(path) != value:
        raise ValueError("A full sampling input is already frozen; do not replace it")
    if not path.exists():
        write_progress_json(path, value)
    return binding(path)


def validate_sampling_input(repair_root, input_path):
    repair_root, input_path = Path(repair_root).resolve(), Path(input_path).resolve()
    value = read_json(input_path)
    full = repair_root.parent
    if (value.get("schema") != "badminton-paired-native-rgb-sampling-v1" or value.get("policy") != POLICY
            or Path(value["root"]).resolve() != full or value["source_input"] != binding(repair_root / "new_source_input.json")
            or value["manifest"] != read_json(full / "pilot_input.json")["manifest"]):
        raise ValueError("The sparse teacher input must bind this exact original full source plan")
    for name in ("manifest", "source_input", "backend_comparison"):
        check_binding(value[name])
    return value


def paired_teacher_rows(repair_root, input_path, session, window, camera):
    value = validate_sampling_input(repair_root, input_path)
    pairs = session.pairs()
    selected = set(window["source_pairs"])
    core = [pair for pair in pairs if pair["lidar_index"] in selected]
    if not core or len(core) != len(selected):
        raise ValueError("The original selected native scans changed")
    context = [pair for pair in pairs
               if window["left_us"] - value["context_margin_us"] <= pair["scan_mid_us"] <= window["right_us"] + value["context_margin_us"]
               and all(pair["cameras"][name]["status"] == "matched" for name in ("cam1", "cam2"))]
    native_rows = {row["record_index"]: row for row in session.camera_rows(camera)}
    rows = normalize_sparse_rows([native_rows[pair["cameras"][camera]["record_index"]] for pair in context])
    return rows, int(core[0]["scan_mid_us"]), binding(input_path)
