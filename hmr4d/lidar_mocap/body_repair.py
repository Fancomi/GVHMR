"""Resolve the exact frozen windows behind a published full-plus-new union."""

import json
from pathlib import Path


def published_windows(union_path):
    union = json.loads(Path(union_path).read_text(encoding="utf-8"))
    result = []
    if union.get("schema") == "badminton-new-source-repair-input-v1":
        import hashlib
        for job in union["jobs"]:
            window = Path(job["source_window"])
            if hashlib.sha256((window / "window.json").read_bytes()).hexdigest() != job["window_json_sha256"]:
                raise ValueError("Prepared new-source window changed")
            result.append((job["source_id"], window))
        if len(set((sid, str(path)) for sid, path in result)) != len(result):
            raise ValueError("Duplicate prepared repair window")
        return sorted(result, key=lambda value: (value[0], value[1].name))
    for published in union["source_manifests"]:
        publication = json.loads(Path(published["path"]).read_text(encoding="utf-8"))
        source = Path(publication["source_dataset_manifest"])
        body = json.loads(source.read_text(encoding="utf-8"))
        for sid, entry in union["exports"].items():
            if Path(entry["source_manifest"]).resolve() != Path(published["path"]).resolve():
                continue
            root = Path(body.get("window_roots", {}).get(sid, source.parent.parent))
            for name in body["selected_windows"][sid]:
                window = root / "sessions" / sid / "windows" / name
                if not (window / "fit_report.json").is_file():
                    raise ValueError(f"Published source window is incomplete: {window}")
                result.append((sid, window))
    return sorted(result, key=lambda value: (value[0], value[1].name))
