"""Resolve the exact frozen windows behind a published full-plus-new union."""

import json
from pathlib import Path


def published_windows(union_path):
    union = json.loads(Path(union_path).read_text(encoding="utf-8"))
    result = []
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
