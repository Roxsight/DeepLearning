from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LABEL_MAP = {
    "realvideo-realaudio": 0,
    "fakevideo-realaudio": 1,
    "realvideo-fakeaudio": 1,
    "fakevideo-fakeaudio": 1,
}

def stable_id(path: Path, root: Path) -> str:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]

def label_from_path(path: Path, root: Path) -> tuple[int, str]:
    try:
        rel_parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts:
        key = part.lower().replace("_", "-").replace(" ", "")
        if key in LABEL_MAP:
            return LABEL_MAP[key], part
    raise ValueError(
        f"Cannot infer real/fake label from path: {path}. "
        "Expected one of: " + ", ".join(LABEL_MAP)
    )

def infer_group(path: Path, root: Path, category: str) -> str:
    """Infer an identity group from nested folder structure, not the filename."""
    try:
        parts = list(path.resolve().relative_to(root.resolve()).parts)
    except ValueError:
        parts = list(path.parts)
    # Remove the category directory and use the deepest parent folder as identity.
    category_index = next(
        (i for i, p in enumerate(parts) if p.lower().replace("_", "-").replace(" ", "") ==
         category.lower().replace("_", "-").replace(" ", "")),
        None
    )
    if category_index is not None:
        after = parts[category_index + 1:-1]
        if after:
            return "/".join(after)
    if len(parts) >= 2:
        return parts[-2]
    return path.stem

def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "relative_path", "label", "category", "group_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def read_manifest(path: Path):
    import csv
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
