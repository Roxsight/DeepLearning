from __future__ import annotations
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import VIDEO_EXTENSIONS, stable_id, label_from_path, infer_group, write_csv, save_json

def assign_group_splits(rows, ratios, seed):
    """Greedy group-level assignment targeting split sizes and class balance."""
    groups = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)

    if len(groups) < 3:
        print(
            "WARNING: fewer than 3 inferred identity groups. "
            "A group-disjoint 3-way split is not possible; using sample-level split. "
            "This is only suitable for a tiny pipeline smoke test, not final evaluation."
        )
        items = rows[:]
        random.Random(seed).shuffle(items)
        n = len(items)
        n_train = max(1, round(n * ratios[0])) if n else 0
        n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
        if n_train + n_val >= n and n >= 3:
            n_train = max(1, n - n_val - 1)
        return {
            "train": items[:n_train],
            "val": items[n_train:n_train+n_val],
            "test": items[n_train+n_val:],
        }

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    total = len(rows)
    total_fake = sum(int(r["label"]) for r in rows)
    targets = {
        "train": (ratios[0] * total, ratios[0] * total_fake),
        "val": (ratios[1] * total, ratios[1] * total_fake),
        "test": (ratios[2] * total, ratios[2] * total_fake),
    }
    counts = {s: [0, 0] for s in targets}
    result = {s: [] for s in targets}

    for _, members in group_items:
        g_n = len(members)
        g_fake = sum(int(m["label"]) for m in members)
        # Assign to the split with the largest normalized deficit.
        scores = {}
        for split in targets:
            target_n, target_fake = targets[split]
            deficit_n = (target_n - counts[split][0]) / max(target_n, 1)
            deficit_fake = (target_fake - counts[split][1]) / max(target_fake, 1)
            scores[split] = deficit_n + 0.5 * deficit_fake
        chosen = max(scores, key=scores.get)
        result[chosen].extend(members)
        counts[chosen][0] += g_n
        counts[chosen][1] += g_fake

    return result

def main():
    p = argparse.ArgumentParser(description="Index videos and create group-aware splits.")
    p.add_argument("--data-root", required=True, help="Root containing FakeAVCeleb category folders.")
    p.add_argument("--out", default="outputs")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(x <= 0 for x in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        p.error("Split ratios must all be positive and sum to 1.")

    root = Path(args.data_root).expanduser().resolve()
    if not root.exists():
        p.error(f"Data root does not exist: {root}")

    paths = sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in VIDEO_EXTENSIONS)
    if not paths:
        p.error(f"No supported video files found under {root}")

    rows, skipped = [], []
    for video in paths:
        try:
            label, category = label_from_path(video, root)
            rows.append({
                "sample_id": stable_id(video, root),
                "relative_path": video.relative_to(root).as_posix(),
                "label": label,
                "category": category,
                "group_id": infer_group(video, root, category),
            })
        except ValueError as e:
            skipped.append((str(video), str(e)))

    if skipped:
        print(f"Skipped {len(skipped)} videos with unrecognized category labels.")
        for path, reason in skipped[:5]:
            print(f"  {path}: {reason}")
    if not rows:
        p.error("No labeled videos found. Check the folder names and LABEL_MAP in src/common.py.")

    out = Path(args.out)
    write_csv(rows, out / "manifest.csv")
    splits = assign_group_splits(rows, ratios, args.seed)

    payload = {
        "seed": args.seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "label_definition": {"0": "real (RealVideo-RealAudio)", "1": "fake (any manipulated modality)"},
        "train": [r["sample_id"] for r in splits["train"]],
        "val": [r["sample_id"] for r in splits["val"]],
        "test": [r["sample_id"] for r in splits["test"]],
        "samples": {r["sample_id"]: r for r in rows},
    }
    save_json(payload, out / "splits.json")

    print(f"Indexed {len(rows)} videos; labels: {dict(Counter(r['label'] for r in rows))}")
    for name, subset in splits.items():
        print(f"{name:5s}: {len(subset):5d} samples | labels {dict(Counter(int(r['label']) for r in subset))}")
    overlap = (set(payload["train"]) & set(payload["val"])) | (set(payload["train"]) & set(payload["test"])) | (set(payload["val"]) & set(payload["test"]))
    print(f"Sample-ID overlap: {len(overlap)}")
    group_sets = {s: {r["group_id"] for r in subset} for s, subset in splits.items()}
    group_overlap = (group_sets["train"] & group_sets["val"]) | (group_sets["train"] & group_sets["test"]) | (group_sets["val"] & group_sets["test"])
    print(f"Identity-group overlap: {len(group_overlap)}")
    if group_overlap:
        print("WARNING: identity groups overlap. Inspect inferred group_id values in manifest.csv.")

if __name__ == "__main__":
    main()
