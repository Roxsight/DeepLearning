import argparse, json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="outputs/splits.json")
    args = p.parse_args()
    data = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    sets = {k: set(data[k]) for k in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        print(f"{a}/{b} sample overlap: {len(overlap)}")
    groups = {
        split: {data["samples"][sid]["group_id"] for sid in ids}
        for split, ids in sets.items()
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        print(f"{a}/{b} group overlap: {len(groups[a] & groups[b])}")
    for split, ids in sets.items():
        labels = [int(data["samples"][sid]["label"]) for sid in ids]
        print(f"{split}: {len(ids)} samples; real={labels.count(0)} fake={labels.count(1)}")

if __name__ == "__main__":
    main()
