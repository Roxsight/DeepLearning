import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

NAMES = {
    (False, False): 'RealVideo-RealAudio',
    (False, True): 'RealVideo-FakeAudio',
    (True, False): 'FakeVideo-RealAudio',
    (True, True): 'FakeVideo-FakeAudio',
}
SPLITS = {'train': 'train', 'dev': 'val', 'val': 'val', 'test': 'test'}


def window_start(item, seconds):
    duration = float(item.get('duration', 0) or 0)
    periods = item.get('fake_periods') or []
    if periods:
        a, b = periods[0]
        center = (float(a) + float(b)) / 2
    else:
        center = duration / 2
    start = center - seconds / 2
    start = min(start, duration - seconds)
    return max(0.0, start)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', required=True)
    p.add_argument('--out', default='splits')
    p.add_argument('--n-train', type=int, default=3000)
    p.add_argument('--n-val', type=int, default=1000)
    p.add_argument('--n-test', type=int, default=1000)
    p.add_argument('--seconds', type=float, default=2.0)
    p.add_argument('--tar-prefix', default='')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    with open(args.metadata) as f:
        meta = json.load(f)
    if isinstance(meta, dict):
        meta = list(meta.values())
    need = ['file', 'split', 'modify_video', 'modify_audio']
    missing = [k for k in need if k not in meta[0]]
    if missing:
        raise SystemExit('metadata is missing keys %s, first entry is: %s' % (missing, meta[0]))

    pools = {}
    for item in meta:
        split = SPLITS.get(item['split'])
        if split is None:
            continue
        cat = NAMES[(bool(item['modify_video']), bool(item['modify_audio']))]
        pools.setdefault((split, cat), []).append(item)

    rng = random.Random(args.seed)
    wanted = {'train': args.n_train, 'val': args.n_val, 'test': args.n_test}
    rows = []
    payload = {'seed': args.seed, 'dataset': 'LAV-DF', 'train': [], 'val': [], 'test': [], 'samples': {}}
    for split, n in wanted.items():
        per_cat = n // 4
        for cat in NAMES.values():
            items = pools.get((split, cat), [])
            rng.shuffle(items)
            for item in items[:per_cat]:
                rel = item['file']
                sid = rel.rsplit('.', 1)[0].replace('/', '_')
                row = {
                    'sample_id': sid,
                    'relative_path': rel,
                    'label': 0 if cat == 'RealVideo-RealAudio' else 1,
                    'category': cat,
                    'group_id': split,
                    'start': '%.3f' % window_start(item, args.seconds),
                }
                rows.append(row)
                payload[split].append(sid)
                payload['samples'][sid] = row

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / 'splits.json', 'w') as f:
        json.dump(payload, f, indent=2)
    with open(out / 'tar_files.txt', 'w') as f:
        for r in rows:
            f.write(args.tar_prefix + r['relative_path'] + '\n')

    for split in wanted:
        cats = Counter(payload['samples'][s]['category'] for s in payload[split])
        print(split, len(payload[split]), dict(cats))
    print('wrote', out / 'manifest.csv', out / 'splits.json', out / 'tar_files.txt')


if __name__ == '__main__':
    main()
