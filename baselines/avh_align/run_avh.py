import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

CATEGORY = {
    'RealVideo-RealAudio': 'ARVR',
    'RealVideo-FakeAudio': 'AFVR',
    'FakeVideo-RealAudio': 'ARVF',
    'FakeVideo-FakeAudio': 'AFVF',
}


def read_meta(path):
    videos = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['type'] not in CATEGORY:
                continue
            folder = row['path'].replace('FakeAVCeleb/', '')
            rel = os.path.join(folder, row['filename'])
            video_id = rel.replace('.mp4', '')
            label = 0 if row['type'] == 'RealVideo-RealAudio' else 1
            videos.append({'video_id': video_id, 'rel': rel, 'label': label, 'category': CATEGORY[row['type']]})
    return videos


def load_model(avh_dir, checkpoint, device):
    sys.path.insert(0, avh_dir)
    from model import FusionModel
    weights = torch.load(checkpoint, map_location=device, weights_only=False)
    model = FusionModel().to(device)
    model.load_state_dict(weights['state_dict'])
    model.eval()
    return model


@torch.no_grad()
def score_video(model, path, device):
    data = np.load(path, allow_pickle=True)
    v = torch.from_numpy(data['visual']).to(device)
    a = torch.from_numpy(data['audio']).to(device)
    v = v / torch.linalg.norm(v, ord=2, dim=-1, keepdim=True)
    a = a / torch.linalg.norm(a, ord=2, dim=-1, keepdim=True)
    out = model(v, a)
    return float(torch.logsumexp(-out, dim=0).cpu().squeeze())


def eer_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float(thr[i]), float((fpr[i] + 1 - tpr[i]) / 2)


def get_metrics(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= 0.5).astype(int)
    m = {}
    m['accuracy'] = accuracy_score(y, pred)
    m['precision'] = precision_score(y, pred, zero_division=0)
    m['recall'] = recall_score(y, pred, zero_division=0)
    m['f1'] = f1_score(y, pred, zero_division=0)
    if len(set(y.tolist())) == 2:
        m['auc'] = roc_auc_score(y, p)
        m['eer'] = eer_threshold(y, p)[1]
    else:
        m['auc'] = float('nan')
        m['eer'] = float('nan')
    return m


def save_preds(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'label', 'category', 'score', 'mean_u', 'mean_g'])
        for r in rows:
            writer.writerow([r['video_id'], r['label'], r['category'], '%.6f' % r['prob'], '', ''])


def log_runs(path, method, seed, split, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['method', 'seed', 'split', 'metric', 'value'])
        for name, value in metrics.items():
            writer.writerow([method, seed, split, name, '%.6f' % value])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--avh_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--features_path', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--splits', default=None)
    parser.add_argument('--out_dir', default='results')
    parser.add_argument('--method', default='avh_align')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.avh_dir, args.checkpoint, device)
    videos = read_meta(args.metadata)

    missing = 0
    scored = []
    for vid in videos:
        path = os.path.join(args.features_path, vid['rel'].replace('.mp4', '.npz'))
        if not os.path.exists(path):
            missing += 1
            continue
        vid['raw'] = score_video(model, path, device)
        scored.append(vid)
    print('scored %d videos, %d missing features' % (len(scored), missing))

    if args.splits:
        with open(args.splits) as f:
            splits = json.load(f)
        groups = {}
        for name in ['val', 'test']:
            ids = set(splits[name])
            groups[name] = [v for v in scored if v['video_id'] in ids]
        thr_split = 'val'
    else:
        print('WARNING: no splits file, using all videos and choosing the threshold on them (only for testing the setup)')
        groups = {'all': scored}
        thr_split = 'all'

    thr_rows = groups[thr_split]
    thr, _ = eer_threshold([v['label'] for v in thr_rows], [v['raw'] for v in thr_rows])
    print('threshold chosen on %s (EER point): %.4f' % (thr_split, thr))

    for name, rows in groups.items():
        for v in rows:
            v['prob'] = 1.0 / (1.0 + np.exp(-(v['raw'] - thr)))
        metrics = get_metrics([v['label'] for v in rows], [v['prob'] for v in rows])
        log_runs(os.path.join(args.out_dir, 'runs.csv'), args.method, args.seed, name, metrics)
        save_preds(os.path.join(args.out_dir, 'preds', '%s_s%d_%s.csv' % (args.method, args.seed, name)), rows)
        print(name, len(rows), {k: round(v, 4) for k, v in metrics.items()})


if __name__ == '__main__':
    main()
