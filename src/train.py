import argparse
import csv
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve

from src.model import FusionModel, get_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DummyData(torch.utils.data.Dataset):
    def __init__(self, n=32, T=16):
        self.n = n
        self.T = T

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            'frames': torch.randn(self.T, 3, 224, 224),
            'mel': torch.randn(128, 200),
            'label': i % 2,
            'category': ['ARVR', 'AFVF'][i % 2],
            'video_id': 'v' + str(i),
        }


def get_loaders(cfg, dummy):
    if dummy:
        loaders = {}
        for name in ['train', 'val', 'test']:
            shuffle = name == 'train'
            loaders[name] = torch.utils.data.DataLoader(DummyData(), batch_size=cfg['batch_size'], shuffle=shuffle)
        return loaders
    from src.data.loader import build_loaders
    return build_loaders(cfg['splits'], cfg['batch_size'], cfg['num_workers'])


def get_eer(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


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
        m['eer'] = get_eer(y, p)
    else:
        m['auc'] = float('nan')
        m['eer'] = float('nan')
    return m


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    scores = defaultdict(list)
    us = defaultdict(list)
    gs = defaultdict(list)
    labels = {}
    cats = {}
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch['frames'].to(device), batch['mel'].to(device))
        probs = torch.sigmoid(out['logit'].float()).cpu()
        u = out['u'].float().mean(1).cpu()
        g = out['g'].float().mean(1).cpu()
        for i, vid in enumerate(batch['video_id']):
            scores[vid].append(probs[i].item())
            us[vid].append(u[i].item())
            gs[vid].append(g[i].item())
            labels[vid] = int(batch['label'][i])
            cats[vid] = batch['category'][i]
    rows = []
    for vid in scores:
        rows.append((vid, labels[vid], cats[vid], float(np.mean(scores[vid])),
                     float(np.mean(us[vid])), float(np.mean(gs[vid]))))
    metrics = get_metrics([r[1] for r in rows], [r[3] for r in rows])
    return metrics, rows


def save_preds(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_id', 'label', 'category', 'score', 'mean_u', 'mean_g'])
        writer.writerows(rows)


def log_runs(path, method, seed, split, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['method', 'seed', 'split', 'metric', 'value'])
        for name, value in metrics.items():
            writer.writerow([method, seed, split, name, '%.6f' % value])


def gate_report(rows):
    real = [r for r in rows if r[1] == 0]
    fake = [r for r in rows if r[1] == 1]

    def avg(rs, j):
        if len(rs) == 0:
            return float('nan')
        return float(np.mean([r[j] for r in rs]))

    return 'u real %.3f fake %.3f | gate real %.3f fake %.3f' % (avg(real, 4), avg(fake, 4), avg(real, 5), avg(fake, 5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/proposed.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--override', nargs='*', default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for item in args.override:
        key, value = item.split('=', 1)
        cfg[key] = yaml.safe_load(value)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = bool(cfg['amp']) and device.type == 'cuda'
    tag = '%s_s%d' % (cfg['method'], args.seed)

    model = FusionModel(d=cfg['d'], d_proj=cfg['d_proj'], T=cfg['T'], heads=cfg['heads'], r=cfg['r'],
                        dropout=cfg['dropout'], backbone=cfg['backbone'],
                        pretrained=cfg['pretrained'] and not args.dummy,
                        freeze_visual=cfg['freeze_visual'], use_gate=cfg['use_gate']).to(device)
    loaders = get_loaders(cfg, args.dummy)

    video_params = list(model.video.backbone.parameters())
    audio_params = list(model.audio.parameters())
    used = set(id(p) for p in video_params + audio_params)
    new_params = [p for p in model.parameters() if id(p) not in used]
    optimizer = torch.optim.AdamW([
        {'params': video_params, 'lr': cfg['lr_visual']},
        {'params': audio_params, 'lr': cfg['lr_audio']},
        {'params': new_params, 'lr': cfg['lr_new']},
    ], weight_decay=cfg['weight_decay'])

    epochs = cfg['warmup_epochs'] + cfg['joint_epochs']
    total_steps = max(1, epochs * math.ceil(len(loaders['train']) / cfg['accum']))
    warm_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warm_steps:
            return (step + 1) / warm_steps
        progress = (step - warm_steps) / max(1, total_steps - warm_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_score = -1.0
    best_path = 'results/ckpt/%s.pt' % tag
    os.makedirs('results/ckpt', exist_ok=True)

    for epoch in range(epochs):
        if not cfg['freeze_visual']:
            model.video.set_trainable(epoch >= cfg['warmup_epochs'])
        model.train()
        optimizer.zero_grad()
        total_cls = 0.0
        total_cons = 0.0
        n_batches = len(loaders['train'])

        for i, batch in enumerate(loaders['train']):
            y = torch.as_tensor(batch['label']).float().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(batch['frames'].to(device), batch['mel'].to(device))
            loss, parts = get_loss(out, y, cfg['lam'], cfg['tau'], cfg['delta'],
                                   cfg['pos_weight'], use_cons=cfg['lam'] > 0)
            scaler.scale(loss / cfg['accum']).backward()
            if (i + 1) % cfg['accum'] == 0 or i + 1 == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            total_cls += parts['cls']
            total_cons += parts['cons']

        metrics, rows = evaluate(model, loaders['val'], device, amp)
        if math.isnan(metrics['auc']):
            score = -total_cls / n_batches
        else:
            score = metrics['auc']
        print('epoch %d/%d cls %.4f cons %.4f | val auc %.4f acc %.4f | %s' % (
            epoch + 1, epochs, total_cls / n_batches, total_cons / n_batches,
            metrics['auc'], metrics['accuracy'], gate_report(rows)), flush=True)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    for split in ['val', 'test']:
        metrics, rows = evaluate(model, loaders[split], device, amp)
        log_runs('results/runs.csv', cfg['method'], args.seed, split, metrics)
        save_preds('results/preds/%s_%s.csv' % (tag, split), rows)
        print(split, {k: round(v, 4) for k, v in metrics.items()})


if __name__ == '__main__':
    main()
