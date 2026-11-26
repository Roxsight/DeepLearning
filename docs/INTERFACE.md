# Interface between members (read before coding)

## Batch contract (A's loader must produce this)
`from src.data.loader import build_loaders` with
`build_loaders(splits_path, batch_size, num_workers) -> {"train": DataLoader, "val": DataLoader, "test": DataLoader}`

Each batch is a dict:
- `frames`: float tensor `[B, 16, 3, 224, 224]` (ImageNet-normalised face crops)
- `mel`: float tensor `[B, 128, 200]` (log-Mel, 16 kHz, 2 s window)
- `label`: `[B]`, **1 = fake, 0 = real**
- `category`: list of str (ARVR, AFVR, ARVF, AFVF, ...)
- `video_id`: list of str. All windows of one video share the id (scores are averaged per video).

Use balanced sampling in the train loader (FakeAVCeleb has ~500 real vs ~19,500 fake), or set `pos_weight` in the config.
Split by speaker/identity, never by window, so no video leaks across train/val/test.

## Outputs (everyone writes the same files)
- `results/runs.csv`: `method,seed,split,metric,value`
- `results/preds/<method>_s<seed>_<split>.csv`: `video_id,label,category,score,mean_u,mean_g`
  (score = P(fake) per video. C and D can compute metrics and ROC curves from these without rerunning models.)

Baselines (A, C, D, and B for AVH-Align): write the same two files with your own `method` name.

## Running the proposed model
    python -m src.train --config configs/proposed.yaml --seed 0
    python -m src.train --config configs/proposed.yaml --seed 0 --override use_gate=false method=no_gate
    python -m src.train --config configs/proposed.yaml --seed 0 --override lam=0 method=no_cons
(D uses the last two lines for the ablations.)
Smoke test without data: add `--dummy --override backbone=vit_tiny_patch16_224`.
Tests: `python -m pytest -q tests`
