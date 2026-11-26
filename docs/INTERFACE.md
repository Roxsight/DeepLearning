# Interface between members

## Data pipeline (A's code, adapted)
1. Put FakeAVCeleb in `data/raw/FakeAVCeleb/` (folders RealVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-RealAudio, FakeVideo-FakeAudio). Never commit it.
2. `python scripts/build_splits.py --data-root data/raw/FakeAVCeleb --out splits`  -> `splits/splits.json` (commit this file) and `splits/manifest.csv`
3. `python scripts/silence_check.py --manifest splits/manifest.csv --data-root data/raw/FakeAVCeleb --out splits/audio_audit.csv`
4. `python scripts/preprocess.py --manifest splits/manifest.csv --data-root data/raw/FakeAVCeleb --out-root data/processed`
   Takes one 2 s window from the middle of each video: 16 frames (224x224) and 2 s of 16 kHz audio from the same time span.
   The middle is used instead of the start to avoid the leading-silence shortcut in fake videos.

## Batch contract (`src/data/loader.py`, `build_loaders(splits_path, batch_size, num_workers)`)
- `frames`: `[B, 16, 3, 224, 224]`, ImageNet-normalised, random horizontal flip on train only
- `mel`: `[B, 128, 200]` log-Mel (16 kHz, 25 ms window, 10 ms hop, FFT 1024)
- `label`: 1 = fake, 0 = real
- `category`: ARVR, AFVR, ARVF, AFVF
- `video_id`: relative path without extension, e.g. `RealVideo-RealAudio/African/men/id00076/00109`
- Train loader is class-balanced (WeightedRandomSampler), so `pos_weight` stays null.
- Splits are by identity folder (no identity in two splits).

## Outputs (every method, including baselines)
- `results/runs.csv`: `method,seed,split,metric,value`
- `results/preds/<method>_s<seed>_<split>.csv`: `video_id,label,category,score,mean_u,mean_g` (score = P(fake))

## Commands
    python -m src.train --seed 0
    python -m src.train --seed 0 --override use_gate=false method=no_gate
    python -m src.train --seed 0 --override lam=0 method=no_cons
    python baselines/avh_align/run_avh.py --avh_dir <AVH-Align> --checkpoint <AVH-Align>/checkpoints/AVH-Align_AV1M.pt --features_path <features> --metadata data/raw/FakeAVCeleb/meta_data.csv --splits splits/splits.json
Tests: `python -m pytest -q tests`
