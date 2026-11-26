# Member A — FakeAVCeleb Data Pipeline

This is a starter implementation for the Member A deliverables:
- dataset indexing and reproducible train/validation/test splits
- video-frame and audio preprocessing
- PyTorch Dataset/DataLoader
- missing/silent audio audit

## 1. Expected input structure

The official FakeAVCeleb layout commonly uses these category folders:

```text
data/raw/FakeAVCeleb/
├── RealVideo-RealAudio/
├── FakeVideo-RealAudio/
├── RealVideo-FakeAudio/
└── FakeVideo-FakeAudio/
```

The scripts recursively search for videos under the root, so nested ethnicity/gender/identity folders are supported.

**Important:** The public sample may have a slightly different layout. Run the index command and inspect `outputs/manifest.csv`. If the category folder names are not present in the path, edit `LABEL_MAP` in `src/common.py` to match the sample's documented labels. Do not guess labels from filenames.

## 2. Install

Python 3.10+ is recommended for this starter code.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Install the PyTorch build appropriate for your machine from https://pytorch.org/get-started/locally/ if the default `pip install torch` does not suit your CPU/GPU.

## 3. Put sample videos in place

Extract the permitted public sample into `data/raw/FakeAVCeleb/`. Do not commit dataset videos, audio, extracted frames, or checkpoints to GitHub. The dataset terms restrict redistribution.

## 4. Build manifest and splits

```bash
python scripts/build_splits.py --data-root data/raw/FakeAVCeleb --out outputs
```

Creates:
- `outputs/manifest.csv`: discovered videos and labels
- `outputs/splits.json`: train/validation/test sample IDs and metadata

The split generator groups by inferred identity folder where possible to reduce identity leakage. For very small samples it may be impossible to place every class in every split; inspect the printed split summary. Never use the test split for tuning.

## 5. Audit missing/silent audio

```bash
python scripts/silence_check.py --manifest outputs/manifest.csv --out outputs/audio_audit.csv
```

Requires FFmpeg available on PATH to extract audio from video. Missing audio is flagged. RMS threshold is configurable and is a screening threshold, not a universal definition of silence.

## 6. Preprocess

```bash
python scripts/preprocess.py   --manifest outputs/manifest.csv   --out-root data/processed   --frames-per-video 8   --image-size 224   --sample-rate 16000   --audio-seconds 5
```

Output per sample:
```text
data/processed/
├── frames/<sample_id>/000.jpg ...
└── audio/<sample_id>.wav
```

The preprocessing script uses FFmpeg to extract audio and OpenCV to sample frames. Install FFmpeg and make sure `ffmpeg` is available in PATH.

## 7. Use the PyTorch loader

```python
import json
from torch.utils.data import DataLoader
from src.dataset import FakeAVCelebDataset

with open("outputs/splits.json", encoding="utf-8") as f:
    splits = json.load(f)

train_ds = FakeAVCelebDataset(
    splits["train"],
    processed_root="data/processed",
    num_frames=8,
    image_size=224,
    sample_rate=16000,
    audio_seconds=5,
)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)

batch = next(iter(train_loader))
print(batch["frames"].shape)  # [B, T, C, H, W]
print(batch["audio"].shape)   # [B, samples]
print(batch["label"].shape)   # [B]
```

Each item returns:
- `frames`: float tensor `[T, C, H, W]`, scaled to `[0, 1]`
- `audio`: float tensor `[sample_rate * audio_seconds]`
- `label`: integer `0` (real) or `1` (fake)
- `sample_id`: stable sample identifier

**Coordinate with Member B:** confirm the number of frames, image size, audio representation, sample rate and duration before freezing this interface. Member B may require log-Mel spectrograms or another audio representation instead of raw waveform.

## 8. Data and evaluation cautions

- Keep audio and video from the same original file paired.
- Keep a fixed split file for all methods and baselines.
- Group by identity/source when possible to reduce leakage.
- Check class counts and identity overlap before training.
- Do not silently discard missing/silent-audio samples; report them and agree on a handling policy with the group.
- This starter uses binary labels: any fake modality = fake. Confirm this matches the project protocol.
- The public sample is for pipeline smoke tests, not for reporting final model performance.

## 9. Suggested smoke test

Run the commands above, then instantiate the dataset and load one batch. Verify:
1. no sample has mismatched audio/video IDs;
2. labels match the source category;
3. frame tensors have the expected shape and finite values;
4. audio tensors have the expected length and finite values;
5. split IDs do not overlap.
