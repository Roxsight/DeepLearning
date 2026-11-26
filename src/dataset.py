from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

class FakeAVCelebDataset(Dataset):
    """
    Returns paired video/audio tensors.
    frames: [T, C, H, W], float32 in [0,1]
    audio: [sample_rate * audio_seconds], float32
    label: scalar long (0=real, 1=fake)
    """
    def __init__(
        self,
        sample_ids,
        processed_root="data/processed",
        samples=None,
        num_frames=8,
        image_size=224,
        sample_rate=16000,
        audio_seconds=5.0,
    ):
        self.root = Path(processed_root)
        self.sample_ids = list(sample_ids)
        self.samples = samples or {}
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.sample_rate = int(sample_rate)
        self.audio_length = int(sample_rate * audio_seconds)

    @classmethod
    def from_splits(cls, split_json, split="train", **kwargs):
        payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        return cls(payload[split], samples=payload["samples"], **kwargs)

    def __len__(self):
        return len(self.sample_ids)

    def _load_frames(self, sample_id):
        folder = self.root / "frames" / sample_id
        paths = sorted(folder.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(f"No extracted frames found for {sample_id}: {folder}")
        # Uniformly sample/pad to fixed frame count.
        idx = np.linspace(0, len(paths) - 1, self.num_frames).round().astype(int)
        frames = []
        for i in idx:
            image = cv2.imread(str(paths[int(i)]))
            if image is None:
                raise RuntimeError(f"Could not read frame: {paths[int(i)]}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            frames.append(image)
        return torch.from_numpy(np.stack(frames))

    def _load_audio(self, sample_id):
        path = self.root / "audio" / f"{sample_id}.wav"
        if not path.exists():
            raise FileNotFoundError(f"Processed audio not found: {path}")
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sr} Hz in {path}")
        if len(audio) > self.audio_length:
            audio = audio[:self.audio_length]
        elif len(audio) < self.audio_length:
            audio = np.pad(audio, (0, self.audio_length - len(audio)))
        return torch.from_numpy(np.asarray(audio, dtype=np.float32))

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        if sample_id not in self.samples:
            raise KeyError(f"{sample_id} is missing from the samples metadata in splits.json")
        meta = self.samples[sample_id]
        frames = self._load_frames(sample_id)
        audio = self._load_audio(sample_id)
        label = torch.tensor(int(meta["label"]), dtype=torch.long)
        return {
            "frames": frames,
            "audio": audio,
            "label": label,
            "sample_id": sample_id,
        }
