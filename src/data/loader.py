import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.dataset import FakeAVCelebDataset

CATEGORY = {
    'realvideo-realaudio': 'ARVR',
    'realvideo-fakeaudio': 'AFVR',
    'fakevideo-realaudio': 'ARVF',
    'fakevideo-fakeaudio': 'AFVF',
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filters(sr=16000, n_fft=1024, n_mels=128):
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - left) / (center - left)
        down = (right - freqs) / (right - center)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return torch.from_numpy(fb)


class LogMel:
    def __init__(self, sr=16000, n_fft=1024, win=400, hop=160, n_mels=128, n_steps=200):
        self.n_fft = n_fft
        self.win = win
        self.hop = hop
        self.n_steps = n_steps
        self.window = torch.hann_window(win)
        self.fb = mel_filters(sr, n_fft, n_mels)

    def __call__(self, audio):
        spec = torch.stft(audio, self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.window, center=True, return_complex=True)
        power = spec.abs() ** 2
        mel = torch.log(self.fb @ power + 1e-6)
        if mel.shape[1] >= self.n_steps:
            mel = mel[:, :self.n_steps]
        else:
            mel = torch.nn.functional.pad(mel, (0, self.n_steps - mel.shape[1]), value=math.log(1e-6))
        return mel


class ModelDataset(Dataset):
    def __init__(self, base, samples, train=False):
        self.base = base
        self.samples = samples
        self.train = train
        self.logmel = LogMel()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        sid = item['sample_id']
        meta = self.samples[sid]
        frames = item['frames']
        if self.train and torch.rand(1).item() < 0.5:
            frames = torch.flip(frames, dims=[3])
        frames = (frames - MEAN) / STD
        mel = self.logmel(item['audio'])
        key = meta['category'].lower().replace('_', '-').replace(' ', '')
        video_id = meta['relative_path'].rsplit('.', 1)[0]
        return {
            'frames': frames,
            'mel': mel,
            'label': int(meta['label']),
            'category': CATEGORY.get(key, meta['category']),
            'video_id': video_id,
        }


def build_loaders(splits_path, batch_size, num_workers, processed_root='data/processed',
                  num_frames=16, audio_seconds=2.0, balance=True):
    with open(splits_path) as f:
        payload = json.load(f)
    samples = payload['samples']
    loaders = {}
    for split in ['train', 'val', 'test']:
        base = FakeAVCelebDataset(payload[split], processed_root=processed_root, samples=samples,
                                  num_frames=num_frames, image_size=224, sample_rate=16000,
                                  audio_seconds=audio_seconds)
        ds = ModelDataset(base, samples, train=(split == 'train'))
        if split == 'train' and balance and len(ds) > 0:
            labels = [int(samples[s]['label']) for s in payload[split]]
            counts = np.bincount(labels, minlength=2)
            weights = [1.0 / max(counts[y], 1) for y in labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
            loaders[split] = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                        num_workers=num_workers, drop_last=len(ds) > batch_size)
        else:
            loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loaders
