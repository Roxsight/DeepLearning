from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import read_manifest

def get_window(video_path: Path, seconds: float):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total / fps if fps > 0 else 0.0
    start = max(0.0, duration / 2 - seconds / 2)
    return start, fps, total

def extract_frames(video_path: Path, output_dir: Path, count: int, size: int, start: float, seconds: float, fps: float, total: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened() or total <= 0:
        cap.release()
        return 0
    first = int(round(start * fps))
    last = min(total - 1, int(round((start + seconds) * fps)) - 1)
    last = max(first, last)
    indices = np.linspace(first, last, num=count).round().astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for out_idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        target = output_dir / f"{out_idx:03d}.jpg"
        if cv2.imwrite(str(target), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            saved += 1
    cap.release()
    return saved

def extract_audio(video_path: Path, wav_path: Path, sample_rate: int, start: float, seconds: float):
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", str(wav_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44

def standardize_audio(wav_path: Path, sample_rate: int, seconds: float):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate and len(audio):
        common = gcd(int(sr), int(sample_rate))
        audio = resample_poly(audio, sample_rate // common, sr // common).astype(np.float32)
    target_len = int(sample_rate * seconds)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return np.asarray(audio, dtype=np.float32)

def main():
    p = argparse.ArgumentParser(description="Extract fixed-count video frames and mono WAV audio.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default="data/raw/FakeAVCeleb")
    p.add_argument("--out-root", default="data/processed")
    p.add_argument("--frames-per-video", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--audio-seconds", type=float, default=2.0)
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        p.error("FFmpeg was not found on PATH. Install FFmpeg before preprocessing.")
    if args.frames_per_video < 1 or args.image_size < 1 or args.audio_seconds <= 0:
        p.error("Frame count, image size and audio duration must be positive.")

    manifest = read_manifest(Path(args.manifest))
    data_root, out_root = Path(args.data_root), Path(args.out_root)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for row in tqdm(manifest, desc="Preprocessing"):
            sid = row["sample_id"]
            video_path = data_root / row["relative_path"]
            frame_dir = out_root / "frames" / sid
            audio_out = out_root / "audio" / f"{sid}.wav"
            if not video_path.exists():
                failures.append((sid, "missing_video"))
                continue
            start, fps, total = get_window(video_path, args.audio_seconds)
            n_frames = extract_frames(video_path, frame_dir, args.frames_per_video, args.image_size, start, args.audio_seconds, fps, total)
            if n_frames == 0:
                failures.append((sid, "frame_extraction_failed"))
                continue
            temp_wav = tmp / f"{sid}.wav"
            if not extract_audio(video_path, temp_wav, args.sample_rate, start, args.audio_seconds):
                failures.append((sid, "audio_extraction_failed"))
                continue
            try:
                audio = standardize_audio(temp_wav, args.sample_rate, args.audio_seconds)
                audio_out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(audio_out, audio, args.sample_rate)
            except Exception as e:
                failures.append((sid, f"audio_processing_failed:{type(e).__name__}"))
                continue

    fail_path = out_root / "preprocessing_failures.csv"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with fail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "reason"])
        writer.writerows(failures)
    print(f"Processed {len(manifest) - len(failures)}/{len(manifest)} samples.")
    print(f"Failures: {len(failures)}; details: {fail_path}")

if __name__ == "__main__":
    main()
