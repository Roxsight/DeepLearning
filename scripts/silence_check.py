from __future__ import annotations
import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import read_manifest

def extract_audio(video_path: Path, wav_path: Path, sample_rate: int):
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", str(wav_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44

def main():
    p = argparse.ArgumentParser(description="Flag missing, silent, or near-silent audio tracks.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default="data/raw/FakeAVCeleb")
    p.add_argument("--out", default="outputs/audio_audit.csv")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--rms-threshold", type=float, default=1e-4)
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        p.error("FFmpeg was not found on PATH. Install FFmpeg before running this script.")

    manifest = read_manifest(Path(args.manifest))
    root = Path(args.data_root)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, row in enumerate(manifest, 1):
            video = root / row["relative_path"]
            wav = tmp / f"{row['sample_id']}.wav"
            status, rms, duration = "ok", 0.0, 0.0
            if not video.exists():
                status = "missing_video"
            elif not extract_audio(video, wav, args.sample_rate):
                status = "missing_or_unreadable_audio"
            else:
                try:
                    audio, sr = sf.read(wav, dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    duration = len(audio) / sr if sr else 0.0
                    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
                    if len(audio) == 0:
                        status = "empty_audio"
                    elif rms < args.rms_threshold:
                        status = "silent_or_near_silent"
                except Exception:
                    status = "unreadable_audio"
            results.append({
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "label": row["label"],
                "audio_status": status,
                "rms": f"{rms:.8f}",
                "duration_seconds": f"{duration:.3f}",
            })
            if i % 25 == 0 or i == len(manifest):
                print(f"Checked {i}/{len(manifest)}")

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)
    counts = {}
    for r in results:
        counts[r["audio_status"]] = counts.get(r["audio_status"], 0) + 1
    print(f"Saved {output}")
    print("Audio status counts:", counts)
    print("Review flagged rows with the group; do not automatically delete them.")

if __name__ == "__main__":
    main()
