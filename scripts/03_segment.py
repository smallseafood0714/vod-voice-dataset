"""Step 3: data/02_vocals or 01_audio/<id>.wav -> data/03_segments/<id>/<id>_<start_ms>.wav

Uses the separated vocals when step 2 was run for a video, otherwise the
step 1 audio (clean streams can skip step 2). Finds speech with Silero VAD
(CPU only) and cuts it into mono clips of --min-sec to --max-sec.
Neighbouring speech separated by a short pause is merged into one clip.
Each video gets a segments.csv listing every clip; videos that already
have one are skipped.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # VAD runs on CPU; keep the GPU untouched

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "01_audio"
VOCAL_DIR = ROOT / "data" / "02_vocals"
OUT_DIR = ROOT / "data" / "03_segments"

VAD_SR = 16000
BLOCK_SEC = 600         # VAD runs on 10-minute blocks to bound memory
MIN_SILENCE_MS = 300    # pauses shorter than this never split speech
SPEECH_PAD_MS = 150     # context kept before and after each speech span
MERGE_GAP_SEC = 0.8     # neighbouring spans this close are merged


def speech_spans(src: Path, model, max_sec: float) -> list[tuple[float, float]]:
    """Speech (start, end) in seconds over the whole file."""
    spans = []
    with sf.SoundFile(src) as f:
        sr = f.samplerate
        offset = 0.0
        for block in f.blocks(blocksize=BLOCK_SEC * sr, dtype="float32",
                              always_2d=True):
            x = block.mean(axis=1)
            if np.any(x):
                wav = AF.resample(torch.from_numpy(x), sr, VAD_SR)
                for ts in get_speech_timestamps(
                        wav, model, sampling_rate=VAD_SR,
                        max_speech_duration_s=max_sec,
                        min_silence_duration_ms=MIN_SILENCE_MS,
                        speech_pad_ms=SPEECH_PAD_MS):
                    spans.append((offset + ts["start"] / VAD_SR,
                                  offset + ts["end"] / VAD_SR))
            offset += len(x) / sr
    return spans


def merge(spans, max_sec: float) -> list[tuple[float, float]]:
    clips: list[list[float]] = []
    for s, e in spans:
        if clips and s - clips[-1][1] <= MERGE_GAP_SEC and e - clips[-1][0] <= max_sec:
            clips[-1][1] = e
        else:
            clips.append([s, e])
    return [(s, e) for s, e in clips]


def db(x: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-10)))


def segment_file(model, src: Path, out_dir: Path,
                 min_sec: float, max_sec: float) -> tuple[int, float, float]:
    spans = speech_spans(src, model, max_sec)
    clips = merge(spans, max_sec)
    speech = sum(e - s for s, e in clips)

    tmp_dir = out_dir.with_name(out_dir.name + ".part")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    sr = sf.info(src).samplerate
    rows, kept = [], 0.0
    for s, e in clips:
        if e - s < min_sec:
            continue
        audio, _ = sf.read(src, start=int(s * sr), stop=int(e * sr),
                           dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        name = f"{src.stem}_{int(s * 1000):09d}.wav"
        sf.write(tmp_dir / name, audio, sr, subtype="PCM_16")
        kept += e - s
        rows.append({
            "file": name,
            "start": f"{s:.3f}",
            "end": f"{e:.3f}",
            "duration": f"{e - s:.3f}",
            "rms_db": f"{db(audio):.1f}",
            "peak_db": f"{20 * np.log10(max(float(np.abs(audio).max()), 1e-10)):.1f}",
            # streams are limited to ~0 dBFS, so peak alone does not mean clipping
            "clip_pct": f"{np.mean(np.abs(audio) >= 0.999) * 100:.3f}",
            "source": src.parent.name,
        })

    with open(tmp_dir / "segments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    return len(rows), kept, speech


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--vocal-dir", type=Path, default=VOCAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-sec", type=float, default=2.0)
    parser.add_argument("--max-sec", type=float, default=15.0)
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    model = load_silero_vad()
    for audio in sorted(args.audio_dir.glob("*.wav")):
        if audio.name.endswith(".part.wav"):
            continue
        out_dir = args.out_dir / audio.stem
        if (out_dir / "segments.csv").exists():
            print(f"[skip] {audio.stem} (exists)")
            continue
        vocals = args.vocal_dir / audio.name
        src = vocals if vocals.exists() else audio
        print(f"[segment] {audio.stem} from {src.parent.name}", flush=True)
        t0 = time.time()
        count, kept, speech = segment_file(model, src, out_dir,
                                           args.min_sec, args.max_sec)
        print(f"[done] {src.stem}: {count} clips, {kept / 60:.1f} min kept "
              f"of {speech / 60:.1f} min speech "
              f"({(speech - kept) / 60:.1f} min in clips under {args.min_sec} s), "
              f"{time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
