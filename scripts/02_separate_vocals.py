"""Step 2: data/01_audio/<id>.wav -> data/02_vocals/<id>.wav (vocals, mono)

Runs a vocal separation model (audio-separator / UVR models).
Only the parts louder than --threshold-db are separated; everything else
is written as silence, so the output stays time-aligned with the input.
Loud parts are packed into chunks of at most --chunk-sec; regions longer
than that are split with a short crossfaded overlap, so memory use does not
grow with file length. Files whose output already exists are skipped.
"""
import os
import sys

if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # never touch the GPU

import argparse
import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from audio_separator.separator import Separator

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "01_audio"
OUT_DIR = ROOT / "data" / "02_vocals"
MODEL_DIR = ROOT / "models"

DEFAULT_MODEL = "vocals_mel_band_roformer.ckpt"

FRAME_SEC = 0.05   # loudness is measured per 50 ms frame
BRIDGE_SEC = 1.0   # quiet gaps shorter than this stay inside a region
PAD_SEC = 0.3      # extra context kept around each region
GAP_SEC = 1.0      # silence between regions packed into one model call


class VocalSeparator:
    def __init__(self, model: str, work_dir: Path, autocast: bool):
        self.work_dir = work_dir
        self.sep = Separator(
            model_file_dir=str(MODEL_DIR),
            output_dir=str(work_dir),
            output_format="WAV",
            output_single_stem="Vocals",
            # only rescales above this peak; 1.0 keeps chunk levels consistent
            normalization_threshold=1.0,
            use_soundfile=True,
            use_autocast=autocast,
            log_level=logging.WARNING,
        )
        self.sep.load_model(model_filename=model)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Stereo float audio in, mono float vocals out (same length)."""
        src = self.work_dir / "chunk_in.wav"
        sf.write(src, audio, sr, subtype="FLOAT")
        outputs = self.sep.separate(str(src), {"Vocals": "chunk_vocals"})
        out_path = self.work_dir / Path(outputs[0]).name
        vocals, _ = sf.read(out_path, dtype="float32", always_2d=True)
        out_path.unlink()
        src.unlink()

        vocals = vocals.mean(axis=1)
        n = len(audio)
        if len(vocals) < n:
            vocals = np.pad(vocals, (0, n - len(vocals)))
        return vocals[:n]


def active_regions(src: Path, threshold_db: float) -> list[tuple[int, int]]:
    """Sample ranges louder than threshold_db (gaps bridged, edges padded)."""
    with sf.SoundFile(src) as f:
        sr, total = f.samplerate, f.frames
        frame = int(FRAME_SEC * sr)
        loud = []
        for block in f.blocks(blocksize=frame * 1200, dtype="float32",
                              always_2d=True):
            x = block.mean(axis=1)
            m = len(x) // frame
            rms = np.sqrt((x[:m * frame].reshape(m, frame) ** 2).mean(axis=1))
            loud.append(rms > 10 ** (threshold_db / 20))
    idx = np.flatnonzero(np.concatenate(loud))
    if idx.size == 0:
        return []

    breaks = np.flatnonzero(np.diff(idx) > BRIDGE_SEC / FRAME_SEC)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    pad = int(PAD_SEC / FRAME_SEC)
    # BRIDGE_SEC > 2 * PAD_SEC, so padded regions never overlap
    return [(max(0, (s - pad) * frame), min(total, (e + pad) * frame))
            for s, e in zip(starts, ends)]


def split_long(regions, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """Split regions longer than chunk into overlapping pieces."""
    pieces = []
    for s, e in regions:
        for start in range(s, e, chunk):
            pieces.append((max(s, start - overlap), min(e, start + chunk)))
    return pieces


def pack(pieces, chunk: int, gap: int) -> list[list[tuple[int, int]]]:
    """Group consecutive pieces so each model call stays under chunk."""
    batches, current, size = [], [], 0
    for s, e in pieces:
        if current and size + gap + (e - s) > chunk:
            batches.append(current)
            current, size = [], 0
        size += (gap if current else 0) + (e - s)
        current.append((s, e))
    if current:
        batches.append(current)
    return batches


def separate_file(model: VocalSeparator, src: Path, dst: Path, work_dir: Path,
                  threshold_db: float, chunk_sec: int, overlap_sec: int) -> float:
    """Returns the number of seconds actually sent to the model."""
    info = sf.info(src)
    sr, total = info.samplerate, info.frames
    chunk, overlap, gap = chunk_sec * sr, overlap_sec * sr, int(GAP_SEC * sr)

    regions = active_regions(src, threshold_db)
    pieces = split_long(regions, chunk, overlap)
    active = sum(e - s for s, e in regions) / sr
    print(f"    {len(regions)} loud regions, {active / 60:.1f} min of "
          f"{total / sr / 60:.1f} min above {threshold_db} dB", flush=True)

    # float32 accumulator on disk, so RAM use does not depend on file length
    acc_path = work_dir / "acc.f32"
    acc = np.memmap(acc_path, dtype=np.float32, mode="w+", shape=(total,))
    written_end = 0
    done = 0.0
    for batch in pack(pieces, chunk, gap):
        parts = [sf.read(src, start=s, stop=e, dtype="float32",
                         always_2d=True)[0] for s, e in batch]
        silence = np.zeros((gap, parts[0].shape[1]), dtype=np.float32)
        joined = np.concatenate(
            [x for p in parts for x in (silence, p)][1:])
        vocals = model(joined, sr)

        pos = 0
        for (s, e), part in zip(batch, parts):
            v = vocals[pos:pos + len(part)]
            pos += len(part) + gap
            if s < written_end:  # overlap with the previous piece: crossfade
                n = written_end - s
                fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
                acc[s:written_end] = acc[s:written_end] * (1 - fade) + v[:n] * fade
                acc[written_end:e] = v[n:]
            else:
                acc[s:e] = v
            written_end = e
        done += sum(e - s for s, e in batch) / sr
        print(f"    {done / 60:6.1f} / {active / 60:.1f} min", flush=True)

    tmp = dst.with_suffix(".part.wav")
    with sf.SoundFile(tmp, "w", sr, 1, subtype="PCM_16") as out:
        for i in range(0, total, chunk):
            out.write(np.asarray(acc[i:i + chunk]))
    del acc
    acc_path.unlink()
    tmp.replace(dst)
    return active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*",
                        help="video IDs to separate (default: every file in --in-dir)")
    parser.add_argument("--in-dir", type=Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold-db", type=float, default=-50.0,
                        help="parts quieter than this are written as silence")
    parser.add_argument("--chunk-sec", type=int, default=600)
    parser.add_argument("--overlap-sec", type=int, default=5)
    parser.add_argument("--autocast", action="store_true",
                        help="mixed precision (faster; crashed the GPU once in testing)")
    parser.add_argument("--cpu", action="store_true",
                        help="run on CPU only and keep the GPU untouched")
    args = parser.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        print("CUDA GPU not available; check nvidia-smi (or pass --cpu)")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(args.in_dir.glob("*.wav"))
            if not p.name.endswith(".part.wav")
            and (not args.ids or p.stem in args.ids)
            and not (args.out_dir / p.name).exists()]
    if not todo:
        print("nothing to do")
        return 0

    work_dir = Path(tempfile.mkdtemp(prefix="sep_"))
    try:
        model = VocalSeparator(args.model, work_dir, args.autocast)
        for src in todo:
            dst = args.out_dir / src.name
            hours = sf.info(src).duration / 3600
            print(f"[separate] {src.name} ({hours:.2f} h)", flush=True)
            t0 = time.time()
            active = separate_file(model, src, dst, work_dir, args.threshold_db,
                                   args.chunk_sec, args.overlap_sec)
            minutes = (time.time() - t0) / 60
            print(f"[done] {src.name}: {minutes:.1f} min, "
                  f"{minutes / hours:.1f} min per hour of video, "
                  f"{minutes / max(active / 3600, 1e-9):.1f} min per hour "
                  f"of loud audio", flush=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
