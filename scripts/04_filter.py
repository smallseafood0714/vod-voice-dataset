"""Step 4: data/03_segments/<id>/*.wav -> data/04_filtered/<id>/*.wav

Keeps clips that sound like the reference speaker and pass level checks.
The reference is ref/*.wav when there is any; otherwise it is the centroid
of all clips, which works because nearly every clip is the streamer.
Every clip's score and the reason it was dropped go to
data/04_filtered/scores.csv. Speaker embeddings (ECAPA, CPU only) are cached
per video in data/04_filtered/_cache/, so re-runs only embed new videos.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only; keep the GPU untouched
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

ROOT = Path(__file__).resolve().parent.parent
SEG_DIR = ROOT / "data" / "03_segments"
OUT_DIR = ROOT / "data" / "04_filtered"
REF_DIR = ROOT / "ref"
MODEL_DIR = ROOT / "models" / "spkrec-ecapa-voxceleb"

EMB_SR = 16000
REF_WINDOW_SEC = 10   # long reference recordings are embedded in windows
CORE_FRACTION = 0.8   # centroid is refined on the most typical 80% of clips


class Embedder:
    def __init__(self):
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(MODEL_DIR),
            run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY)

    @torch.no_grad()
    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        wav = AF.resample(torch.from_numpy(audio), sr, EMB_SR)
        emb = self.model.encode_batch(wav.unsqueeze(0)).squeeze().numpy()
        return emb / np.linalg.norm(emb)


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def video_embeddings(embedder: Embedder, seg_dir: Path,
                     cache_dir: Path) -> tuple[list[dict], np.ndarray]:
    with open(seg_dir / "segments.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["file"] for r in rows]
    cache = cache_dir / f"{seg_dir.name}.npz"
    if cache.exists():
        data = np.load(cache)
        if list(data["names"]) == names:
            return rows, data["emb"]

    t0 = time.time()
    emb = np.stack([embedder(*read_mono(seg_dir / n)) for n in names]) \
        if names else np.zeros((0, 192), dtype=np.float32)
    np.savez(cache, names=np.array(names), emb=emb)
    print(f"[embed] {seg_dir.name}: {len(names)} clips, "
          f"{time.time() - t0:.0f} s", flush=True)
    return rows, emb


def reference(embedder: Embedder, all_emb: np.ndarray) -> tuple[np.ndarray, str]:
    refs = sorted(REF_DIR.glob("*.wav"))
    if refs:
        embs = []
        for path in refs:
            audio, sr = read_mono(path)
            step = REF_WINDOW_SEC * sr
            for i in range(0, max(1, len(audio) - sr), step):
                embs.append(embedder(audio[i:i + step], sr))
        c = np.mean(embs, axis=0)
        return c / np.linalg.norm(c), f"ref/ ({len(refs)} files)"

    c = all_emb.mean(axis=0)
    c /= np.linalg.norm(c)
    sims = all_emb @ c
    core = all_emb[sims >= np.quantile(sims, 1 - CORE_FRACTION)]
    c = core.mean(axis=0)
    return c / np.linalg.norm(c), "centroid of all clips"


def sync_dir(src_dir: Path, dst_dir: Path, keep: set[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in dst_dir.glob("*.wav"):
        if path.name not in keep:
            path.unlink()
    for name in keep:
        if not (dst_dir / name).exists():
            shutil.copy2(src_dir / name, dst_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-sim", type=float, default=0.5,
                        help="minimum cosine similarity to the reference voice")
    parser.add_argument("--min-rms-db", type=float, default=-35.0,
                        help="drop clips quieter than this")
    parser.add_argument("--max-clip-pct", type=float, default=0.1,
                        help="drop clips with more clipped samples than this (%%)")
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    cache_dir = OUT_DIR / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder()

    videos = sorted(p for p in SEG_DIR.iterdir()
                    if (p / "segments.csv").exists())
    per_video = [(v, *video_embeddings(embedder, v, cache_dir)) for v in videos]
    all_emb = np.concatenate([e for _, _, e in per_video])
    ref, ref_source = reference(embedder, all_emb)
    print(f"[reference] {ref_source}")

    out_rows = []
    for seg_dir, rows, emb in per_video:
        keep = set()
        for row, sim in zip(rows, emb @ ref):
            reasons = []
            if sim < args.min_sim:
                reasons.append("speaker")
            if float(row["rms_db"]) < args.min_rms_db:
                reasons.append("quiet")
            if float(row["clip_pct"]) > args.max_clip_pct:
                reasons.append("clipping")
            if not reasons:
                keep.add(row["file"])
            out_rows.append({
                "video": seg_dir.name,
                "file": row["file"],
                "duration": row["duration"],
                "rms_db": row["rms_db"],
                "clip_pct": row["clip_pct"],
                "similarity": f"{sim:.3f}",
                "keep": int(not reasons),
                "drop_reason": "+".join(reasons),
            })
        sync_dir(seg_dir, OUT_DIR / seg_dir.name, keep)
        kept = sum(float(r["duration"]) for r in out_rows
                   if r["video"] == seg_dir.name and r["keep"])
        print(f"[filter] {seg_dir.name}: {len(keep)} / {len(rows)} clips kept, "
              f"{kept / 60:.1f} min")

    with open(OUT_DIR / "scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    kept = [r for r in out_rows if r["keep"]]
    print(f"[total] {len(kept)} / {len(out_rows)} clips, "
          f"{sum(float(r['duration']) for r in kept) / 3600:.2f} h kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
