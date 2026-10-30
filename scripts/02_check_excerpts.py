"""Cut before/after excerpts of step 2 for listening checks.

For every file present in both data/01_audio and data/02_vocals, find the
windows where both speech and removed sound (music / game audio) are loud,
and write each one as a before/after pair to data/02_vocals/_check/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "01_audio"
VOCAL_DIR = ROOT / "data" / "02_vocals"


def window_scores(src: Path, vocals: Path, win_sec: int) -> np.ndarray:
    """Per window: geometric mean of vocal RMS and removed-sound RMS."""
    scores = []
    with sf.SoundFile(src) as fa, sf.SoundFile(vocals) as fv:
        n = win_sec * fa.samplerate
        while True:
            a = fa.read(n, dtype="float32", always_2d=True).mean(axis=1)
            v = fv.read(n, dtype="float32")
            m = min(len(a), len(v))
            if m < n:
                break
            a, v = a[:m], v[:m]
            voc = np.sqrt(np.mean(v ** 2))
            rest = np.sqrt(np.mean((a - v) ** 2))
            scores.append(np.sqrt(voc * rest))
    return np.array(scores)


def pick(scores: np.ndarray, count: int, min_gap: int) -> list[int]:
    chosen: list[int] = []
    for i in np.argsort(scores)[::-1]:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(int(i))
        if len(chosen) == count:
            break
    return sorted(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()

    out_dir = VOCAL_DIR / "_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    for vocals in sorted(VOCAL_DIR.glob("*.wav")):
        if vocals.name.endswith(".part.wav"):
            continue
        src = IN_DIR / vocals.name
        if not src.exists():
            continue
        scores = window_scores(src, vocals, args.seconds)
        # keep picks at least 10 minutes apart so they cover different scenes
        min_gap = max(1, 600 // args.seconds)
        for i in pick(scores, args.count, min_gap):
            start = i * args.seconds
            stamp = f"{start // 3600:02d}h{start % 3600 // 60:02d}m{start % 60:02d}s"
            for path, tag in ((src, "1before"), (vocals, "2after")):
                info = sf.info(path)
                audio, sr = sf.read(path, start=start * info.samplerate,
                                    frames=args.seconds * info.samplerate)
                sf.write(out_dir / f"{vocals.stem}_{stamp}_{tag}.wav", audio, sr)
            print(f"[check] {vocals.stem} @ {stamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
