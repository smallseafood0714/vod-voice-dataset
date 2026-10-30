"""Step 1: data/00_raw/* -> data/01_audio/<videoID>.wav

Extract the audio track of every ffmpeg-readable file and convert it to
44.1 kHz stereo 16-bit WAV (the format separation models expect).
Files whose output already exists are skipped.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "00_raw"
OUT_DIR = ROOT / "data" / "01_audio"

ID_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def video_id(path: Path) -> str:
    """The [...] part of a yt-dlp style filename, else a sanitized stem."""
    match = ID_PATTERN.findall(path.stem)
    if match:
        return match[-1]
    return re.sub(r"[^\w-]+", "_", path.stem).strip("_")


def extract(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".part.wav")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        # plain WAV breaks above 4 GB (~6.7 h at 44.1 kHz stereo); switch to RF64
        "-rf64", "auto",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(p for p in args.raw_dir.iterdir()
                     if p.is_file() and not p.name.startswith("."))

    seen: dict[str, Path] = {}
    for src in sources:
        vid = video_id(src)
        if vid in seen:
            print(f"[warn] {src.name}: same ID as {seen[vid].name}, skipped")
            continue
        seen[vid] = src

        dst = args.out_dir / f"{vid}.wav"
        if dst.exists():
            print(f"[skip] {vid} (exists)")
            continue
        print(f"[extract] {src.name} -> {dst.name}")
        try:
            extract(src, dst)
        except subprocess.CalledProcessError as e:
            print(f"[error] {src.name}: ffmpeg exit {e.returncode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
