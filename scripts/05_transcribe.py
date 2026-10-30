"""Step 5: data/04_filtered/<id>/*.wav -> data/05_dataset/

Transcribes every clip with Paraformer-zh plus its punctuation model
(FunASR, CPU only) and converts the text to Traditional Chinese (OpenCC
s2twp). A second pass with SenseVoiceSmall gives a second-opinion text,
how well the two agree, and SenseVoice's emotion / event tags (laughter,
BGM, ...). Clips where the two models disagree are the ones to check first.

Each video gets its own transcripts/<id>.csv. Transcripts may be corrected
by hand, so an existing one is never overwritten: only videos without a
transcript are transcribed. Clips whose text is empty, only fillers,
Japanese, too slow / too fast for their length, or where the two models
disagree get keep=0 and a reason in the flag column.
Every transcribed clip is copied to wavs/ so the folder is self-contained
for backup. The GPT-SoVITS list (wav_path|speaker|lang|text) is rebuilt from
all transcripts on every run and holds only rows with keep=1, so editing
text or keep and re-running is enough to update it.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only; keep the GPU untouched

import argparse
import csv
import gc
import re
import shutil
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / "models" / "modelscope"))

import numpy as np
import opencc
import soundfile as sf
import torch
import torchaudio.functional as AF
from funasr import AutoModel

IN_DIR = ROOT / "data" / "04_filtered"
OUT_DIR = ROOT / "data" / "05_dataset"

ASR_SR = 16000
# text made only of these (plus punctuation) is a filler, not a sentence
FILLERS = set("嗯恩啊阿呃喔哦噢欸誒唉哎哈呵嘿蛤耶呀吼齁嘛呢吧啦喲哇痾")
TAG = re.compile(r"<\|([^|]*)\|>")
CJK = re.compile(r"[\u3400-\u9fff]")
KANA = re.compile(r"[\u3040-\u30ff]")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d")
VOWELS = re.compile(r"[aeiouy]+")
LATIN = re.compile(r"[A-Za-z]")
FIELDS = ["file", "duration", "text", "lang", "keep", "flag", "cps",
          "confidence", "agree", "alt_text", "english", "emotion", "event"]

CC = opencc.OpenCC("s2twp")


class Paraformer:
    def __init__(self):
        self.model = AutoModel(model="paraformer-zh", punc_model="ct-punc",
                               device="cpu", disable_update=True)
        # keep the decoder posteriors of the last call to compute a confidence
        net = self.model.model
        decode = net._seaco_decode_with_ASF
        def capture(*args, **kwargs):
            self.logits = decode(*args, **kwargs)
            return self.logits
        net._seaco_decode_with_ASF = capture

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> dict:
        self.logits = None
        text = self.model.generate(input=wav, fs=ASR_SR,
                                   disable_pbar=True)[0]["text"]
        conf = ""
        if text and self.logits is not None:
            conf = f"{float(self.logits[0].exp().max(dim=-1).values.mean()):.3f}"
        return {"text": re.sub(r"\s+", " ", CC.convert(text)).strip(),
                "confidence": conf}


class SenseVoice:
    def __init__(self):
        self.model = AutoModel(model="iic/SenseVoiceSmall", device="cpu",
                               disable_update=True)

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> dict:
        # without ITN numbers stay as Chinese characters, like Paraformer's
        raw = self.model.generate(input=wav, fs=ASR_SR, language="zh",
                                  use_itn=False, disable_pbar=True)[0]["text"]
        tags = TAG.findall(raw)
        return {
            "alt_text": CC.convert(TAG.sub("", raw)).strip(),
            "emotion": tags[1] if len(tags) > 1 else "",
            "event": tags[2] if len(tags) > 2 else "",
        }


def load_16k(path: Path) -> tuple[np.ndarray, float]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    return AF.resample(torch.from_numpy(audio), sr, ASR_SR).numpy(), len(audio) / sr


def run_pass(name: str, model, videos: list[Path], results: dict) -> None:
    for v in videos:
        clips = sorted(v.glob("*.wav"))
        t0, audio_sec = time.time(), 0.0
        for path in clips:
            wav, duration = load_16k(path)
            audio_sec += duration
            res = results.setdefault(v.name, {}).setdefault(
                path.name, {"duration": duration})
            res.update(model(wav))
        took = time.time() - t0
        print(f"[{name}] {v.name}: {len(clips)} clips, {audio_sec / 60:.1f} min "
              f"in {took:.0f} s ({audio_sec / took:.0f}x real time)", flush=True)


def spoken(text: str) -> str:
    return "".join(c for c in text.lower() if CJK.match(c) or c.isalnum())


def syllables(text: str) -> int:
    """Rough spoken length: one per Chinese character or digit, and the
    vowel groups of each English word."""
    return len(CJK.findall(text)) + sum(
        len(VOWELS.findall(w.lower())) or 1 for w in WORD.findall(text))


def flag(text: str, alt_text: str, agree: float, duration: float,
         args: argparse.Namespace) -> str:
    chars = spoken(text)
    if not chars:
        return "empty"
    if all(c in FILLERS for c in chars):
        return "filler"
    # SenseVoice hears Japanese (mostly singing) where Paraformer guesses Chinese
    if len(KANA.findall(alt_text)) >= 2:
        return "japanese"
    cps = syllables(text) / duration
    if cps < args.min_cps:
        return "slow"
    if cps > args.max_cps:
        return "fast"
    if agree < args.min_agree:
        return "disagree"
    return ""


def write_transcript(results: dict, out_csv: Path,
                     args: argparse.Namespace) -> list[dict]:
    rows = []
    for name, res in results.items():
        text, duration = res["text"], res["duration"]
        a, b = spoken(text), spoken(res["alt_text"])
        agree = SequenceMatcher(None, a, b).ratio() if a or b else 1.0
        reason = flag(text, res["alt_text"], agree, duration, args)
        # all-English clips go to GPT-SoVITS as en; zh would drop every letter
        lang = "en" if LATIN.search(text) and not CJK.search(text) else "zh"
        rows.append({
            "file": name,
            "duration": f"{duration:.3f}",
            "text": text,
            "lang": lang,
            "keep": int(not reason),
            "flag": reason,
            "cps": f"{syllables(text) / duration:.1f}",
            "confidence": res["confidence"],
            "agree": f"{agree:.2f}",
            "alt_text": res["alt_text"],
            # GPT-SoVITS drops Latin letters from zh text, so these lose words
            "english": int(lang == "zh" and bool(LATIN.search(text))),
            "emotion": res["emotion"],
            "event": res["event"],
        })
    # utf-8-sig so Excel opens it without mojibake; .part -> rename never
    # replaces an existing transcript
    tmp = out_csv.with_suffix(".csv.part")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.rename(out_csv)
    return rows


def read_transcript(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", default="me",
                        help="speaker name written to the .list file (and its file name)")
    parser.add_argument("--min-cps", type=float, default=1.5,
                        help="flag clips with fewer characters per second")
    parser.add_argument("--max-cps", type=float, default=9.0,
                        help="flag clips with more characters per second")
    parser.add_argument("--min-agree", type=float, default=0.5,
                        help="flag clips where Paraformer and SenseVoice agree less")
    args = parser.parse_args()

    tr_dir = OUT_DIR / "transcripts"
    wav_dir = OUT_DIR / "wavs"
    tr_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in IN_DIR.iterdir()
                    if p.is_dir() and not p.name.startswith("_")
                    and any(p.glob("*.wav")))
    todo = [v for v in videos if not (tr_dir / f"{v.name}.csv").exists()]
    for v in videos:
        if v not in todo:
            print(f"[skip] {v.name} (transcript exists)")

    if todo:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        results: dict = {}
        # one model in memory at a time; RAM is often tight
        for name, cls in (("paraformer", Paraformer), ("sensevoice", SenseVoice)):
            model = cls()
            run_pass(name, model, todo, results)
            del model
            gc.collect()
        for v in todo:
            rows = write_transcript(results[v.name], tr_dir / f"{v.name}.csv", args)
            flagged = [r for r in rows if not r["keep"]]
            print(f"[done] {v.name}: {len(rows)} clips, {len(flagged)} flagged")

    lines, kept_sec, missing = [], 0.0, 0
    for path in sorted(tr_dir.glob("*.csv")):
        vid = path.stem
        rows = read_transcript(path)
        new = ({p.name for p in (IN_DIR / vid).glob("*.wav")}
               - {r["file"] for r in rows})
        if new:
            print(f"[note] {vid}: {len(new)} clips in 04_filtered have no "
                  f"transcript row (step 4 was re-run?)")
        for r in rows:
            dst = wav_dir / r["file"]
            if not dst.exists():
                src = IN_DIR / vid / r["file"]
                if not src.exists():
                    missing += 1
                    continue
                shutil.copy2(src, dst)
            text = r["text"].replace("|", " ").strip()
            if r["keep"].strip() == "1" and text:
                lang = r.get("lang", "").strip() or "zh"
                lines.append(f"{dst}|{args.speaker}|{lang}|{text}")
                kept_sec += float(r["duration"])
    if missing:
        print(f"[warn] {missing} transcript rows have no wav "
              f"(neither in wavs/ nor 04_filtered/)")

    # GPT-SoVITS reads the list as plain UTF-8; a BOM would break the first path
    list_path = OUT_DIR / f"{args.speaker}.list"
    tmp = list_path.with_suffix(".list.part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(list_path)
    print(f"[list] {list_path.relative_to(ROOT)}: {len(lines)} clips, "
          f"{kept_sec / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
