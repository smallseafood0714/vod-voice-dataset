"""GPU stress test: can this machine survive hours of training?

Runs a training-like loop (transformer, fp16 autocast, backward + Adam) for
--minutes and logs temperature, power, clocks and throttle reasons every
--log-sec seconds to logs/gpu_stress_<time>.csv. Each line is fsync'ed, so
after a crash (BSOD) the last line shows the state right before it died.

A fixed fp32 matmul is re-run at every log line and compared with the
result from the start; a mismatch means the GPU computed something wrong
without crashing. A non-finite loss is counted the same way.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

SEQ_LEN = 256
D_MODEL = 512

SMI_FIELDS = ["temperature.gpu", "power.draw", "clocks.gr", "clocks.mem",
              "utilization.gpu", "pstate", "clocks_event_reasons.active"]

# bit -> name for clocks_event_reasons.active (nvidia-smi -q -d PERFORMANCE)
THROTTLE_BITS = {
    0x1: "idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal", 0x40: "hw_thermal",
    0x80: "hw_power_brake", 0x100: "display_clocks",
}


def build_model(device: torch.device) -> nn.Module:
    layer = nn.TransformerEncoderLayer(D_MODEL, nhead=8, dim_feedforward=2048,
                                       batch_first=True)
    return nn.Sequential(nn.TransformerEncoder(layer, num_layers=8),
                         nn.Linear(D_MODEL, D_MODEL)).to(device)


class Trainer:
    def __init__(self, device: torch.device):
        self.device = device
        self.amp = device.type == "cuda"
        self.model = build_model(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

    def batch(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(size, SEQ_LEN, D_MODEL, device=self.device)
        return x, torch.randn_like(x)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            loss = nn.functional.mse_loss(self.model(x), y)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss.item()


def pick_batch(trainer: Trainer, vram_fraction: float) -> int:
    """Largest power-of-two batch whose peak memory stays under the target."""
    total = torch.cuda.mem_get_info()[1]
    target = total * vram_fraction
    best, size = 1, 1
    while size <= 1024:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            trainer.step(*trainer.batch(size))
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            break
        if torch.cuda.max_memory_allocated() > target:
            break
        best, size = size, size * 2
    torch.cuda.empty_cache()
    return best


def query_smi() -> dict[str, str]:
    cmd = ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}",
           "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    values = dict(zip(SMI_FIELDS, (v.strip() for v in out.splitlines()[0].split(","))))
    try:
        bits = int(values["clocks_event_reasons.active"], 16)
        values["clocks_event_reasons.active"] = "+".join(
            name for bit, name in THROTTLE_BITS.items() if bits & bit) or "none"
    except (KeyError, ValueError):
        pass
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--load", type=float, default=1.0,
                    help="fraction of time the GPU is busy (sleeps the rest); "
                         "lower it to find a load level that stays stable")
    ap.add_argument("--vram-fraction", type=float, default=0.7,
                    help="target peak memory as a fraction of total VRAM")
    ap.add_argument("--batch", type=int, help="fixed batch size (skips auto pick)")
    ap.add_argument("--log-sec", type=float, default=10)
    ap.add_argument("--cpu", action="store_true",
                    help="run on CPU (only for checking the script itself)")
    args = ap.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        sys.exit("CUDA is not available")
    device = torch.device("cpu" if args.cpu else "cuda")

    torch.manual_seed(0)
    trainer = Trainer(device)
    if args.batch:
        batch_size = args.batch
    elif args.cpu:
        batch_size = 2
    else:
        batch_size = pick_batch(trainer, args.vram_fraction)
    x, y = trainer.batch(batch_size)

    check_in = torch.randn(1024, 1024, device=device)
    check_ref = check_in @ check_in

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"gpu_stress_{datetime.now():%Y%m%d_%H%M%S}.csv"
    header = ["time", "elapsed_s", "steps", "steps_per_s", "loss", "vram_gb",
              "temp_c", "power_w", "clock_mhz", "mem_clock_mhz", "util_pct",
              "pstate", "throttle", "errors"]
    smi_columns = dict(zip(SMI_FIELDS, header[6:13]))

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"batch {batch_size}, load {args.load:.0%}, {args.minutes:g} min")
    print(f"log: {log_path}")

    errors = steps = 0
    loss = float("nan")
    start = last_log = time.monotonic()
    last_steps = 0
    deadline = start + args.minutes * 60
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        try:
            while True:
                t0 = time.monotonic()
                loss = trainer.step(x, y)
                steps += 1
                if loss != loss or loss in (float("inf"), float("-inf")):
                    errors += 1
                if args.load < 1:
                    busy = time.monotonic() - t0
                    time.sleep(busy * (1 - args.load) / args.load)

                now = time.monotonic()
                if steps > 1 and now - last_log < args.log_sec and now < deadline:
                    continue
                if not torch.equal(check_in @ check_in, check_ref):
                    errors += 1
                row = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_s": round(now - start),
                    "steps": steps,
                    "steps_per_s": round((steps - last_steps) / (now - last_log), 2),
                    "loss": round(loss, 4),
                    "vram_gb": (round(torch.cuda.memory_reserved() / 2**30, 2)
                                if device.type == "cuda" else ""),
                    "errors": errors,
                }
                if device.type == "cuda":
                    for field, value in query_smi().items():
                        row[smi_columns[field]] = value
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                print(f"[{row['elapsed_s'] // 60:>3}m{row['elapsed_s'] % 60:02d}s] "
                      f"{row['steps_per_s']} step/s  {row.get('temp_c', '-')} C  "
                      f"{row.get('power_w', '-')} W  {row.get('clock_mhz', '-')} MHz  "
                      f"throttle={row.get('throttle', '-')}  errors={errors}",
                      flush=True)
                last_log, last_steps = now, steps
                if now >= deadline:
                    break
        except KeyboardInterrupt:
            print("interrupted")
        except RuntimeError as e:
            # "CUDA error: unknown error" here usually means the GPU dropped
            # off the bus; nvidia-smi then reports "GPU is lost"
            elapsed = round(time.monotonic() - start)
            writer.writerow({"time": datetime.now().isoformat(timespec="seconds"),
                             "elapsed_s": elapsed, "steps": steps,
                             "errors": f"crash: {str(e).splitlines()[0]}"})
            f.flush()
            os.fsync(f.fileno())
            print(f"CRASHED after {elapsed} s ({steps} steps): {e}")
            return 2

    minutes = (time.monotonic() - start) / 60
    if errors:
        print(f"FAILED: {errors} wrong results in {minutes:.1f} min")
        return 1
    print(f"OK: survived {minutes:.1f} min without errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
