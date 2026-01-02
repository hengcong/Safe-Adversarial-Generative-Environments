#!/usr/bin/env python3
"""
fake_mappo_log_simple.py

Simulate concise MAPPO training logs for an autonomous-driving (CARLA-like) setup.
No PPO epoch/minibatch details. Only clean PyTorch-style training signals + driving metrics.
"""

import argparse
import math
import random
import time
import csv
from datetime import datetime
import shutil
import sys

TERM_WIDTH = shutil.get_terminal_size((120, 20)).columns

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ascii_trend(values, width=40):
    if not values:
        return ""
    low, high = min(values), max(values)
    if high == low:
        high = low + 1e-6
    chars = "▁▂▃▄▅▆▇█"
    out = []
    for v in values[-width:]:
        frac = (v - low) / (high - low)
        idx = int(frac * (len(chars)-1))
        out.append(chars[idx])
    return "".join(out)

def human_seconds(s):
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    return f"{m}m{s-60*m:.0f}s"

def simulate(iters=9766, device="cpu", agents=16, rollout_len=128, csv_out=None):

    random.seed(0)
    lr0 = 3e-4

    losses, values, entropies = [], [], []

    print("=" * TERM_WIDTH)
    print(now_ts(), f"SIMULATED MAPPO TRAINING | device={device} | agents={agents}")
    print("-" * TERM_WIDTH)

    start_time = time.time()
    total_steps_per_iter = agents * rollout_len
    csv_rows = []

    for it in range(1, iters+1):
        iter_start = time.time()
        progress = it / iters
        lr = lr0 * (0.95 ** (it // 1000))

        # rollout summary
        print(f"[Iter {it:04d}] rollout collected: {total_steps_per_iter} steps")

        # RL losses (averaged per iteration rather than per batch)
        policy_loss = 0.25*math.exp(-it/9000) + random.uniform(-0.01,0.01)
        value_loss  = 0.40*math.exp(-it/8500) + random.uniform(-0.02,0.02)
        entropy     = max(0.05, 0.75 - 0.45*progress + random.uniform(-0.02,0.02))
        grad_norm   = max(0.02, 1.0*math.exp(-it/9500) + random.uniform(-0.1,0.1))
        kl          = max(0.0, 0.01*math.exp(-it/7000) + random.uniform(-0.0002,0.0002))

        losses.append(policy_loss)
        values.append(value_loss)
        entropies.append(entropy)

        # Driving metrics (CARLA-like)
        collision_rate = max(0.0, 0.15*(1-progress) + random.uniform(-0.015, 0.015))
        avg_speed      = 20 + 12.0*progress + random.uniform(-0.5,0.5)
        success_rate   = min(1.0, 0.10 + 0.80*progress + random.uniform(-0.02,0.02))

        # clean PyTorch-style iteration summary
        print(f"    RL: policy_loss={policy_loss:.4f}  value_loss={value_loss:.4f}  "
              f"entropy={entropy:.3f}  grad_norm={grad_norm:.3f}  kl={kl:.4f}  lr={lr:.2e}")
        print(f"   avg_speed={avg_speed:.2f}m/s ")

        # ASCII trend
        print(f"    loss_trend: {ascii_trend(losses, width=36)}")
        print(f"    iter_elapsed={human_seconds(time.time()-iter_start)}")

        print("-" * TERM_WIDTH)

        # checkpoint simulation
        if it % max(1, iters//10) == 0 or it <= 3:
            print(f"[Iter {it:04d}] checkpoint saved -> mappo_sim_{it:04d}.pt")

        csv_rows.append({
            "iter": it,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "grad_norm": grad_norm,
            "kl": kl,
            "collision_rate": collision_rate,
            "avg_speed_mps": avg_speed,
            "success_rate": success_rate,
            "lr": lr,
            "timestamp": now_ts()
        })

    total_elapsed = time.time() - start_time
    print("=" * TERM_WIDTH)
    print(f"TRAINING COMPLETE | total_elapsed={human_seconds(total_elapsed)}")
    print(f"final policy_loss={losses[-1]:.4f}  entropy={entropies[-1]:.3f}")

    if csv_out:
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print(f"CSV saved to {csv_out}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--agents", type=int, default=16)
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--csv", type=str, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    simulate(iters=1000000,
             device=args.device,
             agents=args.agents,
             rollout_len=args.rollout_len,
             csv_out=args.csv)
