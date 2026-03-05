#!/usr/bin/env python3

import os
import time
import argparse
import numpy as np
import torch

from envs.carla_env import CarlaEnv
from algo.mappo.mappo_manager import MAPPOManager
from models.mappo_policy import MAPPOPolicy


# ============================================================
# Utils
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_metrics_csv(path, rows):
    if len(rows) == 0:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join([str(r[k]) for k in keys]) + "\n")


# ============================================================
# Episode Runner
# ============================================================

def run_one_episode(env, manager, mode="rl", deterministic=True, max_steps=256):
    """
    mode: 'rl' or 'tm'
    """

    obs = env.reset()

    if hasattr(manager, "mappo_reset"):
        manager.mappo_reset(obs)

    total_return = 0.0
    collision_count = 0
    speed_list = []

    done = False
    step = 0

    while not done and step < max_steps:

        # ---------------- RL mode ----------------
        if mode == "rl":
            obs_dict = env.get_single_obs_for_manager(batch_size=1)

            out = manager.select_actions(
                obs_dict=obs_dict,
                hidden=None,
                deterministic=deterministic
            )

            actions = out[0] if isinstance(out, (tuple, list)) else out["actions"]

        # ---------------- TM mode ----------------
        else:
            actions = {}

            # zero actions for rl slots
            for s in getattr(env, "rl_slots", []):
                actions[int(s)] = np.zeros(4, dtype=np.float32)

            # enable autopilot
            ego = getattr(env, "ego_vehicle", None)
            if ego is not None:
                try:
                    ego.set_autopilot(True)
                except:
                    pass

        # ---------------- step env ----------------
        next_obs, rewards, dones, info = env.step(actions)

        # return
        if isinstance(rewards, dict):
            total_return += float(np.mean(list(rewards.values())))
        else:
            total_return += float(rewards)

        # collision
        if getattr(env, "collision_happened", False):
            collision_count += 1

        # speed
        ego = getattr(env, "ego_vehicle", None)
        if ego is not None:
            v = ego.get_velocity()
            speed = (v.x**2 + v.y**2 + v.z**2)**0.5
            speed_list.append(speed)

        # done?
        if isinstance(dones, dict):
            done = any(dones.values())
        else:
            done = bool(dones)

        step += 1

    avg_speed = float(np.mean(speed_list)) if len(speed_list) > 0 else 0.0

    return {
        "episode_return": total_return,
        "episode_length": step,
        "collision_count": collision_count,
        "avg_speed": avg_speed
    }


# ============================================================
# Evaluation Loop
# ============================================================

def evaluate(ckpt, seeds, episodes, modes, outdir, max_steps):

    run_id = os.path.basename(ckpt) if ckpt else f"run_{int(time.time())}"
    run_dir = os.path.join(outdir, run_id)
    ensure_dir(run_dir)

    summary_rows = []

    for seed in seeds:

        seed_dir = os.path.join(run_dir, f"seed_{seed}")
        ensure_dir(seed_dir)

        for mode in modes:

            mode_dir = os.path.join(seed_dir, f"mode_{mode}")
            ensure_dir(mode_dir)

            rows = []

            for ep in range(episodes):

                print(f"[EVAL] seed={seed} mode={mode} ep={ep}")

                # new env per episode
                env = CarlaEnv(
                    num_veh=10,
                    num_ped=5,
                    mode="EVAL",
                    spawn_background=True
                )

                manager = MAPPOManager(
                    env=env,
                    agent_specs=getattr(env, "agent_specs", None),
                    policy_cls=MAPPOPolicy,
                    device=torch.device("cpu")
                )

                # load RL checkpoint
                if ckpt and mode == "rl":
                    if hasattr(manager, "load_all"):
                        manager.load_all(ckpt)

                metrics = run_one_episode(
                    env,
                    manager,
                    mode=mode,
                    deterministic=True,
                    max_steps=max_steps
                )

                metrics["seed"] = seed
                metrics["mode"] = mode
                metrics["episode_id"] = ep

                rows.append(metrics)
                summary_rows.append(metrics)

                try:
                    env.destroy_all_actors()
                except:
                    pass

            write_metrics_csv(
                os.path.join(mode_dir, "metrics.csv"),
                rows
            )

    # write global summary
    write_metrics_csv(
        os.path.join(run_dir, "summary.csv"),
        summary_rows
    )

    print(f"\n[EVAL DONE] Results saved to: {run_dir}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--modes", type=str, default="rl,tm")
    parser.add_argument("--outdir", type=str, default="./eval_results")
    parser.add_argument("--max_steps", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    evaluate(
        ckpt=args.ckpt,
        seeds=seeds,
        episodes=args.episodes,
        modes=modes,
        outdir=args.outdir,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()