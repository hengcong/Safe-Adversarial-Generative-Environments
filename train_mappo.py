# train_mappo.py  — 完整版（替换你的旧文件）
import os
import time
import argparse
import glob
import torch
import numpy as np
import traceback
import re

# Replace these imports to match your project layout if needed
from envs.carla_env import CarlaEnv
from algo.mappo.mappo_manager import MAPPOManager
from models.mappo_policy import MAPPOPolicy
from train.trainer_mappo import MAPPOTrainer

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7)

def policy_ctor_from_spec(spec):
    bev_in_ch = spec.get("bev_in_ch", 3)
    obs_dim = int(spec["obs_dim"])
    act_dim = int(spec.get("act_dim", spec.get("act_dim_vehicle", 2)))
    act_dim_ped = int(spec.get("act_dim_ped", act_dim))
    return MAPPOPolicy(
        bev_in_ch=bev_in_ch,
        obs_dim=obs_dim,
        act_dim_vehicle=act_dim,
        act_dim_ped=act_dim_ped,
        type_vocab_size=spec.get("type_vocab_size", 8),
        type_emb_dim=spec.get("type_emb_dim", 8),
        hidden_dim=spec.get("hidden_dim", 256),
        recurrent_hidden_dim=spec.get("recurrent_hidden_dim", 256),
        use_bev_gru=spec.get("use_bev_gru", True),
        use_slot_gru=spec.get("use_slot_gru", True),
        global_ctx_dim=spec.get("global_ctx_dim", 256),
        action_scale=spec.get("action_scale", 1.0),
        log_std_init=spec.get("log_std_init", 0.0),
        device=spec.get("device", "cuda"),
    )


def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(manager: MAPPOManager, out_dir: str, step, trainer=None):
    make_dirs(out_dir)
    prefix = os.path.join(out_dir, f"mappo_ckpt_{step}")

    state = {
        "step": step,
        "model": {},
        "optimizer": None,
        "trainer_global_step": None,
        "trainer_epoch": None,
    }

    # save model state (all slots)
    for slot, agent in manager.agents.items():
        state["model"][slot] = agent.state_dict()

    # save optimizer
    if hasattr(manager, "optim") and manager.optim is not None:
        state["optimizer"] = manager.optim.state_dict()

    # save trainer state if provided
    if trainer is not None:
        state["trainer_global_step"] = getattr(trainer, "global_step", None)
        state["trainer_epoch"] = getattr(trainer, "epoch", None)

    torch.save(state, prefix + ".pt")
    print(f"[SAVE] Full checkpoint saved: {prefix}.pt")

def load_checkpoint(manager: MAPPOManager, prefix: str, trainer=None, map_location=None):

    if map_location is None:
        map_location = getattr(manager, "device", "cpu")

    path = prefix if prefix.endswith(".pt") else prefix + ".pt"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)
    # ---- FORCE BUILD BEV FC BEFORE LOADING ----
    if hasattr(manager.policy.bev_enc, "_fc") and manager.policy.bev_enc._fc is None:
        dummy = torch.zeros(1, manager.policy.bev_in_ch, 84, 84).to(map_location)
        with torch.no_grad():
            manager.policy.bev_enc(dummy)

    # restore models
    for slot, agent in manager.agents.items():
        if slot in ckpt["model"]:
            agent.load_state_dict(ckpt["model"][slot])

    # restore optimizer
    if ckpt.get("optimizer") and hasattr(manager, "optim"):
        manager.optim.load_state_dict(ckpt["optimizer"])

    # restore trainer
    if trainer is not None:
        if ckpt.get("trainer_global_step") is not None:
            trainer.global_step = ckpt["trainer_global_step"]
        if ckpt.get("trainer_epoch") is not None:
            trainer.epoch = ckpt["trainer_epoch"]

    print(f"[LOAD] Fully restored checkpoint: {path}")

    return ckpt.get("step", 0)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_steps", type=int, default=256, help="rollout length (T)")
    p.add_argument("--num_iters", type=int, default=200000, help="training iterations")
    p.add_argument("--save_every", type=int, default=200, help="save checkpoint every N iters")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--num_veh", type=int, default=30)
    p.add_argument("--num_ped", type=int, default=0)
    p.add_argument("--mode", default="MAPPO")
    p.add_argument("--shield_mode", type=str, default="none",
                        choices=["none", "soft", "single_cbf", "ecbf_collision"],
                        help="Select the safety shield mode for training.")
    p.add_argument("--resume", type=str, default=None,
                   help="Path prefix to checkpoint to resume from, or 'auto' to pick latest numeric checkpoint")
    p.add_argument("--resume_step", type=int, default=0,
                   help="Training step to resume from (for logging continuity). Only used when resume is a path.")
    p.add_argument("--exp_dir", type=str, default=None)
    return p.parse_args()


def construct_agent_specs(n_agents=8, obs_dim=128, act_dim=2):
    return {
        "vehicle": {
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "buffer_T": 128,
            "bev_in_ch": 3,
        }
    }

def find_latest_checkpoint(out_dir):
    ckpts = glob.glob(os.path.join(out_dir, "mappo_ckpt_*"))
    if not ckpts:
        return None
    max_step = -1
    max_prefix = None
    for c in ckpts:
        base = os.path.basename(c)
        # match mappo_ckpt_{step} or mappo_ckpt_{step}_... (with extension or not)
        m = re.search(r"mappo_ckpt_(\d+)", base)
        if m:
            step = int(m.group(1))
            if step > max_step:
                max_step = step
                max_prefix = os.path.join(out_dir, f"mappo_ckpt_{step}")
    if max_prefix is None:
        return None
    return (max_step, max_prefix)

# Debug helper copied/adapted from earlier (keeps non-invasive signature)
def debug_policy_and_batch_shapes(manager, sample_batch=None, try_resolves: bool = False):
    import torch
    import torch.nn as nn
    import numpy as np
    import traceback

    info = {"batch_shapes": {}, "per_agent": None, "per_flat_size": None, "linear_layers": [], "forward_attempts": [],
            "exception_tracebacks": []}

    print("\n=== debug_policy_and_batch_shapes START ===")

    batch = sample_batch
    if batch is None:
        buf = getattr(manager, "buffer", None)
        if buf is None:
            print("[debug] manager.buffer not found; skipping sample debug.")
            print("=== debug_policy_and_batch_shapes END ===\n")
            return info
        try:
            gen = buf.feed_forward_generator(mini_batch_size=max(1, getattr(manager, "debug_mini_batch_size", 64)))
            batch = next(gen)
            print("[debug] sampled one minibatch from buffer.feed_forward_generator()")
        except Exception as e:
            tb = traceback.format_exc()
            print("[debug] failed to sample from buffer:", e)
            print(tb)
            info["exception_tracebacks"].append(tb)
            print("=== debug_policy_and_batch_shapes END ===\n")
            return info

    def _shape_repr(x):
        try:
            if isinstance(x, torch.Tensor):
                return ("torch", tuple(x.shape))
            elif isinstance(x, np.ndarray):
                return ("nd", tuple(x.shape))
            elif isinstance(x, list):
                return ("list", len(x))
            elif x is None:
                return ("None", None)
            else:
                return (type(x).__name__, None)
        except Exception:
            return ("unknown", None)

    keys = list(batch.keys())
    print("[debug] batch keys:", keys)
    for k in keys:
        s = _shape_repr(batch.get(k))
        print(f" - {k}: {s}")
        info["batch_shapes"][k] = s

    # list first Linear layers
    try:
        linear_layers = []
        for name, module in manager.policy.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, int(module.in_features), int(module.out_features)))
        info["linear_layers"] = linear_layers
        if linear_layers:
            print("[debug] Found linear layers (name, in, out) (first 12):")
            for i, (n, inf, outf) in enumerate(linear_layers[:12]):
                print(f"  [{i}] {n}  in={inf} out={outf}")
    except Exception:
        pass

    print("=== debug_policy_and_batch_shapes END ===\n")
    return info


# ---------------------------
# Main training entry
# ---------------------------
def main():
    args = parse_args()
    device = args.device

    # auto create run directory if not provided
    if getattr(args, "exp_dir", None) is not None:
        exp_dir = args.exp_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(
            "./carla_experiments",
            f"run_{timestamp}"
        )
        exp_dir = base_dir

    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    make_dirs(exp_dir)
    make_dirs(ckpt_dir)

    env = CarlaEnv(
        num_veh=args.num_veh,
        num_ped=args.num_ped,
        mode=args.mode,
        experiment_path=exp_dir
    )

    agent_specs = construct_agent_specs(
        n_agents=16,
        obs_dim=int(env.obs_dim),
        act_dim=2
    )

    manager = MAPPOManager(
        agent_specs=agent_specs,
        policy_ctor=policy_ctor_from_spec,
        device=device
    )

    try:
        manager.policy = next(iter(manager.agents.values()))
    except StopIteration:
        manager.policy = None

    manager.policy.to("cuda")

    trainer = MAPPOTrainer(
        envs=env,
        manager=manager,
        num_steps=args.num_steps,
        device=device
    )

    start_iter = 0

    if args.resume == "auto":
        res = find_latest_checkpoint(ckpt_dir)
        if res is not None:
            latest_step, latest_prefix = res
            start_iter = load_checkpoint(
                manager,
                latest_prefix,
                trainer=trainer,
                map_location=device
            )
            start_iter += 1
            trainer.global_step = start_iter
            trainer.epoch = start_iter

            print(">>>> RESUME DEBUG <<<<")
            print("start_iter after load:", start_iter)
            print("trainer.global_step:", trainer.global_step)
            print(">>>>>>>>>>>>>>>>>>>>>>")
        else:
            print("[RESUME] No checkpoint found, starting fresh.")

    elif args.resume is not None:
        start_iter = load_checkpoint(
            manager,
            args.resume,
            trainer=trainer,
            map_location=device
        )
        start_iter = int(start_iter or 0)


    start_time = time.time()

    for it in range(start_iter, args.num_iters):
        try:
            trainer.collect_rollout()
            torch.cuda.empty_cache()
        except Exception:
            time.sleep(0.5)
            continue

        if it % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"[Iter {it}] elapsed {elapsed:.1f}s")

        if it % args.save_every == 0:
            save_checkpoint(manager, ckpt_dir, it, trainer=trainer)

    save_checkpoint(manager, ckpt_dir, args.num_iters, trainer=trainer)

if __name__ == "__main__":
    main()