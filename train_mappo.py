# train_mappo.py  (FULL file — replace your existing file with this)
import os
import time
import argparse
import torch
import numpy as np

# -------------------------
# Replace these imports if your project uses different module paths.
# These are the direct imports used across our conversation.
from envs.carla_env import CarlaEnv
from algo.mappo.mappo_manager import MAPPOManager
from models.mappo_policy import MAPPOPolicy
from train.trainer_mappo import MAPPOTrainer
# -------------------------

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
        log_std_init=spec.get("log_std_init", -0.5),
        device=spec.get("device", "cpu"),
    )

def make_dirs(path):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(manager, out_dir, step):
    make_dirs(out_dir)
    prefix = os.path.join(out_dir, f"mappo_ckpt_{step}")
    try:
        manager.save_all(prefix)
    except Exception as e:
        print("Warning: manager.save_all failed:", e)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_steps", type=int, default=128, help="rollout length (T)")
    p.add_argument("--num_iters", type=int, default=10000, help="training iterations")
    p.add_argument("--save_every", type=int, default=200, help="save checkpoint every N iters")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--num_veh", type=int, default=30)
    p.add_argument("--num_ped", type=int, default=0)
    p.add_argument("--mode", default="MAPPO")
    return p.parse_args()

def construct_agent_specs(n_agents=8, obs_dim=128, act_dim=2):
    return {
        "vehicle": {
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "buffer_T": 128,
            "bev_in_ch": 3,
            # you can add more defaults here if your policy ctor reads them
        }
    }

# ---------------------------
# Debug helper (single complete function)
# ---------------------------
def debug_policy_and_batch_shapes(manager, sample_batch=None, try_resolves: bool = False):
    """
    Debug helper — single function to locate input_size mismatch between batch and policy.

    Usage:
      - manager: MAPPOManager instance (must have .policy and optionally .buffer)
      - sample_batch: optional dict to use instead of sampling from manager.buffer
      - try_resolves: if True, function will also try a few common reshapes (can be slow)

    This function prints diagnostics to stdout and returns a dict `info` summarizing findings.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    import traceback

    info = {
        "batch_shapes": {},
        "per_agent": None,
        "per_flat_size": None,
        "linear_layers": [],
        "forward_attempts": [],
        "exception_tracebacks": []
    }

    print("\n=== debug_policy_and_batch_shapes START ===")

    # 1) obtain a batch
    batch = sample_batch
    if batch is None:
        buf = getattr(manager, "buffer", None)
        if buf is None:
            print("[debug] manager.buffer not found; provide sample_batch argument instead.")
        else:
            try:
                gen = buf.feed_forward_generator(mini_batch_size=max(1, getattr(manager, "debug_mini_batch_size", 64)))
                batch = next(gen)
                print("[debug] sampled one minibatch from buffer.feed_forward_generator()")
            except Exception as e:
                tb = traceback.format_exc()
                print("[debug] failed to sample from buffer:", e)
                print(tb)
                info["exception_tracebacks"].append(tb)
                batch = None

    if batch is None:
        print("[debug] No batch available. Exiting debug function. Provide a sample_batch to test.")
        print("=== debug_policy_and_batch_shapes END ===\n")
        return info

    # 2) inspect batch keys and shapes
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

    # 3) analyze agent_feats if present
    af = batch.get("agent_feats", None)
    per_flat_size = None
    per_agent = None
    if af is not None:
        try:
            if isinstance(af, torch.Tensor):
                af_np = af.detach().cpu().numpy()
            elif isinstance(af, np.ndarray):
                af_np = af
            else:
                af_np = np.asarray(af)
            if af_np.ndim == 3:
                B, N, F = af_np.shape
                per_agent = (int(B), int(N), int(F))
                per_flat_size = int(N * F)
                print(f"[debug] agent_feats interpreted as [B,N,F] -> B={B}, N={N}, F={F}, N*F={per_flat_size}")
            elif af_np.ndim == 2:
                rows, F = af_np.shape
                N_guess = getattr(getattr(manager, "buffer", None), "N", None)
                if N_guess and rows % N_guess == 0:
                    B = rows // N_guess
                    N = N_guess
                    per_agent = (int(B), int(N), int(F))
                    per_flat_size = int(N * F)
                    print(f"[debug] agent_feats interpreted as flattened rows -> inferred B={B}, N={N}, F={F}, N*F={per_flat_size}")
                else:
                    N = rows
                    per_agent = (1, int(N), int(F))
                    per_flat_size = int(N * F)
                    print(f"[debug] agent_feats interpreted as [N,F] -> N={N}, F={F}, N*F={per_flat_size}")
            else:
                print(f"[debug] agent_feats ndarray has ndim={af_np.ndim}, shape={af_np.shape}")
        except Exception as e:
            tb = traceback.format_exc()
            print("[debug] error while analyzing agent_feats:", e)
            print(tb)
            info["exception_tracebacks"].append(tb)

    info["per_agent"] = per_agent
    info["per_flat_size"] = per_flat_size

    # 4) list policy's first Linear layers
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
        else:
            print("[debug] No nn.Linear modules found in policy.")
    except Exception as e:
        tb = traceback.format_exc()
        print("[debug] failed inspecting policy modules:", e)
        print(tb)
        info["exception_tracebacks"].append(tb)

    # 5) prepare obs for forward/evaluate
    device = next(manager.policy.parameters()).device if any(True for _ in manager.policy.parameters()) else torch.device("cpu")
    obs = {}
    img = None
    if "images" in batch:
        img = batch.get("images")
    elif "image" in batch:
        img = batch.get("image")
    if img is not None:
        try:
            if isinstance(img, np.ndarray):
                arr = img
                if arr.ndim >= 4:
                    arr = arr[-1]
                if arr.ndim == 3:
                    arr = np.expand_dims(arr, 0)
                obs["image"] = torch.as_tensor(arr, dtype=torch.float32, device=device)
            elif isinstance(img, torch.Tensor):
                obs["image"] = img.to(device)
        except Exception as e:
            tb = traceback.format_exc()
            print("[debug] failed to prepare image for obs:", e)
            print(tb)
            info["exception_tracebacks"].append(tb)

    if af is not None:
        try:
            if isinstance(af, np.ndarray):
                af_arr = af
                if af_arr.ndim == 3:
                    obs["agent_feats"] = torch.as_tensor(af_arr, dtype=torch.float32, device=device)
                elif af_arr.ndim == 2:
                    N_guess = getattr(getattr(manager, "buffer", None), "N", None)
                    if N_guess and af_arr.shape[0] % N_guess == 0:
                        B_guess = af_arr.shape[0] // N_guess
                        obs["agent_feats"] = torch.as_tensor(af_arr.reshape(B_guess, N_guess, af_arr.shape[1]), dtype=torch.float32, device=device)
                    else:
                        obs["agent_feats"] = torch.as_tensor(af_arr[np.newaxis, ...], dtype=torch.float32, device=device)
                else:
                    obs["agent_feats"] = torch.as_tensor(af_arr, dtype=torch.float32, device=device)
            elif isinstance(af, torch.Tensor):
                if af.dim() == 3:
                    obs["agent_feats"] = af.to(device)
                elif af.dim() == 2:
                    N_guess = getattr(getattr(manager, "buffer", None), "N", None)
                    if N_guess and af.shape[0] % N_guess == 0:
                        B_guess = af.shape[0] // N_guess
                        obs["agent_feats"] = af.view(B_guess, N_guess, af.shape[1]).to(device)
                    else:
                        obs["agent_feats"] = af.unsqueeze(0).to(device)
                else:
                    obs["agent_feats"] = af.to(device)
        except Exception as e:
            tb = traceback.format_exc()
            print("[debug] failed to prepare agent_feats for obs:", e)
            print(tb)
            info["exception_tracebacks"].append(tb)

    # 6) attempt forward/evaluate with common call styles
    def try_call(desc, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs)
            info["forward_attempts"].append((desc, True, "success"))
            print(f"[attempt] {desc} -> success (returned type: {type(out)})")
            return out
        except Exception as e:
            tb = traceback.format_exc()
            info["forward_attempts"].append((desc, False, str(e)))
            info["exception_tracebacks"].append(tb)
            print(f"[attempt] {desc} -> FAILED: {e}")
            print(tb)
            return None

    print("[debug] starting forward/evaluate attempts...")
    # a) evaluate_actions(agent_feats, actions)
    if hasattr(manager.policy, "evaluate_actions"):
        candidate_af = obs.get("agent_feats", None)
        out = try_call("evaluate_actions(agent_feats_pos, actions_pos)",
                       manager.policy.evaluate_actions,
                       candidate_af, batch.get("actions", None))
        if out is not None:
            print("=== debug_policy_and_batch_shapes END ===\n")
            return info

        out = try_call("evaluate_actions(obs_pos, actions_pos)",
                       manager.policy.evaluate_actions,
                       obs, batch.get("actions", None))
        if out is not None:
            print("=== debug_policy_and_batch_shapes END ===\n")
            return info

        out = try_call("evaluate_actions(agent_feats=..., actions=..., image=...)",
                       manager.policy.evaluate_actions,
                       agent_feats=candidate_af, actions=batch.get("actions", None), image=obs.get("image", None))
        if out is not None:
            print("=== debug_policy_and_batch_shapes END ===\n")
            return info

    # b) try forward(obs=...)
    out = try_call("policy.forward(obs=..., mode='step')",
                   manager.policy.forward,
                   obs=obs, mode="step", deterministic=True)
    if out is not None:
        print("=== debug_policy_and_batch_shapes END ===\n")
        return info

    # 7) optional reshape trials
    if try_resolves:
        print("[debug] attempting common reshape heuristics on agent_feats...")
        trials = []
        try:
            if af is not None:
                if isinstance(af, np.ndarray) and af.ndim == 3:
                    B, N, F = af.shape
                    trials.append(("reshape_BxN_to_rows", torch.as_tensor(af.reshape(B*N, F), device=device)))
                elif isinstance(af, torch.Tensor) and af.dim() == 3:
                    B, N, F = af.shape
                    trials.append(("reshape_BxN_to_rows_tensor", af.view(B*N, F).to(device)))
                if isinstance(af, (np.ndarray, torch.Tensor)):
                    trials.append(("take_first_row", torch.as_tensor(af[0:1], device=device) if isinstance(af, np.ndarray) else af[0:1].to(device)))
        except Exception:
            pass

        for desc, candidate in trials:
            try_obs = {"agent_feats": candidate}
            if "image" in obs:
                try_obs["image"] = obs["image"]
            out = try_call(f"trial_forward_{desc}", manager.policy.forward, obs=try_obs, mode="step", deterministic=True)
            if out is not None:
                print(f"[debug] trial {desc} succeeded — indicates reshape that fits policy.")
                print("=== debug_policy_and_batch_shapes END ===\n")
                return info

    print("[debug] All attempts done. See info dict for details.")
    print("=== debug_policy_and_batch_shapes END ===\n")
    return info

# ---------------------------
# Main training entry
# ---------------------------
def main():
    args = parse_args()
    device = args.device

    # ---- Environment ----
    env = CarlaEnv(num_veh=args.num_veh, num_ped=args.num_ped, mode=args.mode)
    # If your CarlaEnv has a different constructor, adjust above.

    # ---- Manager & Policy ----
    agent_specs = construct_agent_specs(n_agents=16, obs_dim=128, act_dim=2)
    manager = MAPPOManager(agent_specs=agent_specs, policy_ctor=policy_ctor_from_spec, device=device)
    # ensure manager.policy points to a representative policy for trainer utils
    manager.policy = next(iter(manager.agents.values()))

    # ---- QUICK DEBUG (one-time) ----
    # run input-shape diagnostics once before starting trainer; set try_resolves=False to avoid heavy reshape attempts
    debug_info = debug_policy_and_batch_shapes(manager, sample_batch=None, try_resolves=False)
    # optionally: inspect debug_info programmatically:
    # print("DEBUG INFO SUMMARY:", debug_info)

    # ---- Trainer ----
    trainer = MAPPOTrainer(envs=env, manager=manager, num_steps=args.num_steps, device=device)

    print("Starting training loop")
    start_time = time.time()

    for it in range(1, args.num_iters + 1):
        t0 = time.time()
        try:
            trainer.collect_rollout()
            trainer.finish_and_update()
        except Exception as e:
            print(f"[Iter {it}] Error during rollout/update: {e}")
            import traceback as tb
            tb.print_exc()
            # decide whether to continue or break; we continue to attempt recovery
            time.sleep(0.5)
            continue

        if it % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"[Iter {it}] elapsed {elapsed:.1f}s")

        if it % args.save_every == 0:
            save_checkpoint(manager, args.out_dir, it)
            print(f"[Iter {it}] checkpoint saved to {args.out_dir}")

    # final save
    save_checkpoint(manager, args.out_dir, f"final")
    print("Training finished.")

if __name__ == "__main__":
    main()
