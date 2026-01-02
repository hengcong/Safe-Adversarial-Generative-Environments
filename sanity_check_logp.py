# sanity_check_logp.py
import torch
import numpy as np
import traceback

from algo.mappo.mappo_manager import MAPPOManager
from models.mappo_policy import MAPPOPolicy

EPS = 1e-8

def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x + 1e-12) + 1e-12)

def run_sanity(manager):
    policy = manager.policy
    device = manager.device
    slots = manager.agent_slots
    N = len(slots)

    feat_dim = getattr(manager, "obs_dim", 128)
    act_dim = getattr(policy, "act_dim_vehicle", getattr(policy, "act_dim", 2))

    agent_feats = torch.randn(1, N, feat_dim, device=device)

    # build obs_dict expected by manager.select_actions
    obs_dict = {
        slot: {
            "agent_feats": agent_feats[:, i, :],        # [B=1, feat]
            "type_id": torch.zeros(1, dtype=torch.long, device=device),
            # no 'image' here — manager.select_actions only requires agent_feats in this test
        }
        for i, slot in enumerate(slots)
    }

    # 1) select_actions -> manager logps & actions
    acts, vals, logps, _, _, _ = manager.select_actions(obs_dict)
    manager_logps = torch.tensor([logps[s] for s in slots], device=device).unsqueeze(0)  # [1, N]
    actions_tensor = torch.tensor([acts[s] for s in slots], device=device).unsqueeze(0)   # [1, N, A]

    # 2) policy.evaluate_actions (assumes signature evaluate_actions(agent_feats, actions))
    # Some policy implementations return tuple (log_probs, values, entropy)
    try:
        eval_out = policy.evaluate_actions(agent_feats, actions_tensor)
        if isinstance(eval_out, tuple) or isinstance(eval_out, list):
            eval_logp = eval_out[0]
            eval_values = eval_out[1] if len(eval_out) > 1 else None
            eval_entropy = eval_out[2] if len(eval_out) > 2 else None
        elif isinstance(eval_out, dict):
            eval_logp = eval_out.get("log_probs") or eval_out.get("logp")
            eval_values = eval_out.get("values")
            eval_entropy = eval_out.get("entropy")
        else:
            raise RuntimeError("evaluate_actions returned unexpected type")
    except Exception as e:
        print("policy.evaluate_actions failed:", e)
        traceback.print_exc()
        return

    # 3) compute corrected logp from pre-tanh Normal (if policy supports actor_head & compute_slot_features)
    try:
        slot_emb = policy.compute_slot_features(agent_feats)   # expected [B,N,hidden]
        mu, log_std = policy.actor_head(slot_emb)              # actor_head -> (mu, log_std)
        std = log_std.exp()
        # clamp actions for numerics
        a_clamped = actions_tensor.clamp(-0.999999, 0.999999)
        pre_t = atanh(a_clamped)
        dist_pre = torch.distributions.Normal(mu, std)
        logp_pre = dist_pre.log_prob(pre_t).sum(-1)   # sum over action dims -> [B,N]
        jac = torch.log(1 - a_clamped**2 + EPS).sum(-1)  # sum over action dims
        corrected_logp = logp_pre - jac
    except Exception as e:
        print("Could not compute corrected_logp (policy missing helpers):", e)
        corrected_logp = None

    # Print results
    print("=== MANAGER LOGP ===")
    print(manager_logps.detach().cpu().numpy())

    print("\n=== EVAL LOGP (uncorrected) ===")
    print(eval_logp.detach().cpu().numpy())

    if corrected_logp is not None:
        print("\n=== EVAL LOGP (corrected pre-tanh + jacobian) ===")
        print(corrected_logp.detach().cpu().numpy())
    else:
        print("\n=== EVAL CORRECTED LOGP SKIPPED ===")

    # Differences
    try:
        diff1 = (eval_logp - manager_logps).abs().mean().item()
        print("\nMean abs diff (eval - manager):", diff1)
        if corrected_logp is not None:
            diff2 = (corrected_logp - manager_logps).abs().mean().item()
            print("Mean abs diff (corrected - manager):", diff2)
            if diff2 < diff1:
                print("\n⚠ Policy uses tanh-squash; evaluate_actions needs pre-tanh + jacobian correction.")
            else:
                print("\n✓ evaluate_actions consistent with manager.select_actions (no tanh correction needed).")
    except Exception:
        pass

def policy_ctor_from_spec(spec):
    # construct MAPPOPolicy with explicit args expected by its __init__
    # minimal required: bev_in_ch, obs_dim, act_dim_vehicle, act_dim_ped, device
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

if __name__ == "__main__":
    # build a minimal agent_specs like CarlaEnv would provide
    agent_specs = {
        "vehicle": {
            "n_agents": 4,
            "obs_dim": 128,
            "act_dim": 2,
            "buffer_T": 128,
            # optional BEV ch if you want to override
            "bev_in_ch": 3,
        }
    }

    manager = MAPPOManager(agent_specs, policy_ctor_from_spec, device="cpu")
    # manager.policy should be set to first agent's policy by manager init
    manager.policy = next(iter(manager.agents.values()))
    run_sanity(manager)
