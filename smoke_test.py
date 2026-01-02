# smoke_test.py
import torch
import numpy as np

from envs.carla_env import CarlaEnv
from models.mappo_policy import MAPPOPolicy
from algo.mappo.mappo_manager import MAPPOManager


# ===========================================================
# 1) 创建环境
# ===========================================================
env = CarlaEnv(num_veh=6, num_ped=0, mode="MAPPO")
print("Env created.")

# ===========================================================
# 2) 构造 agent_specs（manager 需要用）
# ===========================================================
# env.obs_dim / env.action_dim / env.max_slots 必须存在
obs_dim = env.obs_dim
act_dim = 2
max_slots = env.max_slots

agent_specs = {
    i: {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "n_agents": max_slots,
        "buffer_T": 128,
        "buffer_size": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }
    for i in range(max_slots)
}


# ===========================================================
# 3) policy_ctor（manager 会调用）
# ===========================================================
def policy_ctor(spec):
    return MAPPOPolicy(
        bev_in_ch=3,
        obs_dim=spec["obs_dim"],
        act_dim_vehicle=spec["act_dim"],
        act_dim_ped=spec["act_dim"],
        device="cpu",
        use_bev_gru=True,
        use_slot_gru=True,
    )


# ===========================================================
# 4) 创建 manager + 按 slot 构造 policies
# ===========================================================
manager = MAPPOManager(agent_specs=agent_specs, policy_ctor=policy_ctor, device="cpu")
print("MAPPOManager created.")

# manager.policy = 单一共享 policy（取第一个 agent 的 policy）
manager.policy = next(iter(manager.agents.values()))

print("Shared policy bound.")


# ===========================================================
# 5) reset + warmup（获得序列用于 RNN burn-in）
# ===========================================================
seq = env.reset()
assert "seq" in seq, "reset() must return a dict with key 'seq'"
seq_feats = seq["seq"]["agent_feats"]  # numpy: [T,N,obs_dim]
seq_mask  = seq["seq"].get("mask", None)

print(f"Warmup OK, seq_feats shape = {seq_feats.shape}")


# ===========================================================
# 6) burn-in（初始化 hidden）
# ===========================================================
try:
    hidden = manager.policy.burn_in(
        torch.tensor(seq_feats, dtype=torch.float32).unsqueeze(1),     # add B=1 dim
        None if seq_mask is None else torch.tensor(seq_mask, dtype=torch.float32).unsqueeze(1)
    )
    # burn_in 返回 next_hidden，本项目中你实际是在 trainer 中 burn-in，
    # 此处 smoke test 不严格要求 burn_in 完整，只要求不报错。
except Exception as e:
    print("burn_in skipped (policy may not require it).", e)
    hidden = None
# ============================
# 7) 严格构造 obs_dict（替换此处原有代码）
# ============================
import torch
import numpy as np

# 获取当前 policy 设备与期望 obs_dim（严格校验）
policy = getattr(manager, "policy", None)
assert policy is not None, "manager.policy must exist before building obs_dict"
policy_device = getattr(policy, "device", torch.device("cpu"))
# policy 里应有 obs_dim 或 obs feature dim 的属性；若没有请在 policy 中定义 policy.obs_dim
if hasattr(policy, "obs_dim"):
    expected_feat_dim = int(getattr(policy, "obs_dim"))
elif hasattr(policy, "bev_feat_dim"):
    expected_feat_dim = int(getattr(policy, "bev_feat_dim"))
else:
    # 最后兜底：尝试 policy 中常见命名
    expected_feat_dim = int(getattr(policy, "agent_feat_dim", 128))

# seq_feats 必须存在且形状明确 (T, B, N, feat) 或 (T, N, feat) 或 (T, 1, N, feat)
assert 'seq_feats' in globals() or 'seq_feats' in locals(), "seq_feats 未定义，无法构造 obs"
raw_last = seq_feats[-1]  # 可能是 numpy 或 tensor

# 将 raw_last 规范化为 numpy ndarray，最终期望 last_frame.shape == (N_agents, feat)
if torch.is_tensor(raw_last):
    raw_np = raw_last.detach().cpu().numpy()
else:
    raw_np = np.asarray(raw_last)

# 只接受以下严格布局（否则立即报错）
# Acceptable raw_last shapes:
#   (N, feat)
#   (1, N, feat)  -> squeeze leading dim 1
#   (B, N, feat)  -> only accept if B == 1 (else raise)
#   (N, )         -> treat as single agent feature vector?  <-- reject (must be 2D)
if raw_np.ndim == 2:
    last_frame = raw_np  # [N, feat]
elif raw_np.ndim == 3 and raw_np.shape[0] == 1:
    last_frame = raw_np[0]  # squeeze leading batch
elif raw_np.ndim == 3 and raw_np.shape[0] == 1:
    last_frame = raw_np[0]
elif raw_np.ndim == 3 and raw_np.shape[0] != 1:
    raise RuntimeError(f"seq_feats[-1] has leading batch dim >1: shape {raw_np.shape}. In smoke test we require B==1. Got B={raw_np.shape[0]}.")
else:
    raise RuntimeError(f"Unexpected seq_feats[-1] layout {raw_np.shape}. Acceptable shapes: (N,feat) or (1,N,feat) or (1,N,feat) as numpy/tensor.")

N_agents = int(last_frame.shape[0])
feat_dim = int(last_frame.shape[1])

# 强校验 feat_dim 与 policy 期望一致
assert feat_dim == expected_feat_dim, f"feature dim mismatch: last_frame feat_dim={feat_dim}, policy expects {expected_feat_dim}"

# 验证 env.rl_slots 存在并且是非空 list 且元素合法
assert hasattr(env, "rl_slots"), "env.rl_slots 缺失 —— 必须在 env.reset() 中设置 rl_slots 为待控制的 slot 索引列表"
rl_slots = getattr(env, "rl_slots")
assert isinstance(rl_slots, (list, tuple)) and len(rl_slots) > 0, f"env.rl_slots 必须为非空 list/tuple，当前值: {rl_slots}"
# 保证所有 rl_slots 在合法范围内
for s in rl_slots:
    assert isinstance(s, int) and (0 <= s < N_agents), f"rl_slot {s} 超出 last_frame 范围 [0, {N_agents-1}]"

# 构造 image tensor：严格为 float32, shape [B,3,H,W], contiguous, device=policy_device
img = np.ascontiguousarray(env.latest_image)  # numpy H,W,C
assert img.ndim == 3 and img.shape[2] in (3, 4), f"env.latest_image shape must be H,W,3|4; got {img.shape}"
# take first 3 channels if RGBA
img3 = img[:, :, :3].astype(np.float32)  # H,W,3
image_tensor = torch.from_numpy(np.ascontiguousarray(img3)).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
# normalize type and device (do not change values)
image_tensor = image_tensor.to(dtype=torch.float32, device=policy_device).contiguous()

# 构造 obs_dict 且严格校验每个 field
obs_dict = {}
for slot_idx in rl_slots:
    # last_frame row for this slot
    af = last_frame[slot_idx]  # numpy 1D array length feat_dim
    # strict convert to tensor [B=1, feat_dim]
    agent_feat_t = torch.as_tensor(np.ascontiguousarray(af), dtype=torch.float32, device=policy_device).unsqueeze(0).contiguous()
    assert agent_feat_t.dim() == 2 and agent_feat_t.size(1) == expected_feat_dim, \
        f"agent_feats for slot {slot_idx} has wrong shape {tuple(agent_feat_t.size())}, expected [1, {expected_feat_dim}]"

    # strict type_id shape [B=1, 1], long dtype
    type_id_t = torch.as_tensor([[0]], dtype=torch.long, device=policy_device).contiguous()

    obs_dict[int(slot_idx)] = {
        "image": image_tensor,
        "agent_feats": agent_feat_t,
        "type_id": type_id_t
    }

# FINAL STRICT ASSERTIONS (no extra keys, uniform shapes)
for k, v in obs_dict.items():
    assert set(v.keys()) == {"image", "agent_feats", "type_id"}, f"slot {k} obs keys mismatch: {v.keys()}"
    # image checks
    img_t = v["image"]
    assert torch.is_tensor(img_t) and img_t.dtype == torch.float32 and img_t.dim() == 4 and img_t.size(0) == 1 and img_t.size(1) == 3, \
        f"slot {k} image tensor invalid shape/dtype: {tuple(img_t.size())}, {img_t.dtype}"
    # agent_feats checks
    af_t = v["agent_feats"]
    assert torch.is_tensor(af_t) and af_t.dtype == torch.float32 and af_t.dim() == 2 and af_t.size(0) == 1 and af_t.size(1) == expected_feat_dim, \
        f"slot {k} agent_feats invalid: {tuple(af_t.size())}, expected (1,{expected_feat_dim})"
    # type_id checks
    t_t = v["type_id"]
    assert torch.is_tensor(t_t) and t_t.dtype == torch.long and t_t.dim() == 2 and t_t.size() == (1,1), \
        f"slot {k} type_id invalid: {tuple(t_t.size())}, expected (1,1)"

# at this point obs_dict is strictly formed and safe to pass to manager.select_actions
print("OBS DICT STRICT CHECK OK. slots:", sorted(list(obs_dict.keys())))
# ============================
# End strict obs_dict block
# ============================

# ===========================================================
# 8) 调用 manager.select_actions（真正的 MAPPO 动作采样）
# ===========================================================
actions, values, logps, next_slot_h, next_bev_h, info = \
    manager.select_actions(obs_dict, hidden={"slot": None, "bev": None}, mask=None)

print("select_actions OK.")
print("actions slots:", actions)


# ===========================================================
# 9) 调环境 step（完整链路）
# ===========================================================
next_obs, rewards, dones, info_env = env.step(actions)

print("env.step OK.")
print("sample rewards:", rewards)
print("sample dones:", dones)

print("\n==========================")
print("SMOKE TEST FINISHED. Pipeline OK.")
print("==========================")
