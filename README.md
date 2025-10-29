# MAPPO Network Architecture for Scenario Generation

## Symbol Definition
- B = Batch
- N = Slot (Agent)
- C, H, W = BEV Channel, Height, Width
- T = Time steps

---

## BEV Encoder

Input: BEV [T, B, C, H, W]

Conv1: 7×7, 32ch, stride=2, ReLU, GroupNorm  
Conv2: 5×5, 64ch, stride=2, ReLU, GroupNorm  
Conv3: 3×3, 128ch, stride=2, ReLU, GroupNorm  
Conv4: 3×3, 256ch, stride=2, ReLU, GroupNorm  
Conv5: 3×3, 256ch, stride=2,ReLU, GroupNorm
Flatten → FC_shared(512) → bev_emb [T, B, 512]  
LayerNorm → GRU(input=512, hidden=256, use_layernorm=True)  
GRU input shape [T, B, 512] → seq_feat [T, B, 256]  
Hidden reset at episode boundary: h[:, done_idxs] = 0 (use done mask)  
bev_t = seq_feat[t] → [B, 256]

---

### Data Flow
1. **Input** → BEV image + scalar speed  
2. **BEVFeatureExtractor** → CNN (Conv1–Conv5) → Flatten → FC(512) → optional GRU(512→256)  
3. **SB3 Policy** → Actor & Critic heads use the extracted embedding  
4. **Algorithm Layer (PPO/TD3)** → manages rollouts, loss computation, and optimization  
5. **Output** → optimized policy capable of controlling the CARLA environment

## Per-Agent Branch

Inputs:
- feats_agents: [T, B, N, F_agent]
- types: [T, B, N] or [B, N]
- mask: [T, B, N]

bev_t expand → bev_t_expanded [T, B, N, 256]  
type_emb: embed(types) → [T, B, N, E] or [B, N, E]  
per_agent_feat = concat(feats_agents, bev_t_expanded, type_emb) → [T, B, N, F_agent + 256 + E]

Optional per-slot GRU:
per_agent_feat → GRU_per_slot(hidden=256) → slot_seq_feat [T, B, N, 256]  
hidden reset using mask per slot

---

## Actor Heads

Vehicle head (shared for all vehicles):  
MLP → mu_vehicle [T, B, N_v, act_dim]  
log_std_vehicle: learnable param [act_dim] (per-type, per-dim)

Pedestrian head (shared for all peds):  
MLP → mu_ped [T, B, N_p, act_dim]  
log_std_ped: learnable param [act_dim]

Action sampling:  
u ~ Normal(mu, exp(log_std))  
action = tanh(u) * action_scale  
log_prob = Normal.log_prob(u) - Σ log(1 - tanh(u)² + eps)  
mask-out pad agents: log_prob *= mask

---

## Critic (Per-Agent DeepSets)

Input: concat(agent_feats, global_ctx_expanded, bev_t_expanded) → [T, B, N, ...]  
MLP → V_per_agent_raw [T, B, N]  
V_per_agent = V_per_agent_raw * mask  
Optionally: value clipping when computing value loss

---

## Rollout and PPO Loss

Store trajectories with shapes [T, B, N, ...], masks [T, B, N], dones [T, B, N]

δ_t = r_t + γ * V_{t+1} * mask_{t+1} - V_t  
A_t = δ_t + γ * λ * A_{t+1} * mask_{t+1}

Normalize advantages: adv = (adv - mean_masked) / std_masked  
PPO actor loss, value loss, entropy aggregated with mask: sum(... * mask) / mask.sum()

---

## Initialization and Training

Final layers orthogonal init; biases = 0; log_std init ≈ -0.5  
Keep network in train() during rollout  
GroupNorm replaces BatchNorm

---

## Input Summary

| Input | Shape | Description |
|-------|--------|-------------|
| obs_agents | [B, N, D_obs] | Per-agent local numeric features |
| types | [B, N] | 0 = vehicle, 1 = pedestrian |
| mask | [B, N] | 1 = valid slot, 0 = pad |
| bev_local | [B, C, H, W] | Ego-centric local BEV |
| bev_global | [B, Cg, Hg, Wg] | Optional global BEV |

---

## Shared Modules

BEV Encoder: ConvNet → bev_feat [B, 256]  
Agent Encoder: MLP(obs_dim → F) → feats_agents [B, N, F]

Pooling / Context:  
ctx_elems = ctx_mlp(feats_agents) → [B, N, F]  
global_ctx = masked_mean(ctx_elems, mask) → [B, F]  
global_ctx_expanded = global_ctx.unsqueeze(1).expand(-1, N, -1) → [B, N, F]

Per-agent combined feature:  
per_agent_feat = concat(feats_agents, global_ctx_expanded, type_emb) → [B, N, F*2 + E]

Critic (Alternative):  
input: concat(feats_agents, global_ctx_expanded) → [B, N, 2F]  
MLP → V_per_agent [B, N]  
V = V * mask

---

## PPO Training

Compute returns/advantages per agent [T, B, N]  
Mask all loss terms  
Aggregate: sum(... * mask) / mask.sum()
