# models/mappo_policy.py
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

_LOG_2PI = float(torch.log(torch.tensor(2.0 * torch.pi)))
_EPS = 1e-8
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

# reuse the BEV and SlotGRU implementations you have (or will add)
from .bev_feature_extractor import BEVFeatureExtractor
from .temporal_encoder import SlotGRU
from .actor_critic_heads import ActorHeads, CriticPerAgent

# small util: orthogonal init for modules (linear/conv)
def orthogonal_init_module(module: nn.Module, gain: float = 1.0):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class MAPPOPolicy(nn.Module):
    """
    MAPPOPolicy initializer only.

    This __init__ focuses on:
      - clear separation of core MAPPO elements (encoder, actor, critic)
      - SAGE-specific elements (BEV encoder, per-slot GRU, heterogeneous action heads)
      - algorithm controls (action scaling, initial log-std, device)
    Forward / action sampling / evaluate_actions are intentionally NOT implemented here:
    you'll instruct me which method to implement next.

    Parameters (high-level):
      bev_in_ch: channels of BEV image input (SAGE-specific)
      obs_dim: per-agent observation vector dimension (numeric features)
      act_dim_vehicle / act_dim_ped: action dims per type (SAGE-specific)
      type_vocab_size / type_emb_dim: categorical type handling
      hidden_dim: MLP hidden sizes and general embedding dims
      recurrent_hidden_dim: GRU hidden size (both BEV GRU and slot GRU will project to this)
      use_bev_gru: whether BEV encoder includes a temporal GRU (scene-level memory)
      use_slot_gru: whether to use per-slot (per-agent) GRU
      global_ctx_dim: dimensionality of pooled global context used by critic
      action_scale: scale applied to final tanh actions
      log_std_init: initial value for learnable log-std parameters
      dropout / norm_cfg left minimal here; can extend as needed.
    """

    def __init__(self,
                 # SAGE / environment specific
                 bev_in_ch: int,
                 obs_dim: int,
                 act_dim_vehicle= 2,
                 act_dim_ped: Optional[int] = 2,

                 # basic MAPPO elements
                 type_vocab_size: int = 2,
                 type_emb_dim: int = 8,
                 hidden_dim: int = 256,
                 recurrent_hidden_dim: int = 256,
                 use_bev_gru: bool = True,
                 use_slot_gru: bool = True,
                 global_ctx_dim: int = 256,

                 # algorithmic / training controls
                 action_scale: float = 5.0,
                 log_std_init: float = -1.5,

                 # misc / engineering
                 device: str = "cpu") -> None:
        super().__init__()

        if act_dim_ped is None:
            act_dim_ped = act_dim_vehicle
        # -------------------------
        # Save core hyperparams
        # -------------------------
        self.device = torch.device(device)
        # action scale and initial log_std are algorithmic knobs (trainer controls these)
        self.action_scale = torch.as_tensor(
            action_scale,
            dtype=torch.float32,
            device=self.device
        )

        self.log_std_init = float(log_std_init)

        # dims & flags
        self.obs_dim = int(obs_dim)
        self.bev_in_ch = int(bev_in_ch)
        self.type_vocab_size = int(type_vocab_size)
        self.type_emb_dim = int(type_emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.use_bev_gru = bool(use_bev_gru)
        self.use_slot_gru = bool(use_slot_gru)
        self.global_ctx_dim = int(global_ctx_dim)

        # -------------------------
        # 1) BEV feature extractor (scene-level encoder)
        #    - produces a 256-d (by design) per-timestep BEV embedding
        #    - optionally contains its own GRU to model scene dynamics
        #    SAGE-specific: BEV images are central to your env representation.
        # -------------------------
        # BEVFeatureExtractor should map BEV -> 256-d (or (T,B,256) if time-major seq)
        self.bev_enc = BEVFeatureExtractor(
            in_channels=self.bev_in_ch,
            use_gru=self.use_bev_gru,
            device=self.device
        )

        # -------------------------
        # 2) Type embedding (heterogeneous agents)
        #    - encodes agent category (vehicle/ped/etc) into small dense vector
        #    - used in per-agent feature concatenation so network knows agent semantics
        # -------------------------
        self.type_emb = nn.Embedding(self.type_vocab_size, self.type_emb_dim)

        # -------------------------
        # 3) Slot-level temporal encoder (per-agent RNN)
        #    - if enabled, keeps per-slot hidden states
        #    - implemented as GRUCell-based module (SlotGRU)
        #    - SAGE-specific: helps capture agent-level temporal dynamics
        # -------------------------
        if self.use_slot_gru:
            # input dim to slot GRU = obs_dim + bev_emb(256) + type_emb
            slot_input_dim = self.obs_dim + 256 + self.type_emb_dim
            self.slot_gru = SlotGRU(input_size=slot_input_dim,
                                    hidden_size=self.recurrent_hidden_dim)
        else:
            self.slot_gru = None

        # -------------------------
        # 4) Context MLP (global pooling / critic context)
        #    - maps per-agent numeric features into a context embedding for pooling
        #    - pooled global context is concatenated into critic input (DeepSets-style)
        #    - design choice: small MLP to keep global_ctx_dim
        # -------------------------
        self.ctx_mlp = nn.Sequential(
            nn.Linear(self.obs_dim, self.global_ctx_dim),
            nn.ReLU(),
            nn.Linear(self.global_ctx_dim, self.global_ctx_dim)
        )

        # BEV feature flat dim (per design BEVFeatureExtractor -> 256-d)
        self.bev_feat_dim = 256
        # project BEV vector to global_ctx_dim for critic
        self.bev_to_ctx = nn.Linear(self.bev_feat_dim, self.global_ctx_dim)
        # initialize
        nn.init.orthogonal_(self.bev_to_ctx.weight, gain=1.0)
        if self.bev_to_ctx.bias is not None:
            nn.init.constant_(self.bev_to_ctx.bias, 0.0)

        # -------------------------
        # 5) Actor heads (heterogeneous action outputs)
        #    - separate heads for vehicle and pedestrian (can be extended by type)
        #    - ActorHeads returns mu for each type; log_std are learnable params
        # -------------------------
        # actor input dim: if using slot_gru -> recurrent_hidden_dim, else per-agent concat dim
        if self.use_slot_gru:
            actor_in_dim = self.recurrent_hidden_dim
        else:
            actor_in_dim = self.obs_dim + 256 + self.type_emb_dim

        self.actor = ActorHeads(
            in_dim=actor_in_dim,
            act_dim_vehicle=act_dim_vehicle,
            act_dim_ped=act_dim_ped,
            hidden=[self.hidden_dim, self.hidden_dim],
            init_log_std=self.log_std_init
        )

        # -------------------------
        # 6) Critic per-agent (DeepSets style)
        #    - critic input: per-agent-feat (+ pooled global ctx)
        #    - if using slot_gru, per_agent_feats dim == recurrent_hidden_dim
        #    - critic_in_dim = per_agent_feats + global_ctx_dim
        # -------------------------
        if self.use_slot_gru:
            critic_in_dim = self.recurrent_hidden_dim + self.global_ctx_dim
        else:
            critic_in_dim = (self.obs_dim + 256 + self.type_emb_dim) + self.global_ctx_dim

        self.critic = CriticPerAgent(in_dim=critic_in_dim, hidden=[self.hidden_dim, self.hidden_dim])

        # -------------------------
        # 7) bookkeeping: initial hidden placeholders & device move
        #    - note: actual hidden tensors are managed by trainer (per-env),
        #      but policy should provide helper methods to init/reset them.
        # -------------------------
        # We'll keep placeholders (None) here; trainer will call policy.get_initial_hidden(...)
        self._bev_hidden_shape = (1, None, self.recurrent_hidden_dim)  # (num_layers, B, H) with B set by trainer
        self._slot_hidden_shape = (1, None, None, self.recurrent_hidden_dim)  # (1, B, N, H)

        # move modules to device
        self.to(self.device)

        # apply orthogonal initialization to final MLPs (stability)
        orthogonal_init_module(self.ctx_mlp)
        orthogonal_init_module(self.actor)

        # initialize actor.log_std param if present (ensure consistent starting variance)
        if hasattr(self, "actor") and hasattr(self.actor, "log_std"):
            with torch.no_grad():
                try:
                    self.actor.log_std.fill_(float(self.log_std_init))
                except Exception:
                    # some actor implementations store log_std under different name/shape — ignore safely
                    pass

        orthogonal_init_module(self.critic)

        # ---- final notes stored as attributes for trainer usage ----
        self.act_dim_vehicle = int(act_dim_vehicle)
        self.act_dim_ped = int(act_dim_ped)

        # expose a small config dict for convenient external inspection/logging
        self.cfg = {
            "bev_in_ch": self.bev_in_ch,
            "obs_dim": self.obs_dim,
            "type_vocab_size": self.type_vocab_size,
            "type_emb_dim": self.type_emb_dim,
            "hidden_dim": self.hidden_dim,
            "recurrent_hidden_dim": self.recurrent_hidden_dim,
            "use_bev_gru": self.use_bev_gru,
            "use_slot_gru": self.use_slot_gru,
            "global_ctx_dim": self.global_ctx_dim,
            "action_scale": self.action_scale,
            "log_std_init": self.log_std_init,
            "device": str(self.device),
        }

        # ensure hidden-dim attribute names expected by get_initial_hidden
        self.slot_hidden_dim = self.recurrent_hidden_dim
        self.bev_hidden_dim = self.bev_feat_dim

        # ensure save() writes vehicle/ped act dims (fix mismatch)
        # (you can also update save() instead; setting act_dim attr keeps backward compat)
        self.act_dim = max(self.act_dim_vehicle, self.act_dim_ped)

    def forward(
            self,
            obs: Dict[str, torch.Tensor],
            slot_hidden: Optional[torch.Tensor] = None,
            bev_hidden: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            mode: str = "seq",
            deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        assert mode in ("seq", "step"), "mode must be 'seq' or 'step'"

        image = obs.get("image", None)
        agent_feats = obs.get("agent_feats", None)
        type_id = obs.get("type_id", None)

        if image is None or agent_feats is None:
            raise ValueError("obs must contain 'image' and 'agent_feats'")

        is_step = (mode == "step")

        # --- normalize image to time-major [T,B,C,H,W] ---
        if is_step:
            if image.dim() == 4:  # [B,C,H,W]
                image = image.unsqueeze(0)  # -> [1,B,C,H,W]
            elif image.dim() == 5 and image.shape[0] != 1 and image.shape[1] == 1:
                image = image.transpose(0, 1)
        else:
            if image.dim() == 5 and agent_feats is not None and agent_feats.dim() == 4:
                if agent_feats.shape[0] == image.shape[0] and agent_feats.shape[1] == image.shape[1]:
                    image = image.transpose(0, 1)

        # --- normalize agent_feats to time-major [T,B,N,feat] ---
        if is_step:
            if agent_feats.dim() == 3:  # [B,N,feat]
                agent_feats = agent_feats.unsqueeze(0)
            elif agent_feats.dim() == 4 and agent_feats.shape[0] != 1 and agent_feats.shape[1] == 1:
                agent_feats = agent_feats.transpose(0, 1)
        else:
            if agent_feats.dim() == 4 and image.dim() == 5:
                if agent_feats.shape[0] == image.shape[1]:
                    agent_feats = agent_feats.transpose(0, 1)

        # --- normalize type_id to [T,B,N] if provided ---
        if type_id is not None:
            if is_step:
                if type_id.dim() == 2:
                    type_id = type_id.unsqueeze(0)
            else:
                if type_id.dim() == 2:
                    type_id = type_id.unsqueeze(0)
                elif type_id.dim() == 3 and image is not None and type_id.shape[0] == image.shape[1]:
                    type_id = type_id.transpose(0, 1)

        # --- normalize mask to [T,B,N] if provided ---
        if mask is not None:
            if is_step:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                elif mask.dim() == 3 and mask.shape[0] != 1 and mask.shape[1] == 1:
                    mask = mask.transpose(0, 1)
            else:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                elif mask.dim() == 3 and image is not None and mask.shape[0] == image.shape[1]:
                    mask = mask.transpose(0, 1)

        # final shape checks
        if image.dim() != 5:
            raise ValueError(f"image should be 5D after normalization, got {image.shape}")
        if agent_feats.dim() != 4:
            raise ValueError(f"agent_feats should be 4D after normalization, got {agent_feats.shape}")

        T, B = image.shape[0], image.shape[1]
        _, _, N, obs_dim = agent_feats.shape

        # --- slot_hidden normalization ---
        if slot_hidden is None or bev_hidden is None:
            slot_hidden, bev_hidden = self.get_initial_hidden(batch_size=B, n_agent=N)
        else:
            if slot_hidden.dim() == 4:
                slot_hidden = slot_hidden[-1]
            if slot_hidden.dim() != 3:
                raise ValueError("slot_hidden must be [B,N,H] or [T,B,N,H]")

        if bev_hidden is not None and bev_hidden.dim() == 4:
            bev_hidden = bev_hidden[-1]

        # --- precompute BEV embeddings for seq mode with robust normalization ---
        bev_embed_seq = None
        if not is_step:
            bev_out = self._call_bev_encoder_seq(image, bev_hidden) if hasattr(self,
                                                                               "_call_bev_encoder_seq") else self.bev_enc(
                image
            )

            if isinstance(bev_out, tuple) and len(bev_out) == 2:
                bev_embed_seq, bev_h = bev_out
            else:
                bev_embed_seq = bev_out

            if not torch.is_tensor(bev_embed_seq):
                bev_embed_seq = torch.as_tensor(bev_embed_seq, device=image.device)

            if bev_embed_seq.dim() == 2:
                if bev_embed_seq.shape[0] == T * B:
                    bev_embed_seq = bev_embed_seq.contiguous().view(T, B, -1)
                elif bev_embed_seq.shape[0] == T:
                    bev_embed_seq = bev_embed_seq.contiguous().view(T, 1, -1).expand(T, B, -1).contiguous()
                else:
                    bev_embed_seq = bev_embed_seq.contiguous().view(T, B, -1)
            elif bev_embed_seq.dim() == 3:
                if bev_embed_seq.shape[0] == T:
                    pass
                elif bev_embed_seq.shape[1] == T:
                    bev_embed_seq = bev_embed_seq.transpose(0, 1).contiguous()
                else:
                    bev_embed_seq = bev_embed_seq.contiguous().view(T, B, -1)
            else:
                bev_embed_seq = bev_embed_seq.contiguous().view(T, B, -1)

        # --- prepare outputs ---
        mus = []
        log_stds = []
        pre_tanz = []
        actions_out = []
        log_probs = []
        values_out = []
        entropies = []

        slot_h = slot_hidden
        bev_h = bev_hidden

        for t in range(T):
            img_t = image[t]  # [B,C,H,W]
            feat_t = agent_feats[t]  # [B,N,obs]
            mask_t = mask[t] if mask is not None else torch.ones(B, N, device=feat_t.device)
            type_t = type_id[t] if type_id is not None else None

            # BEV embedding per-step (ensure shape [B,D])
            if is_step:
                bev_embed, bev_h = self._call_bev_encoder_step(img_t, bev_h)
                if not torch.is_tensor(bev_embed):
                    bev_embed = torch.as_tensor(bev_embed, device=img_t.device)
                if bev_embed.dim() == 1:
                    bev_embed = bev_embed.unsqueeze(0)
                if bev_embed.dim() == 2 and bev_embed.shape[0] != B:
                    if bev_embed.shape[0] == 1 and B > 1:
                        bev_embed = bev_embed.expand(B, bev_embed.shape[-1]).contiguous()
                    else:
                        bev_embed = bev_embed.contiguous().view(B, -1)
            else:
                bev_embed = bev_embed_seq[t]
                if not torch.is_tensor(bev_embed):
                    bev_embed = torch.as_tensor(bev_embed, device=img_t.device)
                if bev_embed.dim() == 1:
                    bev_embed = bev_embed.unsqueeze(0)
                if bev_embed.dim() == 2 and bev_embed.shape[0] != B:
                    if bev_embed.shape[0] == T and B == 1:
                        bev_embed = bev_embed[0].unsqueeze(0)
                    else:
                        bev_embed = bev_embed.contiguous().view(B, -1)

            # slot GRU step
            slot_emb, slot_h = self._call_slot_gru_step(feat_t, slot_h, bev_embed, mask_t)

            # actor head
            mu_t, log_std_t = self._call_actor_head(slot_emb, type_t)

            # compute std and entropy
            std = torch.exp(log_std_t)
            dist = torch.distributions.Normal(mu_t, std)
            entropy_t = dist.entropy().sum(dim=-1)  # [B,N]

            # sample/deterministic
            if deterministic:
                pre_t = mu_t
            else:
                eps = torch.randn_like(mu_t)
                pre_t = mu_t + eps * std

            action_t = torch.tanh(pre_t) * self.action_scale

            # log prob
            logprob_t = self._call_logprob(pre_t, mu_t, log_std_t, action_scale=self.action_scale)
            # critic
            v_t = self._call_critic(slot_emb, feat_t, bev_embed, mask_t)

            # mask out inactive agents
            mask_t_ = mask_t.unsqueeze(-1)
            v_t = v_t * mask_t_
            action_t = action_t * mask_t_.expand_as(action_t)
            logprob_t = logprob_t * mask_t
            entropy_t = entropy_t * mask_t

            # collect
            mus.append(mu_t.unsqueeze(0))
            log_stds.append(log_std_t.unsqueeze(0))
            pre_tanz.append(pre_t.unsqueeze(0))
            actions_out.append(action_t.unsqueeze(0))
            log_probs.append(logprob_t.unsqueeze(0))
            values_out.append(v_t.unsqueeze(0))
            entropies.append(entropy_t.unsqueeze(0))

        # stack
        mu = torch.cat(mus, dim=0)
        log_std = torch.cat(log_stds, dim=0)
        pre_tanh = torch.cat(pre_tanz, dim=0)
        actions = torch.cat(actions_out, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        values = torch.cat(values_out, dim=0)
        entropy = torch.cat(entropies, dim=0)

        # ensure values shape [T,B,N] if last dim==1
        if values.dim() == 4 and values.shape[-1] == 1:
            values = values.squeeze(-1)

        outputs = {
            "mu": mu,
            "log_std": log_std,
            "pre_tanh": pre_tanh,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy,
            "next_slot_hidden": slot_h,
            "next_bev_hidden": bev_h,
        }

        if is_step:
            def _squeeze0(x):
                return x.squeeze(0) if torch.is_tensor(x) and x.shape[0] == 1 else x

            outputs = {k: _squeeze0(v) for k, v in outputs.items()}

        return outputs

    def burn_in(self, obs_seq: Dict[str, torch.Tensor],
                slot_hidden: Optional[torch.Tensor] = None,
                bev_hidden: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                detach: bool = True) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Warm up policy hidden states from an observation sequence by reusing forward(mode='seq').

        Args:
            obs_seq: dict in the same format forward(..., mode='seq') expects, e.g.
                     {"image": Tensor[T,B,C,H,W], "agent_feats": Tensor[T,B,N,obs_dim], "type_id": ...}
            slot_hidden: optional initial [B,N,H] or [T,B,N,H]
            bev_hidden: optional initial bev hidden
            mask: optional mask tensor (T,B,N) or (B,N)
            detach: if True, detach returned hidden tensors

        Returns:
            (next_slot_hidden, next_bev_hidden) where next_slot_hidden is [B,N,H] or None.
        """
        # ensure obs_seq tensor devices match policy device
        # move obs_seq tensors to policy device if they are torch tensors
        obs_on_device = {}
        if torch.is_tensor(obs_seq):
            obs_seq = {"agent_feats": obs_seq.to(self.device)}
        elif isinstance(obs_seq, np.ndarray):
            obs_seq = {"agent_feats": torch.tensor(obs_seq, device=self.device)}
        for k, v in obs_seq.items():
            if torch.is_tensor(v):
                obs_on_device[k] = v.to(self.device)
            else:
                obs_on_device[k] = v

        # call forward in seq mode under no_grad by default (burn-in shouldn't create graph)
        with torch.no_grad():
            out = self.forward(
                obs=obs_on_device,
                slot_hidden=slot_hidden,
                bev_hidden=bev_hidden,
                mask=mask,
                mode="seq",
                deterministic=True
            )

        # extract hidden states from forward outputs (support multiple key names)
        next_slot = out.get("next_slot_hidden", out.get("slot_hidden", None))
        next_bev = out.get("next_bev_hidden", out.get("bev_hidden", None))

        # If forward returned time-major sequence for hidden, take the last timestep
        if isinstance(next_slot, torch.Tensor):
            # possible shapes: [T,B,N,H] or [1,B,N,H] or [B,N,H]
            if next_slot.ndim == 4:
                # assume time-major -> take last
                next_slot = next_slot[-1]
            # ensure shape [B,N,H]
            if next_slot.ndim != 3:
                raise RuntimeError(f"burn_in: unexpected slot hidden shape {next_slot.shape}")

        if isinstance(next_bev, torch.Tensor):
            # possible shapes: [T,B,D] or [1,B,D] or [B,D]
            if next_bev.ndim == 3:
                next_bev = next_bev[-1]
            # allow [B,D] or None

        if detach:
            if isinstance(next_slot, torch.Tensor):
                next_slot = next_slot.detach()
            if isinstance(next_bev, torch.Tensor):
                next_bev = next_bev.detach()

        # ensure device consistency
        if isinstance(next_slot, torch.Tensor):
            next_slot = next_slot.to(self.device)
        if isinstance(next_bev, torch.Tensor):
            next_bev = next_bev.to(self.device)

        return next_slot, next_bev

    def select_action(self,
                      obs_step,
                      slot_hidden=None,
                      bev_hidden=None,
                      mask=None,
                      deterministic=False):

        device = next(self.parameters()).device

        with torch.no_grad():
            out = self.forward(
                obs=obs_step,
                slot_hidden=slot_hidden,
                bev_hidden=bev_hidden,
                mask=mask,
                mode="step",
                deterministic=deterministic
            )

        mu = out["mu"]
        log_std = out["log_std"]
        values = out.get("values", out.get("value"))
        next_slot_hidden = out.get("next_slot_hidden", None)
        next_bev_hidden = out.get("next_bev_hidden", None)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        if deterministic:
            pre_t = mu
        else:
            eps = torch.randn_like(mu)
            pre_t = mu + eps * std

        tanh_pre = torch.tanh(pre_t)

        action_scale = getattr(self, "action_scale", 1.0)
        action_scale_t = torch.as_tensor(
            action_scale, device=device, dtype=tanh_pre.dtype
        )

        action_scaled = tanh_pre * action_scale_t

        a_trim = action_scaled[..., :2]

        a0_unscaled = a_trim[..., 0] / (action_scale_t + _EPS)
        throttle = (a0_unscaled + 1.0) / 2.0
        steer = a_trim[..., 1]

        actions = torch.stack([throttle, steer], dim=-1)
        actions[..., 0] = actions[..., 0].clamp(0.0, 1.0)

        logp = self._call_logprob(pre_t, mu, log_std, action_scale=action_scale_t)

        info = {
            "mu": mu,
            "log_std": log_std,
            "pre_t": pre_t
        }

        return actions, values, logp, next_slot_hidden, next_bev_hidden, info

    def get_initial_hidden(self, batch_size: int, n_agent: int,
                           randomize: bool = False, eps: float = 1e-3):
        """
        Return initial hidden states for slot and BEV encoders.

        Args:
            batch_size (int): number of parallel environments (B)
            n_agent (int): number of agents (N)
            randomize (bool): if True, initialize with small Gaussian noise
            eps (float): standard deviation for random initialization

        Returns:
            tuple(torch.Tensor, Optional[torch.Tensor]):
                slot_hidden [B, N, H_slot],
                bev_hidden [B, bev_h_dim] or None
        """
        H_slot = getattr(self, "slot_hidden_dim", None)
        bev_h_dim = getattr(self, "bev_hidden_dim", None)
        if H_slot is None:
            raise RuntimeError("slot_hidden_dim must be set in the policy")

        device = getattr(self, "device", torch.device("cpu"))

        if randomize:
            slot_h = torch.randn(batch_size, n_agent, H_slot, device=device, dtype=torch.float32) * float(eps)
        else:
            slot_h = torch.zeros(batch_size, n_agent, H_slot, device=device, dtype=torch.float32)

        bev_h = None
        if bev_h_dim is not None:
            if randomize:
                bev_h = torch.randn(batch_size, bev_h_dim, device=device, dtype=torch.float32) * float(eps)
            else:
                bev_h = torch.zeros(batch_size, bev_h_dim, device=device, dtype=torch.float32)

        return slot_h, bev_h

    def reset_hidden_for_agents(self, slot_hidden: torch.Tensor,
                                respawn_mask: torch.Tensor,
                                randomize: bool = False, eps: float = 1e-3) -> torch.Tensor:
        """
        Reset hidden states for agents indicated by respawn_mask.

        Args:
            slot_hidden: [B, N, H_slot]
            respawn_mask: [B, N] tensor with 1 where hidden should be reset
            randomize: if True, use small Gaussian noise instead of zeros
            eps: std deviation for random noise

        Returns:
            new_slot_hidden: [B, N, H_slot]
        """
        if slot_hidden.dim() != 3:
            raise ValueError("slot_hidden must be [B, N, H]")

        device = slot_hidden.device
        B, N, H = slot_hidden.shape
        mask = respawn_mask.to(device=device, dtype=torch.float32).unsqueeze(-1)

        if randomize:
            new_vals = torch.randn(B, N, H, device=device, dtype=slot_hidden.dtype) * float(eps)
        else:
            new_vals = torch.zeros(B, N, H, device=device, dtype=slot_hidden.dtype)

        new_slot_hidden = slot_hidden * (1.0 - mask) + new_vals * mask
        return new_slot_hidden

    def get_value(self,
                  obs: Dict[str, torch.Tensor],
                  slot_hidden: Optional[torch.Tensor] = None,
                  bev_hidden: Optional[torch.Tensor] = None,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute value estimates for given observations.

        Args:
            obs: dict with keys "image" [B,C,H,W] and "agent_feats" [B,N,obs_dim]
            slot_hidden: [B,N,H_slot] or None
            bev_hidden: [B,bev_h_dim] or None
            mask: [B,N] boolean or 0/1 tensor

        Returns:
            values: torch.Tensor [B,N,1]
        """
        if "agent_feats" not in obs:
            raise ValueError("obs must contain 'agent_feats'")

        agent_feats = obs["agent_feats"]
        device = agent_feats.device
        B, N, _ = agent_feats.shape

        if mask is None:
            mask = torch.ones(B, N, device=device, dtype=torch.float32)
        else:
            mask = mask.to(device=device, dtype=torch.float32)

        # if hidden not provided, initialize
        if slot_hidden is None or bev_hidden is None:
            slot_hidden, bev_hidden = self.get_initial_hidden(B, N)

        # reuse forward to compute value only
        with torch.no_grad():
            out = self.forward(
                obs=obs,
                slot_hidden=slot_hidden,
                bev_hidden=bev_hidden,
                mask=mask,
                mode="step",
                deterministic=True
            )

        values = out.get("values", out.get("value", None))
        if values is None:
            raise RuntimeError("forward() must return 'values' or 'value' field")

        # ensure [B,N,1]
        if values.dim() == 2:
            values = values.unsqueeze(-1)

        values = values.to(device) * mask.unsqueeze(-1)
        return values

    def save(self, path_prefix: str) -> None:
        """
        Save model parameters and configuration.

        Args:
            path_prefix: file path prefix, e.g. "checkpoints/policy_step100"
                         will create "policy_step100.pt" and "policy_step100_config.json"
        """
        import os, json, torch

        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)

        # save model parameters
        torch.save(self.state_dict(), f"{path_prefix}.pt")

        # save minimal configuration
        config = {
            "slot_hidden_dim": getattr(self, "slot_hidden_dim", None),
            "bev_hidden_dim": getattr(self, "bev_hidden_dim", None),
            "act_dim_vehicle": getattr(self, "act_dim_vehicle", None),
            "act_dim_ped": getattr(self, "act_dim_ped", None),
            "obs_dim": getattr(self, "obs_dim", None),
            "device": str(getattr(self, "device", "cpu")),
            "class": self.__class__.__name__,
        }
        with open(f"{path_prefix}_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load(self, path_prefix: str, map_location: Optional[str] = None) -> None:
        """
        Load model parameters and configuration.

        Args:
            path_prefix: file path prefix used during save(), e.g. "checkpoints/policy_step100"
            map_location: device mapping for torch.load (e.g. "cpu", "cuda:0")
        """
        import json, torch, os

        model_path = f"{path_prefix}.pt"
        config_path = f"{path_prefix}_config.json"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # load state dict
        state_dict = torch.load(model_path, map_location=map_location or getattr(self, "device", "cpu"))
        self.load_state_dict(state_dict)

        # load config if available
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if hasattr(self, k) and v is not None:
                    setattr(self, k, v)
        device = map_location or getattr(self, "device", "cpu")
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.to(self.device)

    def _call_bev_encoder_step(self, img_t, bev_h):
        """
        img_t: [B, C, H, W]
        bev_h: previous bev hidden (or None)
        Return:
          bev_embed: [B, D]
          next_bev_h
        """
        device = img_t.device
        B = img_t.shape[0]

        # Always call encoder on the single timestep input (force step-like behavior)
        # Do NOT rely on a `step()` API that might internally use cached full sequences.
        out = None
        try:
            out = self.bev_enc(img_t)
        except Exception:
            # as a last resort, if bev_enc has step, call it
            if hasattr(self.bev_enc, "step"):
                out = self.bev_enc.step(img_t, bev_h)
            else:
                out = self.bev_enc(img_t)

        if isinstance(out, tuple) and len(out) == 2:
            bev_embed, next_bev_h = out
        else:
            bev_embed = out
            next_bev_h = bev_h

        # normalize to [B, D]
        if bev_embed is None:
            D = getattr(self, "bev_feat_dim", 256)
            bev_embed = torch.zeros(B, D, device=device)
        else:
            # if returned flattened time*batch (T*B, D), try to reshape and select first batch slice
            if bev_embed.dim() == 2 and bev_embed.shape[0] != B:
                if bev_embed.shape[0] % B == 0:
                    TB = bev_embed.shape[0]
                    T = TB // B
                    bev_embed = bev_embed.view(T, B, bev_embed.shape[-1])[0]  # take t=0
                else:
                    # fallback: take first B rows
                    bev_embed = bev_embed[:B]
            # if higher-dim, reshape to [B, -1]
            if bev_embed.dim() != 2:
                bev_embed = bev_embed.reshape(B, -1)

        return bev_embed, next_bev_h

    def _call_bev_encoder_seq(self, image_seq, bev_h):
        """
        image_seq: [T, B, C, H, W]
        return: [T, B, D]
        """

        T, B = image_seq.shape[0], image_seq.shape[1]

        # reshape to 4D for CNN
        img_flat = image_seq.view(T * B, image_seq.shape[2], image_seq.shape[3], image_seq.shape[4])

        bev_flat = self.bev_enc(img_flat)  # [T*B, D]

        # ensure 2D
        if bev_flat.dim() != 2:
            bev_flat = bev_flat.view(T * B, -1)

        bev_embed_seq = bev_flat.view(T, B, -1)

        return bev_embed_seq, bev_h

    def _call_slot_gru_step(self, feat_t, slot_h, bev_embed, mask_t):

        dev = next(self.parameters()).device

        if not isinstance(feat_t, torch.Tensor):
            feat_t = torch.as_tensor(feat_t, dtype=torch.float32, device=dev)
        feat_t = feat_t.to(dev)
        if feat_t.dim() != 3:
            raise RuntimeError(f"feat_t must be [B,N,F], got {tuple(feat_t.shape)}")
        B, N, F_feat = feat_t.shape

        if bev_embed is None:
            raise RuntimeError("bev_embed cannot be None")
        if not isinstance(bev_embed, torch.Tensor):
            bev_embed = torch.as_tensor(bev_embed, dtype=torch.float32, device=dev)
        bev_embed = bev_embed.to(dev)
        if bev_embed.dim() == 2:
            if bev_embed.shape[0] != B:
                # try to handle case where bev_embed rows == B*N
                if bev_embed.shape[0] == B * N:
                    bev_embed = bev_embed.view(B, N, bev_embed.shape[1])
                else:
                    raise RuntimeError(f"bev_embed batch mismatch {tuple(bev_embed.shape)} vs B={B}")
            bev_embed = bev_embed.unsqueeze(1).expand(B, N, -1).contiguous()
        elif bev_embed.dim() == 3:
            if tuple(bev_embed.shape[:2]) != (B, N):
                if bev_embed.shape[0] * bev_embed.shape[1] == B * N:
                    bev_embed = bev_embed.view(B, N, bev_embed.shape[2])
                else:
                    raise RuntimeError(f"bev_embed shape mismatch {tuple(bev_embed.shape)} vs expected (B,N)=({B},{N})")
        else:
            raise RuntimeError(f"bev_embed must be [B,F] or [B,N,F], got {tuple(bev_embed.shape)}")
        F_bev = bev_embed.shape[2]

        if mask_t is None:
            mask_t = torch.ones(B, N, device=dev)
        if not isinstance(mask_t, torch.Tensor):
            mask_t = torch.as_tensor(mask_t, dtype=torch.bool, device=dev)
        mask_t = mask_t.to(dev)
        if mask_t.dim() == 1 and mask_t.shape[0] == B * N:
            mask_t = mask_t.view(B, N)
        if mask_t.dim() != 2:
            raise RuntimeError(f"mask_t must be [B,N], got {tuple(mask_t.shape)}")

        pieces = [feat_t.reshape(B * N, F_feat), bev_embed.reshape(B * N, F_bev)]

        slot_input = torch.cat(pieces, dim=-1)  # [B*N, total_dim]
        total_dim = slot_input.shape[1]

        # ensure slot_input matches expected input size (existing code kept)
        expected_in = None
        try:
            expected_in = int(
                getattr(self.slot_gru, "input_size", None) or getattr(self.slot_gru, "gru_cell").input_size)
        except Exception:
            expected_in = None

        if expected_in is not None:
            if total_dim < expected_in:
                pad_dim = expected_in - total_dim
                pad = torch.zeros(B * N, pad_dim, device=dev, dtype=slot_input.dtype)
                slot_input = torch.cat([slot_input, pad], dim=-1)
                total_dim = expected_in
            elif total_dim > expected_in:
                raise RuntimeError(
                    f"[slot_gru] input mismatch: total_dim={total_dim} > expected_in={expected_in}; "
                    f"feat_t={tuple(feat_t.shape)}, bev_embed={tuple(bev_embed.shape)}, mask_t={tuple(mask_t.shape)}"
                )

        # --- Prepare flattened hidden and mask so they match slot_input batch dim [B*N, ...] ---
        # Normalize slot_h to flat [B*N, H]
        if slot_h is None:
            # 初始化 hidden
            H = getattr(self, "slot_hidden_dim", None)
            if H is None:
                # 从 gru_cell 读 hidden_size
                try:
                    H = self.slot_gru.gru_cell.hidden_size
                except Exception:
                    raise RuntimeError("Cannot infer slot hidden size")
            slot_h_flat = torch.zeros(B * N, H, device=dev, dtype=slot_input.dtype)

        else:

            if slot_h.dim() == 3:
                if slot_h.shape[1] == N:
                    # If it's a single-sequence hidden state, expand it
                    if slot_h.shape[0] == 1 and B > 1:
                        slot_h = slot_h.expand(B, -1, -1).contiguous()

                    # Use the actual batch size of the tensor
                    curr_batch_size = slot_h.shape[0]
                    slot_h_flat = slot_h.view(curr_batch_size * N, -1)
                else:
                    raise RuntimeError(
                        f"slot_h agents mismatch: got {slot_h.shape[1]}, expected {N}"
                    )
                # # 情况1: 正常 [B, N, H]
                # if tuple(slot_h.shape[:2]) == (B, N):
                #     slot_h_flat = slot_h.view(B * N, -1)
                #
                # # 情况2: rollout阶段常见 [1, N, H] → 需要 expand 到 [B,N,H]
                # elif slot_h.shape[0] == 1 and slot_h.shape[1] == N:
                #     slot_h = slot_h.expand(B, -1, -1).contiguous()
                #     slot_h_flat = slot_h.view(B * N, -1)
                #
                # else:
                #     raise RuntimeError(
                #         f"slot_h shape mismatch: got {tuple(slot_h.shape)} expected (B,N,H)=({B},{N},H)"
                #     )

            elif slot_h.dim() == 2 and slot_h.shape[0] == B * N:
                slot_h_flat = slot_h

            else:
                raise RuntimeError(
                    f"Unable to flatten slot_h with shape {tuple(slot_h.shape)}"
                )

        # Normalize mask: keep both forms: mask_2d [B,N] and mask_flat [B*N]
        mask_2d = None
        mask_flat = None
        if mask_t is None:
            mask_2d = torch.ones(B, N, device=dev, dtype=slot_input.dtype)
            mask_flat = mask_2d.reshape(B * N)
        else:
            if mask_t.dim() == 2 and tuple(mask_t.shape) == (B, N):
                mask_2d = mask_t
                mask_flat = mask_t.reshape(B * N)
            elif mask_t.dim() == 1 and mask_t.shape[0] == B * N:
                mask_flat = mask_t
                mask_2d = mask_t.view(B, N)
            else:
                # try to coerce
                try:
                    if mask_t.numel() == B * N:
                        mask_flat = mask_t.view(B * N)
                        mask_2d = mask_flat.view(B, N)
                    else:
                        raise RuntimeError(f"mask_t has incompatible numel {mask_t.numel()} vs B*N {B*N}")
                except Exception as e:
                    raise RuntimeError(f"mask_t shape not compatible: {tuple(mask_t.shape)}") from e

        # Try calling slot_gru.step with common candidate signatures robustly
        call_errors = []
        slot_emb_flat = None
        next_slot_h_out = None

        # Candidate 1: (slot_input, slot_h_flat, mask_flat) -> most likely
        try:
            slot_emb_flat, next_slot_h_out = self.slot_gru.step(slot_input, slot_h_flat, mask_flat)
        except Exception as e1:
            call_errors.append(("flat_h, flat_mask", e1))
            # Candidate 2: (slot_input, slot_h_flat, mask_2d)
            try:
                slot_emb_flat, next_slot_h_out = self.slot_gru.step(slot_input, slot_h_flat, mask_2d)
            except Exception as e2:
                call_errors.append(("flat_h, 2d_mask", e2))
                # Candidate 3: (slot_input, slot_h, mask_flat) where slot_h is 3D [B,N,H]
                try:
                    slot_emb_flat, next_slot_h_out = self.slot_gru.step(slot_input, slot_h, mask_flat)
                except Exception as e3:
                    call_errors.append(("3d_h, flat_mask", e3))
                    # Candidate 4: (slot_input, slot_h, mask_2d)
                    try:
                        slot_emb_flat, next_slot_h_out = self.slot_gru.step(slot_input, slot_h, mask_2d)
                    except Exception as e4:
                        call_errors.append(("3d_h, 2d_mask", e4))
                        # All failed → raise a detailed error
                        err_msgs = "\n".join([f"{k}: {v}" for k, v in call_errors])
                        raise RuntimeError(
                            f"[slot_gru.step] all candidate call signatures failed. Diagnostics:\n"
                            f"B={B}, N={N}, slot_input.shape={tuple(slot_input.shape)}, "
                            f"slot_h.shape={tuple(slot_h.shape) if slot_h is not None else None}, "
                            f"mask_2d.shape={tuple(mask_2d.shape) if mask_2d is not None else None}, "
                            f"mask_flat.shape={tuple(mask_flat.shape) if mask_flat is not None else None}\n"
                            f"Errors:\n{err_msgs}"
                        ) from e4

        # --- Normalize outputs to expected shapes [B,N,H] ---
        if isinstance(slot_emb_flat, torch.Tensor):
            if slot_emb_flat.dim() == 2 and slot_emb_flat.shape[0] == B * N:
                H_out = slot_emb_flat.shape[1]
                slot_emb = slot_emb_flat.view(B, N, H_out)
            elif slot_emb_flat.dim() == 3 and tuple(slot_emb_flat.shape[:2]) == (B, N):
                slot_emb = slot_emb_flat
            else:
                try:
                    slot_emb = slot_emb_flat.contiguous().view(B, N, -1)
                except Exception as e:
                    raise RuntimeError(f"Unexpected slot_emb_flat shape {tuple(slot_emb_flat.shape)}") from e
        else:
            raise RuntimeError("slot_gru.step returned unexpected slot_emb type")

        # next_slot_h_out -> normalize to [B, N, H]
        if isinstance(next_slot_h_out, torch.Tensor):
            if next_slot_h_out.dim() == 2 and next_slot_h_out.shape[0] == B * N:
                next_slot_h = next_slot_h_out.view(B, N, -1)
            elif next_slot_h_out.dim() == 3 and tuple(next_slot_h_out.shape[:2]) == (B, N):
                next_slot_h = next_slot_h_out
            else:
                try:
                    next_slot_h = next_slot_h_out.contiguous().view(B, N, -1)
                except Exception as e:
                    raise RuntimeError(f"Unexpected next_slot_h_out shape {tuple(next_slot_h_out.shape)}") from e
        else:
            raise RuntimeError("slot_gru.step returned unexpected next_slot_h type")

        return slot_emb, next_slot_h


    def _call_actor_head(self, slot_emb, type_t=None):
        """
        slot_emb: [B, N, H]
        type_t: optional [B, N] type ids

        Returns:
          mu: [B, N, A]
          log_std: [B, N, A]
        """

        B, N, H = slot_emb.shape
        device = slot_emb.device

        # flattened per-agent features (B*N, H)
        flat = slot_emb.contiguous().view(B * N, H)

        # 1) Call actor with preferred signature(s)
        if type_t is not None:
            # prefer (slot_emb, type_t) signature
            try:
                out = self.actor(slot_emb, type_t)
            except TypeError:
                # fallback to (flat, type_flat)
                try:
                    type_flat = type_t.contiguous().view(B * N)
                    out = self.actor(flat, type_flat)
                except Exception:
                    # ultimate fallback: actor(flat)
                    out = self.actor(flat)
        else:
            try:
                out = self.actor(flat)
            except Exception:
                out = self.actor(slot_emb)

        # 2) Normalize actor output into mu_flat and log_std_flat tensors
        mu_flat = None
        log_std_flat = None

        # tuple / list => (mu, log_std?) or single mu
        if isinstance(out, (tuple, list)):
            if len(out) >= 2:
                mu_flat, log_std_flat = out[0], out[1]
            elif len(out) == 1:
                mu_flat = out[0]
        # dict => try common keys
        elif isinstance(out, dict):
            # look for mean keys
            for k in ("mu", "mean", "loc", "action", "mu_flat"):
                if k in out:
                    mu_flat = out[k]
                    break
            # fallback: take first tensor value
            if mu_flat is None:
                for v in out.values():
                    if torch.is_tensor(v):
                        mu_flat = v
                        break
            # log_std / std keys
            if "log_std" in out:
                log_std_flat = out["log_std"]
            elif "logvar" in out:
                # if logvar provided, convert to log_std = 0.5 * logvar
                lv = out["logvar"]
                if torch.is_tensor(lv):
                    log_std_flat = 0.5 * lv
            elif "std" in out:
                stdv = out["std"]
                if torch.is_tensor(stdv):
                    log_std_flat = torch.log(stdv.clamp(min=1e-12))
        # tensor => treat as mu
        elif torch.is_tensor(out):
            mu_flat = out
        else:
            # last resort: try to convert to tensor
            try:
                mu_flat = torch.as_tensor(out, device=device)
            except Exception:
                raise RuntimeError(f"Unsupported actor output type: {type(out)}")

        # ensure mu_flat is a tensor
        if mu_flat is None or not torch.is_tensor(mu_flat):
            # helpful debug info
            if isinstance(out, dict):
                keys = list(out.keys())
            else:
                keys = None
            raise RuntimeError(f"actor returned unsupported output (no mean found). out_type={type(out)}, keys={keys}")

        # 3) Normalize shapes:
        # mu_flat may be either:
        #  - [B, N, A]  (3D) -> flatten to [B*N, A]
        #  - [B*N, A]   (2D) -> ok
        #  - [S, A]     (2D) where S == B*N -> ok
        # Do conversions accordingly and keep A.
        if mu_flat.dim() == 3:
            b_, n_, A = mu_flat.size()
            if b_ != B or n_ != N:
                # try to handle swapped dims if bizarre, but prefer raising
                raise RuntimeError(
                    f"actor returned mu with shape {tuple(mu_flat.size())} inconsistent with expected (B,N,*)=({B},{N},*)")
            mu_flat = mu_flat.contiguous().view(B * N, A)
        elif mu_flat.dim() == 2:
            A = mu_flat.size(-1)
            # assume already flat [B*N, A] or [S, A] where S == B*N
            if mu_flat.size(0) != B * N:
                # if shapes mismatch, but equals B or N maybe actor returned [B, A] or [N, A]
                if mu_flat.size(0) == B:
                    # [B, A] -> expand across N agents? This is ambiguous; raise to enforce strictness
                    raise RuntimeError(
                        f"actor returned mu of shape [B, A] ({tuple(mu_flat.size())}) — ambiguous for per-agent output. Expected [B*N, A].")
                elif mu_flat.size(0) == N:
                    raise RuntimeError(
                        f"actor returned mu of shape [N, A] ({tuple(mu_flat.size())}) — ambiguous for batch handling. Expected [B*N, A].")
                else:
                    raise RuntimeError(
                        f"actor returned mu with unexpected first dim {mu_flat.size(0)}; expected {B * N}")
        else:
            raise RuntimeError(f"actor returned mu with unsupported dim {mu_flat.dim()} (expected 2 or 3)")

        # 4) Normalize log_std_flat
        if log_std_flat is None:
            # if actor has learnable log_std param, try to use it
            if hasattr(self.actor, "log_std"):
                # actor.log_std might be shape [A] or [1, A] or [B*N, A]
                ls = self.actor.log_std
                if torch.is_tensor(ls):
                    # expand to [B*N, A] if needed
                    if ls.dim() == 1:
                        log_std_flat = ls.unsqueeze(0).expand(B * N, -1).to(device)
                    elif ls.dim() == 2 and ls.size(0) == 1:
                        log_std_flat = ls.expand(B * N, -1).to(device)
                    elif ls.dim() == 2 and ls.size(0) == B * N:
                        log_std_flat = ls.to(device)
                    else:
                        # fallback: flatten and expand
                        log_std_flat = ls.view(-1).unsqueeze(0).expand(B * N, -1).to(device)
                else:
                    # not tensor, coerce
                    log_std_flat = torch.as_tensor(float(self.log_std_init), device=device).unsqueeze(0).expand(B * N,
                                                                                                                1)
            else:
                # use configured init value
                log_std_flat = torch.full((B * N, A), fill_value=float(getattr(self, "log_std_init", 0.0)),
                                          device=device)
        else:
            # if provided, coerce to tensor and normalize dims
            if not torch.is_tensor(log_std_flat):
                log_std_flat = torch.as_tensor(log_std_flat, device=device)
            if log_std_flat.dim() == 3:
                b_, n_, _ = log_std_flat.size()
                if b_ != B or n_ != N:
                    raise RuntimeError(
                        f"actor returned log_std with shape {tuple(log_std_flat.size())} inconsistent with expected (B,N,*)")
                log_std_flat = log_std_flat.contiguous().view(B * N, -1)
            elif log_std_flat.dim() == 2:
                # OK if matches [B*N, A] or [B*N, 1] (will broadcast later)
                if log_std_flat.size(0) != B * N:
                    # if it's [B, A] or [1, A], try to match strictly only when unambiguous
                    if log_std_flat.size(0) == B and log_std_flat.size(1) == A:
                        raise RuntimeError(
                            "actor returned log_std shaped [B,A] which is ambiguous; prefer [B*N,A] or [1,A]")
                    elif log_std_flat.size(0) == 1 and log_std_flat.size(1) == A:
                        log_std_flat = log_std_flat.expand(B * N, -1)
                    else:
                        raise RuntimeError(
                            f"actor returned log_std with unexpected first dim {log_std_flat.size(0)}; expected {B * N}")
            elif log_std_flat.dim() == 1:
                # [A] -> expand
                log_std_flat = log_std_flat.view(1, -1).expand(B * N, -1)
            else:
                raise RuntimeError(f"actor returned log_std with unsupported dim {log_std_flat.dim()}")

        # 5) Final reshape to [B, N, A]
        mu = mu_flat.view(B, N, A)
        log_std = log_std_flat.view(B, N, A)

        log_std = torch.clamp(log_std, -5.0, 2.0)

        return mu, log_std

    def _call_logprob(self, pre_t, mu, log_std, action_scale=None):
        pre_t = pre_t.clamp(-20.0, 20.0)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = torch.exp(log_std)
        var = std * std + _EPS

        sq = (pre_t - mu) ** 2
        logp_per_dim = -0.5 * (sq / var + 2.0 * log_std + _LOG_2PI)
        logp = logp_per_dim.sum(dim=-1)

        log2 = torch.log(torch.tensor(2.0, device=pre_t.device, dtype=pre_t.dtype))
        correction_per_dim = 2.0 * (log2 - pre_t - F.softplus(-2.0 * pre_t))
        correction = correction_per_dim.sum(dim=-1)

        if action_scale is None:
            scale_log = 0.0
        else:
            action_scale_t = torch.as_tensor(
                action_scale, device=pre_t.device, dtype=pre_t.dtype
            )
            log_abs_scale = torch.log(torch.abs(action_scale_t) + _EPS)

            while log_abs_scale.dim() < pre_t.dim():
                log_abs_scale = log_abs_scale.unsqueeze(0)

            scale_log = log_abs_scale.sum(dim=-1)

        return logp - correction - scale_log

    def _call_critic(self, slot_emb: torch.Tensor, agent_feats: torch.Tensor, bev_embed: Optional[torch.Tensor], mask_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        slot_emb: [B, N, H]
        agent_feats: [B, N, obs_dim]  (raw per-agent numeric features)
        bev_embed: [B, bev_feat_dim] or other
        mask_t: [B, N] or None

        Returns: v: [B, N, 1]
        Uses: self.ctx_mlp (per-agent -> global_ctx pool) and self.bev_to_ctx if bev available.
        """
        B, N, H = slot_emb.shape
        device = slot_emb.device
        G = self.global_ctx_dim  # target global ctx dim

        # 1) per-agent context via ctx_mlp -> [B, N, G]
        agent_ctx = self.ctx_mlp(agent_feats)  # expects agent_feats shape [B,N,obs_dim] -> yields [B,N,G]

        # 2) masked mean pooling over agents to get pooled_ctx [B, G]
        if mask_t is None:
            pooled = agent_ctx.mean(dim=1)  # [B, G]
        else:
            m = mask_t.float()  # [B,N]
            if m.dim() == 3:
                m = m.squeeze(-1)
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
            weighted = (agent_ctx * m.unsqueeze(-1)).sum(dim=1)  # [B,G]
            pooled = weighted / denom  # [B,G]

        # 3) optionally incorporate BEV vector (project to G and add)
        if bev_embed is not None and bev_embed.dim() == 2 and bev_embed.shape[0] == B:
            # if bev_embed is larger/smaller than expected, try to adapt safely
            if bev_embed.shape[1] == self.bev_feat_dim:
                bev_ctx = self.bev_to_ctx(bev_embed)  # [B, G]
            elif bev_embed.shape[1] > self.bev_feat_dim:
                bev_trim = bev_embed[:, : self.bev_feat_dim]
                bev_ctx = self.bev_to_ctx(bev_trim)
            else:
                # bev smaller than expected -> pad then project
                pad = torch.zeros(B, self.bev_feat_dim - bev_embed.shape[1], device=device, dtype=bev_embed.dtype)
                bev_pad = torch.cat([bev_embed, pad], dim=1)
                bev_ctx = self.bev_to_ctx(bev_pad)
            global_ctx = pooled + bev_ctx
        else:
            global_ctx = pooled  # [B, G]

        # 4) expand to per-agent and concat with slot_emb -> [B, N, H+G]
        global_rep = global_ctx.unsqueeze(1).expand(-1, N, -1)
        critic_in = torch.cat([slot_emb, global_rep], dim=-1)  # [B,N,H+G]

        # 5) flatten -> feed to self.critic (assumes [B*N, in_dim] -> [B*N,1])
        critic_in_flat = critic_in.view(B * N, -1)
        v_flat = self.critic(critic_in_flat)  # [B*N,1]
        v = v_flat.view(B, N, 1)

        # 6) apply mask (zero-out inactive)
        if mask_t is not None:
            m = mask_t.float().unsqueeze(-1) if mask_t.dim() == 2 else mask_t.float()
            v = v * m.to(v.dtype)

        return v

    def mappo_reset(self, init_obs: Optional[dict] = None):
        """
        Reset per-policy runtime state called by MAPPOManager after env.reset().

        Behavior:
          - clear runtime hidden/caches to avoid stale references across episodes
          - if init_obs provided and contains tensor 'agent_feats', infer (B, N)
            and create initial hidden via get_initial_hidden() storing them on the policy.

        Notes:
          - Manager typically calls this on each agent; returning a value is optional.
          - We store *_runtime_* attributes as placeholders; trainer/manager may choose to read them.
        """
        # 1) clear runtime placeholders / caches
        self._runtime_slot_hidden = None
        self._runtime_bev_hidden = None

        # 2) if init_obs is a dict and contains agent_feats (torch.Tensor), create initial hidden
        try:
            if isinstance(init_obs, dict):
                af = init_obs.get("agent_feats", None)
                # accept a single-step [B,N,obs] or stepless [N,obs] or numpy arrays
                if af is not None:
                    # convert numpy -> torch if needed (non-destructive)
                    if not torch.is_tensor(af):
                        try:
                            import numpy as _np
                            if isinstance(af, _np.ndarray):
                                af = torch.from_numpy(af)
                        except Exception:
                            af = None

                    if torch.is_tensor(af):
                        # possible shapes: [B,N,obs], [N,obs], [B,obs] etc.
                        if af.dim() == 3:
                            B, N, _ = af.shape
                        elif af.dim() == 2:
                            # ambiguous: treat as [1, N, obs] if first dim equals N else [B, N] fallback
                            # we assume [N, obs] -> single batch
                            N = af.shape[0]
                            B = 1
                        else:
                            # fallback: can't infer
                            B = None
                            N = None

                        if B is not None and N is not None:
                            slot_h, bev_h = self.get_initial_hidden(batch_size=B, n_agent=N)
                            # store as runtime defaults (trainer/manager can read them)
                            self._runtime_slot_hidden = slot_h
                            self._runtime_bev_hidden = bev_h
        except Exception:
            # don't raise: reset should be lightweight and robust
            self._runtime_slot_hidden = None
            self._runtime_bev_hidden = None

        # 3) Also clear any other potential runtime attrs if present
        for attr in ("_last_obs", "_running_mean", "_some_temp_buffer"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

        return None

    def evaluate_actions(self,
                         obs,
                         actions,
                         mask=None,
                         slot_hidden=None,
                         bev_hidden=None,
                         mode="seq",
                         pre_t=None):  # <--- [Modification 1] Added optional pre_t parameter
        eps = 1e-6
        device = next(self.parameters()).device

        images = obs.get("image", None)
        agent_feats = obs["agent_feats"]

        is_step = False
        if agent_feats.dim() == 3:
            agent_feats = agent_feats.unsqueeze(0)
            is_step = True
        if images is not None and images.dim() == 4:
            images = images.unsqueeze(0)
        if actions.dim() == 3:
            actions = actions.unsqueeze(0)
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # [Modification 2] Adjust dimensions for pre_t if provided
        if pre_t is not None:
            if pre_t.dim() == 3:  # [B, N, A] -> [1, B, N, A]
                pre_t = pre_t.unsqueeze(0)
            pre_t = pre_t.to(device)

        T, B, N, _ = agent_feats.shape

        # Hidden Init
        if slot_hidden is None or bev_hidden is None:
            slot_h, bev_h = self.get_initial_hidden(batch_size=B, n_agent=N)
        else:
            slot_h = slot_hidden.to(device)
            bev_h = bev_hidden.to(device)

        logp_seq = []
        value_seq = []
        entropy_seq = []
        logstd_seq = []

        action_scale = getattr(self, "action_scale", 1.0)
        action_scale_t = torch.as_tensor(action_scale, device=device, dtype=actions.dtype)

        for t in range(T):
            feat_t = agent_feats[t].to(device)
            # act_t = actions[t].to(device) # Not strictly needed for reconstruction if pre_t exists
            img_t = images[t].to(device) if images is not None else None
            mask_t = mask[t].to(device) if mask is not None else torch.ones(B, N, device=device)

            bev_embed, bev_h = self._call_bev_encoder_step(img_t, bev_h)
            slot_emb, slot_h = self._call_slot_gru_step(feat_t, slot_h, bev_embed, mask_t)

            actor_out = self.actor(slot_emb)
            mu = actor_out["mu"]
            log_std = torch.clamp(actor_out["log_std"], LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)

            # --- [Modification 3] Core Fix Logic ---
            if pre_t is not None:
                # Plan A: Use the stored pre_t directly from buffer (Most robust)
                pre_t_step = pre_t[t]
            else:
                # Plan B: Fallback to reconstruction if pre_t is missing
                act_t = actions[t].to(device)
                A_dim = mu.shape[-1]
                tanh_pre = torch.zeros((B, N, A_dim), device=device, dtype=mu.dtype)

                # Inverse Tanh Logic
                tanh_pre[..., 0] = act_t[..., 0] * 2.0 - 1.0
                if A_dim > 1:
                    tanh_pre[..., 1] = act_t[..., 1] / (action_scale_t + eps)

                # Clamp to prevent NaN
                tanh_pre = torch.clamp(tanh_pre, -1.0 + eps, 1.0 - eps)
                pre_t_step = 0.5 * (torch.log1p(tanh_pre) - torch.log1p(-tanh_pre))

            # Calculate LogP
            logp = self._call_logprob(pre_t_step, mu, log_std, action_scale=action_scale_t)

            dist = torch.distributions.Normal(mu, std)
            entropy = dist.entropy().sum(dim=-1)

            # Critic
            bev_ctx = self.bev_to_ctx(bev_embed)
            bev_ctx_exp = bev_ctx.unsqueeze(1).expand(-1, N, -1)
            critic_input = torch.cat([slot_emb, bev_ctx_exp], dim=-1)
            values = self.critic(critic_input)
            if values.dim() == 3:
                values = values.squeeze(-1)

            entropy = entropy * mask_t
            values = values * mask_t

            logp_seq.append(logp.unsqueeze(0))
            value_seq.append(values.unsqueeze(0))
            entropy_seq.append(entropy.unsqueeze(0))
            logstd_seq.append(log_std.unsqueeze(0))

        # Stack output
        logp_out = torch.cat(logp_seq, dim=0)
        value_out = torch.cat(value_seq, dim=0)
        entropy_out = torch.cat(entropy_seq, dim=0)
        logstd_out = torch.cat(logstd_seq, dim=0)

        if is_step:
            logp_out = logp_out.squeeze(0)
            value_out = value_out.squeeze(0)
            entropy_out = entropy_out.squeeze(0)
            logstd_out = logstd_out.squeeze(0)

        return {
            "log_probs": logp_out,
            "values": value_out,
            "entropy": entropy_out,
            "log_std": logstd_out
        }

    def compute_slot_features(self, agent_feats):
        """
        Robust slot feature computation.

        Input:
          agent_feats: tensor in shape [B, N, F] OR [B*N, F] OR [B, F] (interpreted as N=1)
        Output:
          slot_emb: tensor [B, N, E] (E = self.slot_emb_dim or fallback 128)
        Behavior:
          - Will try to use self.slot_mlp / self.slot_encoder if present.
          - Otherwise create a lightweight fallback MLP stored as self._slot_mlp.
        """
        if agent_feats is None:
            raise ValueError("compute_slot_features: agent_feats is None")

        # ensure tensor
        if not torch.is_tensor(agent_feats):
            agent_feats = torch.as_tensor(agent_feats)

        # infer shapes
        if agent_feats.dim() == 3:
            B, N, F = agent_feats.shape
            flat = False
            flat_feats = agent_feats.view(B * N, F)
        elif agent_feats.dim() == 2:
            # could be [B*N, F] or [B, F] (N==1)
            # try to guess: if self has known slot count, use it; else assume N==1
            F = agent_feats.shape[1]
            guessed_N = getattr(self, "slot_count", None) or getattr(self, "num_agents", None) or getattr(self, "N",
                                                                                                          None)
            if guessed_N and agent_feats.shape[0] % guessed_N == 0:
                B = agent_feats.shape[0] // guessed_N
                N = guessed_N
                flat = True
                flat_feats = agent_feats.view(B * N, F)
            else:
                # treat as [B, F] => single slot
                B = agent_feats.shape[0]
                N = 1
                flat = False
                flat_feats = agent_feats.view(B * N, F)
                agent_feats = flat_feats.view(B, N, F)
        else:
            raise RuntimeError(f"compute_slot_features: unsupported agent_feats dim {agent_feats.dim()}")

        device = agent_feats.device

        # determine embedding dim
        slot_emb_dim = getattr(self, "slot_emb_dim", None)
        if slot_emb_dim is None:
            # try common names
            slot_emb_dim = getattr(self, "slot_h_dim", None) or getattr(self, "slot_hidden_dim", None) or 128
        slot_emb_dim = int(slot_emb_dim)

        # try to use existing encoder if available
        encoder = None
        for cand in ("slot_mlp", "slot_encoder", "slot_proj", "slot_net"):
            if hasattr(self, cand):
                encoder = getattr(self, cand)
                break

        # If no encoder module, create a fallback and attach to self (so parameters persist)
        if encoder is None:
            if not hasattr(self, "_slot_mlp"):
                # small MLP fallback
                in_dim = int(flat_feats.shape[1])
                hidden = max(64, min(256, in_dim * 2))
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, slot_emb_dim)
                )
                # move to correct device and dtype
                mlp.to(device)
                self._slot_mlp = mlp
            encoder = getattr(self, "_slot_mlp")

        # If encoder expects [B,N,F] vs [B*N, F] - handle both
        try:
            # preferred: pass flat_feats [B*N, F]
            slot_emb_flat = encoder(flat_feats)
            # ensure it's a tensor
            if not torch.is_tensor(slot_emb_flat):
                slot_emb_flat = torch.as_tensor(slot_emb_flat, device=device)
        except Exception:
            # try passing [B, N, F] if encoder can accept 3D
            try:
                slot_emb = encoder(agent_feats)  # expect [B,N,E]
                if slot_emb.dim() == 2 and slot_emb.shape[0] == B * N:
                    slot_emb = slot_emb.view(B, N, -1)
                elif slot_emb.dim() == 3 and slot_emb.shape[0] == B and slot_emb.shape[1] == N:
                    pass
                else:
                    # try to flatten and re-run
                    flat_feats2 = agent_feats.view(B * N, agent_feats.shape[-1])
                    slot_emb_flat = encoder(flat_feats2)
                    if slot_emb_flat.dim() == 2:
                        slot_emb = slot_emb_flat.view(B, N, -1)
                    else:
                        # last resort: try squeeze/reshape
                        slot_emb = slot_emb
            except Exception as e:
                # propagate informative error
                raise RuntimeError(f"compute_slot_features failed to run encoder: {e}")

        # if we got flat result, reshape to [B,N,E]
        if 'slot_emb_flat' in locals():
            se = slot_emb_flat
            if se.dim() == 2 and se.shape[0] == B * N:
                slot_emb = se.view(B, N, se.shape[1])
            elif se.dim() == 3 and se.shape[0] == B and se.shape[1] == N:
                slot_emb = se
            else:
                # try to coerce
                try:
                    slot_emb = se.view(B, N, -1)
                except Exception:
                    raise RuntimeError(f"compute_slot_features: unexpected encoder output shape {tuple(se.shape)}")
        # ensure final shape
        if slot_emb.dim() != 3:
            raise RuntimeError(f"compute_slot_features: final slot_emb dim unexpected {slot_emb.dim()}")

        return slot_emb

    def _actor_head_fallback(self, slot_emb):
        """
        Produce (mu, log_std) from slot_emb.
        slot_emb: [B, N, E] or [B*N, E]
        returns mu, log_std both shaped [B, N, A]
        """
        if slot_emb.dim() == 3:
            B, N, E = slot_emb.shape
            flat = False
            flat_in = slot_emb.view(B * N, E)
        elif slot_emb.dim() == 2:
            flat_in = slot_emb
            B = None;
            N = None
        else:
            raise RuntimeError("actor_head_fallback: unsupported slot_emb dim")

        # get action dim if known
        A = getattr(self, "action_dim", None) or getattr(self, "act_dim", None) or 2
        A = int(A)

        # create fallback heads if missing
        if not hasattr(self, "_actor_mu_head"):
            hidden = max(64, min(256, flat_in.shape[1] * 2))
            self._actor_mu_head = nn.Sequential(nn.Linear(flat_in.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, A))
            # logstd head
            self._actor_logstd = nn.Parameter(torch.zeros((1, A), device=flat_in.device))
        mu_flat = self._actor_mu_head(flat_in)  # [B*N, A] or [B, A]
        # broadcast logstd
        logstd = self._actor_logstd.expand(mu_flat.shape[0], -1)

        if slot_emb.dim() == 3:
            mu = mu_flat.view(B, N, A)
            log_std = logstd.view(B, N, A)
        else:
            mu = mu_flat
            log_std = logstd
        return mu, log_std

    def _critic_head_fallback(self, slot_emb):
        """
        Produce values [B,N] (or [B*N] flattened) from slot_emb.
        """
        if slot_emb.dim() == 3:
            B, N, E = slot_emb.shape
            flat = True
            flat_in = slot_emb.view(B * N, E)
        elif slot_emb.dim() == 2:
            flat_in = slot_emb
        else:
            raise RuntimeError("critic_head_fallback: unsupported slot_emb dim")

        if not hasattr(self, "_critic_head_fc"):
            hidden = max(64, min(256, flat_in.shape[1] * 2))
            self._critic_head_fc = nn.Sequential(nn.Linear(flat_in.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self._critic_head_fc.to(flat_in.device)

        v_flat = self._critic_head_fc(flat_in)  # [B*N, 1] or [B,1]
        # reshape back
        if slot_emb.dim() == 3:
            values = v_flat.view(B, N)
        else:
            values = v_flat.squeeze(-1)
        return values



