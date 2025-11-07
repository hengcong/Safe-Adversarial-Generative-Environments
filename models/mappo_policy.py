# models/mappo_policy.py
from typing import Optional, Tuple
import torch
import torch.nn as nn

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
                 act_dim_vehicle: int,
                 act_dim_ped: int,

                 # basic MAPPO elements
                 type_vocab_size: int = 2,
                 type_emb_dim: int = 8,
                 hidden_dim: int = 256,
                 recurrent_hidden_dim: int = 256,
                 use_bev_gru: bool = True,
                 use_slot_gru: bool = True,
                 global_ctx_dim: int = 256,

                 # algorithmic / training controls
                 action_scale: float = 1.0,
                 log_std_init: float = -0.5,

                 # misc / engineering
                 device: str = "cpu") -> None:
        super().__init__()

        # -------------------------
        # Save core hyperparams
        # -------------------------
        self.device = torch.device(device)
        # action scale and initial log_std are algorithmic knobs (trainer controls these)
        self.action_scale = float(action_scale)
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
