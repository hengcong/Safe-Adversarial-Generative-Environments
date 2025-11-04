import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Normal

from algo.mappo.rollout_buffer import RolloutBuffer

class MAPPOAgent(object):
    def __init__(self, sb3_policy:ActorCriticPolicy):
        super().__init__()
        self.policy = sb3_policy
        self.features_extractor = self.policy.features_extractor
        self.mlp_extractor = self.policy.mlp_extractor
        self.action_net = self.policy.action_net
        self.value_net = self.policy.value_net
        # try to find log_std parameter in SB3 policy; otherwise create one
        if hasattr(self.policy, "log_std"):
            self.log_std = self.policy.log_std
        else:
            found = None
            for n, p in self.policy.named_parameters():
                if "log_std" in n:
                    found = p
                    break
            if found is not None:
                self.log_std = found
            else:
                # infer action dim from action_net weights
                try:
                    act_dim = self.action_net.out_features
                except Exception:
                    act_dim = list(self.action_net.parameters())[0].shape[0]
                self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
                # register param on policy module so save/load works
                self.policy.register_parameter("log_std_extra", self.log_std)

        # device alignment
        try:
            self.device = next(self.policy.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        self.policy.to(self.device)
        # optimizer for training (can be replaced externally)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # PPO hyperparams (tune as needed)
        self.clip_eps = 0.2
        self.epochs = 10
        self.batch_size = 64
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.max_grad_norm = 0.5

        # runtime placeholders
        self.last_obs = None
        self.rnn_state = None

    # ---------- core forward using SB3 internals ----------
    def _forward(self, obs_tensor: torch.Tensor):
        """
        obs_tensor: (B, *obs_shape) on correct device
        returns: mean(B, act_dim), std(B, act_dim), value(B,)
        """
        features = self.features_extractor(obs_tensor)
        pi_latent, vf_latent = self.mlp_extractor(features)
        mean = self.action_net(pi_latent)
        logstd = self.log_std
        if logstd.dim() == 1:
            std = torch.exp(logstd).unsqueeze(0).expand_as(mean)
        else:
            std = torch.exp(logstd).expand_as(mean)
        value = self.value_net(vf_latent).squeeze(-1)
        return mean, std, value

    # ---------- action selection ----------
    def select_action(self, obs_np: np.ndarray):
        """
        obs_np: single-observation numpy array (obs_dim,) or batch (first entry used)
        returns: action (numpy), value (float), logp (float)
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            mean, std, val = self._forward(obs_t)
            dist = Normal(mean, std)
            act_t = dist.sample()
            logp_t = dist.log_prob(act_t).sum(-1)
        action = act_t.cpu().numpy().squeeze(0)
        return action, float(val.cpu().item()), float(logp_t.cpu().item())

    # ---------- training (manager provides data) ----------
    def update_from_data(self, data: Dict[str, torch.Tensor]) -> None:
        """
        data should be tensors already on self.device with keys:
          'obs', 'acts', 'advs', 'rets', 'logps', 'vals'
        Shapes: (N, ...)
        """
        obs = data["obs"].to(self.device)
        acts = data["acts"].to(self.device)
        advs = data["advs"].to(self.device)
        rets = data["rets"].to(self.device)
        oldlogp = data["logps"].to(self.device)

        N = obs.shape[0]
        for _ in range(self.epochs):
            idxs = torch.randperm(N, device=self.device)
            for start in range(0, N, self.batch_size):
                mb = idxs[start: start + self.batch_size]
                mb_obs = obs[mb]
                mb_acts = acts[mb]
                mb_advs = advs[mb]
                mb_rets = rets[mb]
                mb_oldlogp = oldlogp[mb]

                mean, std, vals = self._forward(mb_obs)
                dist = Normal(mean, std)
                newlogp = dist.log_prob(mb_acts).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (newlogp - mb_oldlogp).exp()
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advs
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_v = 0.5 * ((vals - mb_rets) ** 2).mean()
                loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    # ---------- reset ----------
    def mappo_reset(self, initial_obs: Any = None) -> None:
        """
        Called at environment reset to clear any runtime state.
        Manager typically calls this with the initial obs dict value for the agent.
        """
        self.last_obs = None if initial_obs is None else np.array(initial_obs, dtype=np.float32)
        if self.rnn_state is not None:
            self.rnn_state = torch.zeros_like(self.rnn_state)

    # ---------- save / load ----------
    def save(self, path_prefix: str) -> None:
        torch.save(self.policy.state_dict(), f"{path_prefix}_policy.pth")
        torch.save(self.optimizer.state_dict(), f"{path_prefix}_optim.pth")

    def load(self, path_prefix: str, map_location=None) -> None:
        self.policy.load_state_dict(torch.load(f"{path_prefix}_policy.pth", map_location=map_location))
        opt_state = torch.load(f"{path_prefix}_optim.pth", map_location=map_location)
        self.optimizer.load_state_dict(opt_state)

