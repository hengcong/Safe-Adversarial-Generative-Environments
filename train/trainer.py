
import time
import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any


class Trainer(ABC):
    """
    Generic Trainer base class.
    Subclasses must implement:
      - collect_rollout()
      - finish_and_update()
    Trainer provides:
      - epoch/train loop
      - logging hook
      - checkpoint save hook
    """

    def __init__(self,
                 envs,
                 manager,
                 device: str = "cpu",
                 log_interval: int = 1,
                 checkpoint_dir: Optional[str] = None,
                 save_fn: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Args:
            envs: environment manager (vectorized or list-like). Expected methods:
                  - reset()
                  - step(env_idx, action) OR step(actions_batch) if vectorized
                  - optionally get_obs_all() or get_last_obs_all()
            manager: MAPPOManager-like object with methods used by subclass
            device: device string
            log_interval: epochs between prints
            checkpoint_dir: directory where to save checkpoints
            save_fn: optional user-supplied function(state_dict) for saving
        """
        self.envs = envs
        self.manager = manager
        self.device = device
        self.log_interval = int(log_interval)
        self.checkpoint_dir = checkpoint_dir
        self.save_fn = save_fn

        self.epoch = 0
        self.global_step = 0
        self.start_time = time.time()

    @abstractmethod
    def collect_rollout(self) -> None:
        """Collect rollout(s) and store into manager / buffers. Must update self.global_step appropriately."""
        raise NotImplementedError

    @abstractmethod
    def finish_and_update(self) -> None:
        """Finish rollouts (compute GAE) and update policies."""
        raise NotImplementedError

    def save_checkpoint(self, name: Optional[str] = None) -> None:
        """Save a checkpoint with epoch/global_step and manager state if available."""
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        # attempt to get manager state
        try:
            if hasattr(self.manager, "state_dict"):
                state["manager_state"] = self.manager.state_dict()
        except Exception:
            pass

        if self.save_fn is not None:
            try:
                self.save_fn(state)
            except Exception:
                # do not crash training on save error
                pass

        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            name = name or f"checkpoint_epoch{self.epoch}_step{self.global_step}.pt"
            path = os.path.join(self.checkpoint_dir, name)
            try:
                # manager may implement its own save
                if hasattr(self.manager, "save"):
                    self.manager.save(path.replace(".pt", ""))
                else:
                    import torch
                    torch.save(state, path)
            except Exception:
                pass

    def log(self, message: str) -> None:
        elapsed = time.time() - self.start_time
        print(f"[Epoch {self.epoch} Step {self.global_step}] {message} (elapsed {elapsed:.1f}s)")

    def train(self, num_epochs: int = 1000, print_fn=print, save_every: Optional[int] = None) -> None:
        """
        Run training loop.
        Args:
            num_epochs: number of outer epochs (each epoch does collect + update)
            print_fn: function used to print logs
            save_every: save checkpoint every N epochs (if checkpoint_dir set)
        """
        for _ in range(int(num_epochs)):
            t0 = time.time()
            self.collect_rollout()
            self.finish_and_update()
            self.epoch += 1
            epoch_time = time.time() - t0

            if (self.epoch % self.log_interval) == 0:
                print_fn(f"[Epoch {self.epoch}] global_step={self.global_step} epoch_time={epoch_time:.3f}s")

            if save_every is not None and self.checkpoint_dir is not None and (self.epoch % save_every) == 0:
                self.save_checkpoint()

        # final save
        if self.checkpoint_dir is not None:
            self.save_checkpoint(f"final_epoch{self.epoch}_step{self.global_step}.pt")
