from envs.carla_env import CarlaEnv
from algo.mappo.mappo_manager import MAPPOManager
from train.trainer_mappo import MAPPOTrainer
from models.mappo_policy import MAPPOPolicy

if __name__ == "__main__":
    env = CarlaEnv(num_veh=4, num_ped=0)

    agent_specs = {
        "veh_0": dict(obs_dim=64, act_dim=2, buffer_T=128, n_agents=4)
    }

    def policy_ctor(spec):
        return MAPPOPolicy(
            bev_in_ch=3,
            obs_dim=spec["obs_dim"],
            act_dim_vehicle=2,
            act_dim_ped=2,
            device="cuda"
        )

    manager = MAPPOManager(agent_specs=agent_specs, policy_ctor=policy_ctor, device="cuda")
    trainer = MAPPOTrainer(envs=env, manager=manager, num_steps=128, device="cuda")

    trainer.train(num_epochs=1000)
