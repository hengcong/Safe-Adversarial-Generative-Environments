from envs.carla_env import CarlaEnv

# 测试参数
NUM_RESETS = 50
STEPS_PER_EPISODE = 200

env = CarlaEnv(num_veh=30, num_ped=10, mode="MAPPO")

for episode in range(NUM_RESETS):
    print(f"\n========== Episode {episode} ==========")
    try:
        obs = env.reset()
    except Exception as e:
        print(f"[ERROR] reset failed at episode {episode}: {e}")
        break

    for t in range(STEPS_PER_EPISODE):
        # 确保 env.agent_ids 是最新的（reset 后才有）
        if not env.agent_ids:
            print("[WARN] no agent_ids, skipping step")
            break

        # 随机动作（每个 agent 一个）
        actions = {
            aid: env.action_space.sample()[aid]
            for aid in env.agent_ids
        }

        try:
            obs, rewards, dones, info = env.step(actions)
        except Exception as e:
            print(f"[ERROR] step failed at episode {episode}, step {t}: {e}")
            break

        if any(dones.values()):
            print(f"Episode {episode} terminated early at step {t}")
            break

print("\n==== Test Completed ====")

