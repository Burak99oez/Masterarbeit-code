import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
obs, _ = env.reset()
print("Start state:", obs)

for _ in range(5):
    obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
    print(obs, r, terminated, truncated)
    if terminated or truncated:
        break

env.close()
