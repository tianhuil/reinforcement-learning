import os
from time import sleep

from src.logistics import Logistics

DELAY = 0.5

env = Logistics(
    n_rows=4,
    n_cols=4,
    palette_types=4,
    prob_loading=0.2,
    prob_unloading=0.2,
)

obs, _ = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

# Hardcoded best agent: always go left!
n_steps = 20
for step in range(n_steps):
    print(f"Step {step + 1}")
    obs, reward, terminated, truncated, info = env.step((2, 2, 0))
    done = terminated or truncated
    print("reward=", reward, "done=", done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break
    sleep(DELAY)
    os.system("clear")
