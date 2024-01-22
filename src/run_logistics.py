import os
from time import sleep

from src.logistics import Logistics

DELAY = 0.5

env = Logistics(
    n_steps=20,
    n_rows=4,
    n_cols=4,
    palette_types=4,
    prob_loading=0.2,
    prob_unloading=0.2,
)

obs, _ = env.reset()
# Hardcoded best agent: always go left!
step = 0
while True:
    print(env.observation_space)
    print(env.action_space)
    action = env.action_space.sample()
    print(action)

    print(f"Step {step + 1}")
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print("reward=", reward, "done=", done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break
    sleep(DELAY)
    os.system("clear")
    step += 1
