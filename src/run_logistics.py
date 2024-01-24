import os
from time import sleep

from src.logistics import Logistics

DELAY = 0.2

env = Logistics(
    n_rows=3,
    n_cols=2,
    palette_types=2,
    prob_loading=0.1,
    prob_unloading=0.2,
    n_steps=100,
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
        env.reset()
    sleep(DELAY)
    os.system("clear")
    step += 1
