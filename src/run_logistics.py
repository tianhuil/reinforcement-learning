import os
from time import sleep

from stable_baselines3 import A2C

from src.logistics import Logistics

DELAY = 0.2


def run_logistics_with_model(model_dir: str | None):
    env = Logistics(
        n_rows=3,
        n_cols=2,
        palette_types=2,
        prob_loading=0.1,
        prob_unloading=0.2,
        n_steps=100,
    )

    obs, _ = env.reset()
    step = 0

    model = (
        A2C.load(model_dir) if model_dir else A2C("MultiInputPolicy", env, verbose=1)
    )

    while True:
        print(env.observation_space)
        print(env.action_space)
        action, _ = model.predict(obs, deterministic=True)
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


if __name__ == "__main__":
    run_logistics_with_model(None)
