import os
from time import sleep

from src.config import Model, new_logistics

DELAY = 0.2


def run_logistics_with_model(model_dir: str | None):
    env = new_logistics()

    obs, _ = env.reset()
    step = 0

    model = (
        Model.load(model_dir)
        if model_dir
        else Model("MultiInputPolicy", env, verbose=1)
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
