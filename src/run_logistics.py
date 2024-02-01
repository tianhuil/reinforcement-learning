import os
from time import sleep

from src.config import Model, new_logistics

DELAY = 0.2


def run_logistics_with_model(model_dir: str | None, input_advance: bool = True):
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
        action = (
            model.predict(obs, deterministic=True)[0]
            if model_dir
            else model.action_space.sample()
        )

        print(f"Step {step + 1}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(
            "action={}".format(
                env.render_action(action),
            ),
            "reward={: 1.6f}".format(reward),
            "done={}".format(done),
        )

        print("")
        env.render()
        print("")

        if done:
            env.reset()

        if input_advance:
            input("Press any key to continue...")
        else:
            sleep(DELAY)

        os.system("clear")
        step += 1


if __name__ == "__main__":
    run_logistics_with_model(None)
