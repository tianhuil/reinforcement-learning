import time

from random_word import RandomWords
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.config import Model, new_logistics

STEPS = 1_00_000

timestamp = int(time.time())
timestamp_base64 = f"{timestamp:b}"
r = RandomWords()
id_str = "{}-{}-{}".format(timestamp, r.get_random_word(), r.get_random_word())
log_dir = f"./data/tb/{id_str}"
model_dir = f"./data/model/{id_str}"

env = new_logistics()
check_env(env)
env = Monitor(env, log_dir, allow_early_resets=True)


def linear_schedule(initial_value: float, final_value: float) -> float:
    def f(remaining: float) -> float:
        return remaining * initial_value + (1 - remaining) * final_value

    return f


checkpoint_callback = CheckpointCallback(save_freq=STEPS // 5, save_path=model_dir)

model = Model(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=linear_schedule(1e-3, 1e-5),
).learn(
    STEPS,
    callback=checkpoint_callback,
)
print(f"Model saved under {model_dir}")
