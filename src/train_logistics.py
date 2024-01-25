import time

from random_word import RandomWords
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.config import new_logistics

r = RandomWords()
id_str = "{}-{}".format(r.get_random_word(), r.get_random_word())

env = new_logistics()
check_env(env)
log_dir = "/tmp/gym/{}".format(int(time.time()))
env = Monitor(env, log_dir, allow_early_resets=True)

model = A2C(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=f"./data/tb/{id_str}",
    learning_rate=2e-3,
).learn(200_000)
model.save(f"./data/model/{id_str}")
print(f"Model saved under ./data/model/{id_str}")
