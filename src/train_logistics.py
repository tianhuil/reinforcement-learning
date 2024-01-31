import time
from typing import Dict

from random_word import RandomWords
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.config import N_ENVS, STEPS, Model, new_logistics

timestamp = int(time.time())
timestamp_base64 = f"{timestamp:b}"
r = RandomWords()
id_str = "{}-{}-{}".format(timestamp, r.get_random_word(), r.get_random_word())
log_dir = f"./data/tb/{id_str}"
model_dir = f"./data/model/{id_str}"

env = new_logistics()
check_env(env)
env = Monitor(env, log_dir, allow_early_resets=True)
parameters = env.parameters()


def linear_schedule(initial_value: float, final_value: float) -> float:
    def f(remaining: float) -> float:
        return remaining * initial_value + (1 - remaining) * final_value

    return f


checkpoint_callback = CheckpointCallback(
    save_freq=max(STEPS // N_ENVS // 5, 1), save_path=model_dir
)


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict: Dict[str, str | float] = {
            "algorithm": self.model.__class__.__name__,
            **{
                "optimizer_kwargs_" + key: value
                for key, value in self.model.policy_kwargs["optimizer_kwargs"].items()
            },
            **parameters,
        }
        if isinstance(
            self.model.learning_rate, float
        ):  # Can also be Schedule, in that case, we don't report
            hparam_dict["learning rate"] = self.model.learning_rate
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict: Dict[str, float] = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv", "optimizer_class"),
        )

    def _on_step(self) -> bool:
        return True


if __name__ == "__main__":
    env = make_vec_env(new_logistics, n_envs=N_ENVS, seed=42, vec_env_cls=SubprocVecEnv)

    model = Model(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-3,
    ).learn(
        STEPS,
        callback=[checkpoint_callback, HParamCallback()],
    )

    print(f"Model saved under {model_dir}")
