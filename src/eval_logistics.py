from stable_baselines3.common.evaluation import evaluate_policy

from src.config import Model, new_logistics

DELAY = 0.2


def eval_logistics_with_model(model_dir: str | None, n_steps: int = 100):
    env = new_logistics()

    obs, _ = env.reset()

    model = (
        Model.load(model_dir)
        if model_dir
        else Model("MultiInputPolicy", env, verbose=1)
    )

    ep_rew_mean, ep_rew_std = evaluate_policy(
        model, env, n_eval_episodes=n_steps, render=False
    )
    print("Mean Reward = {:.2f} +/- {:.2f}".format(ep_rew_mean, ep_rew_std))
