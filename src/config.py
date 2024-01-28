from stable_baselines3 import A2C

from src.logistics import Logistics

STEPS = 30_000_000
N_ENVS = 32


def new_logistics():
    return Logistics(
        n_rows=4,
        n_cols=4,
        palette_types=2,
        prob_loading=0.2,
        prob_unloading=0.3,
        n_steps=50,
        early_termination=False,
    )


Model = A2C
