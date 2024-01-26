from stable_baselines3 import A2C

from src.logistics import Logistics


def new_logistics():
    return Logistics(
        n_rows=3,
        n_cols=2,
        palette_types=2,
        prob_loading=0.2,
        prob_unloading=0.3,
        n_steps=30,
    )


Model = A2C
