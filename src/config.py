from src.logistics import Logistics


def new_logistics():
    return Logistics(
        n_rows=3,
        n_cols=2,
        palette_types=2,
        prob_loading=0.1,
        prob_unloading=0.2,
        n_steps=100,
    )
