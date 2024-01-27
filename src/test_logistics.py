import numpy as np
from gymnasium.utils.env_checker import check_env

from src.logistics import SMALL, Logistics, Unloading, move_palette


def test_transfer():
    assert move_palette(1, 0) == (0, 1, True)
    assert move_palette(0, 0) == (0, 0, False)
    assert move_palette(0, 1) == (0, 1, False)
    assert move_palette(1, 1) == (1, 1, False)


def test_unload():
    assert Unloading.unload(1, 0) == (1, 0, False)
    assert Unloading.unload(0, 0) == (0, 0, False)
    assert Unloading.unload(0, 1) == (0, 1, False)
    assert Unloading.unload(1, 1) == (0, 0, True)


def test_check_env():
    env = Logistics()
    check_env(env, skip_render_check=True)


def compute_shaping_reward(unloading: np.ndarray, grid: np.ndarray) -> float:
    """
    Compute the shaping reward for the given unloading state and grid.

    :param unloading: the unloading state
    :param grid: the grid
    :return: the shaping reward
    """
    n_rows, n_cols = grid.shape
    assert (n_cols,) == unloading.shape
    palette_types = max(grid.max(), unloading.max())

    env = Logistics(n_rows=n_rows, n_cols=n_cols, palette_types=palette_types)
    env.unloading.state = unloading
    env.grid = grid
    return env._shaping_reward()


def test_shaping_reward_simple():
    assert (
        compute_shaping_reward(
            np.array([0, 1, 0, 0]),
            np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
        )
        == -1 * SMALL
    )

    assert (
        compute_shaping_reward(
            np.array([0, 1, 0, 0]),
            np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
        )
        == -2 * SMALL
    )

    assert (
        compute_shaping_reward(
            np.array([0, 2, 0, 0]),
            np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
        )
        == -2 * SMALL
    )


def test_shaping_reward_multiple():
    assert (
        compute_shaping_reward(
            np.array([0, 2, 0, 0]),
            np.array([[0, 0, 0, 0], [0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]]),
        )
        == -4 * SMALL
    )

    assert (
        compute_shaping_reward(
            np.array([0, 2, 0, 1]),
            np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
        )
        == -3 * SMALL
    )
