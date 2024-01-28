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


def build_env(unloading: np.ndarray, grid: np.ndarray) -> Logistics:
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
    return env


def test_shaping_reward_unload_simple():
    env = build_env(
        np.array([0, 1, 0, 0]),
        np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
    )
    assert env._shaping_reward_unloading() == -1 * SMALL

    env = build_env(
        np.array([0, 1, 0, 0]),
        np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
    )
    assert env._shaping_reward_unloading() == -2 * SMALL

    env = build_env(
        np.array([0, 2, 0, 0]),
        np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
    )
    assert env._shaping_reward_unloading() == -2 * SMALL


def test_shaping_reward_unload_multiple():
    env = build_env(
        np.array([0, 2, 0, 0]),
        np.array([[0, 0, 0, 0], [0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]]),
    )
    assert env._shaping_reward_unloading() == -4 * SMALL

    env = build_env(
        np.array([0, 2, 0, 1]),
        np.array([[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
    )
    assert env._shaping_reward_unloading() == -3 * SMALL


def test_shaping_reward_loading():
    env = build_env(
        np.array([0, 0, 0, 0]),
        np.array([[0, 0, 0, 0], [0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]]),
    )
    assert env._shaping_reward_loading() == 0.0

    env = build_env(
        np.array([0, 0, 0, 0]),
        np.array([[1, 0, 0, 0], [0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]]),
    )
    assert env._shaping_reward_loading() == -1 * SMALL

    env = build_env(
        np.array([0, 0, 0, 0]),
        np.array([[0, 1, 0, 1], [0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 1, 0]]),
    )
    assert env._shaping_reward_loading() == -2 * SMALL
