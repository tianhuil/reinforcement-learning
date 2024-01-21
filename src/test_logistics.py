from gymnasium.utils.env_checker import check_env

from src.logistics import Logistics, Unloading, move_palette


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
