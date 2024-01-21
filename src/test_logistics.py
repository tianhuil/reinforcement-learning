from src.logistics import Unloading, move_palette


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
