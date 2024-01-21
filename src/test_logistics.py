from src.logistics import Logistics


def test_transfer():
    assert Logistics.transfer(1, 0) == (0, 1, True)
    assert Logistics.transfer(0, 0) == (0, 0, False)
    assert Logistics.transfer(0, 1) == (0, 1, False)
    assert Logistics.transfer(1, 1) == (1, 1, False)


def test_unload():
    assert Logistics.unload(1, 0) == (1, 0, False)
    assert Logistics.unload(0, 0) == (0, 0, False)
    assert Logistics.unload(0, 1) == (0, 1, False)
    assert Logistics.unload(1, 1) == (0, 0, True)
