import numpy as np
from ..model import XYModel


def test_init():
    m = XYModel(N=2)

    assert (m.state == np.zeros(4)).all()
    assert (
        m.bonds == [[0, 1], [1, 0], [2, 3], [3, 2], [0, 2], [1, 3], [2, 0], [3, 1]]
    ).all()

    assert m.M == 1
    assert m.E == -8
