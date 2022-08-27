import numpy as np
from ..model import XYModel


def test_init():
    m = XYModel(N=2)

    assert (m.state[:,0] == np.zeros(4)).all()
    assert (m.state[:,1] == np.ones(4)).all()
    assert (
        m.bonds == [[0, 1], [1, 0], [2, 3], [3, 2], [0, 2], [1, 3], [2, 0], [3, 1]]
    ).all()

    assert m.M == 1
    assert m.E == -8

    assert m.i2xy(2) == (0, 1)
    assert m.xy2i(1, 1) == 3

    assert (m.neighbours(0) == [1, 1, 2, 2]).all()

def test_steps():

    m = XYModel(N=10)  

    assert m._proposed_DE(0, [1, 0]) == 4.0