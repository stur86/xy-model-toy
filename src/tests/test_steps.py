import numpy as np

from ..model import XYModel

def test_metropolis():
    
    m = XYModel(10, 1, seed=1)
    m.metropolis_step()

    assert m.state[47][1] == np.cos(0.08216181435011584)
    assert m.state[47][0] == np.sin(0.08216181435011584)

def test_wolff():

    m = XYModel(10, 1, XYModel.InitialState.RANDOM, seed=1)

    m.wolff_step()

    assert m.E == 9.48755294586184
