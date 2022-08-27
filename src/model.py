from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class XYModel:
    class InitialState(Enum):
        ALL_UP = 0
        RANDOM = 1

    def __init__(
        self, N: int, T: float = 1.0, initial_state: InitialState = InitialState.ALL_UP,
        seed: Optional[int] = None
    ) -> None:

        self._N: int = N
        self._N2: int = N*N
        self.T: float = T

        self._rgen: np.random.Generator = np.random.Generator(np.random.PCG64(seed))

        # Initialise the model
        init = self.InitialState(initial_state)

        self._state = np.zeros((N * N, 2))
        
        if init == self.InitialState.ALL_UP:
            self._state[:,1] = 1.0
        if init == self.InitialState.RANDOM:
            phi = self._rgen.random(N*N)*2*np.pi
            self._state[:,0] = np.cos(phi)
            self._state[:,1] = np.sin(phi)

        # Adjacency map
        nrange = np.arange(N)

        rowc = np.array(list(zip(nrange, np.roll(nrange, -1))))
        colc = np.array(list(zip(nrange, nrange + N)))

        allrowc = np.tile(rowc, reps=(N, 1)) + np.repeat(nrange * N, N)[:, None]
        allcolc = (np.tile(colc, reps=(N, 1)) + np.repeat(nrange * N, N)[:, None]) % (
            N * N
        )

        self._adjmap = np.concatenate([allrowc, allcolc], axis=0)

        # Neighbour list
        self._neighs: List[np.ndarray] = []
        for i in range(N*N):
            ibonds = np.where(self._adjmap == i)
            self._neighs.append(self._adjmap[ibonds[0], 1-ibonds[1]])

    @property
    def N(self) -> int:
        return self._N

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def bonds(self) -> np.ndarray:
        return self._adjmap.copy()

    @property
    def M(self) -> np.ndarray:
        return np.average(self._state[:,1])

    @property
    def E(self) -> np.ndarray:
        s = self._state
        return -np.sum(np.prod(s[self._adjmap], axis=1))
    
    def i2xy(self, i: int) -> Tuple[int, int]:
        return (i%self.N, i//self.N)
    
    def xy2i(self, x: int, y: int) -> int:
        return x+y*self.N
    
    def neighbours(self, i: int) -> np.ndarray:
        return self._neighs[i].copy()
    
    def _proposed_DE(self, i: int, s: Tuple[float, float]) -> float:
        s0 = self._state[i]
        # Average field?
        f = np.sum(self._state[self._neighs[i]], axis=0)
        E0 = f@s0
        E1 = f@s
        return E1-E0
    
    def metropolis_step(self, dphisigma: float = 0.1):
        # Pick a random spin
        i = self._rgen.integers(0, self._N2)
        # Pick an angle shift
        dphi = self._rgen.normal(scale=dphisigma)

        # Current state
        s0 = self._state[i]
        # New state?
        c = np.cos(dphi)
        s = np.sin(dphi)
        s = np.array([c*s0[0]+s*s0[1], c*s0[1]-s*s0[0]])

        DE = self._proposed_DE(i, s)

        if DE < 0 or self._rgen.random() < np.exp(-DE/self.T):
            # Accept
            self._state[i] = s
        


    


