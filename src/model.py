from enum import Enum
import numpy as np


class XYModel:
    class InitialState(Enum):
        ALL_UP = 0
        RANDOM = 1

    def __init__(
        self, N: int, T: float = 1.0, initial_state: InitialState = InitialState.ALL_UP
    ) -> None:

        self._N: int = N
        self.T: float = T

        # Initialise the model
        init = self.InitialState(initial_state)

        self._state = np.zeros(N * N)

        if init == self.InitialState.RANDOM:
            self._state = np.random.random(N * N) * 2 * np.pi

        # Adjacency map
        nrange = np.arange(N)

        rowc = np.array(list(zip(nrange, np.roll(nrange, -1))))
        colc = np.array(list(zip(nrange, nrange + N)))

        allrowc = np.tile(rowc, reps=(N, 1)) + np.repeat(nrange * N, N)[:, None]
        allcolc = (np.tile(colc, reps=(N, 1)) + np.repeat(nrange * N, N)[:, None]) % (
            N * N
        )

        self._adjmap = np.concatenate([allrowc, allcolc], axis=0)

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
        return np.average(np.cos(self._state))

    @property
    def E(self) -> np.ndarray:
        s = self._state
        return -np.sum(np.cos(np.diff(s[self._adjmap], axis=1)))
